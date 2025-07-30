"""
Drug name to ChEMBL ID resolver using hybrid approach with LLM normalization
"""
import re
import json
import logging
import os
from typing import Optional, Dict, List
from .http_client import RateLimitedClient
from .cache import APICache

# Try to import OpenAI (optional dependency)
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)


class DrugResolver:
    """Resolves drug names to ChEMBL IDs using multiple strategies"""
    
    def __init__(self, http_client: RateLimitedClient = None, cache: APICache = None):
        self.http = http_client or RateLimitedClient()
        self.cache = cache or APICache()
        self.has_openai = HAS_OPENAI and os.getenv('OPENAI_API_KEY')
        
    # Regex patterns for aggressive cleaning
    DOSE_ADMIN_RGX = re.compile(
        r"(\b\d+(\.\d+)?\s*(mg(?:/m2)?|mcg|µg|g|%|ml|units?|iu)\b)|"  # doses
        r"(\biv\b|\bsc\b|\bim\b|\bpo\b|intravenous|oral|subcutaneous|intramuscular|topical)|"  # admin routes
        r"(- iv|- sc|- oral|- po)",  # dash-separated admin routes
        re.I,
    )
    
    # Non-drugs to short-circuit immediately
    NON_DRUGS = {
        "placebo", "saline", "vehicle", "sham control", "saline solution",
        "normal saline", "control", "dummy", "inactive"
    }
    
    def _clean_drug_name(self, name: str) -> Optional[str]:
        """Clean drug name for better matching, return None for non-drugs"""
        if not name or not name.strip():
            return None
            
        cleaned = name.strip().lower()
        
        # Short-circuit obvious non-drugs
        for non_drug in self.NON_DRUGS:
            if non_drug in cleaned:
                return None
        
        # Handle imaging tracers separately (optional: could map via PubChem only)
        if re.search(r'\[[0-9]+[a-z]\]', cleaned):  # e.g., [123I], [18F]
            logger.debug(f"Skipping imaging tracer: {name}")
            return None
            
        # Apply aggressive dose/admin cleaning
        cleaned = self.DOSE_ADMIN_RGX.sub("", cleaned)
        
        # Remove brackets and parenthetical content
        cleaned = re.sub(r"[\(\[].*?[\)\]]", "", cleaned)
        
        # Remove common salt suffixes
        suffixes = ['hydrochloride', 'hcl', 'sodium', 'potassium', 'sulfate', 'acetate', 'tartrate', 'citrate']
        for suffix in suffixes:
            cleaned = re.sub(rf'\b{suffix}\b', '', cleaned, flags=re.I)
        
        # Clean up whitespace and normalize
        cleaned = ' '.join(cleaned.split())
        
        # Skip if too short or empty after cleaning
        if not cleaned or len(cleaned) < 2:
            return None
            
        return cleaned
    
    async def _normalize_drug_name_with_llm(self, drug_name: str) -> Optional[str]:
        """Use LLM to normalize drug names (research codes to generic names)"""
        if not self.has_openai:
            return None
            
        # Check cache first
        cache_key = f"llm_normalize:{drug_name.lower()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        prompt = f"""
        You are a pharmaceutical expert. Given a drug intervention string, identify the actual drug name that would be found in databases like ChEMBL.
        
        Convert research codes, formulations, and complex strings to their standard drug names.
        
        Examples:
        "JNJ-54861911 160 mg" → "Atabecestat"
        "LY2886721" → "LY2886721" (if no better name known)
        "Carboplatin injection" → "Carboplatin"
        "Placebo" → "Placebo"
        "TPI-287 20 mg/m2" → "Abeotaxane"
        
        Drug string: "{drug_name}"
        
        Respond with ONLY the normalized drug name, nothing else. If it's clearly not a drug (placebo, saline, etc.), respond with "NOT_A_DRUG".
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a pharmaceutical database expert. Respond with only the drug name, nothing else."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            normalized = response.choices[0].message.content.strip()
            
            # Handle special responses
            if normalized.upper() == "NOT_A_DRUG":
                normalized = None
            
            # Cache the result
            self.cache.set(cache_key, normalized, ttl_hours=48)
            
            if normalized and normalized != drug_name:
                logger.info(f"LLM normalized: '{drug_name}' → '{normalized}'")
            
            return normalized
            
        except Exception as e:
            logger.debug(f"LLM normalization failed for {drug_name}: {e}")
            return None
    
    async def _search_chembl_direct(self, query: str) -> Optional[Dict[str, any]]:
        """Search ChEMBL using multiple endpoints (search is unreliable)"""
        
        # Strategy 1: Try exact synonym match first
        try:
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_synonyms__molecule_synonym__iexact={query}&limit=1"
            response = await self.http.get(url, cache_ttl_hours=48)
            
            if response.get('molecules') and len(response['molecules']) > 0:
                molecule = response['molecules'][0]
                pref_name = molecule.get('pref_name')
                chembl_id = molecule.get('molecule_chembl_id')
                
                logger.debug(f"Found molecule: {chembl_id}, pref_name: '{pref_name}', type: {type(pref_name)}")
                
                # Validate it's not a generic result
                if pref_name and pref_name != 'None' and pref_name != None:
                    logger.debug(f"Validation passed for {chembl_id}")
                    return {
                        'chembl_id': chembl_id,
                        'pref_name': pref_name,
                        'source': 'chembl_synonym_exact'
                    }
                else:
                    logger.debug(f"Validation failed: pref_name='{pref_name}', bool={bool(pref_name)}")
        except Exception as e:
            logger.debug(f"ChEMBL exact synonym search failed for {query}: {e}")
        
        # Strategy 2: Try case-insensitive synonym match
        try:
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?molecule_synonyms__molecule_synonym__icontains={query}&limit=3"
            response = await self.http.get(url, cache_ttl_hours=48)
            
            if response.get('molecules'):
                # Find best match by checking synonyms
                for molecule in response['molecules']:
                    if molecule.get('pref_name') and molecule.get('pref_name') != 'None':
                        # Check if any synonym matches our query closely
                        synonyms = molecule.get('molecule_synonyms', [])
                        for syn in synonyms:
                            syn_name = syn.get('molecule_synonym', '').lower()
                            if query.lower() in syn_name or syn_name in query.lower():
                                return {
                                    'chembl_id': molecule.get('molecule_chembl_id'),
                                    'pref_name': molecule.get('pref_name'),
                                    'source': 'chembl_synonym_contains'
                                }
        except Exception as e:
            logger.debug(f"ChEMBL synonym contains search failed for {query}: {e}")
        
        # Strategy 3: Try the general search endpoint (but validate results)
        try:
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule.json?search={query}&limit=3"
            response = await self.http.get(url, cache_ttl_hours=48)
            
            if response.get('molecules'):
                for molecule in response['molecules']:
                    # Skip generic results without proper names
                    if molecule.get('pref_name') and molecule.get('pref_name') not in ['None', '']:
                        return {
                            'chembl_id': molecule.get('molecule_chembl_id'),
                            'pref_name': molecule.get('pref_name'),
                            'source': 'chembl_search'
                        }
        except Exception as e:
            logger.debug(f"ChEMBL general search failed for {query}: {e}")
        
        return None
    
    async def resolve_to_chembl_id(self, drug_name: str) -> Optional[Dict[str, any]]:
        """Resolve drug name to ChEMBL ID using hybrid approach"""
        
        # Check cache first
        cache_key = f"drug_resolver:{drug_name.lower()}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        # Strategy 1: Try LLM normalization first (if available)
        llm_normalized = await self._normalize_drug_name_with_llm(drug_name)
        if llm_normalized:
            result = await self._search_chembl_direct(llm_normalized)
            if result:
                result['source'] = f"{result['source']}_via_llm"
                self.cache.set(cache_key, result, ttl_hours=48)
                return result
        elif llm_normalized is None and self.has_openai:
            # LLM said it's not a drug
            result = None
            self.cache.set(cache_key, result, ttl_hours=24)
            return result
        
        # Strategy 2: Clean the drug name with regex
        cleaned_name = self._clean_drug_name(drug_name)
        if not cleaned_name:
            # Cache the non-drug result to avoid repeated processing
            result = None
            self.cache.set(cache_key, result, ttl_hours=24)
            return result
        
        # Strategy 3: Try ChEMBL search with cleaned name
        result = await self._search_chembl_direct(cleaned_name)
        if result:
            self.cache.set(cache_key, result, ttl_hours=48)
            return result
        
        # Also try with original (uncleaned) name in case cleaning was too aggressive
        if cleaned_name != drug_name.strip().lower():
            result = await self._search_chembl_direct(drug_name.strip())
            if result:
                self.cache.set(cache_key, result, ttl_hours=48)
                return result
        
        # Strategy 2: Fallback to PubChem → UniChem crosswalk
        pubchem_result = await self._resolve_via_pubchem(cleaned_name)
        if pubchem_result:
            self.cache.set(cache_key, pubchem_result, ttl_hours=48)
            return pubchem_result
        
        # More informative logging for unresolved drugs
        if "placebo" not in drug_name.lower():
            logger.warning(f"Unresolved drug (likely no ChEMBL entry): {drug_name}")
        else:
            logger.debug(f"Non-drug skipped: {drug_name}")
        
        # Cache the negative result
        self.cache.set(cache_key, None, ttl_hours=24)
        return None
    
    async def _resolve_via_pubchem(self, name: str) -> Optional[Dict]:
        """Route B: PubChem → UniChem crosswalk"""
        try:
            # Step 1: Get PubChem CID
            pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
            pubchem_response = await self.http.get(pubchem_url, timeout=10)
            
            if not pubchem_response or "IdentifierList" not in pubchem_response:
                return None
            
            cids = pubchem_response["IdentifierList"]["CID"]
            if not cids:
                return None
            
            # Use first CID
            cid = cids[0]
            
            # Step 2: Use UniChem to get ChEMBL ID
            # 22 = PubChem source ID, 1 = ChEMBL target ID
            unichem_url = f"https://www.ebi.ac.uk/unichem/rest/src_compound_id/{cid}/22"
            unichem_response = await self.http.get(unichem_url, timeout=10)
            
            if unichem_response:
                # UniChem returns a list, look for ChEMBL entry
                for entry in unichem_response:
                    if entry.get("src_id") == "1":  # ChEMBL
                        chembl_id = f"CHEMBL{entry.get('src_compound_id')}"
                        
                        # Get full ChEMBL record for max_phase
                        chembl_data = await self._get_chembl_molecule(chembl_id)
                        
                        return {
                            "chembl_id": chembl_id,
                            "pref_name": chembl_data.get("pref_name") if chembl_data else name,
                            "max_phase": chembl_data.get("max_phase") if chembl_data else None,
                            "pubchem_cid": cid,
                            "source": "pubchem_unichem"
                        }
        except Exception as e:
            logger.debug(f"PubChem/UniChem resolution failed for {name}: {e}")
        
        return None
    
    async def _get_chembl_molecule(self, chembl_id: str) -> Optional[Dict]:
        """Get full ChEMBL molecule record"""
        try:
            url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
            response = await self.http.get(url, timeout=10)
            return response
        except:
            return None
    
    async def resolve_batch(self, drug_names: List[str], chunk_size: int = 20) -> Dict[str, Dict]:
        """Resolve multiple drug names efficiently"""
        results = {}
        
        # Process in chunks to avoid overwhelming APIs
        for i in range(0, len(drug_names), chunk_size):
            chunk = drug_names[i:i + chunk_size]
            
            # Try to resolve each drug in parallel
            import asyncio
            tasks = [self.resolve_to_chembl_id(name) for name in chunk]
            chunk_results = await asyncio.gather(*tasks)
            
            # Map results
            for name, result in zip(chunk, chunk_results):
                results[name] = result
        
        return results
