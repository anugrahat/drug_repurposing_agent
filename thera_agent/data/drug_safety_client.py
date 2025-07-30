#!/usr/bin/env python3

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from .http_client import RateLimitedClient
from .cache import APICache

logger = logging.getLogger(__name__)

class DrugSafetyClient:
    """Comprehensive drug safety and pharmacological data client"""
    
    def __init__(self):
        self.http = RateLimitedClient()
        self.cache = APICache()
        
        # API endpoints
        self.openfda_base = "https://api.fda.gov"
        self.dailymed_base = "https://dailymed.nlm.nih.gov/dailymed"
        self.rxnav_base = "https://rxnav.nlm.nih.gov/REST"
        self.drugbank_base = "https://go.drugbank.com/public_api/v1"  # Requires API key
    
    async def get_comprehensive_safety_profile(self, drug_name: str, chembl_id: str = None) -> Dict[str, Any]:
        """Get comprehensive safety profile from multiple sources"""
        
        profile = {
            "drug_name": drug_name,
            "chembl_id": chembl_id,
            "fda_adverse_events": await self._get_fda_adverse_events(drug_name),
            "fda_drug_labels": await self._get_fda_drug_labels(drug_name),
            "contraindications": await self._get_contraindications(drug_name),
            "drug_interactions": await self._get_drug_interactions(drug_name),
            "black_box_warnings": await self._get_black_box_warnings(drug_name),
            "pharmacology": await self._get_pharmacology_data(drug_name),
            "allergies_cross_sensitivity": await self._get_allergy_data(drug_name)
        }
        
        return profile
    
    async def _get_fda_adverse_events(self, drug_name: str) -> Dict[str, Any]:
        """Get FDA adverse event reports from OpenFDA"""
        
        cache_key = f"fda_ae_{drug_name}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # OpenFDA Drug Adverse Events API
            url = f"{self.openfda_base}/drug/event.json"
            params = {
                "search": f'patient.drug.medicinalproduct:"{drug_name}"',
                "count": "patient.reaction.reactionmeddrapt.exact",
                "limit": 20
            }
            
            response = await self.http.get(url, params=params, timeout=10)
            
            if response and response.get("results"):
                # Process adverse events with frequencies
                adverse_events = []
                for result in response["results"]:
                    adverse_events.append({
                        "reaction": result["term"],
                        "count": result["count"],
                        "percentage": round(result["count"] / sum(r["count"] for r in response["results"]) * 100, 2)
                    })
                
                ae_data = {
                    "total_reports": sum(r["count"] for r in response["results"]),
                    "top_adverse_events": adverse_events[:10],
                    "data_source": "FDA OpenFDA"
                }
                
                self.cache.set(cache_key, ae_data, ttl_hours=24)
                return ae_data
                
        except Exception as e:
            logger.error(f"FDA adverse events lookup failed for {drug_name}: {e}")
        
        return {"total_reports": 0, "top_adverse_events": [], "data_source": "Not available"}
    
    async def _get_fda_drug_labels(self, drug_name: str) -> Dict[str, Any]:
        """Get FDA drug label information"""
        
        cache_key = f"fda_labels_{drug_name}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # OpenFDA Drug Labels API
            url = f"{self.openfda_base}/drug/label.json"
            params = {
                "search": f'openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}"',
                "limit": 1
            }
            
            response = await self.http.get(url, params=params, timeout=10)
            
            if response and response.get("results"):
                label = response["results"][0]
                
                label_data = {
                    "brand_names": label.get("openfda", {}).get("brand_name", []),
                    "generic_name": label.get("openfda", {}).get("generic_name", []),
                    "warnings": label.get("warnings", []),
                    "contraindications": label.get("contraindications", []),
                    "adverse_reactions": label.get("adverse_reactions", []),
                    "drug_interactions": label.get("drug_interactions", []),
                    "clinical_pharmacology": label.get("clinical_pharmacology", []),
                    "indications_and_usage": label.get("indications_and_usage", []),
                    "boxed_warning": label.get("boxed_warning", []),
                    "manufacturer": label.get("openfda", {}).get("manufacturer_name", [])
                }
                
                self.cache.set(cache_key, label_data, ttl_hours=72)
                return label_data
                
        except Exception as e:
            logger.error(f"FDA drug labels lookup failed for {drug_name}: {e}")
        
        return {"brand_names": [], "warnings": [], "contraindications": []}
    
    async def _get_contraindications(self, drug_name: str) -> List[Dict[str, Any]]:
        """Get detailed contraindications"""
        
        # This would typically come from FDA labels or DailyMed
        fda_labels = await self._get_fda_drug_labels(drug_name)
        
        contraindications = []
        for contraindication in fda_labels.get("contraindications", []):
            if contraindication.strip():
                contraindications.append({
                    "condition": contraindication[:200] + "..." if len(contraindication) > 200 else contraindication,
                    "severity": "absolute",  # From FDA labels, these are absolute
                    "source": "FDA Drug Label"
                })
        
        return contraindications
    
    async def _get_drug_interactions(self, drug_name: str) -> List[Dict[str, Any]]:
        """Get drug-drug interactions from RxNav"""
        
        cache_key = f"interactions_{drug_name}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # First get RxCUI for the drug
            url = f"{self.rxnav_base}/rxcui.json"
            params = {"name": drug_name}
            
            response = await self.http.get(url, params=params, timeout=10)
            
            if response and response.get("idGroup", {}).get("rxnormId"):
                rxcui = response["idGroup"]["rxnormId"][0]
                
                # Get interactions using correct RxNav endpoint 
                interaction_url = f"{self.rxnav_base}/interaction/list.json"
                params = {"rxcuis": rxcui}
                
                interaction_response = await self.http.get(interaction_url, params=params, timeout=10)
                
                interactions = []
                if interaction_response and interaction_response.get("interactionTypeGroup"):
                    for group in interaction_response["interactionTypeGroup"]:
                        for pair in group.get("interactionType", []):
                            for interaction in pair.get("interactionPair", []):
                                interactions.append({
                                    "interacting_drug": interaction.get("interactionConcept", [{}])[1].get("minConceptItem", {}).get("name", "Unknown"),
                                    "severity": interaction.get("severity", "Unknown"),
                                    "description": interaction.get("description", "No description available")[:300] + "...",
                                    "source": "RxNav/NLM"
                                })
                
                self.cache.set(cache_key, interactions[:20], ttl_hours=48)  # Cache top 20
                return interactions[:20]
                
        except Exception as e:
            logger.warning(f"RxNav drug interactions unavailable for {drug_name}: {e}")
            # Return empty list for now - could add other interaction databases
            return []
    
    async def _get_black_box_warnings(self, drug_name: str) -> List[Dict[str, Any]]:
        """Extract black box warnings from FDA labels"""
        
        fda_labels = await self._get_fda_drug_labels(drug_name)
        
        warnings = []
        for warning in fda_labels.get("boxed_warning", []):
            if warning.strip():
                warnings.append({
                    "warning": warning[:500] + "..." if len(warning) > 500 else warning,
                    "type": "Black Box Warning",
                    "source": "FDA Drug Label"
                })
        
        # Also check general warnings for severity indicators
        for warning in fda_labels.get("warnings", []):
            if any(keyword in warning.lower() for keyword in ["death", "fatal", "serious", "life-threatening"]):
                warnings.append({
                    "warning": warning[:500] + "..." if len(warning) > 500 else warning,
                    "type": "Serious Warning",
                    "source": "FDA Drug Label"
                })
        
        return warnings
    
    async def _get_pharmacology_data(self, drug_name: str) -> Dict[str, Any]:
        """Get pharmacological data from FDA labels"""
        
        fda_labels = await self._get_fda_drug_labels(drug_name)
        
        pharmacology = {
            "mechanism_of_action": [],
            "pharmacokinetics": [],
            "pharmacodynamics": [],
            "metabolism": [],
            "absorption": [],
            "distribution": [],
            "elimination": []
        }
        
        # Extract from clinical pharmacology section
        for section in fda_labels.get("clinical_pharmacology", []):
            section_lower = section.lower()
            
            if "mechanism" in section_lower or "action" in section_lower:
                pharmacology["mechanism_of_action"].append(section[:300] + "...")
            elif "pharmacokinetic" in section_lower or "absorption" in section_lower:
                pharmacology["pharmacokinetics"].append(section[:300] + "...")
            elif "metabolism" in section_lower or "metaboli" in section_lower:
                pharmacology["metabolism"].append(section[:300] + "...")
        
        return pharmacology
    
    async def _get_allergy_data(self, drug_name: str) -> Dict[str, Any]:
        """Get allergy and cross-sensitivity data"""
        
        # This would typically require specialized allergy databases
        # For now, extract from FDA adverse events and labels
        
        fda_ae = await self._get_fda_adverse_events(drug_name)
        fda_labels = await self._get_fda_drug_labels(drug_name)
        
        allergic_reactions = []
        
        # Look for allergic reactions in adverse events
        for event in fda_ae.get("top_adverse_events", []):
            if any(keyword in event["reaction"].lower() for keyword in ["allerg", "rash", "hypersensitiv", "anaphyla"]):
                allergic_reactions.append({
                    "reaction": event["reaction"],
                    "frequency": f"{event['percentage']}%",
                    "source": "FDA Adverse Events"
                })
        
        return {
            "allergic_reactions": allergic_reactions,
            "cross_sensitivity": [],  # Would need specialized database
            "incidence_rate": "Variable based on population"
        }

    async def get_clinical_trial_safety_summary(self, nct_id: str) -> Dict[str, Any]:
        """Get safety summary from a specific clinical trial"""
        
        # This would integrate with the existing clinical trials client
        from .clinical_trials_client import ClinicalTrialsClient
        ct_client = ClinicalTrialsClient()
        
        try:
            results = await ct_client.get_trial_results(nct_id)
            
            if results and results.get("adverse_events"):
                ae_data = results["adverse_events"]
                
                return {
                    "nct_id": nct_id,
                    "serious_adverse_events": ae_data.get("seriousEvents", {}),
                    "other_adverse_events": ae_data.get("otherEvents", {}),
                    "deaths": ae_data.get("deaths", {}),
                    "participants_affected": ae_data.get("frequencyThreshold", "Not specified"),
                    "data_source": "ClinicalTrials.gov"
                }
        except Exception as e:
            logger.error(f"Clinical trial safety summary failed for {nct_id}: {e}")
        
        return {"nct_id": nct_id, "data_available": False}
