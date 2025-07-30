"""
Drug Repurposing Agent using Clinical Trials data and LLM analysis
"""
import asyncio
import json
import logging
import os
from typing import List, Dict, Optional, Any
from collections import defaultdict
from datetime import datetime

from .data.clinical_trials_client import ClinicalTrialsClient
from .data.chembl_client import ChEMBLClient
from .data.pubmed_client import PubMedClient
from .data.http_client import RateLimitedClient
from .data.cache import APICache
from .data.drug_resolver import DrugResolver
from .data.drug_safety_client import DrugSafetyClient
from .data.pharma_intelligence_client import PharmaIntelligenceClient
from .query_parser import QueryParser

try:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

logger = logging.getLogger(__name__)

class DrugRepurposingAgent:
    """Agent for analyzing drug repurposing opportunities"""
    
    def __init__(self):
        self.http = RateLimitedClient()
        self.cache = APICache()
        self.ct_client = ClinicalTrialsClient(self.http, self.cache)
        self.chembl_client = ChEMBLClient()
        self.pubmed_client = PubMedClient()
        self.drug_resolver = DrugResolver(self.http, self.cache)
        self.drug_safety_client = DrugSafetyClient()
        self.pharma_intel_client = PharmaIntelligenceClient()
        self.query_parser = QueryParser()
        
        # Initialize OpenAI client
        if HAS_OPENAI:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                openai.api_key = openai_api_key
                self.openai_client = True
            else:
                self.openai_client = None
        else:
            self.openai_client = None
    
    async def _llm_query(self, prompt: str) -> str:
        """Make an LLM query with timeout"""
        if not HAS_OPENAI:
            raise Exception("OpenAI not available")
        
        try:
            import asyncio
            
            # Use old OpenAI API with timeout
            response = await asyncio.wait_for(
                openai.ChatCompletion.acreate(
                    model="gpt-4o-mini",  # Use faster model
                    messages=[
                        {"role": "system", "content": "You are a drug discovery expert. Respond in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1500
                ),
                timeout=30  # 30 second timeout
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            raise Exception(f"LLM analysis failed: {e}")
    
    async def analyze_disease_failures(
        self,
        disease: str,
        target: Optional[str] = None,
        max_trials: Optional[int] = None
    ) -> Dict:
        """Analyze failed trials for a disease and suggest alternatives"""
        
        logger.info(f"Analyzing failed trials for {disease}")
        
        # Find failed trials
        if target:
            failed_trials = await self.ct_client.find_failed_trials_by_target(
                target=target,
                disease=disease
            )
        else:
            # Get all failed trials for the disease
            failed_trials = await self.ct_client.search_trials(
                condition=disease,
                status=["TERMINATED", "WITHDRAWN", "SUSPENDED"],
                max_results=max_trials or 100
            )
        
        # Analyze failure patterns
        print(f"📊 Analyzing failure patterns from {len(failed_trials)} trials...")
        failure_patterns = await self.ct_client.analyze_failure_patterns(failed_trials)
        
        # Get repurposing candidates
        print(f"🔍 Finding repurposing candidates...")
        candidates = await self.ct_client.get_drug_repurposing_candidates(
            disease=disease,
            exclude_targets=[target] if target else []
        )
        
        # Enrich candidates with ChEMBL data first
        candidates = await self._enrich_candidates_with_chembl(candidates)
        
        # Enrich candidates with target information
        candidates_with_targets = await self._enrich_with_target_info(candidates)
        
        # Get enriched repurposing candidates with ownership
        enriched_candidates = await self._enrich_with_ownership_info(candidates_with_targets)
        
        # Get total trial count for context
        all_trials = await self.ct_client.search_trials(
            condition=disease,
            max_results=200  # Get broader context
        )
        
        # Extract side effects and adverse events
        print(f"⚠️ Analyzing side effects and adverse events...")
        side_effects_analysis = await self._analyze_side_effects(
            failed_trials=failed_trials
        )
        
        # Use LLM to analyze why drugs failed and suggest alternatives
        print(f"🤖 Analyzing failure patterns with LLM...")
        failure_analysis = await self._analyze_failures_with_llm(
            disease=disease,
            failed_trials=failed_trials,
            failure_patterns=failure_patterns,
            side_effects=side_effects_analysis
        )
        
        # Get alternative targets based on failure analysis
        print(f"🎯 Identifying alternative targets...")
        alternative_targets = await self._suggest_alternative_targets(
            disease=disease,
            current_target=target,
            failure_analysis=failure_analysis,
            total_trials=len(all_trials),
            failed_count=len(failed_trials),
            failure_rate=len(failed_trials) / len(all_trials) if all_trials else 0
        )
        
        # Analyze each alternative target
        target_analyses = []
        for alt_target in alternative_targets[:5]:  # Top 5 alternatives
            analysis = await self._analyze_target(alt_target["target"])
            analysis["rationale"] = alt_target["rationale"]
            analysis["confidence"] = alt_target["confidence"]
            target_analyses.append(analysis)
        
        # Get comprehensive safety profiles for top candidates
        print(f"💊 Getting comprehensive safety profiles...")
        candidate_safety_profiles = await self._get_candidate_safety_profiles(
            candidates[:5]  # Top 5 candidates
        )
        
        # Add asset ownership information
        candidates = await self._enrich_with_ownership_info(
            candidates
        )
        
        # Categorize into repurposing vs rescue opportunities
        categorized_candidates = self._categorize_drug_opportunities(candidates)
        
        return {
            "disease": disease,
            "original_target": target,
            "failed_trials_count": len(failed_trials),
            "failure_patterns": {
                k: len(v) for k, v in failure_patterns.items()
            },
            "side_effects_analysis": side_effects_analysis,
            "failure_analysis": failure_analysis,
            "repurposing_candidates": candidates[:10],  # Keep for backward compatibility
            "drug_repurposing": categorized_candidates["drug_repurposing"],
            "drug_rescue": categorized_candidates["drug_rescue"],
            "candidate_safety_profiles": candidate_safety_profiles,
            "alternative_targets": target_analyses,
            "failed_trials_sample": failed_trials[:5]  # Sample of failed trials
        }
    
    async def _analyze_failures_with_llm(
        self,
        disease: str,
        failed_trials: List[Dict],
        failure_patterns: Dict[str, List[Dict]],
        side_effects: Dict
    ) -> Dict:
        """Use LLM to understand why trials failed"""
        
        # Prepare detailed trial summaries with actual drug data
        trial_summaries = []
        for trial in failed_trials[:10]:  # Analyze top 10
            # Extract all drug interventions
            drugs = []
            for intervention in trial.get("interventions", []):
                if intervention.get("type") == "DRUG":
                    drugs.append({
                        "name": intervention["name"],
                        "description": intervention.get("description", "")
                    })
            
            summary = {
                "nct_id": trial["nct_id"],
                "title": trial.get("title", "")[:100] + "...",
                "drugs": drugs,
                "phase": trial.get("phase", []),
                "why_stopped": trial.get("why_stopped", "Not specified"),
                "status": trial["status"],
                "conditions": trial.get("conditions", [])
            }
            trial_summaries.append(summary)
        
        prompt = f"""
        Analyze these ACTUAL failed drugs in {disease} clinical trials:
        
        Failed Trials with Drug Details:
        {json.dumps(trial_summaries, indent=2)}
        
        Side Effects & Safety Analysis:
        Safety Terminations: {side_effects.get('safety_terminations_count', 0)}
        Why Stopped Categories: {side_effects.get('why_stopped_categories', {})}
        Side Effect Patterns: {side_effects.get('side_effects_patterns', {}).get('patterns', [])}
        Affected Organ Systems: {side_effects.get('side_effects_patterns', {}).get('organ_systems', [])}
        
        Your task:
        1. Identify the specific drugs that failed and their likely mechanisms of action
        2. Determine biological reasons for failure (including safety/side effects)
        3. Understand what these failures reveal about {disease} pathology
        4. Analyze side effect patterns to understand drug class limitations
        5. Suggest what alternative mechanisms might work
        
        Focus on the ACTUAL DRUGS that failed, not generic categories.
        
        Return JSON:
        {{
            "failed_drugs_analysis": [
                {{
                    "drug_name": "actual_drug_name",
                    "mechanism": "drug_mechanism_of_action",
                    "failure_type": "safety/efficacy/other",
                    "biological_insight": "why_this_mechanism_failed_in_disease"
                }}
            ],
            "disease_insights": "what_failures_reveal_about_{disease.replace(' ', '_')}_biology",
            "failed_pathways": ["pathway1", "pathway2"],
            "alternative_mechanisms": ["promising_alternative1", "promising_alternative2"],
            "key_biological_challenges": ["challenge1", "challenge2"]
        }}
        """
        
        response = await self._llm_query(prompt)
        
        # Clean JSON if needed
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        try:
            analysis = json.loads(response)
            logger.info(f"LLM Analysis Success: {analysis}")
            return analysis
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw response: {response}")
            # Print to console for debugging
            print(f"🔍 DEBUG - LLM Raw Response: {response[:500]}...")
            raise Exception(f"LLM failure analysis failed: {e}")
    
    async def _suggest_alternative_targets(
        self,
        disease: str,
        current_target: Optional[str],
        failure_analysis: Dict,
        total_trials: int,
        failed_count: int,
        failure_rate: float
    ) -> List[Dict]:
        """Suggest alternative targets based on failure analysis"""
        
        prompt = f"""Based on ACTUAL failed drug analysis for {disease}:
        
        Failed Drugs Analysis:
        {json.dumps(failure_analysis.get('failed_drugs_analysis', []), indent=2)}
        
        Disease Biology Insights:
        {failure_analysis.get('disease_insights', 'Limited insights available')}
        
        Failed Pathways: {failure_analysis.get('failed_pathways', [])}
        Alternative Mechanisms Suggested: {failure_analysis.get('alternative_mechanisms', [])}
        
        Clinical Context:
        - Total trials analyzed: {total_trials}
        - Failed trials: {failed_count}
        - Biological insights: {failure_analysis.get('biological_insights', '')}
        - Success hints: {failure_analysis.get('successful_mechanism_hints', '')}
        
        Suggest 5 alternative therapeutic targets specifically for treating {disease}.
        These targets should address the failures observed with {current_target} in {disease} treatment.
        Consider targets that:
        1. Use different mechanisms than those that failed
        2. Address the root causes of {disease} pathology
        3. Have emerging evidence in {disease} research
        4. Could overcome the key challenges in {disease} treatment
        
        Return JSON array:
        [
            {{
                "target": "GENE_SYMBOL",
                "rationale": "Why this target addresses the failures in {disease}",
                "mechanism": "How it works differently for {disease}",
                "evidence": "Supporting evidence in {disease}",
                "confidence": 0.0-1.0
            }}
        ]
        """
        
        response = await self._llm_query(prompt)
        
        try:
            # Clean JSON if needed
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
                
            targets = json.loads(response)
            return targets[:5]
        except Exception as e:
            logger.error(f"Failed to parse alternative targets response: {e}")
            logger.error(f"Raw response: {response}")
            raise Exception(f"Alternative target suggestion failed: {e}")
    
    async def _analyze_side_effects(self, failed_trials: List[Dict]) -> Dict:
        """Analyze side effects and adverse events from failed trials"""
        
        safety_terminations = []
        adverse_events_summary = {}
        why_stopped_analysis = {
            "safety_related": [],
            "efficacy_related": [],
            "business_related": [],
            "other": []
        }
        
        # Analyze "why stopped" reasons for safety patterns
        for trial in failed_trials:
            why_stopped = trial.get("why_stopped", "").lower()
            nct_id = trial.get("nct_id")
            
            if why_stopped:
                # Categorize stopping reasons - avoid false positives
                # First check for explicit non-safety phrases
                if any(phrase in why_stopped for phrase in ["not related to safety", "unrelated to safety", "not safety", "no safety"]):
                    why_stopped_analysis["business_related"].append(trial.get("why_stopped"))
                elif any(term in why_stopped for term in ["safety concern", "adverse event", "toxicity", "side effect", "serious adverse", "safety issue", "safety signal"]):
                    safety_terminations.append({
                        "nct_id": nct_id,
                        "reason": trial.get("why_stopped"),
                        "interventions": [i.get("name") for i in trial.get("interventions", [])]
                    })
                    why_stopped_analysis["safety_related"].append(trial.get("why_stopped"))
                elif any(term in why_stopped for term in ["ineffective", "efficacy", "futility", "no benefit", "lack of response"]):
                    why_stopped_analysis["efficacy_related"].append(trial.get("why_stopped"))
                elif any(term in why_stopped for term in ["funding", "sponsor", "business", "resource", "financial"]):
                    why_stopped_analysis["business_related"].append(trial.get("why_stopped"))
                else:
                    why_stopped_analysis["other"].append(trial.get("why_stopped"))
                    
                # Try to get detailed adverse events if available
                if trial.get("results_available"):
                    try:
                        detailed_results = await self.ct_client.get_trial_results(nct_id)
                        if detailed_results and detailed_results.get("adverse_events"):
                            ae_data = detailed_results["adverse_events"]
                            adverse_events_summary[nct_id] = {
                                "serious_events": ae_data.get("seriousEvents", {}),
                                "other_events": ae_data.get("otherEvents", {}),
                                "deaths": ae_data.get("deaths", {})
                            }
                    except Exception as e:
                        logger.debug(f"Could not get adverse events for {nct_id}: {e}")
        
        # Generate side effects insights
        side_effects_patterns = await self._extract_side_effect_patterns(
            safety_terminations, adverse_events_summary
        )
        
        return {
            "safety_terminations_count": len(safety_terminations),
            "safety_terminations": safety_terminations,
            "why_stopped_categories": {
                k: len(v) for k, v in why_stopped_analysis.items()
            },
            "why_stopped_details": why_stopped_analysis,
            "adverse_events_summary": adverse_events_summary,
            "side_effects_patterns": side_effects_patterns,
            "repurposing_opportunities": await self._identify_side_effect_repurposing(
                side_effects_patterns
            )
        }
    
    async def _extract_side_effect_patterns(self, safety_terminations: List[Dict], adverse_events: Dict) -> Dict:
        """Extract common side effect patterns using LLM"""
        
        if not safety_terminations and not adverse_events:
            return {"patterns": [], "drug_classes_affected": [], "organ_systems": []}
        
        prompt = f"""Analyze these drug safety terminations and adverse events to identify patterns:
        
        Safety Terminations:
        {json.dumps(safety_terminations, indent=2)}
        
        Adverse Events Summary:
        {json.dumps(list(adverse_events.keys())[:3], indent=2)}  # Limit to prevent token overflow
        
        Extract:
        1. Common side effect patterns
        2. Affected organ systems
        3. Drug classes/mechanisms most problematic
        4. Severity patterns (mild, moderate, severe)
        
        Return JSON:
        {{
            "patterns": ["list of common side effect patterns"],
            "organ_systems": ["affected organ systems"],
            "drug_classes_affected": ["drug classes with issues"],
            "severity_distribution": {{"mild": 0, "moderate": 0, "severe": 0}}
        }}
        """
        
        try:
            response = await self._llm_query(prompt)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response)
        except Exception as e:
            logger.error(f"Side effect pattern analysis failed: {e}")
            return {"patterns": [], "drug_classes_affected": [], "organ_systems": []}
    
    async def _identify_side_effect_repurposing(self, side_effects_patterns: Dict) -> List[Dict]:
        """Identify repurposing opportunities based on side effects"""
        
        if not side_effects_patterns.get("patterns"):
            return []
        
        prompt = f"""Based on these side effect patterns, identify potential repurposing opportunities:
        
        Side Effect Patterns: {side_effects_patterns.get('patterns', [])}
        Affected Organ Systems: {side_effects_patterns.get('organ_systems', [])}
        
        For each side effect, suggest diseases where this "side effect" could be therapeutic.
        Examples:
        - Drowsiness → Sleep disorders
        - Weight loss → Obesity
        - Blood pressure changes → Hypertension/hypotension
        - Mood changes → Depression/anxiety
        
        Return JSON array of opportunities:
        [
            {{
                "side_effect": "specific side effect",
                "repurposing_disease": "disease where this could be therapeutic",
                "rationale": "why this makes biological sense",
                "confidence": 0.0-1.0
            }}
        ]
        """
        
        try:
            response = await self._llm_query(prompt)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            return json.loads(response)
        except Exception as e:
            logger.error(f"Side effect repurposing analysis failed: {e}")
            return []
    
    async def _get_candidate_safety_profiles(self, candidates: List[Dict]) -> List[Dict]:
        """Get comprehensive safety profiles for repurposing candidates"""
        
        safety_profiles = []
        
        for candidate in candidates:
            drug_name = candidate.get("drug", "")
            chembl_id = candidate.get("chembl_id")
            
            if not drug_name:
                continue
                
            # Get comprehensive safety data from real APIs
            comprehensive_profile = await self.drug_safety_client.get_comprehensive_safety_profile(drug_name, chembl_id)
        
            profile = {
                "drug_name": drug_name,
                "chembl_id": chembl_id,
                "fda_adverse_events": comprehensive_profile.get("fda_adverse_events", {}),
                "drug_interactions": comprehensive_profile.get("drug_interactions", {}),
                "contraindications": comprehensive_profile.get("contraindications", {}),
                "mechanism_summary": comprehensive_profile.get("mechanism_summary", {}),
                "safety_score": candidate.get("repurposing_score", 0)
            }
            
            safety_profiles.append(profile)
        
        return safety_profiles
    
    async def _enrich_with_ownership_info(self, candidates: List[Dict]) -> List[Dict]:
        """Enrich candidates with ownership and availability information"""
        
        for candidate in candidates:
            drug_name = candidate.get("drug", "")
            
            if not drug_name:
                continue
                
            try:
                # Check asset availability using pharma intelligence
                availability_info = await self.pharma_intel_client.get_asset_availability_status(drug_name)
                
                # Extract ownership info from clinical trials
                ct_status = availability_info.get("clinical_trials_status", {})
                sponsors = ct_status.get("recent_sponsors", [])
                primary_sponsor = sponsors[0] if sponsors else "Unknown"
                
                candidate["current_owner"] = primary_sponsor
                candidate["last_activity"] = ct_status.get("latest_activity", "Unknown")
                candidate["availability_status"] = availability_info.get("availability_assessment", "unknown")
                
            except Exception as e:
                logger.warning(f"Failed to get ownership info for {drug_name}: {e}")
                candidate["current_owner"] = "Unknown"
                candidate["last_activity"] = "Unknown"
                candidate["availability_status"] = "unknown"
                
        return candidates
    
    async def _enrich_with_target_info(self, candidates: List[Dict]) -> List[Dict]:
        """LLM-first comprehensive drug analysis with ChEMBL validation"""
        
        for candidate in candidates:
            drug_name = candidate.get("drug", "")
            chembl_id = candidate.get("chembl_id")
            
            # Step 1: LLM comprehensive drug analysis (PRIMARY intelligence)
            drug_analysis = await self._comprehensive_drug_analysis_llm(drug_name)
            
            # Step 2: ChEMBL validation/enrichment
            if chembl_id and drug_analysis.get("primary_targets"):
                # Use LLM targets to make smarter ChEMBL queries
                chembl_targets = await self._validate_targets_with_chembl(
                    chembl_id, 
                    drug_analysis["primary_targets"]
                )
                # Merge LLM intelligence with ChEMBL facts
                final_targets = self._merge_target_intelligence(drug_analysis["primary_targets"], chembl_targets)
                confidence = "validated" if chembl_targets else "inferred"
            else:
                # Use LLM analysis as primary source
                final_targets = drug_analysis.get("primary_targets", ["Unknown"])
                confidence = drug_analysis.get("confidence", "inferred")
            
            # Update candidate with comprehensive analysis
            candidate.update({
                "primary_targets": final_targets[:3],  # Top 3 targets
                "target_confidence": confidence,
                "drug_class": drug_analysis.get("drug_class", "unknown"),
                "mechanism": drug_analysis.get("mechanism", "unknown"),
                "normalized_name": drug_analysis.get("normalized_name", drug_name)
            })
            
        return candidates
    
    async def _comprehensive_drug_analysis_llm(self, drug_name: str) -> Dict[str, Any]:
        """LLM-powered comprehensive drug analysis"""
        if not self.openai_client:
            return {
                "primary_targets": ["Unknown"],
                "drug_class": "unknown",
                "mechanism": "unknown",
                "normalized_name": drug_name,
                "confidence": "low",
                "chembl_search_terms": [drug_name]
            }
            
        try:
            prompt = f"""Analyze this drug comprehensively: "{drug_name}"

Provide a detailed JSON analysis with:
1. normalized_name: Clean, standard drug name
2. primary_targets: List of 1-3 main protein targets (gene symbols/protein names)
3. drug_class: Category (e.g., "kinase_inhibitor", "gpcr_agonist", "monoclonal_antibody")
4. mechanism: Brief mechanism of action
5. confidence: "high", "medium", or "low" based on name clarity
6. chembl_search_terms: 2-3 terms for ChEMBL validation

Examples:
- "PF-04991532" → Pfizer compound (PF-prefix), likely small molecule
- "BioChaperone insulin lispro" → Modified insulin targeting insulin receptor
- "Empagliflozin" → SGLT2 inhibitor
- "anti-VEGF antibody" → Monoclonal antibody targeting VEGF

JSON format:
{{
  "normalized_name": "Standard name",
  "primary_targets": ["TARGET1", "TARGET2"],
  "drug_class": "category",
  "mechanism": "brief description",
  "confidence": "high/medium/low",
  "chembl_search_terms": ["term1", "term2"]
}}"""
            
            response = await self._llm_query(prompt)
            # Clean response - remove markdown code blocks if present
            clean_response = self._extract_json_from_llm_response(response)
            analysis = json.loads(clean_response)
            
            # Validate and set defaults
            if not isinstance(analysis.get("primary_targets"), list):
                analysis["primary_targets"] = ["Unknown"]
            if not analysis.get("normalized_name"):
                analysis["normalized_name"] = drug_name
            if not analysis.get("confidence"):
                analysis["confidence"] = "medium"
                
            return analysis
                
        except Exception as e:
            logger.warning(f"LLM comprehensive drug analysis failed for {drug_name}: {e}")
            return {
                "primary_targets": ["Unknown"],
                "drug_class": "unknown",
                "mechanism": "unknown",
                "normalized_name": drug_name,
                "confidence": "low",
                "chembl_search_terms": [drug_name]
            }
    
    async def _validate_targets_with_chembl(self, chembl_id: str, llm_targets: List[str]) -> List[str]:
        """Use LLM targets to make smarter ChEMBL queries"""
        try:
            # Get bioactivities using existing method
            bioactivities = await self._get_drug_targets_from_chembl(chembl_id)
            
            # Cross-validate LLM targets with ChEMBL data
            validated_targets = []
            for chembl_target in bioactivities:
                # Check if ChEMBL target matches or relates to LLM predictions
                for llm_target in llm_targets:
                    if (llm_target.lower() in chembl_target.lower() or 
                        chembl_target.lower() in llm_target.lower()):
                        validated_targets.append(chembl_target)
                        break
                else:
                    # Add ChEMBL target even if not predicted (new discovery)
                    validated_targets.append(chembl_target)
            
            return validated_targets[:5]  # Top 5 validated targets
            
        except Exception as e:
            logger.warning(f"ChEMBL target validation failed for {chembl_id}: {e}")
            return []
    
    def _merge_target_intelligence(self, llm_targets: List[str], chembl_targets: List[str]) -> List[str]:
        """Intelligently merge LLM predictions with ChEMBL facts"""
        merged = []
        
        # Priority 1: Targets confirmed by both LLM and ChEMBL
        for llm_target in llm_targets:
            for chembl_target in chembl_targets:
                if (llm_target.lower() in chembl_target.lower() or 
                    chembl_target.lower() in llm_target.lower()):
                    merged.append(chembl_target)  # Use ChEMBL naming
                    break
        
        # Priority 2: ChEMBL targets not predicted by LLM (discoveries)
        for chembl_target in chembl_targets:
            if not any(ct.lower() in chembl_target.lower() or chembl_target.lower() in ct.lower() 
                      for ct in merged):
                merged.append(chembl_target)
        
        # Priority 3: LLM targets not found in ChEMBL (novel predictions)
        for llm_target in llm_targets:
            if not any(llm_target.lower() in merged_target.lower() or merged_target.lower() in llm_target.lower() 
                      for merged_target in merged):
                merged.append(f"{llm_target} (predicted)")
        
        return merged
    
    def _extract_json_from_llm_response(self, response: str) -> str:
        """Extract JSON from LLM response that may be wrapped in markdown code blocks"""
        if not response:
            return "{}"
        
        # Remove markdown code blocks
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]  # Remove ```json
        if response.startswith("```"):
            response = response[3:]   # Remove ```
        if response.endswith("```"):
            response = response[:-3]  # Remove ending ```
        
        return response.strip()
    
    async def _get_drug_targets_from_chembl(self, chembl_id: str) -> List[str]:
        """Get drug targets from ChEMBL using high-quality filters and mechanism fallback"""
        try:
            # Primary: High-quality bioactivity data (SINGLE PROTEIN, confidence >= 8)
            bioactivities = await self.chembl_client.get_bioactivities(
                chembl_id=chembl_id,
                limit=20
            )
            
            # If no high-quality targets found, try mechanism endpoint
            if not bioactivities:
                logger.info(f"No high-quality targets found for {chembl_id}, trying mechanism endpoint")
                mechanisms = await self.chembl_client.get_drug_mechanisms(chembl_id)
                return self._extract_targets_from_mechanisms(mechanisms)
            
            # Score and prioritize targets by bioactivity quality
            target_scores = {}
            for activity in bioactivities:
                target_chembl_id = activity.get("target_chembl_id")
                target_name = activity.get("target_pref_name", "")
                
                # Skip low-quality targets
                if (not target_chembl_id or 
                    self._is_organism_target(target_name) or
                    activity.get("standard_type") in ["Body weight", "Survival"]):
                    continue
                
                # Score bioactivity quality
                score = self._score_bioactivity_quality(activity, target_name)
                if score > 0:  # Only include targets with positive scores
                    if target_chembl_id not in target_scores:
                        target_scores[target_chembl_id] = score
                    else:
                        target_scores[target_chembl_id] = max(target_scores[target_chembl_id], score)
            
            # Sort targets by score (highest first)
            sorted_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get actual protein names from top-scored targets
            protein_targets = []
            for target_id, score in sorted_targets[:5]:  # Top 5 by score
                protein_name = await self._get_protein_name_from_target_id(target_id)
                if protein_name and protein_name not in protein_targets:
                    protein_targets.append(protein_name)
            
            return protein_targets
                    
        except Exception as e:
            logger.warning(f"ChEMBL target lookup failed for {chembl_id}: {e}")
            return []
    
    def _is_organism_target(self, target_name: str) -> bool:
        """Check if target name is an organism, cell line, or non-protein target"""
        if not target_name:
            return True
            
        # Organism indicators
        organism_indicators = [
            "musculus", "norvegicus", "sapiens", "elegans", "melanogaster",
            "cerevisiae", "pombe", "thaliana", "Mus", "Rattus", "Homo", 
            "Caenorhabditis", "Drosophila", "Saccharomyces", "Arabidopsis"
        ]
        
        # Cell line patterns (common cancer cell lines)
        cell_line_patterns = [
            "HCT-116", "A549", "MCF7", "HeLa", "HL-60", "P388", "LFCl2A",
            "WiDr", "IGROV-1", "K562", "U937", "Jurkat", "PC-3", "MDA-MB",
            "T47D", "SK-", "SW-", "DU-", "HT-", "BT-", "COLO", "OVCAR",
            "NCI-", "CCRF", "MOLT", "RPMI", "L1210", "B16", "LLC",
            "5637", "A2780", "H322", "M109", "U266", "THP-1", "KG-1"
        ]
        
        # Non-protein target indicators
        non_protein_indicators = [
            "NON-PROTEIN TARGET", "No relevant target", "Uncategorized",
            "whole organism", "tissue", "organ", "cell culture", "extract"
        ]
        
        target_upper = target_name.upper()
        
        # Check all exclusion criteria
        return (any(indicator in target_name for indicator in organism_indicators) or
                any(pattern.upper() in target_upper for pattern in cell_line_patterns) or
                any(indicator.upper() in target_upper for indicator in non_protein_indicators))
    
    def _score_bioactivity_quality(self, activity: dict, target_name: str) -> float:
        """Score bioactivity quality to prioritize relevant protein targets"""
        score = 0.0
        
        # Base score for having a target
        if target_name:
            score += 1.0
        
        # High-value activity types (binding, inhibition)
        activity_type = activity.get("standard_type", "").upper()
        high_value_types = ["IC50", "KI", "KD", "EC50", "POTENCY", "INHIBITION"]
        if any(vtype in activity_type for vtype in high_value_types):
            score += 3.0
        elif activity_type in ["ACTIVITY", "PERCENT_INHIBITION", "RATIO"]:
            score += 1.0
        
        # Known drug target keywords (higher priority)
        target_upper = target_name.upper()
        known_targets = [
            "TOPOISOMERASE", "TUBULIN", "THYMIDYLATE", "SYNTHASE", "KINASE", 
            "RECEPTOR", "CHANNEL", "TRANSPORTER", "PROTEASE", "POLYMERASE",
            "CYCLASE", "OXIDASE", "REDUCTASE", "TRANSFERASE", "LIGASE", "HYDROLASE"
        ]
        if any(keyword in target_upper for keyword in known_targets):
            score += 2.0
        
        # Penalty for generic/unclear names
        generic_terms = ["ADMET", "UNKNOWN", "UNDEFINED", "GENERAL", "MISC"]
        if any(term in target_upper for term in generic_terms):
            score -= 1.0
            
        # Penalty for organism-specific entries that passed initial filter
        organism_terms = ["SCROFA", "MUSCULUS", "NORVEGICUS"]
        if any(term in target_upper for term in organism_terms):
            score -= 2.0
        
        return max(0.0, score)  # Never return negative scores
    
    def _extract_targets_from_mechanisms(self, mechanisms: List[Dict]) -> List[str]:
        """Extract target information from ChEMBL mechanism data"""
        targets = []
        
        for mechanism in mechanisms:
            mechanism_of_action = mechanism.get("mechanism_of_action", "")
            
            # For cytotoxic agents like busulfan (DNA alkylation)
            if any(term in mechanism_of_action.lower() for term in 
                   ["dna alkyl", "cytotoxic", "crosslink"]):
                targets.append(f"DNA alkylation (cytotoxic)")
            
            # For antimetabolites like azacitidine
            elif any(term in mechanism_of_action.lower() for term in 
                     ["methyl", "dnmt", "antimetabolite"]):
                targets.append("DNA methyltransferase inhibitor")
            
            # For nucleoside analogs like fludarabine
            elif any(term in mechanism_of_action.lower() for term in 
                     ["nucleoside", "purine", "dna synthesis"]):
                targets.append("DNA polymerase / nucleoside metabolism")
            
            # Generic mechanism
            elif mechanism_of_action:
                targets.append(mechanism_of_action)
        
        return targets[:3]  # Return top 3 mechanisms
    
    async def _get_protein_name_from_target_id(self, target_chembl_id: str) -> str:
        """Get protein name from ChEMBL target ID"""
        try:
            client = await self.chembl_client.http_client if hasattr(self.chembl_client, 'http_client') else None
            if not client:
                import httpx
                async with httpx.AsyncClient(timeout=10) as temp_client:
                    url = f"https://www.ebi.ac.uk/chembl/api/data/target/{target_chembl_id}.json"
                    response = await temp_client.get(url)
                    data = response.json()
            else:
                url = f"https://www.ebi.ac.uk/chembl/api/data/target/{target_chembl_id}.json"
                response = await client.get(url)
                data = response.json()
            
            # Extract protein name from target details
            target_components = data.get("target_components", [])
            if target_components:
                # Get the first component's protein name
                component = target_components[0]
                protein_name = (
                    component.get("component_synonym") or 
                    component.get("component_name") or
                    data.get("pref_name", "Unknown")
                )
                return protein_name
                
            return data.get("pref_name", "Unknown")
            
        except Exception as e:
            logger.warning(f"Failed to get protein name for {target_chembl_id}: {e}")
            return "Unknown"
    
    async def _infer_drug_targets_llm(self, drug_name: str) -> List[str]:
        """Use LLM to infer likely targets for unknown drugs"""
        if not self.openai_client:
            return ["Unknown"]
            
        try:
            prompt = f"""Based on the drug name '{drug_name}', what are the most likely protein targets?
            
Provide a JSON list of 1-3 most probable target proteins (gene symbols or protein names).
If the name suggests a specific mechanism, include that target.
If it's unclear, return ["Unknown"].
            
Examples:
- "PF-04991532" (Pfizer compound) → might target specific pathway
- "BioChaperone insulin" → "Insulin Receptor"
- "Empagliflozin" → "SGLT2"
            
JSON format: ["TARGET1", "TARGET2"]"""
            
            response = await self._llm_query(prompt)
            targets = json.loads(response)
            
            if isinstance(targets, list) and len(targets) > 0:
                return targets[:3]
            else:
                return ["Unknown"]
                
        except Exception as e:
            logger.warning(f"LLM target inference failed for {drug_name}: {e}")
            return ["Unknown"]
    
    def _categorize_drug_opportunities(self, candidates: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize drugs into repurposing (approved) vs rescue (experimental) opportunities"""
        
        approved_repurposing = []
        experimental_rescue = []
        
        for candidate in candidates:
            max_phase = candidate.get("max_phase")
            
            # Consider phase 4 (post-market) or phase null with high confidence as approved
            # Phase 4 indicates post-market surveillance (already approved)
            if max_phase == 4 or (max_phase is None and candidate.get("repurposing_score", 0) > 100):
                candidate["opportunity_type"] = "repurposing"
                approved_repurposing.append(candidate)
            else:
                candidate["opportunity_type"] = "rescue"
                experimental_rescue.append(candidate)
        
        return {
            "drug_repurposing": approved_repurposing,  # Approved → new indication
            "drug_rescue": experimental_rescue        # Failed/experimental → different indication
        }
    
    # Removed hardcoded safety methods - now using real DrugSafetyClient APIs
    
    async def _analyze_target(self, target: str) -> Dict:
        """Analyze a specific target for drug discovery"""
        
        # Import PDBClient for structure data
        try:
            from .data.pdb_client import PDBClient
            pdb_client = PDBClient()
        except:
            pdb_client = None
        
        # Get inhibitors from ChEMBL
        inhibitors = await self.chembl_client.get_inhibitors_for_target(
            gene_symbol=target,
            max_ic50_nm=1000  # Broader range for repurposing
        )
        
        # Get literature
        papers = await self.pubmed_client.search_articles(
            query=f"{target} inhibitors clinical",
            max_results=10
        )
        
        # Get clinical trials for this target
        trials = await self.ct_client.search_trials(
            intervention=target,
            max_results=20
        )
        
        # Categorize trials
        trial_stats = {
            "total": len(trials),
            "recruiting": 0,
            "completed": 0,
            "failed": 0,
            "phases": {"PHASE1": 0, "PHASE2": 0, "PHASE3": 0}
        }
        
        for trial in trials:
            if trial["status"] in ["RECRUITING", "ACTIVE_NOT_RECRUITING"]:
                trial_stats["recruiting"] += 1
            elif trial["status"] == "COMPLETED":
                trial_stats["completed"] += 1
            elif trial["status"] in ["TERMINATED", "WITHDRAWN", "SUSPENDED"]:
                trial_stats["failed"] += 1
            
            for phase in trial.get("phase", []):
                if phase in trial_stats["phases"]:
                    trial_stats["phases"][phase] += 1
        
        # Find most advanced inhibitor
        most_advanced = None
        if inhibitors:
            for inh in inhibitors:
                if inh.get("max_phase"):
                    if not most_advanced or inh["max_phase"] > most_advanced.get("max_phase", 0):
                        most_advanced = inh
        
        # Get PDB structures if available (with timeout)
        pdb_structures = []
        if pdb_client:
            try:
                # Add timeout to prevent hanging
                import asyncio
                structures = await asyncio.wait_for(
                    pdb_client.search_structures(target, limit=5),
                    timeout=10  # 10 second timeout
                )
                pdb_structures = [
                    {
                        "pdb_id": s.get("pdb_id", "N/A"),
                        "title": s.get("title", "N/A"),
                        "resolution": s.get("resolution", "N/A"),
                        "method": s.get("experimental_method", "N/A")
                    }
                    for s in structures
                ]
            except:
                pass
        
        return {
            "target": target,
            "inhibitor_count": len(inhibitors),
            "most_potent_ic50": min([i["standard_value_nm"] for i in inhibitors if "standard_value_nm" in i]) if inhibitors else None,
            "most_advanced_compound": {
                "chembl_id": most_advanced["molecule_chembl_id"] if most_advanced else None,
                "phase": most_advanced.get("max_phase") if most_advanced else None
            },
            "clinical_trials": trial_stats,
            "literature_count": len(papers),
            "development_score": self._calculate_development_score(
                inhibitors=len(inhibitors),
                trials=trial_stats,
                papers=len(papers)
            ),
            "pdb_structures": pdb_structures
        }
    
    def _calculate_development_score(
        self,
        inhibitors: int,
        trials: Dict,
        papers: int
    ) -> float:
        """Calculate how promising a target is for development"""
        
        score = 0.0
        
        # Inhibitor availability (max 30 points)
        score += min(inhibitors * 2, 30)
        
        # Clinical validation (max 40 points)
        if trials["completed"] > 0:
            score += 20
        if trials["recruiting"] > 0:
            score += 10
        if trials["phases"]["PHASE3"] > 0:
            score += 10
        
        # Literature support (max 20 points)
        score += min(papers, 20)
        
        # Penalty for failures (max -10 points)
        if trials["total"] > 0:
            failure_rate = trials["failed"] / trials["total"]
            score -= failure_rate * 10
        
        return round(score / 100, 2)  # Normalize to 0-1
    
    async def _enrich_candidates_with_chembl(self, candidates: List[Dict]) -> List[Dict]:
        """Enrich drug candidates with ChEMBL IDs and phase data"""
        
        # Extract unique drug names
        drug_names = list(set(c["drug"] for c in candidates))
        
        # Resolve to ChEMBL IDs using the hybrid resolver
        chembl_map = await self.drug_resolver.resolve_batch(drug_names)
        
        # Enrich each candidate
        for candidate in candidates:
            drug_name = candidate["drug"]
            chembl_data = chembl_map.get(drug_name)
            
            if chembl_data:
                candidate["chembl_id"] = chembl_data.get("chembl_id")
                candidate["max_phase"] = chembl_data.get("max_phase")
                candidate["resolution_source"] = chembl_data.get("source")
                
                # If no max_phase from ChEMBL, infer from trial phases
                if not candidate["max_phase"] and candidate.get("phases"):
                    phase_map = {"PHASE3": 3, "PHASE2": 2, "PHASE1": 1}
                    max_phase = max((phase_map.get(p, 0) for p in candidate["phases"]), default=0)
                    candidate["max_phase"] = max_phase if max_phase > 0 else None
            else:
                candidate["chembl_id"] = None
                candidate["max_phase"] = None
                candidate["resolution_source"] = "not_found"
        
        return candidates

async def main():
    """Example usage"""
    agent = DrugRepurposingAgent()
    
    # Analyze glioblastoma failures and find alternatives
    results = await agent.analyze_disease_failures(
        disease="glioblastoma",
        target="EGFR"
    )
    
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
