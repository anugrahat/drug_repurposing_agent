"""
ClinicalTrials.gov API client for drug repurposing analysis
"""
import asyncio
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..data.http_client import RateLimitedClient
from ..data.cache import APICache

logger = logging.getLogger(__name__)

class ClinicalTrialsClient:
    """Client for ClinicalTrials.gov API v2"""
    
    BASE_URL = "https://clinicaltrials.gov/api/v2"
    
    def __init__(self, http_client: RateLimitedClient, cache_manager: APICache):
        self.http = http_client
        self.cache = cache_manager
    
    async def search_trials(
        self, 
        query: str = None,
        condition: str = None,
        intervention: str = None,
        status: List[str] = None,
        phase: List[str] = None,
        max_results: int = 100
    ) -> List[Dict]:
        """Search clinical trials with filters using API v2"""
        
        # Build query parameters for v2 API
        params = {
            "format": "json",
            "pageSize": min(max_results, 100)
        }
        
        # Build query string using search areas
        query_parts = []
        if condition:
            query_parts.append(f"AREA[ConditionSearch]{condition}")
        if intervention:
            query_parts.append(f"AREA[InterventionSearch]{intervention}")
        if query:
            query_parts.append(query)
        
        if query_parts:
            params["query.cond"] = " AND ".join(query_parts)
        
        # Add filters using v2 format
        if status:
            params["filter.overallStatus"] = "|".join(status)
        if phase:
            params["filter.phase"] = "|".join(phase)
        
        # Check cache
        cached = self.cache.get(f"{self.BASE_URL}/studies", params)
        if cached:
            return cached
        
        # Make API request
        url = f"{self.BASE_URL}/studies"
        try:
            data = await self.http.get(url, params=params)
            studies = data.get("studies", [])
            
            # Extract relevant fields
            results = []
            for study in studies:
                protocol = study.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                design_module = protocol.get("designModule", {})
                arms_module = protocol.get("armsInterventionsModule", {})
                outcomes_module = protocol.get("outcomesModule", {})
                
                result = {
                    "nct_id": id_module.get("nctId"),
                    "title": id_module.get("briefTitle"),
                    "status": status_module.get("overallStatus"),
                    "phase": design_module.get("phases", []),
                    "conditions": protocol.get("conditionsModule", {}).get("conditions", []),
                    "interventions": [],
                    "start_date": status_module.get("startDateStruct", {}).get("date"),
                    "completion_date": status_module.get("completionDateStruct", {}).get("date"),
                    "enrollment": design_module.get("enrollmentInfo", {}).get("count"),
                    "primary_outcomes": [],
                    "why_stopped": status_module.get("whyStopped", ""),
                    "results_available": study.get("hasResults", False)
                }
                
                # Extract interventions with details
                for intervention in arms_module.get("interventions", []):
                    result["interventions"].append({
                        "type": intervention.get("type"),
                        "name": intervention.get("name"),
                        "description": intervention.get("description", "")
                    })
                
                # Extract primary outcomes
                for outcome in outcomes_module.get("primaryOutcomes", []):
                    result["primary_outcomes"].append({
                        "measure": outcome.get("measure"),
                        "description": outcome.get("description", ""),
                        "time_frame": outcome.get("timeFrame", "")
                    })
                
                results.append(result)
            
            # Cache results
            self.cache.set(f"{self.BASE_URL}/studies", results, params=params, ttl_hours=24)
            return results
        except Exception as e:
            logger.error(f"Error searching trials: {e}")
            return []
    
    async def get_trial_results(self, nct_id: str) -> Optional[Dict]:
        """Get detailed results for a specific trial"""
        
        url = f"{self.BASE_URL}/studies/{nct_id}"
        params = {"format": "json", "fields": "ResultsSection"}
        
        cached = self.cache.get(url, params)
        if cached:
            return cached
        
        try:
            data = await self.http.get(url, params=params)
            study = data.get("studies", [{}])[0]
            results_section = study.get("resultsSection", {})
            
            if results_section:
                # Extract key results
                result = {
                    "nct_id": nct_id,
                    "participant_flow": results_section.get("participantFlowModule", {}),
                    "baseline": results_section.get("baselineCharacteristicsModule", {}),
                    "outcome_measures": results_section.get("outcomeMeasuresModule", {}),
                    "adverse_events": results_section.get("adverseEventsModule", {}),
                    "more_info": results_section.get("moreInfoModule", {})
                }
                
                self.cache.set(url, result, params=params, ttl_hours=48)
                return result
        except Exception as e:
            logger.error(f"Error getting trial results for {nct_id}: {e}")
        
        return None
    
    async def find_failed_trials_by_target(
        self,
        target: str,
        disease: str = None,
        include_withdrawn: bool = True,
        include_terminated: bool = True,
        include_suspended: bool = True
    ) -> List[Dict]:
        """Find failed/stopped trials for a specific target"""
        
        # Define failure statuses
        failure_statuses = []
        if include_withdrawn:
            failure_statuses.append("WITHDRAWN")
        if include_terminated:
            failure_statuses.append("TERMINATED")
        if include_suspended:
            failure_statuses.append("SUSPENDED")
        
        # Also include completed trials that might have failed
        failure_statuses.append("COMPLETED")
        
        # Search for trials
        trials = await self.search_trials(
            intervention=target,
            condition=disease,
            status=failure_statuses,
            max_results=100
        )
        
        # Filter and enrich results
        failed_trials = []
        for trial in trials:
            # Skip if no clear failure reason and status is completed
            if trial["status"] == "COMPLETED" and not trial.get("why_stopped"):
                continue
            
            # Get detailed results if available
            if trial.get("results_available"):
                results = await self.get_trial_results(trial["nct_id"])
                if results:
                    trial["detailed_results"] = results
            
            failed_trials.append(trial)
        
        return failed_trials
    
    async def analyze_failure_patterns(
        self,
        trials: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """Analyze patterns in trial failures"""
        
        patterns = {
            "safety_issues": [],
            "efficacy_issues": [],
            "recruitment_issues": [],
            "business_reasons": [],
            "other_reasons": [],
            "unknown": []
        }
        
        # Keywords for categorization
        safety_keywords = ["safety", "adverse", "toxicity", "side effect", "SAE", "death"]
        efficacy_keywords = ["efficacy", "ineffective", "no benefit", "futility", "endpoint not met"]
        recruitment_keywords = ["enrollment", "recruitment", "accrual", "participants"]
        business_keywords = ["sponsor", "funding", "business", "strategic", "priorit"]
        
        for trial in trials:
            why_stopped = trial.get("why_stopped", "").lower()
            categorized = False
            
            # Check safety issues
            if any(keyword in why_stopped for keyword in safety_keywords):
                patterns["safety_issues"].append(trial)
                categorized = True
            
            # Check efficacy issues
            elif any(keyword in why_stopped for keyword in efficacy_keywords):
                patterns["efficacy_issues"].append(trial)
                categorized = True
            
            # Check recruitment issues  
            elif any(keyword in why_stopped for keyword in recruitment_keywords):
                patterns["recruitment_issues"].append(trial)
                categorized = True
            
            # Check business reasons
            elif any(keyword in why_stopped for keyword in business_keywords):
                patterns["business_reasons"].append(trial)
                categorized = True
            
            # Other reasons
            elif why_stopped and len(why_stopped) > 10:
                patterns["other_reasons"].append(trial)
                categorized = True
            
            # Unknown
            if not categorized:
                patterns["unknown"].append(trial)
        
        return patterns
    
    async def get_drug_repurposing_candidates(
        self,
        disease: str,
        exclude_targets: List[str] = None
    ) -> List[Dict]:
        """Find drug repurposing candidates for a disease"""
        
        # Search for all trials in the disease
        all_trials = await self.search_trials(
            condition=disease,
            max_results=200
        )
        
        # Group by intervention
        intervention_stats = {}
        
        for trial in all_trials:
            for intervention in trial.get("interventions", []):
                if intervention["type"] != "DRUG":
                    continue
                
                name = intervention["name"]
                if exclude_targets and any(target in name for target in exclude_targets):
                    continue
                
                # Filter out placebo and non-drug interventions
                name_lower = name.lower()
                non_drug_terms = [
                    "placebo", "saline", "standard care", "usual care", 
                    "observation", "control", "sham", "vehicle",
                    "no treatment", "supportive care", "best supportive care"
                ]
                if any(term in name_lower for term in non_drug_terms):
                    continue
                
                if name not in intervention_stats:
                    intervention_stats[name] = {
                        "drug": name,
                        "total_trials": 0,
                        "completed": 0,
                        "failed": 0,
                        "ongoing": 0,
                        "phases": set(),
                        "trials": [],
                        # Track failure reasons
                        "safety_failures": 0,
                        "efficacy_failures": 0,
                        "recruitment_failures": 0,
                        "business_failures": 0,
                        "other_failures": 0
                    }
                
                stats = intervention_stats[name]
                stats["total_trials"] += 1
                stats["trials"].append(trial["nct_id"])
                
                # Update phase info
                for phase in trial.get("phase", []):
                    stats["phases"].add(phase)
                
                # Categorize by status
                status = trial["status"]
                if status == "COMPLETED":
                    stats["completed"] += 1
                elif status in ["TERMINATED", "WITHDRAWN", "SUSPENDED"]:
                    stats["failed"] += 1
                    
                    # Track failure reason for scoring
                    why_stopped = trial.get("why_stopped", "").lower()
                    
                    # Categorize failure reason
                    safety_keywords = ["safety", "adverse", "toxicity", "dose limiting", "tolerability", "side effect"]
                    efficacy_keywords = ["efficacy", "ineffective", "futility", "lack of efficacy", "no improvement", "failed to meet"]
                    recruitment_keywords = ["recruitment", "enrollment", "accrual", "low enrollment", "slow recruitment"]
                    business_keywords = ["business", "funding", "sponsor", "strategic", "company decision", "commercial"]
                    
                    if any(keyword in why_stopped for keyword in safety_keywords):
                        stats["safety_failures"] += 1
                    elif any(keyword in why_stopped for keyword in efficacy_keywords):
                        stats["efficacy_failures"] += 1
                    elif any(keyword in why_stopped for keyword in recruitment_keywords):
                        stats["recruitment_failures"] += 1
                    elif any(keyword in why_stopped for keyword in business_keywords):
                        stats["business_failures"] += 1
                    else:
                        stats["other_failures"] += 1
                        
                elif status in ["RECRUITING", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING"]:
                    stats["ongoing"] += 1
        
        # Convert to list and calculate scores
        candidates = []
        for name, stats in intervention_stats.items():
            # Calculate repurposing score
            # Higher score for drugs that have been tested but not in late stage
            score = 0
            
            # Points for having trials
            score += min(stats["total_trials"] * 10, 50)
            
            # Points for early phase trials (more opportunity)
            if "PHASE1" in stats["phases"]:
                score += 20
            if "PHASE2" in stats["phases"]:
                score += 15
            if "PHASE3" not in stats["phases"]:  # Bonus if hasn't reached phase 3
                score += 10
            
            # Points for mixed results (some success)
            if stats["completed"] > 0:
                score += 15
            
            # Penalty for many failures
            failure_rate = stats["failed"] / max(stats["total_trials"], 1)
            score -= failure_rate * 20
            
            # BONUS for good failure reasons (drug wasn't the problem)
            # Get failure reason breakdown from trials
            reason_bonuses = {
                "recruitment": 50,    # BEST: Drug is fine, just couldn't find patients
                "business": 40,       # EXCELLENT: Drug works, just funding issues  
                "other": 10,          # GOOD: Non-drug issues
                "efficacy": -30,      # BAD: Drug doesn't work
                "safety": -50         # WORST: Drug is dangerous
            }
            
            # Apply failure reason bonuses
            for reason, bonus in reason_bonuses.items():
                reason_count = stats.get(f"{reason}_failures", 0)
                if reason_count > 0:
                    # Weight by proportion of trials with this reason
                    reason_weight = reason_count / max(stats["total_trials"], 1)
                    score += bonus * reason_weight
            
            stats["phases"] = list(stats["phases"])
            stats["repurposing_score"] = score
            
            candidates.append(stats)
        
        # Sort by score
        candidates.sort(key=lambda x: x["repurposing_score"], reverse=True)
        
        return candidates[:20]  # Top 20 candidates
