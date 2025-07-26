"""
Production-quality Therapeutic Target Agent
"""
import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .data.pubmed_client import PubMedClient
from .data.chembl_client import ChEMBLClient  
from .data.pdb_client import PDBClient


class TherapeuticTargetAgent:
    """Production therapeutic target discovery agent"""
    
    def __init__(self, ncbi_api_key: Optional[str] = None, cache_dir: Path = Path("cache")):
        self.pubmed_client = PubMedClient(api_key=ncbi_api_key)
        self.chembl_client = ChEMBLClient()
        self.pdb_client = PDBClient()
        self.cache_dir = cache_dir
        
    async def analyze_target(self, gene_symbol: str, 
                           max_ic50_nm: Optional[float] = None,
                           min_ic50_nm: Optional[float] = None) -> Dict[str, Any]:
        """Comprehensive target analysis with async data gathering"""
        
        print(f"🎯 Analyzing target: {gene_symbol}")
        
        # Parse IC50 range if not provided
        if max_ic50_nm is None and min_ic50_nm is None:
            min_ic50_nm, max_ic50_nm = self._parse_ic50_range(gene_symbol)
        
        # Gather data from all sources concurrently
        tasks = {
            "literature": self.pubmed_client.search_articles(f"{gene_symbol} inhibitors"),
            "inhibitors": self.chembl_client.get_inhibitors_for_target(
                gene_symbol, max_ic50_nm, min_ic50_nm
            ),
            "structures": self.pdb_client.search_structures(gene_symbol)
        }
        
        print("📊 Gathering data from PubMed, ChEMBL, and PDB...")
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
                print(f"✅ {name.title()}: {len(results[name])} results")
            except Exception as e:
                print(f"❌ {name.title()} error: {e}")
                results[name] = []
        
        # Cross-validate inhibitors with PDB structures
        if results["inhibitors"]:
            print("🔍 Cross-validating inhibitors with PDB structures...")
            chembl_ids = [inh.get("molecule_chembl_id") for inh in results["inhibitors"]]
            ligand_structures = await self.pdb_client.get_ligand_structures(chembl_ids)
            
            # Add PDB structure info to inhibitors
            for inhibitor in results["inhibitors"]:
                chembl_id = inhibitor.get("molecule_chembl_id")
                inhibitor["pdb_structures"] = ligand_structures.get(chembl_id, [])
        
        # Calculate comprehensive target score
        target_score = self._calculate_target_score(results)
        
        # Generate summary
        summary = self._generate_target_summary(gene_symbol, results, target_score)
        
        return {
            "gene_symbol": gene_symbol,
            "analysis_timestamp": datetime.now().isoformat(),
            "target_score": target_score,
            "summary": summary,
            "literature": results["literature"],
            "inhibitors": results["inhibitors"], 
            "structures": results["structures"],
            "query_filters": {
                "min_ic50_nm": min_ic50_nm,
                "max_ic50_nm": max_ic50_nm
            }
        }
    
    async def multi_target_analysis(self, gene_symbols: List[str],
                                  max_ic50_nm: Optional[float] = None,
                                  min_ic50_nm: Optional[float] = None) -> Dict[str, Any]:
        """Analyze multiple targets concurrently"""
        
        print(f"🎯 Multi-target analysis: {', '.join(gene_symbols)}")
        
        # Analyze all targets concurrently
        tasks = []
        for gene_symbol in gene_symbols:
            task = self.analyze_target(gene_symbol, max_ic50_nm, min_ic50_nm)
            tasks.append((gene_symbol, task))
        
        target_results = []
        for gene_symbol, task in tasks:
            try:
                result = await task
                target_results.append(result)
            except Exception as e:
                print(f"❌ Error analyzing {gene_symbol}: {e}")
                continue
        
        # Rank targets by score
        target_results.sort(key=lambda x: x.get("target_score", 0), reverse=True)
        
        # Generate multi-target summary
        multi_summary = self._generate_multi_target_summary(target_results)
        
        return {
            "query": f"Multi-target analysis: {', '.join(gene_symbols)}",
            "analysis_timestamp": datetime.now().isoformat(),
            "targets": target_results,
            "summary": multi_summary,
            "query_filters": {
                "min_ic50_nm": min_ic50_nm,
                "max_ic50_nm": max_ic50_nm
            }
        }
    
    def _parse_ic50_range(self, query: str) -> Tuple[Optional[float], Optional[float]]:
        """Parse IC50 range from natural language query"""
        query_lower = query.lower()
        
        # Range patterns
        range_patterns = [
            r'between\s+(\d*\.?\d+)\s*n?m\s+and\s+(\d*\.?\d+)\s*n?m',
            r'between\s+(\d*\.?\d+)\s*and\s+(\d*\.?\d+)\s*n?m',
            r'from\s+(\d*\.?\d+)\s*to\s+(\d*\.?\d+)\s*n?m',
            r'(\d*\.?\d+)\s*-\s*(\d*\.?\d+)\s*n?m'
        ]
        
        for pattern in range_patterns:
            match = re.search(pattern, query_lower)
            if match:
                min_val, max_val = map(float, match.groups())
                return (min_val, max_val)
        
        # Upper bound patterns
        upper_patterns = [
            r'under\s+(\d*\.?\d+)\s*n?m',
            r'below\s+(\d*\.?\d+)\s*n?m',
            r'less\s+than\s+(\d*\.?\d+)\s*n?m',
            r'<\s*(\d*\.?\d+)\s*n?m'
        ]
        
        for pattern in upper_patterns:
            match = re.search(pattern, query_lower)
            if match:
                max_val = float(match.group(1))
                return (None, max_val)
        
        return (None, None)
    
    def _calculate_target_score(self, results: Dict[str, List]) -> float:
        """Calculate comprehensive target druggability score (0-10)"""
        score = 0.0
        
        # Literature evidence (0-3 points)
        literature_count = len(results.get("literature", []))
        if literature_count >= 20:
            score += 3.0
        elif literature_count >= 10:
            score += 2.0
        elif literature_count >= 5:
            score += 1.0
        elif literature_count >= 1:
            score += 0.5
        
        # Inhibitor potency and diversity (0-4 points)
        inhibitors = results.get("inhibitors", [])
        if inhibitors:
            # Potency score based on best inhibitor
            best_ic50 = min([inh.get("standard_value_nm", float('inf')) for inh in inhibitors])
            if best_ic50 <= 1:
                score += 2.0
            elif best_ic50 <= 10:
                score += 1.5
            elif best_ic50 <= 50:
                score += 1.0
            elif best_ic50 <= 100:
                score += 0.5
            
            # We could add diversity score based on scaffold diversity here
            # For now, just reward having multiple inhibitors
            if len(inhibitors) >= 10:
                score += 1.0
            elif len(inhibitors) >= 5:
                score += 0.5
            
            # Quality bonus for high-quality assays
            avg_quality = sum([inh.get("quality_score", 0) for inh in inhibitors]) / len(inhibitors)
            score += avg_quality * 1.0
        
        # Structural data (0-3 points)
        structures = results.get("structures", [])
        if structures:
            # Base structural bonus
            score += 1.0
            
            # Quality bonus
            avg_structure_quality = sum([struct.get("quality_score", 0) for struct in structures]) / len(structures)
            score += avg_structure_quality * 1.0
            
            # Ligand-bound structure bonus
            ligand_structures = [s for s in structures if s.get("ligands")]
            if ligand_structures:
                score += 1.0
        
        return min(10.0, score)
    
    def _generate_target_summary(self, gene_symbol: str, results: Dict[str, List], score: float) -> str:
        """Generate human-readable target summary"""
        
        inhibitor_count = len(results.get("inhibitors", []))
        literature_count = len(results.get("literature", []))
        structure_count = len(results.get("structures", []))
        
        # Potency summary
        potency_summary = "No inhibitors found"
        if results.get("inhibitors"):
            best_ic50 = min([inh.get("standard_value_nm", float('inf')) for inh in results["inhibitors"]])
            potency_summary = f"Best inhibitor: {best_ic50:.2f} nM"
        
        # Structure quality summary
        structure_summary = "No structures available"
        if results.get("structures"):
            high_quality = [s for s in results["structures"] if s.get("quality_score", 0) >= 0.7]
            ligand_bound = [s for s in results["structures"] if s.get("ligands")]
            structure_summary = f"{structure_count} structures ({len(high_quality)} high-quality, {len(ligand_bound)} ligand-bound)"
        
        # Overall assessment
        if score >= 8.0:
            assessment = "Excellent drug target"
        elif score >= 6.0:
            assessment = "Good drug target"
        elif score >= 4.0:
            assessment = "Moderate drug target"
        elif score >= 2.0:
            assessment = "Challenging drug target"
        else:
            assessment = "Poor drug target"
        
        return f"""
{gene_symbol} Target Analysis (Score: {score:.1f}/10.0)
Assessment: {assessment}

📊 Data Summary:
• Literature: {literature_count} papers
• Inhibitors: {inhibitor_count} compounds
• {potency_summary}
• Structures: {structure_summary}
        """.strip()
    
    def _generate_multi_target_summary(self, target_results: List[Dict]) -> str:
        """Generate summary for multi-target analysis"""
        
        total_targets = len(target_results)
        if total_targets == 0:
            return "No targets successfully analyzed"
        
        avg_score = sum([t.get("target_score", 0) for t in target_results]) / total_targets
        
        # Categorize targets
        excellent = [t for t in target_results if t.get("target_score", 0) >= 8.0]
        good = [t for t in target_results if 6.0 <= t.get("target_score", 0) < 8.0]
        moderate = [t for t in target_results if 4.0 <= t.get("target_score", 0) < 6.0]
        challenging = [t for t in target_results if t.get("target_score", 0) < 4.0]
        
        summary = f"""
Multi-Target Analysis Summary
Analyzed {total_targets} targets (Average score: {avg_score:.1f}/10.0)

🎯 Target Classification:
• Excellent targets ({len(excellent)}): {', '.join([t['gene_symbol'] for t in excellent])}
• Good targets ({len(good)}): {', '.join([t['gene_symbol'] for t in good])}
• Moderate targets ({len(moderate)}): {', '.join([t['gene_symbol'] for t in moderate])}
• Challenging targets ({len(challenging)}): {', '.join([t['gene_symbol'] for t in challenging])}
        """.strip()
        
        return summary
    
    async def save_results(self, results: Dict[str, Any], output_file: Path):
        """Save analysis results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Results saved to {output_file}")
    
    async def cleanup(self):
        """Cleanup resources"""
        # Close HTTP clients if needed
        pass
