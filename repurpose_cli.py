#!/usr/bin/env python3
"""
CLI for Drug Repurposing Analysis using Clinical Trials data
"""
import asyncio
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from thera_agent.repurposing_agent import DrugRepurposingAgent

def create_parser():
    parser = argparse.ArgumentParser(
        description="Omics Oracle Drug Repurposing Agent - Analyze clinical failures and find new opportunities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all failures for a disease
  python repurpose_cli.py "glioblastoma"
  
  # Analyze failures for a specific target in a disease  
  python repurpose_cli.py "glioblastoma" --target EGFR
  
  # Output to JSON file
  python repurpose_cli.py "COVID-19" --target "3CLpro" --output covid_repurposing.json
  
  # Show only top alternatives
  python repurpose_cli.py "melanoma" --target BRAF --top 3
        """
    )
    
    parser.add_argument(
        "disease",
        help="Disease to analyze (e.g., 'glioblastoma', 'COVID-19', 'melanoma')"
    )
    
    parser.add_argument(
        "--target",
        help="Specific target that failed (e.g., 'EGFR', 'BRAF', 'PD1')"
    )
    
    parser.add_argument(
        "--output",
        help="Output JSON file for results"
    )
    
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top alternatives to show (default: 5)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed analysis"
    )
    
    parser.add_argument(
        "--max-trials",
        type=int,
        help="Maximum number of trials to analyze for faster testing"
    )
    
    return parser

def format_trial_summary(trial):
    """Format a single trial summary"""
    interventions = ", ".join([i["name"] for i in trial.get("interventions", [])])
    phases = ", ".join(trial.get("phase", ["Unknown"]))
    
    return f"""
  NCT ID: {trial['nct_id']}
  Title: {trial.get('title', 'N/A')[:80]}...
  Status: {trial['status']}
  Phase: {phases}
  Intervention: {interventions}
  Why Stopped: {trial.get('why_stopped', 'Not specified')[:100]}...
    """

def print_results(results, top_n=5, verbose=False):
    """Print formatted results"""
    
    print("\n" + "="*80)
    print(f"🔬 DRUG REPURPOSING ANALYSIS: {results['disease'].upper()}")
    print("="*80)
    
    if results['original_target']:
        print(f"\n❌ Failed Target: {results['original_target']}")
    
    # Failure Summary
    print(f"\n📊 CLINICAL TRIAL FAILURES")
    print(f"Total Failed Trials: {results['failed_trials_count']}")
    print("\nFailure Patterns:")
    for pattern, count in results['failure_patterns'].items():
        if count > 0:
            print(f"  • {pattern.replace('_', ' ').title()}: {count} trials")
    
    # Side Effects Analysis  
    side_effects = results.get('side_effects_analysis', {})
    if side_effects:
        print(f"\n⚠️ SIDE EFFECTS & SAFETY ANALYSIS")
        print(f"Safety-Related Terminations: {side_effects.get('safety_terminations_count', 0)}")
        
        # Why stopped categories
        why_stopped = side_effects.get('why_stopped_categories', {})
        if any(why_stopped.values()):
            print("\nTermination Reasons:")
            for category, count in why_stopped.items():
                if count > 0:
                    print(f"  • {category.replace('_', ' ').title()}: {count} trials")
        
        # Side effect patterns
        patterns = side_effects.get('side_effects_patterns', {})
        if patterns.get('patterns'):
            print(f"\nCommon Side Effect Patterns:")
            for pattern in patterns['patterns'][:3]:
                print(f"  • {pattern}")
        
        if patterns.get('organ_systems'):
            print(f"\nAffected Organ Systems: {', '.join(patterns['organ_systems'])}")
        
        # Repurposing opportunities from side effects
        repurposing_opps = side_effects.get('repurposing_opportunities', [])
        if repurposing_opps:
            print(f"\n💡 SIDE EFFECT → REPURPOSING OPPORTUNITIES:")
            for opp in repurposing_opps[:3]:
                print(f"  • {opp['side_effect']} → {opp['repurposing_disease']}")
                print(f"    Rationale: {opp['rationale'][:80]}...")
    
    # Failure Analysis
    analysis = results['failure_analysis']
    print(f"\n🧬 BIOLOGICAL INSIGHTS")
    
    # Display failed drugs analysis
    if analysis.get('failed_drugs_analysis'):
        print(f"Failed Drug Analysis:")
        for drug in analysis['failed_drugs_analysis'][:3]:
            print(f"  • {drug['drug_name']} ({drug['mechanism']})")
            print(f"    Failure: {drug['failure_type']} - {drug['biological_insight'][:100]}...\n")
    
    if analysis.get('disease_insights'):
        print(f"Disease Insights: {analysis['disease_insights'][:200]}...")
    
    if analysis.get('failed_pathways'):
        print(f"\nFailed Pathways: {', '.join(analysis['failed_pathways'][:3])}")
    
    if analysis.get('alternative_mechanisms'):
        print(f"Promising Alternatives: {', '.join(analysis['alternative_mechanisms'][:3])}")
    
    # Sample Failed Trials
    if verbose and results.get('failed_trials_sample'):
        print(f"\n📋 SAMPLE FAILED TRIALS")
        for trial in results['failed_trials_sample'][:3]:
            print(format_trial_summary(trial))
    
    # Drug Repurposing vs Rescue Opportunities
    drug_repurposing = results.get('drug_repurposing', [])
    drug_rescue = results.get('drug_rescue', [])
    
    # Display FDA-Approved Repurposing Opportunities
    if drug_repurposing:
        print(f"\n🔄 DRUG REPURPOSING OPPORTUNITIES (FDA-Approved → New Indication)")
        print(f"{'Drug':<25} {'Targets':<35} {'Owner':<25} {'Trials':<8} {'Score':<8} {'Phases':<15} {'Availability':<12}")
        print("-" * 145)
        
        for candidate in drug_repurposing[:top_n]:
            phases = ", ".join(sorted(set(candidate.get('phases', []))))
            owner = candidate.get('current_owner', 'Unknown')[:23]
            availability = candidate.get('availability_status', 'unknown')[:10]
            targets = ", ".join(candidate.get('primary_targets', ['Unknown'])[:2])[:33]
            print(f"{candidate.get('drug', 'Unknown'):<25} {targets:<35} {owner:<25} {candidate.get('total_trials', 0):<8} {candidate.get('repurposing_score', 0):<8.1f} {phases:<15} {availability:<12}")
    
    # Display Experimental Drug Rescue Opportunities  
    if drug_rescue:
        print(f"\n🚑 DRUG RESCUE OPPORTUNITIES (Failed/Experimental → Different Indication)")
        print(f"{'Drug':<25} {'Targets':<35} {'Owner':<25} {'Trials':<8} {'Score':<8} {'Phases':<15} {'Availability':<12}")
        print("-" * 145)
        
        for candidate in drug_rescue[:top_n]:
            phases = ", ".join(sorted(set(candidate.get('phases', []))))
            owner = candidate.get('current_owner', 'Unknown')[:23]
            availability = candidate.get('availability_status', 'unknown')[:10]
            targets = ", ".join(candidate.get('primary_targets', ['Unknown'])[:2])[:33]
            print(f"{candidate.get('drug', 'Unknown'):<25} {targets:<35} {owner:<25} {candidate.get('total_trials', 0):<8} {candidate.get('repurposing_score', 0):<8.1f} {phases:<15} {availability:<12}")
    
    # Safety profiles for top candidates
    safety_profiles = results.get('candidate_safety_profiles', [])
    if safety_profiles:
        print(f"\n🛡️ COMPREHENSIVE SAFETY PROFILES")
        print("=" * 80)
        
        for i, profile in enumerate(safety_profiles[:3], 1):
            drug_name = profile['drug_name']
            print(f"\n{i}. {drug_name.upper()}")
            print("-" * 50)
            
            # Adverse events
            ae_data = profile.get('fda_adverse_events', {})
            ae_list = ae_data.get('top_adverse_events', [])
            if ae_list and ae_list[0] != "Would be retrieved from FDA OpenFDA API":
                print(f"⚠️ Top Adverse Events:")
                for event in ae_list[:3]:
                    print(f"  • {event}")
            
            # Drug interactions
            interactions = profile.get('drug_interactions', [])
            if interactions and interactions[0] != "Would be retrieved from RxNav API":
                print(f"💊 Major Drug Interactions:")
                for interaction in interactions[:2]:
                    print(f"  • {interaction}")
            
            # Contraindications
            contras = profile.get('contraindications', [])
            if contras and contras[0] != "Would be retrieved from FDA drug labels":
                print(f"🚫 Key Contraindications:")
                for contra in contras[:2]:
                    print(f"  • {contra}")
            
            # Mechanism
            mechanism = profile.get('mechanism_summary', '')
            if mechanism and not mechanism.startswith("Would be retrieved"):
                print(f"🧬 Mechanism: {mechanism}")
            
            # ChEMBL ID if available
            if profile.get('chembl_id'):
                print(f"🔗 ChEMBL ID: {profile['chembl_id']}")
    
    # Alternative Targets
    print(f"\n🎯 ALTERNATIVE THERAPEUTIC TARGETS")
    alternatives = results.get('alternative_targets', [])[:top_n]
    
    if alternatives:
        for i, alt in enumerate(alternatives, 1):
            print(f"\n{i}. {alt['target']} (Confidence: {alt.get('confidence', 0):.1%})")
            print(f"   Rationale: {alt.get('rationale', 'N/A')[:150]}...")
            
            # Target metrics
            if 'inhibitor_count' in alt:
                print(f"   • Inhibitors: {alt['inhibitor_count']}")
                if alt.get('most_potent_ic50'):
                    print(f"   • Most Potent IC50: {alt['most_potent_ic50']:.1f} nM")
                
                # Show most advanced compound
                compound = alt.get('most_advanced_compound', {})
                if compound.get('chembl_id'):
                    phase = compound.get('phase', 'N/A')
                    phase_str = f"Phase {phase}" if phase else "Preclinical"
                    print(f"   • Most Advanced: {compound['chembl_id']} ({phase_str})")
                
                trials = alt.get('clinical_trials', {})
                if trials:
                    print(f"   • Clinical Trials: {trials['total']} total "
                          f"({trials['recruiting']} recruiting, "
                          f"{trials['completed']} completed)")
                
                print(f"   • Development Score: {alt.get('development_score', 0):.2f}/1.00")
                
                # Show PDB structures if available
                pdb_structures = alt.get('pdb_structures', [])
                if pdb_structures:
                    print(f"   • PDB Structures: {len(pdb_structures)} available")
                    for pdb in pdb_structures[:2]:  # Show first 2
                        print(f"     - {pdb.get('pdb_id', 'N/A')}: {pdb.get('title', 'N/A')[:60]}...")
                        if pdb.get('resolution') != 'N/A':
                            print(f"       Resolution: {pdb.get('resolution')} Å, Method: {pdb.get('method', 'N/A')}")
    
    print("\n" + "="*80)

async def main():
    parser = create_parser()
    args = parser.parse_args()
    
    print("🚀 Initializing Omics Oracle Repurposing Agent...")
    
    try:
        agent = DrugRepurposingAgent()
        
        print(f"🔍 Analyzing {args.disease} failures" + 
              (f" for target {args.target}" if args.target else "") + "...")
        
        # Run analysis
        results = await agent.analyze_disease_failures(
            disease=args.disease,
            target=args.target,
            max_trials=args.max_trials
        )
        
        # Print results
        print_results(results, top_n=args.top, verbose=args.verbose)
        
        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata
            results["metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "query": {
                    "disease": args.disease,
                    "target": args.target
                },
                "version": "1.0"
            }
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✅ Results saved to: {output_path}")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
