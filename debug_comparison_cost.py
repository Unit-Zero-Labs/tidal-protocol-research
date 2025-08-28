#!/usr/bin/env python3
"""
Debug script to understand comparison cost calculation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tidal_protocol_sim.stress_testing.comparison_scenarios import HighTideVsAaveComparison, ComparisonConfig

def main():
    print("=" * 60)
    print("DEBUGGING COMPARISON COST CALCULATION")
    print("=" * 60)
    
    # Run a small comparison
    config = ComparisonConfig()
    config.num_monte_carlo_runs = 3  # Just 3 runs for debugging
    config.scenario_name = "Debug_Comparison"
    
    comparison = HighTideVsAaveComparison(config)
    
    print("Running 3 debug runs...")
    
    # Run each scenario manually to debug
    for run_id in range(3):
        print(f"\n{'='*40}")
        print(f"DEBUG RUN {run_id + 1}")
        print(f"{'='*40}")
        
        run_seed = 42 + run_id
        
        # Run AAVE scenario
        aave_results = comparison._run_aave_scenario(run_id, run_seed)
        
        cost_analysis = aave_results.get("cost_analysis", {})
        agent_outcomes = aave_results.get("agent_outcomes", [])
        liquidation_activity = aave_results.get("liquidation_activity", {})
        
        print(f"\n🏛️ AAVE Results for Run {run_id + 1}:")
        print(f"  Total agents: {len(agent_outcomes)}")
        print(f"  Liquidation events: {liquidation_activity.get('total_liquidation_events', 0)}")
        print(f"  Total penalties collected: ${liquidation_activity.get('total_penalties_collected', 0):,.2f}")
        
        # Count liquidated agents
        liquidated_agents = [a for a in agent_outcomes if a.get("liquidation_events", 0) > 0]
        print(f"  Liquidated agents: {len(liquidated_agents)}")
        
        if liquidated_agents:
            penalties = [a.get("liquidation_penalties", 0) for a in liquidated_agents]
            manual_avg = sum(penalties) / len(penalties)
            print(f"  Individual penalties: {[f'${p:.0f}' for p in penalties]}")
            print(f"  Manual average: ${manual_avg:.2f}")
        
        print(f"  Cost analysis average_cost_per_agent: ${cost_analysis.get('average_cost_per_agent', 0):.2f}")
        print(f"  Cost analysis average_cost_per_all_agents: ${cost_analysis.get('average_cost_per_all_agents', 0):.2f}")
    
    print(f"\n{'='*60}")
    print("RUNNING FULL COMPARISON...")
    print(f"{'='*60}")
    
    # Now run the full comparison
    results = comparison.run_comparison_analysis()
    
    # Extract the statistics
    comparison_stats = results.get("comparison_statistics", {})
    cost_stats = comparison_stats.get("cost_per_agent", {})
    
    print(f"\n📊 Final Comparison Statistics:")
    print(f"  AAVE cost mean: ${cost_stats.get('aave', {}).get('mean', 0):.2f}")
    print(f"  AAVE cost std: ${cost_stats.get('aave', {}).get('std', 0):.2f}")
    print(f"  AAVE all runs: {cost_stats.get('aave', {}).get('values', [])}")

if __name__ == "__main__":
    main()
