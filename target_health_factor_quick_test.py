#!/usr/bin/env python3
"""
Target Health Factor Quick Test

Quick validation of the corrected Target Health Factor analysis with:
- Randomized agent initial HFs (1.2-1.5) 
- Discrete target HF testing: 1.01, 1.05, 1.075, 1.1, 1.15
- Fewer runs for rapid feedback

This validates the statistical design before running the full analysis.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tidal_protocol_sim.simulation.high_tide_engine import HighTideConfig, HighTideSimulationEngine
from tidal_protocol_sim.simulation.aave_engine import AaveConfig, AaveSimulationEngine


def create_varied_agents_for_target_hf_test(target_hf: float, num_agents: int, agent_type: str) -> List:
    """Create agents with randomized initial HFs (1.2-1.5) and fixed target HF"""
    import random
    from tidal_protocol_sim.agents.high_tide_agent import HighTideAgent
    from tidal_protocol_sim.agents.aave_agent import AaveAgent
    
    agents = []
    
    for i in range(num_agents):
        # Randomize initial health factor between 1.2-1.5 for proper variation
        initial_hf = random.uniform(1.2, 1.5)
        
        if agent_type == "high_tide":
            agent = HighTideAgent(
                f"quick_test_ht_{target_hf}_{i}",
                initial_hf,
                target_hf
            )
        else:  # aave
            agent = AaveAgent(
                f"quick_test_aave_{target_hf}_{i}",
                initial_hf,
                target_hf
            )
        
        agents.append(agent)
    
    return agents


def run_quick_target_hf_test():
    """Run quick Target Health Factor validation test"""
    
    print("=" * 60)
    print("TARGET HEALTH FACTOR QUICK TEST")
    print("=" * 60)
    print("Testing design: Randomized Initial HFs (1.2-1.5) with discrete Target HFs")
    print("Target HFs: 1.05, 1.1 (quick validation)")
    print()
    
    # Quick test with just 2 target HFs and fewer runs
    target_hfs = [1.05, 1.1]
    monte_carlo_runs = 5  # Quick validation
    
    results = []
    
    for target_hf in target_hfs:
        print(f"🎯 Testing Target Health Factor: {target_hf}")
        print(f"   Running {monte_carlo_runs} Monte Carlo runs with varied agents")
        
        # Store results from all runs
        ht_runs = []
        aave_runs = []
        
        for run_num in range(monte_carlo_runs):
            print(f"      Run {run_num + 1}/{monte_carlo_runs}", end=" ")
            
            # High Tide simulation
            ht_config = HighTideConfig()
            ht_config.num_high_tide_agents = 0
            ht_config.btc_decline_duration = 60
            ht_config.moet_btc_pool_size = 250_000
            ht_config.moet_yield_pool_size = 250_000
            
            # Create varied High Tide agents
            custom_ht_agents = create_varied_agents_for_target_hf_test(
                target_hf, num_agents=10, agent_type="high_tide"
            )
            
            ht_engine = HighTideSimulationEngine(ht_config)
            ht_engine.high_tide_agents = custom_ht_agents
            
            for agent in custom_ht_agents:
                ht_engine.agents[agent.agent_id] = agent
            
            ht_results = ht_engine.run_high_tide_simulation()
            ht_runs.append(ht_results)
            
            # Aave simulation
            aave_config = AaveConfig()
            aave_config.num_aave_agents = 0
            aave_config.btc_decline_duration = 60
            aave_config.moet_btc_pool_size = 250_000
            aave_config.moet_yield_pool_size = 250_000
            
            custom_aave_agents = create_varied_agents_for_target_hf_test(
                target_hf, num_agents=10, agent_type="aave"
            )
            
            aave_engine = AaveSimulationEngine(aave_config)
            aave_engine.aave_agents = custom_aave_agents
            
            for agent in custom_aave_agents:
                aave_engine.agents[agent.agent_id] = agent
            
            aave_results = aave_engine.run_aave_simulation()
            aave_runs.append(aave_results)
            
            print("✓")
        
        # Aggregate results for this target HF
        scenario_results = aggregate_quick_test_results(ht_runs, aave_runs, target_hf)
        results.append(scenario_results)
        
        print(f"   Results: HT {scenario_results['ht_survival_rate']:.1%} survival, "
              f"Aave {scenario_results['aave_survival_rate']:.1%} survival")
        print()
    
    # Save quick test results
    save_quick_test_results(results)
    
    print("✅ Quick test completed! Ready for full analysis.")
    return results


def aggregate_quick_test_results(ht_runs: List, aave_runs: List, target_hf: float) -> Dict:
    """Aggregate results for quick test"""
    
    # High Tide metrics
    ht_survival_rates = [run.get("survival_statistics", {}).get("survival_rate", 0.0) for run in ht_runs]
    ht_liquidations = []
    
    for run in ht_runs:
        agent_outcomes = run.get("agent_outcomes", [])
        total_liquidations = sum(outcome.get("emergency_liquidations", 0) for outcome in agent_outcomes)
        ht_liquidations.append(total_liquidations)
    
    # Aave metrics
    aave_survival_rates = [run.get("survival_statistics", {}).get("survival_rate", 0.0) for run in aave_runs]
    aave_liquidations = []
    
    for run in aave_runs:
        liquidation_activity = run.get("liquidation_activity", {})
        aave_liquidations.append(liquidation_activity.get("total_liquidation_events", 0))
    
    return {
        "target_hf": target_hf,
        "ht_survival_rate": np.mean(ht_survival_rates),
        "ht_liquidation_frequency": np.mean(ht_liquidations) / 10,  # Per agent
        "aave_survival_rate": np.mean(aave_survival_rates),
        "aave_liquidation_frequency": np.mean(aave_liquidations) / 10,  # Per agent
        "design_validation": {
            "agents_varied": True,
            "initial_hf_range": [1.2, 1.5],
            "fixed_target_hf": target_hf,
            "liquidation_spread": len(set(ht_liquidations)) > 1  # Different liquidation outcomes
        }
    }


def save_quick_test_results(results: List[Dict]):
    """Save quick test results"""
    
    output_dir = Path("tidal_protocol_sim/results/target_health_factor_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    quick_test_results = {
        "test_type": "Quick_Validation_Test",
        "timestamp": datetime.now().isoformat(),
        "design_fix": "Randomized initial HFs (1.2-1.5) with fixed target HFs",
        "target_hfs_tested": [r["target_hf"] for r in results],
        "results": results
    }
    
    results_path = output_dir / "quick_test_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(quick_test_results, f, indent=2, default=str)
    
    print(f"📁 Quick test results saved to: {results_path}")


def main():
    """Main execution function"""
    try:
        results = run_quick_target_hf_test()
        return results
    except KeyboardInterrupt:
        print("\n\nQuick test interrupted by user.")
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()