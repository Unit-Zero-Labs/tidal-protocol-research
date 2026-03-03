#!/usr/bin/env python3
"""
Test script for Monte Carlo framework
Runs 10 MC simulations to verify the framework
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sim_tests.base_case_ht_vs_aave_comparison import BaseCaseConfig, BaseCaseComparison


def main():
    """Run Monte Carlo test"""
    
    print("=" * 80)
    print("🎲 MONTE CARLO FRAMEWORK TEST")
    print("=" * 80)
    print()
    
    # Create MC configuration
    config = BaseCaseConfig()
    config.num_monte_carlo_runs = 10  # 10 MC runs
    config.random_seed_base = 42
    config.num_agents = 20  # Keep small for faster testing
    config.simulation_duration_days = 90
    
    print(f"Configuration:")
    print(f"  - Monte Carlo Runs: {config.num_monte_carlo_runs}")
    print(f"  - Random Seed Base: {config.random_seed_base}")
    print(f"  - Agents per run: {config.num_agents}")
    print(f"  - Duration: {config.simulation_duration_days} days")
    print()
    
    # Run comparison
    comparison = BaseCaseComparison(config)
    results = comparison.run_comparison()
    
    print("\n" + "=" * 80)
    print("✅ MONTE CARLO TEST COMPLETED")
    print("=" * 80)
    
    # Print key results
    if config.num_monte_carlo_runs > 1:
        ht_metrics = results["high_tide_results"]["final_metrics"]
        aave_metrics = results["aave_results"]["final_metrics"]
        
        print(f"\n📊 HIGH TIDE RESULTS (Averaged over {config.num_monte_carlo_runs} runs):")
        print(f"   Avg Net APY: {ht_metrics['avg_net_apy']:.2%} ± {ht_metrics['std_net_apy']:.2%}")
        print(f"   Range: {ht_metrics['min_net_apy']:.2%} - {ht_metrics['max_net_apy']:.2%}")
        print(f"   Avg Survival: {ht_metrics['avg_survival_rate']:.1%} ± {ht_metrics['std_survival_rate']:.1%}")
        
        print(f"\n📊 AAVE RESULTS (Averaged over {config.num_monte_carlo_runs} runs):")
        print(f"   Avg Net APY: {aave_metrics['avg_net_apy']:.2%} ± {aave_metrics['std_net_apy']:.2%}")
        print(f"   Range: {aave_metrics['min_net_apy']:.2%} - {aave_metrics['max_net_apy']:.2%}")
        print(f"   Avg Survival: {aave_metrics['avg_survival_rate']:.1%} ± {aave_metrics['std_survival_rate']:.1%}")
        
        print(f"\n🔬 DELTA:")
        print(f"   Net APY: {(ht_metrics['avg_net_apy'] - aave_metrics['avg_net_apy']):.2%}")
        print(f"   Survival: {(ht_metrics['avg_survival_rate'] - aave_metrics['avg_survival_rate']):.1%}")
    
    return results


if __name__ == "__main__":
    main()


