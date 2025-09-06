#!/usr/bin/env python3
"""
High Tide vs AAVE Comparison Runner

Executes the complete comparison analysis between High Tide active rebalancing
and AAVE-style traditional liquidation mechanisms with Monte Carlo analysis.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from tidal_protocol_sim.stress_testing.comparison_scenarios import HighTideVsAaveComparison, ComparisonConfig
    from tidal_protocol_sim.analysis.report_builder import generate_liquidation_comparison_report
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def run_quick_comparison_demo():
    """Run a quick demonstration with fewer Monte Carlo runs"""
    
    print("=" * 80)
    print("HIGH TIDE vs AAVE QUICK COMPARISON DEMO")
    print("=" * 80)
    print()
    
    # Create quick demo configuration
    config = ComparisonConfig()
    config.num_monte_carlo_runs = 5  # Quick demo with 5 runs
    config.scenario_name = "High_Tide_vs_Aave_Quick_Demo"
    
    print(f"Running {config.num_monte_carlo_runs} Monte Carlo iterations for demo...")
    print("⚠️  Note: This is a quick demo. Full analysis requires 30+ runs for statistical significance.")
    print()
    
    # Run comparison
    comparison = HighTideVsAaveComparison(config)
    results = comparison.run_comparison_analysis()
    
    # Generate and display summary
    print_results_summary(results)
    
    # Generate report
    report_content = generate_liquidation_comparison_report(results)
    
    # Save report
    output_dir = Path("tidal_protocol_sim/results") / config.scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "comparison_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n📄 Comparison report saved to: {report_path}")
    
    return results


def run_full_statistical_analysis():
    """Run full statistical analysis with sufficient Monte Carlo runs"""
    
    print("=" * 80)
    print("HIGH TIDE vs AAVE FULL STATISTICAL ANALYSIS")
    print("=" * 80)
    print()
    
    # Create full analysis configuration
    config = ComparisonConfig()
    config.num_monte_carlo_runs = 50  # Full statistical analysis
    config.scenario_name = "High_Tide_vs_Aave_Statistical_Analysis"
    
    print(f"Running {config.num_monte_carlo_runs} Monte Carlo iterations for statistical significance...")
    print("⏱️  This may take several minutes to complete.")
    print()
    
    # Run comparison
    comparison = HighTideVsAaveComparison(config)
    results = comparison.run_comparison_analysis()
    
    # Generate and display summary
    print_results_summary(results)
    
    # Generate comprehensive report
    report_content = generate_liquidation_comparison_report(results)
    
    # Save report
    output_dir = Path("tidal_protocol_sim/results") / config.scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "statistical_analysis_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n📄 Statistical analysis report saved to: {report_path}")
    
    return results


def print_results_summary(results: dict):
    """Print a summary of the comparison results"""
    
    stats = results.get("comparison_statistics", {})
    performance = stats.get("performance_summary", {})
    
    survival_comparison = stats.get("survival_rate", {})
    cost_comparison = stats.get("cost_per_agent", {})
    
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS SUMMARY")
    print("=" * 60)
    
    # Survival rate comparison
    if survival_comparison:
        ht_survival = survival_comparison.get("high_tide", {}).get("mean", 0) * 100
        aave_survival = survival_comparison.get("aave", {}).get("mean", 0) * 100
        survival_improvement = survival_comparison.get("performance_difference", {}).get("percentage_improvement", 0)
        
        print(f"\n🎯 Survival Rate Analysis:")
        print(f"   High Tide:  {ht_survival:.1f}%")
        print(f"   AAVE:       {aave_survival:.1f}%")
        print(f"   Improvement: +{survival_improvement:.1f}%")
    
    # Cost comparison
    if cost_comparison:
        ht_cost = cost_comparison.get("high_tide", {}).get("mean", 0)
        aave_cost = cost_comparison.get("aave", {}).get("mean", 0)
        cost_reduction = abs(cost_comparison.get("performance_difference", {}).get("percentage_improvement", 0))
        
        print(f"\n💰 Cost Per Agent Analysis:")
        print(f"   High Tide:  ${ht_cost:,.0f}")
        print(f"   AAVE:       ${aave_cost:,.0f}")
        print(f"   Reduction:  -{cost_reduction:.1f}%")
    
    # Overall performance
    if performance:
        win_rate = performance.get("overall_win_rate", 0) * 100
        total_runs = performance.get("total_runs", 0)
        
        print(f"\n📊 Overall Performance:")
        print(f"   High Tide Win Rate: {win_rate:.0f}% ({total_runs} runs)")
        print(f"   Statistical Power:  {performance.get('statistical_power', 'Unknown')}")
    
    # Statistical significance
    survival_sig = survival_comparison.get("statistical_significance", {})
    cost_sig = cost_comparison.get("statistical_significance", {})
    
    print(f"\n🔬 Statistical Significance:")
    print(f"   Survival Rate: {survival_sig.get('confidence_level', 'Not Available')}")
    print(f"   Cost Reduction: {cost_sig.get('confidence_level', 'Not Available')}")
    
    # Risk profile analysis
    risk_analysis = stats.get("risk_profile_analysis", {})
    if risk_analysis:
        print(f"\n📈 Risk Profile Performance:")
        for profile in ["conservative", "moderate", "aggressive"]:
            if profile in risk_analysis:
                profile_stats = risk_analysis[profile]
                improvement = profile_stats.get("performance_difference", {}).get("percentage_improvement", 0)
                print(f"   {profile.title()}: +{improvement:.1f}% survival improvement")
    
    print("\n" + "=" * 60)


def test_individual_scenarios():
    """Test individual High Tide and AAVE scenarios"""
    
    print("=" * 80)
    print("TESTING INDIVIDUAL SCENARIOS")
    print("=" * 80)
    
    # Test High Tide scenario
    print("\n🌊 Testing High Tide scenario...")
    try:
        from tidal_protocol_sim.simulation.high_tide_vault_engine import HighTideVaultEngine, HighTideConfig
        
        config = HighTideConfig()
        config.num_high_tide_agents = 10  # Small test
        config.btc_decline_duration = 10   # Short test
        
        engine = HighTideVaultEngine(config)
        ht_results = engine.run_simulation()
        
        # Check key results
        agent_outcomes = ht_results.get("agent_outcomes", [])
        survival_rate = sum(1 for a in agent_outcomes if a["survived"]) / len(agent_outcomes) if agent_outcomes else 0
        
        print(f"   ✅ High Tide test completed")
        print(f"   📊 {len(agent_outcomes)} agents, {survival_rate:.1%} survival rate")
        
    except Exception as e:
        print(f"   ❌ High Tide test failed: {e}")
        return False
    
    # Test AAVE scenario
    print("\n🏛️ Testing AAVE scenario...")
    try:
        from tidal_protocol_sim.simulation.aave_protocol_engine import AaveProtocolEngine, AaveConfig
        
        config = AaveConfig()
        config.num_aave_agents = 10   # Small test
        config.btc_decline_duration = 10  # Short test
        
        engine = AaveProtocolEngine(config)
        aave_results = engine.run_simulation()
        
        # Check key results
        agent_outcomes = aave_results.get("agent_outcomes", [])
        survival_rate = sum(1 for a in agent_outcomes if a["survived"]) / len(agent_outcomes) if agent_outcomes else 0
        liquidation_events = aave_results.get("liquidation_activity", {}).get("total_liquidation_events", 0)
        
        print(f"   ✅ AAVE test completed")
        print(f"   📊 {len(agent_outcomes)} agents, {survival_rate:.1%} survival rate")
        print(f"   ⚠️  {liquidation_events} liquidation events")
        
    except Exception as e:
        print(f"   ❌ AAVE test failed: {e}")
        return False
    
    print("\n✅ All individual scenario tests passed!")
    return True


def main():
    """Main execution function with user choice"""
    
    print("High Tide vs AAVE Liquidation Mechanism Comparison")
    print("=" * 50)
    print()
    print("Choose analysis type:")
    print("1. Quick Demo (5 runs) - Fast demonstration")
    print("2. Full Statistical Analysis (50 runs) - Complete study")  
    print("3. Test Individual Scenarios - Verify implementations")
    print("4. All of the above")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        run_quick_comparison_demo()
    elif choice == "2":
        run_full_statistical_analysis()
    elif choice == "3":
        test_individual_scenarios()
    elif choice == "4":
        print("Running complete analysis suite...\n")
        
        # Test individual scenarios first
        if test_individual_scenarios():
            # Run quick demo
            run_quick_comparison_demo()
            
            # Ask if user wants full analysis
            print("\nQuick demo completed. Run full statistical analysis? (y/n): ", end="")
            if input().strip().lower() in ['y', 'yes']:
                run_full_statistical_analysis()
        else:
            print("❌ Individual scenario tests failed. Please check implementation.")
    else:
        print("Invalid choice. Please run the script again and select 1-4.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
