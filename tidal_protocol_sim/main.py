#!/usr/bin/env python3
"""
Tidal Protocol Stress Testing - Main Entry Point

Streamlined entry point for comprehensive stress testing and liquidity analysis
as specified in the refactoring requirements.
"""

import sys
import argparse
import time
from typing import Dict, List

import sys
import os

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tidal_protocol_sim.stress_testing.runner import StressTestRunner, QuickStressTest
from tidal_protocol_sim.stress_testing.scenarios import TidalStressTestSuite  
from tidal_protocol_sim.simulation.config import SimulationConfig
from tidal_protocol_sim.simulation.engine import TidalSimulationEngine
from tidal_protocol_sim.analysis.metrics import TidalMetricsCalculator


def main():
    """Main entry point with command-line interface"""
    
    parser = argparse.ArgumentParser(
        description="Tidal Protocol Stress Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --quick                           # Quick stress test
  python main.py --scenario ETH_Flash_Crash       # Run specific scenario
  python main.py --full-suite --monte-carlo 500   # Full test suite with 500 MC runs
  python main.py --baseline                       # Generate baseline metrics
        """
    )
    
    # Test type arguments
    parser.add_argument('--quick', action='store_true',
                       help='Run quick stress tests for development')
    
    parser.add_argument('--scenario', type=str,
                       help='Run specific stress test scenario')
    
    parser.add_argument('--full-suite', action='store_true',
                       help='Run complete stress test suite')
    
    parser.add_argument('--baseline', action='store_true',
                       help='Generate baseline protocol metrics')
    
    parser.add_argument('--list-scenarios', action='store_true',
                       help='List all available stress test scenarios')
    
    # Configuration arguments
    parser.add_argument('--monte-carlo', type=int, default=100,
                       help='Number of Monte Carlo runs (default: 100)')
    
    parser.add_argument('--steps', type=int, default=1000,
                       help='Number of simulation steps (default: 1000)')
    
    parser.add_argument('--agents', type=int, nargs=3, default=[5, 3, 2],
                       help='Number of agents: lenders traders liquidators (default: 5 3 2)',
                       metavar=('LENDERS', 'TRADERS', 'LIQUIDATORS'))
    
    parser.add_argument('--output', type=str,
                       help='Export results to JSON file')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.quick, args.scenario, args.full_suite, args.baseline, args.list_scenarios]):
        parser.print_help()
        return 1
    
    # Configure simulation
    config = create_simulation_config(args)
    
    try:
        if args.list_scenarios:
            list_scenarios()
            return 0
        
        elif args.quick:
            print("Running Quick Stress Tests")
            print("=" * 50)
            return run_quick_tests(args.verbose)
        
        elif args.scenario:
            print(f"Running Stress Test Scenario: {args.scenario}")
            print("=" * 60)
            return run_single_scenario(args.scenario, config, args)
        
        elif args.full_suite:
            print("Running Full Stress Test Suite")
            print("=" * 50)
            return run_full_suite(config, args)
        
        elif args.baseline:
            print("Generating Baseline Protocol Metrics")
            print("=" * 50)
            return generate_baseline(config, args)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def create_simulation_config(args) -> SimulationConfig:
    """Create simulation configuration from command-line arguments"""
    
    config = SimulationConfig()
    
    # Apply command-line overrides
    config.simulation_steps = args.steps
    config.num_lenders, config.num_traders, config.num_liquidators = args.agents
    
    return config


def list_scenarios():
    """List all available stress test scenarios"""
    
    test_suite = TidalStressTestSuite()
    scenario_names = test_suite.get_scenario_names()
    
    print("Available Stress Test Scenarios:")
    print("-" * 40)
    
    for i, name in enumerate(scenario_names, 1):
        scenario = next(s for s in test_suite.scenarios if s.name == name)
        print(f"{i:2d}. {name}")
        print(f"    {scenario.description}")
        print()


def run_quick_tests(verbose: bool = False) -> int:
    """Run quick stress tests for development"""
    
    print("1. Quick Liquidation Test")
    print("-" * 25)
    
    start_time = time.time()
    liquidation_result = QuickStressTest.run_liquidation_test(100)
    
    print(f"Liquidations triggered: {liquidation_result['liquidations']}")
    print(f"Protocol treasury: ${liquidation_result['protocol_treasury']:,.2f}")
    print(f"Min health factor: {min(liquidation_result['final_health_factors']):.3f}")
    
    print(f"\n2. Quick Debt Cap Test")
    print("-" * 20)
    
    debt_cap_result = QuickStressTest.run_debt_cap_test(50)
    
    print(f"Initial debt cap: ${debt_cap_result['initial_debt_cap']:,.0f}")
    print(f"Final debt cap: ${debt_cap_result['final_debt_cap']:,.0f}")
    print(f"Debt cap change: {debt_cap_result['debt_cap_change']:.1%}")
    print(f"Total borrowed: ${debt_cap_result['total_borrowed']:,.0f}")
    
    elapsed = time.time() - start_time
    print(f"\nQuick tests completed in {elapsed:.1f}s")
    
    return 0


def run_single_scenario(scenario_name: str, config: SimulationConfig, args) -> int:
    """Run a single stress test scenario"""
    
    runner = StressTestRunner(config)
    
    try:
        if args.monte_carlo > 1:
            print(f"Running Monte Carlo analysis ({args.monte_carlo} runs)")
            results = runner.run_monte_carlo_stress_test(scenario_name, args.monte_carlo)
        else:
            print("Running single scenario")
            results = runner.run_targeted_scenario(scenario_name)
        
        # Display results
        display_scenario_results(scenario_name, results, args.verbose)
        
        # Export if requested
        if args.output:
            export_results({scenario_name: results}, args.output)
        
        return 0
        
    except ValueError as e:
        print(f"Error: {str(e)}")
        print("\nUse --list-scenarios to see available scenarios")
        return 1


def run_full_suite(config: SimulationConfig, args) -> int:
    """Run complete stress test suite"""
    
    runner = StressTestRunner(config)
    
    print(f"Configuration:")
    print(f"  Monte Carlo runs: {args.monte_carlo}")
    print(f"  Simulation steps: {args.steps}")
    print(f"  Agents: {args.agents[0]} lenders, {args.agents[1]} traders, {args.agents[2]} liquidators")
    print()
    
    start_time = time.time()
    
    # Run full suite
    results = runner.run_full_stress_test_suite(args.monte_carlo)
    
    # Display summary
    display_suite_summary(results['suite_summary'], args.verbose)
    
    # Export if requested
    if args.output:
        export_results(results, args.output)
    
    elapsed = time.time() - start_time
    print(f"\nFull stress test suite completed in {elapsed/60:.1f} minutes")
    
    return 0


def generate_baseline(config: SimulationConfig, args) -> int:
    """Generate baseline protocol metrics"""
    
    print("Initializing Tidal Protocol...")
    
    engine = TidalSimulationEngine(config)
    calculator = TidalMetricsCalculator(engine.protocol)
    
    # Run short simulation to get baseline
    print("Running baseline simulation (100 steps)...")
    results = engine.run_simulation(100)
    
    # Calculate comprehensive metrics
    print("Calculating protocol metrics...")
    
    current_prices = engine.state.current_prices
    
    # Protocol health
    health_metrics = calculator.calculate_protocol_health_score(current_prices)
    
    # Debt cap metrics
    debt_cap_metrics = calculator.calculate_debt_cap_metrics(current_prices)
    
    # MOET stability
    moet_metrics = calculator.calculate_moet_stability_metrics()
    
    # Utilization metrics
    utilization_metrics = calculator.calculate_utilization_metrics()
    
    # Revenue metrics
    revenue_metrics = calculator.calculate_protocol_revenue_metrics()
    
    # Display baseline metrics
    display_baseline_metrics({
        "health": health_metrics,
        "debt_cap": debt_cap_metrics,
        "moet": moet_metrics,
        "utilization": utilization_metrics,
        "revenue": revenue_metrics
    }, args.verbose)
    
    # Export if requested
    if args.output:
        baseline_data = {
            "baseline_metrics": {
                "health": health_metrics,
                "debt_cap": debt_cap_metrics,
                "moet": moet_metrics,
                "utilization": utilization_metrics,
                "revenue": revenue_metrics
            },
            "simulation_results": results
        }
        export_results(baseline_data, args.output)
    
    return 0


def display_scenario_results(scenario_name: str, results: Dict, verbose: bool):
    """Display results for single scenario"""
    
    print(f"\nResults for {scenario_name}:")
    print("=" * (len(scenario_name) + 12))
    
    if "assessment" in results:
        assessment = results["assessment"]
        print(f"Overall Score: {assessment['overall_score']:.3f}")
        print(f"Risk Level: {assessment['risk_level']}")
        
        if "key_concerns" in assessment and assessment["key_concerns"]:
            print(f"\nKey Concerns:")
            for concern in assessment["key_concerns"]:
                print(f"  • {concern}")
    
    if "risk_metrics" in results:
        risk_metrics = results["risk_metrics"]
        print(f"\nRisk Metrics:")
        for metric, value in risk_metrics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value:.3f}")
    
    if verbose and "statistics" in results:
        stats = results["statistics"]
        print(f"\nDetailed Statistics:")
        for metric, stat_dict in stats.items():
            if isinstance(stat_dict, dict) and "mean" in stat_dict:
                print(f"  {metric.replace('_', ' ').title()}:")
                print(f"    Mean: {stat_dict['mean']:.2f}")
                print(f"    Range: {stat_dict['min']:.2f} - {stat_dict['max']:.2f}")


def display_suite_summary(summary: Dict, verbose: bool):
    """Display summary for full test suite"""
    
    print("\nSTRESS TEST SUITE SUMMARY")
    print("=" * 30)
    
    suite_stats = summary.get("suite_statistics", {})
    print(f"Scenarios run: {suite_stats.get('total_scenarios', 0)}")
    print(f"Success rate: {suite_stats.get('success_rate', 0):.1%}")
    
    overall_metrics = summary.get("overall_metrics", {})
    print(f"Average resilience: {overall_metrics.get('average_resilience_score', 0):.3f}")
    
    # Rankings
    rankings = summary.get("scenario_rankings", {})
    
    if "highest_risk" in rankings and rankings["highest_risk"]:
        print(f"\nHighest Risk Scenarios:")
        for i, (name, score) in enumerate(rankings["highest_risk"][:3], 1):
            print(f"  {i}. {name} (score: {score:.3f})")
    
    if "lowest_risk" in rankings and rankings["lowest_risk"]:
        print(f"\nLowest Risk Scenarios:")
        for i, (name, score) in enumerate(rankings["lowest_risk"][-3:], 1):
            print(f"  {i}. {name} (score: {score:.3f})")
    
    # Recommendations
    recommendations = summary.get("recommendations", [])
    if recommendations:
        print(f"\nRecommendations:")
        for rec in recommendations[:3]:
            print(f"  • {rec}")
    
    # Critical findings
    critical_findings = summary.get("critical_findings", [])
    if critical_findings:
        print(f"\nCritical Findings:")
        for finding in critical_findings:
            print(f"  ⚠️  {finding}")


def display_baseline_metrics(metrics: Dict, verbose: bool):
    """Display baseline protocol metrics"""
    
    print("\nBASELINE PROTOCOL METRICS")
    print("=" * 30)
    
    # Protocol Health
    health = metrics["health"]
    print(f"Overall Health Score: {health['overall_health_score']:.3f} ({health['health_status']})")
    
    if verbose:
        print("  Component Scores:")
        for component, score in health["component_scores"].items():
            print(f"    {component.replace('_', ' ').title()}: {score:.3f}")
    
    # Debt Cap
    debt_cap = metrics["debt_cap"]
    print(f"\nDebt Cap Utilization: {debt_cap['utilization_rate']:.1%} ({debt_cap['utilization_status']})")
    print(f"Available Capacity: ${debt_cap['available_capacity']:,.0f}")
    
    # MOET Stability
    moet = metrics["moet"]
    print(f"\nMOET Price: ${moet['current_price']:.4f}")
    print(f"Peg Stable: {'✓' if moet['is_peg_stable'] else '✗'}")
    print(f"Stability Score: {moet['stability_score']:.3f}")
    
    # Revenue
    revenue = metrics["revenue"]
    print(f"\nTreasury Balance: ${revenue['treasury_balance']:,.2f}")
    print(f"Monthly Revenue Est.: ${revenue['estimated_monthly_revenue']:,.2f}")


def export_results(results: Dict, filepath: str):
    """Export results to JSON file"""
    
    import json
    
    # Convert numpy types and other non-serializable objects
    def make_serializable(obj):
        if hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif hasattr(obj, '__dict__'):  # Custom objects
            return obj.__dict__
        else:
            return str(obj)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=make_serializable)
        print(f"\nResults exported to: {filepath}")
    except Exception as e:
        print(f"Warning: Could not export results - {str(e)}")


if __name__ == "__main__":
    sys.exit(main())