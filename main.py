#!/usr/bin/env python3
"""
Main entry point for the modular Tidal Protocol simulation.

This script demonstrates the new Agent-Action-Market architecture and
provides examples of how to use the modular system.
"""

import sys
import argparse
from typing import Optional
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.schemas.tidal_config import TidalProtocolConfig, create_default_config
from src.core.simulation.factory import SimulationFactory, MonteCarloSimulator
from src.analysis.metrics.calculator import TokenomicsMetricsCalculator


def run_single_simulation(config: TidalProtocolConfig, verbose: bool = True) -> dict:
    """Run a single simulation with the given configuration"""
    if verbose:
        print(f"Running single simulation: {config.client_name}")
        print(f"Configuration: {config.simulation.name}")
        print(f"Agents: {sum(p.count for p in config.simulation.agent_policies)}")
        print(f"Markets: {len(config.simulation.markets)}")
        print(f"Max Days: {config.simulation.max_days}")
        print()
    
    # Create simulation
    simulation = SimulationFactory.create_simulation(config)
    
    # Run simulation
    results = simulation.run_simulation(
        max_days=config.simulation.max_days,
        verbose=verbose
    )
    
    return results


def run_monte_carlo_simulation(config: TidalProtocolConfig, n_runs: Optional[int] = None, verbose: bool = True) -> dict:
    """Run Monte Carlo simulation with the given configuration"""
    if verbose:
        print(f"Running Monte Carlo simulation: {config.client_name}")
        print(f"Configuration: {config.simulation.name}")
        print(f"Number of runs: {n_runs or config.simulation.monte_carlo_runs}")
        print()
    
    # Create Monte Carlo simulator
    simulator = MonteCarloSimulator(config, n_runs)
    
    # Run simulations
    results = simulator.run_monte_carlo(verbose=verbose)
    
    # Get summary
    summary = simulator.get_results_summary()
    
    return summary


def print_results_summary(results: dict):
    """Print a formatted summary of simulation results"""
    print("\n" + "="*80)
    print("SIMULATION RESULTS SUMMARY")
    print("="*80)
    
    # Check if it's Monte Carlo results
    if "monte_carlo_summary" in results:
        mc_summary = results["monte_carlo_summary"]
        print(f"Monte Carlo Runs: {mc_summary['total_simulations']}")
        print(f"Successful Runs: {mc_summary['successful_simulations']}")
        print(f"Configuration: {mc_summary['configuration']}")
        print()
        
        # Print metrics if available
        metrics = results.get("metrics")
        if metrics:
            print("KEY METRICS:")
            
            # Price metrics
            price_metrics = metrics.price_metrics
            final_price = price_metrics['final_price_stats']
            print(f"  Final MOET Price:")
            print(f"    Mean: ${final_price['mean']:.4f}")
            print(f"    95th Percentile: ${final_price['percentiles']['95th']:.4f}")
            print(f"    5th Percentile: ${final_price['percentiles']['5th']:.4f}")
            
            appreciation = price_metrics['price_appreciation']
            print(f"  Price Appreciation:")
            print(f"    Probability of Positive Returns: {appreciation['positive_return_probability']:.1%}")
            print(f"    Mean Return: {appreciation['mean_change']:.2%}")
            
            # Protocol metrics
            protocol_metrics = metrics.protocol_metrics
            treasury = protocol_metrics['treasury_health']
            print(f"  Protocol Treasury:")
            print(f"    Positive Balance Probability: {treasury['positive_balance_probability']:.1%}")
            print(f"    Bankruptcy Risk: {treasury['bankruptcy_risk']:.1%}")
            
            # Risk metrics
            risk_metrics = metrics.risk_metrics
            var = risk_metrics['value_at_risk']
            print(f"  Risk Analysis:")
            print(f"    Value at Risk (95%): {var['var_95']:.2%}")
            print(f"    Value at Risk (99%): {var['var_99']:.2%}")
    
    else:
        # Single simulation results
        sim_summary = results.get("simulation_summary", {})
        final_state = results.get("final_state", {})
        performance = results.get("performance_metrics", {})
        
        print(f"Days Simulated: {sim_summary.get('days_simulated', 0)}")
        print(f"Total Agents: {sim_summary.get('total_agents', 0)}")
        print(f"Total Events: {sim_summary.get('total_events', 0)}")
        print()
        
        print("FINAL STATE:")
        print(f"  MOET Price: ${final_state.get('moet_price', 0):.4f}")
        print(f"  Market Cap: ${final_state.get('market_cap', 0):,.0f}")
        print(f"  Protocol Treasury: ${final_state.get('protocol_treasury', 0):,.0f}")
        print(f"  Total Liquidity: ${final_state.get('total_liquidity', 0):,.0f}")
        print()
        
        print("PERFORMANCE:")
        print(f"  Total Return: {performance.get('total_return', 0):.2%}")
        print(f"  Volatility: {performance.get('volatility', 0):.2%}")
        print(f"  Max Price: ${performance.get('max_price', 0):.4f}")
        print(f"  Min Price: ${performance.get('min_price', 0):.4f}")
    
    print("="*80)


def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description="Tidal Protocol Simulation")
    parser.add_argument("--mode", choices=["single", "monte-carlo"], default="single",
                       help="Simulation mode")
    parser.add_argument("--config", type=str, help="Path to configuration file (YAML)")
    parser.add_argument("--runs", type=int, help="Number of Monte Carlo runs")
    parser.add_argument("--days", type=int, help="Number of days to simulate")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    try:
        # Load or create configuration
        if args.config:
            # TODO: Implement configuration loading from file
            print(f"Loading configuration from {args.config}...")
            config = create_default_config()  # Placeholder
        else:
            print("Using default configuration...")
            config = create_default_config()
        
        # Override configuration with command-line arguments
        if args.days:
            config.simulation.max_days = args.days
        if args.runs:
            config.simulation.monte_carlo_runs = args.runs
        if args.seed:
            config.simulation.random_seed = args.seed
        
        # Validate configuration
        config.validate_config()
        
        print(f"Configuration validated successfully!")
        print(f"Client: {config.client_name}")
        print(f"Simulation: {config.simulation.name}")
        print()
        
        # Run simulation based on mode
        if args.mode == "single":
            results = run_single_simulation(config, verbose=args.verbose)
        else:
            results = run_monte_carlo_simulation(config, args.runs, verbose=args.verbose)
        
        # Print results
        print_results_summary(results)
        
        print("\n🎉 Simulation completed successfully!")
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def example_usage():
    """Demonstrate various ways to use the modular system"""
    print("=== TIDAL PROTOCOL MODULAR SIMULATION EXAMPLES ===\n")
    
    # Example 1: Default configuration
    print("Example 1: Default Configuration")
    print("-" * 40)
    
    config = create_default_config()
    config.simulation.max_days = 30  # Short simulation for demo
    
    # Run single simulation
    results = run_single_simulation(config, verbose=False)
    print(f"Single simulation completed!")
    print(f"Final MOET price: ${results['final_state']['moet_price']:.4f}")
    print(f"Total return: {results['performance_metrics']['total_return']:.2%}")
    print()
    
    # Example 2: Monte Carlo with fewer runs
    print("Example 2: Monte Carlo Simulation")
    print("-" * 40)
    
    config.simulation.monte_carlo_runs = 10  # Small number for demo
    mc_results = run_monte_carlo_simulation(config, verbose=False)
    
    print(f"Monte Carlo completed!")
    if "metrics" in mc_results:
        price_stats = mc_results["metrics"].price_metrics['final_price_stats']
        print(f"Mean final price: ${price_stats['mean']:.4f}")
        print(f"Price volatility: {price_stats['std']:.4f}")
    print()
    
    # Example 3: Custom agent policies
    print("Example 3: Custom Policy Configuration")
    print("-" * 40)
    
    from src.config.schemas.tidal_config import PolicyConfig, PolicyType
    
    # Create config with different agent mix
    custom_config = create_default_config()
    custom_config.simulation.agent_policies = [
        PolicyConfig(
            type=PolicyType.TRADER,
            count=80,  # More traders
            params={"trading_frequency": 0.2, "risk_tolerance": 0.7},  # More aggressive
            initial_balance_usd=15000
        ),
        PolicyConfig(
            type=PolicyType.LENDER,
            count=20,  # Fewer lenders
            params={"min_supply_apy": 0.03, "target_health_factor": 1.8},
            initial_balance_usd=25000
        )
    ]
    
    custom_config.simulation.max_days = 30
    custom_results = run_single_simulation(custom_config, verbose=False)
    
    print(f"Custom policy simulation completed!")
    print(f"Total agents: {sum(p.count for p in custom_config.simulation.agent_policies)}")
    print(f"Trader ratio: {80/100:.1%}")
    print(f"Final price: ${custom_results['final_state']['moet_price']:.4f}")
    print()
    
    print("=== ALL EXAMPLES COMPLETED ===")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - run examples
        example_usage()
    else:
        # Command-line mode
        main()
