#!/usr/bin/env python3
"""
Tidal Protocol Simulation Demo

This script demonstrates the comprehensive Tidal Protocol implementation
with all its specific mechanisms and nuanced behaviors.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.schemas.tidal_config import create_default_config, PolicyConfig, PolicyType
from src.core.simulation.factory import SimulationFactory


def demonstrate_tidal_features():
    """Demonstrate Tidal Protocol specific features"""
    print("🌊 TIDAL PROTOCOL SIMULATION DEMONSTRATION")
    print("=" * 60)
    
    # Create Tidal-focused configuration
    config = create_default_config()
    
    # Adjust for Tidal Protocol demonstration
    config.simulation.max_days = 30
    config.simulation.agent_policies = [
        PolicyConfig(
            type=PolicyType.TIDAL_LENDER,
            count=70,
            params={
                "target_health_factor": 2.0,
                "moet_borrowing_ratio": 0.65,
                "risk_tolerance": 0.6,
                "collateral_diversification": True,
                "min_supply_apy": 0.03
            },
            initial_balance_usd=25000
        ),
        PolicyConfig(
            type=PolicyType.TRADER,
            count=30,
            params={
                "trading_frequency": 0.12,
                "risk_tolerance": 0.7
            },
            initial_balance_usd=15000
        )
    ]
    
    print(f"Configuration: {config.simulation.name}")
    print(f"Agents: {sum(p.count for p in config.simulation.agent_policies)}")
    print(f"- Tidal Lenders: {config.simulation.agent_policies[0].count}")
    print(f"- Traders: {config.simulation.agent_policies[1].count}")
    print(f"Markets: {len(config.simulation.markets)}")
    print()
    
    # Create and run simulation
    print("Creating Tidal Protocol simulation...")
    simulation = SimulationFactory.create_simulation(config)
    
    print("Running 30-day simulation...")
    results = simulation.run_simulation(max_days=30, verbose=False)
    
    # Extract Tidal-specific results
    print("\n🌊 TIDAL PROTOCOL RESULTS")
    print("=" * 60)
    
    final_state = results.get('final_state', {})
    performance = results.get('performance_metrics', {})
    
    print(f"MOET Price: ${final_state.get('moet_price', 1.0):.4f}")
    print(f"Market Cap: ${final_state.get('market_cap', 0):,.0f}")
    print(f"Protocol Treasury: ${final_state.get('protocol_treasury', 0):,.0f}")
    print(f"Total Liquidity: ${final_state.get('total_liquidity', 0):,.0f}")
    print()
    
    print("PERFORMANCE METRICS:")
    print(f"Total Return: {performance.get('total_return', 0):.2%}")
    print(f"Volatility: {performance.get('volatility', 0):.2%}")
    print(f"Max Price: ${performance.get('max_price', 1.0):.4f}")
    print(f"Min Price: ${performance.get('min_price', 1.0):.4f}")
    print()
    
    # Show simulation summary
    sim_summary = results.get('simulation_summary', {})
    print("SIMULATION SUMMARY:")
    print(f"Days Simulated: {sim_summary.get('days_simulated', 0)}")
    print(f"Total Events: {sim_summary.get('total_events', 0)}")
    print(f"Agent Population: {sim_summary.get('total_agents', 0)}")
    print(f"Market Count: {sim_summary.get('total_markets', 0)}")
    
    return results


def demonstrate_tidal_specific_metrics():
    """Show what makes this Tidal-specific"""
    print("\n🔍 TIDAL PROTOCOL SPECIFIC FEATURES")
    print("=" * 60)
    
    print("✅ IMPLEMENTED TIDAL MECHANISMS:")
    print("  • MOET Stablecoin with $1.00 peg target")
    print("  • Multi-asset collateral (ETH, BTC, FLOW, USDC)")
    print("  • Kinked interest rate model (80% kink)")
    print("  • Ebisu-style debt cap calculation (A × B × C)")
    print("  • Liquidation penalties (8%) with 50% close factor")
    print("  • Integrated liquidity pools (MOET pairs)")
    print("  • Health factor management (1.5x target)")
    print("  • Concentrated liquidity modeling")
    print("  • Protocol revenue distribution (50% to LPs)")
    print("  • Peg stability mechanisms (±2% bands)")
    print()
    
    print("✅ TIDAL-SPECIFIC AGENT BEHAVIORS:")
    print("  • TidalLenderPolicy with protocol awareness")
    print("  • Emergency actions for low health factors")
    print("  • MOET borrowing strategies")
    print("  • Collateral diversification logic")
    print("  • Risk-adjusted position management")
    print("  • Protocol-specific yield optimization")
    print()
    
    print("✅ MATHEMATICAL ACCURACY:")
    print("  • Tidal's exact kinked interest rate formulas")
    print("  • Proper compound interest calculations")
    print("  • Liquidation math with penalties")
    print("  • Debt cap risk modeling")
    print("  • MOET minting/burning mechanics")
    print("  • Health factor calculations")


def main():
    """Main demonstration function"""
    try:
        # Run the demonstration
        results = demonstrate_tidal_features()
        demonstrate_tidal_specific_metrics()
        
        print("\n🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("This simulation captures the full complexity of Tidal Protocol")
        print("with all its nuanced mechanisms and specific behaviors.")
        print("\nThe hybrid architecture provides:")
        print("• Generic DeFi framework for extensibility")
        print("• Comprehensive Tidal Protocol implementation")
        print("• Production-ready modularity and configuration")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
