#!/usr/bin/env python3
"""
Example Usage of Tidal Protocol Simulation

This script demonstrates various ways to use the simulation for:
- Single scenario analysis
- Custom parameter testing
- Risk assessment
- Protocol optimization
"""

from tidal_protocol_simulation import (
    TidalProtocol, MonteCarloSimulator, Asset, 
    print_protocol_state, INITIAL_PRICES, INITIAL_LIQUIDITY
)
import numpy as np

def example_single_scenario():
    """Example: Analyze a single scenario with specific parameters"""
    print("="*60)
    print("EXAMPLE 1: Single Scenario Analysis")
    print("="*60)
    
    # Initialize protocol
    protocol = TidalProtocol()
    
    # Apply specific price shocks
    protocol.apply_price_shock(Asset.ETH, -0.20)  # ETH drops 20%
    protocol.apply_price_shock(Asset.BTC, -0.15)  # BTC drops 15%
    protocol.apply_price_shock(Asset.FLOW, -0.40) # FLOW drops 40%
    
    # Simulate some borrowing activity
    protocol.simulate_borrowing(Asset.ETH, 2_000_000)  # Borrow $2M against ETH
    protocol.simulate_borrowing(Asset.BTC, 1_000_000)  # Borrow $1M against BTC
    
    # Get and display state
    state = protocol.get_protocol_state()
    print_protocol_state(state)
    
    return protocol, state

def example_utilization_analysis():
    """Example: Analyze how utilization affects interest rates"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Utilization vs Interest Rate Analysis")
    print("="*60)
    
    protocol = TidalProtocol()
    
    # Test different utilization levels
    utilization_levels = [0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9]
    
    print(f"{'Utilization':<12} {'Borrow Rate':<15} {'Supply Rate':<15}")
    print("-" * 50)
    
    for util in utilization_levels:
        # Set utilization by adjusting borrowed amount
        target_borrowed = util * INITIAL_LIQUIDITY[Asset.ETH]
        protocol.total_borrowed[Asset.ETH] = target_borrowed
        
        borrow_rate = protocol.calculate_borrow_rate(Asset.ETH)
        supply_rate = protocol.calculate_supply_rate(Asset.ETH)
        
        print(f"{util:<12.1%} {borrow_rate:<15.2%} {supply_rate:<15.2%}")

def example_risk_assessment():
    """Example: Assess risk under different scenarios"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Risk Assessment Scenarios")
    print("="*60)
    
    scenarios = [
        ("Baseline", {}),
        ("ETH Crash", {Asset.ETH: -0.30}),
        ("BTC Crash", {Asset.BTC: -0.25}),
        ("FLOW Crash", {Asset.FLOW: -0.50}),
        ("Market Crash", {
            Asset.ETH: -0.25, 
            Asset.BTC: -0.20, 
            Asset.FLOW: -0.45, 
            Asset.USDC: -0.10
        })
    ]
    
    print(f"{'Scenario':<15} {'Health Factor':<15} {'Debt Cap':<15} {'Risk Level':<10}")
    print("-" * 60)
    
    for scenario_name, price_shocks in scenarios:
        protocol = TidalProtocol()
        
        # Apply price shocks
        for asset, shock in price_shocks.items():
            protocol.apply_price_shock(asset, shock)
        
        # Simulate moderate borrowing (50% utilization)
        for asset in [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]:
            target_borrowed = 0.5 * INITIAL_LIQUIDITY[asset]
            protocol.total_borrowed[asset] = target_borrowed
        
        state = protocol.get_protocol_state()
        
        # Calculate average health factor
        avg_health_factor = np.mean(list(state.health_factors.values()))
        debt_cap = state.debt_cap
        
        # Determine risk level
        if avg_health_factor < 1.2:
            risk_level = "HIGH"
        elif avg_health_factor < 1.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        print(f"{scenario_name:<15} {avg_health_factor:<15.2f} ${debt_cap:<14,.0f} {risk_level:<10}")

def example_custom_monte_carlo():
    """Example: Custom Monte Carlo with specific parameters"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Monte Carlo Simulation")
    print("="*60)
    
    protocol = TidalProtocol()
    
    # Create custom simulator with fewer runs for demonstration
    simulator = MonteCarloSimulator(protocol, n_simulations=100)
    
    print("Running custom Monte Carlo simulation...")
    results = simulator.run_simulation()
    
    # Custom analysis
    print(f"\nResults Summary:")
    print(f"Total simulations: {len(results)}")
    
    # Analyze extreme scenarios
    extreme_eth_drops = [
        r for r in results 
        if (r.asset_prices[Asset.ETH] - INITIAL_PRICES[Asset.ETH]) / INITIAL_PRICES[Asset.ETH] < -0.20
    ]
    
    print(f"Scenarios with ETH >20% drop: {len(extreme_eth_drops)}")
    
    if extreme_eth_drops:
        avg_debt_cap = np.mean([r.debt_cap for r in extreme_eth_drops])
        print(f"Average debt cap in extreme ETH scenarios: ${avg_debt_cap:,.0f}")
    
    # Generate summary statistics
    summary = simulator.generate_summary_statistics()
    print(f"\nSummary Statistics:")
    print(summary)
    
    return simulator, results

def example_protocol_optimization():
    """Example: Test different protocol parameters"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Protocol Parameter Optimization")
    print("="*60)
    
    # Test different collateral factors
    collateral_factor_tests = [0.6, 0.7, 0.8, 0.9]
    
    print(f"{'Collateral Factor':<18} {'Max Borrow':<15} {'Risk Level':<10}")
    print("-" * 50)
    
    for cf in collateral_factor_tests:
        protocol = TidalProtocol()
        
        # Temporarily modify collateral factor for ETH
        original_cf = protocol.pools[Asset.ETH].asset.collateral_factor if hasattr(protocol.pools[Asset.ETH].asset, 'collateral_factor') else 0.75
        
        # Calculate max borrow with this collateral factor
        eth_supplied = INITIAL_LIQUIDITY[Asset.ETH]
        eth_price = INITIAL_PRICES[Asset.ETH]
        max_borrow = (eth_supplied * eth_price * cf) / 1.5  # 1.5 is target health factor
        
        # Determine risk level
        if cf < 0.7:
            risk_level = "HIGH"
        elif cf < 0.8:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        print(f"{cf:<18.1%} ${max_borrow:<14,.0f} {risk_level:<10}")

def main():
    """Run all examples"""
    print("Tidal Protocol Simulation - Example Usage")
    print("="*60)
    
    try:
        # Example 1: Single scenario
        protocol1, state1 = example_single_scenario()
        
        # Example 2: Utilization analysis
        example_utilization_analysis()
        
        # Example 3: Risk assessment
        example_risk_assessment()
        
        # Example 4: Custom Monte Carlo
        simulator, results = example_custom_monte_carlo()
        
        # Example 5: Protocol optimization
        example_protocol_optimization()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
