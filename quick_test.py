#!/usr/bin/env python3
"""
Quick test of Tidal Protocol with proper parameters
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.schemas.tidal_config import create_default_config
from src.core.simulation.factory import SimulationFactory


def test_tidal_protocol():
    """Test basic Tidal Protocol functionality"""
    print("🌊 Quick Tidal Protocol Test")
    print("=" * 40)
    
    # Create configuration
    config = create_default_config()
    
    # Verify configuration
    tidal_market = None
    for market in config.simulation.markets:
        if market.market_id == "tidal_protocol":
            tidal_market = market
            break
    
    print("Configuration:")
    print(f"  Target Health Factor: {tidal_market.target_health_factor}")
    print(f"  ETH Collateral Factor: {tidal_market.collateral_factors['ETH']}")
    print(f"  Agents: {sum(p.count for p in config.simulation.agent_policies)}")
    print()
    
    # Create simulation
    print("Creating simulation...")
    simulation = SimulationFactory.create_simulation(config)
    
    # Check initial state
    print("Initial state:")
    print(f"  Agents created: {len(simulation.agent_population)}")
    print(f"  Markets registered: {len(simulation.market_registry.markets)}")
    
    # Check agent balances
    if len(simulation.agent_population) > 0:
        first_agent = simulation.agent_population.agents[0]
        from src.core.simulation.primitives import Asset
        print(f"  First agent balance (ETH): {first_agent.state.token_balances.get(Asset.ETH, 0):.4f}")
        print(f"  First agent balance (USDC): {first_agent.state.token_balances.get(Asset.USDC, 0):.2f}")
        print(f"  First agent token_balances keys: {list(first_agent.state.token_balances.keys())}")
    
    print()
    
    # Run short simulation
    print("Running 3-day simulation...")
    results = simulation.run_simulation(max_days=3, verbose=True)
    
    # Check results
    print("\nResults:")
    final_state = results.get('final_state', {})
    print(f"  MOET Price: ${final_state.get('moet_price', 1.0):.4f}")
    print(f"  Protocol Treasury: ${final_state.get('protocol_treasury', 0):,.0f}")
    
    # Check market data
    market_data = simulation.market_registry.get_all_market_data()
    tidal_data = market_data.get('tidal_protocol', {})
    if tidal_data:
        print(f"  Debt Cap: ${tidal_data.get('debt_cap', 0):,.0f}")
        print(f"  Asset Pools: {len(tidal_data.get('asset_pools', {}))}")
    
    print("\n✅ Test completed!")


if __name__ == "__main__":
    test_tidal_protocol()
