#!/usr/bin/env python3
"""
Debug script to see exactly what happens at minute 0 with a test agent
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from sim_tests.full_year_sim import FullYearSimConfig, FullYearSimulation
from tidal_protocol_sim.agents.aave_agent import AaveAgent
from tidal_protocol_sim.core.protocol import Asset

# Create config
config = FullYearSimConfig()
config.test_name = "Debug_Minute_Zero"
config.year = 2022
config.btc_data_file = "btc-usd-max.csv"
config.aave_historical_rates = True
config.simulation_duration_minutes = 1440  # Just 1 day
config.num_aave_agents = 1

# Load data
config.load_market_data()

# Create simulation
sim = FullYearSimulation(config)
aave_engine = sim._create_aave_engine()

print("=" * 80)
print("BEFORE AGENT REPLACEMENT")
print("=" * 80)
print(f"Number of agents: {len(aave_engine.aave_agents)}")
print(f"BTC in protocol: {aave_engine.protocol.asset_pools[Asset.BTC].total_supplied:,.2f}")
print(f"MOET supply: {aave_engine.protocol.moet_system.total_supply:,.2f}")
print()

# Create test agent with HF 1.525
test_hf = 1.525
btc_price = config.btc_initial_price

test_agent = AaveAgent(
    agent_id="debug_agent",
    initial_hf=test_hf,
    rebalancing_hf=test_hf,
    target_hf=test_hf,
    initial_balance=btc_price
)

print("=" * 80)
print("TEST AGENT CREATED")
print("=" * 80)
print(f"Target HF: {test_hf}")
print(f"BTC Amount: {test_agent.state.btc_amount}")
print(f"BTC Supplied: {test_agent.state.supplied_balances.get(Asset.BTC, 0)}")
print(f"MOET Debt (state): {test_agent.state.moet_debt:,.2f}")
print(f"MOET Borrowed: {test_agent.state.borrowed_balances.get(Asset.MOET, 0):,.2f}")
print(f"MOET in Wallet: {test_agent.state.token_balances.get(Asset.MOET, 0):,.2f}")
print()

# Clear protocol and replace agents
btc_pool = aave_engine.protocol.asset_pools[Asset.BTC]
btc_pool.total_supplied = 0.0
aave_engine.protocol.moet_system.total_supply = 0.0

aave_engine.aave_agents = [test_agent]
aave_engine.agents = {test_agent.agent_id: test_agent}

print("=" * 80)
print("AFTER PROTOCOL CLEARED")
print("=" * 80)
print(f"BTC in protocol: {aave_engine.protocol.asset_pools[Asset.BTC].total_supplied:,.2f}")
print(f"MOET supply: {aave_engine.protocol.moet_system.total_supply:,.2f}")
print()

# Re-initialize positions
aave_engine._setup_aave_positions()

print("=" * 80)
print("AFTER _setup_aave_positions()")
print("=" * 80)
print(f"BTC in protocol: {aave_engine.protocol.asset_pools[Asset.BTC].total_supplied:,.2f}")
print(f"MOET supply: {aave_engine.protocol.moet_system.total_supply:,.2f}")
print(f"Agent HF: {test_agent.state.health_factor:.4f}")

# Manual HF calculation to compare
btc_in_supplied = test_agent.state.supplied_balances.get(Asset.BTC, 0)
btc_price_now = aave_engine.state.current_prices.get(Asset.BTC, 0)
collateral_value_calc = btc_in_supplied * btc_price_now * 0.85
debt_calc = test_agent.state.moet_debt
manual_hf_calc = collateral_value_calc / debt_calc if debt_calc > 0 else float('inf')

print(f"\nManual HF Verification:")
print(f"  BTC in supplied_balances: {btc_in_supplied}")
print(f"  BTC price: ${btc_price_now:,.2f}")
print(f"  Collateral (× 0.85): ${collateral_value_calc:,.2f}")
print(f"  Debt: ${debt_calc:,.2f}")
print(f"  Manual HF: {manual_hf_calc:.4f}")
print(f"  Agent HF: {test_agent.state.health_factor:.4f}")
print(f"  Match: {'✅' if abs(manual_hf_calc - test_agent.state.health_factor) < 0.01 else '❌ MISMATCH!'}")
print()

# Simulate minute 0 - process agents (buy YT)
print("=" * 80)
print("MINUTE 0: Processing Agents (should buy YT)")
print("=" * 80)
aave_engine._process_aave_agents(0)

print(f"YT owned: {len(test_agent.state.yield_token_manager.yield_tokens)}")
print(f"MOET in Wallet after YT purchase: {test_agent.state.token_balances.get(Asset.MOET, 0):,.2f}")
print(f"MOET Debt after YT purchase: {test_agent.state.moet_debt:,.2f}")
print()

# Update HF
test_agent._update_health_factor(aave_engine.state.current_prices)
print(f"Agent HF after YT purchase: {test_agent.state.health_factor:.4f}")
print()

# Check for liquidation
print("=" * 80)
print("MINUTE 0: Checking for Liquidations")
print("=" * 80)
print(f"Pre-check HF: {test_agent.state.health_factor:.4f}")
print(f"Liquidation threshold: HF <= 1.0")

if test_agent.state.health_factor <= 1.0:
    print(f"❌ WOULD LIQUIDATE! HF {test_agent.state.health_factor:.4f} <= 1.0")
else:
    print(f"✅ SAFE! HF {test_agent.state.health_factor:.4f} > 1.0")

print()
print("=" * 80)
print("DIAGNOSTIC CALCULATIONS")
print("=" * 80)

# Manual HF calculation
btc_value = test_agent.state.supplied_balances.get(Asset.BTC, 0) * btc_price
effective_collateral = btc_value * 0.85
debt = test_agent.state.moet_debt
manual_hf = effective_collateral / debt if debt > 0 else float('inf')

print(f"BTC Supplied: {test_agent.state.supplied_balances.get(Asset.BTC, 0)} BTC")
print(f"BTC Price: ${btc_price:,.0f}")
print(f"BTC Value: ${btc_value:,.2f}")
print(f"Effective Collateral (× 0.85): ${effective_collateral:,.2f}")
print(f"MOET Debt: ${debt:,.2f}")
print(f"Manual HF Calculation: {manual_hf:.4f}")
print(f"Agent's HF: {test_agent.state.health_factor:.4f}")
print(f"Match: {'✅' if abs(manual_hf - test_agent.state.health_factor) < 0.01 else '❌'}")
print("=" * 80)

