#!/usr/bin/env python3
"""Diagnostic script to check agent initialization"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tidal_protocol_sim.engine.high_tide_vault_engine import HighTideVaultEngine, HighTideConfig
from tidal_protocol_sim.agents.high_tide_agent import HighTideAgent
from tidal_protocol_sim.core.protocol import Asset

# Create minimal config
ht_config = HighTideConfig()
ht_config.num_high_tide_agents = 0
ht_config.btc_initial_price = 100_000.0
ht_config.moet_yield_pool_size = 500_000.0
ht_config.yield_token_concentration = 0.95
ht_config.yield_token_ratio = 0.75

# Create engine
engine = HighTideVaultEngine(ht_config)
print(f"✅ Engine created")
print(f"   BTC price: ${engine.state.current_prices[Asset.BTC]:,.0f}")

# Create one agent
agent = HighTideAgent(
    agent_id="test_agent",
    initial_hf=1.05,
    rebalancing_hf=1.015,
    target_hf=1.03,
    initial_balance=100_000.0,
    yield_token_pool=engine.yield_token_pool
)
agent.engine = engine

print(f"\n📊 AGENT STATE AFTER CREATION:")
print(f"   BTC amount: {agent.state.btc_amount}")
print(f"   BTC supplied: {agent.state.supplied_balances[Asset.BTC]}")
print(f"   MOET debt: ${agent.state.moet_debt:,.2f}")
print(f"   Health Factor: {agent.state.health_factor:.6f}")
print(f"   Initial HF target: {agent.state.initial_health_factor}")

# Update health factor manually
btc_price = engine.state.current_prices[Asset.BTC]
collateral_value = agent.state.btc_amount * btc_price * 0.80
debt_value = agent.state.moet_debt
calculated_hf = collateral_value / debt_value if debt_value > 0 else float('inf')

print(f"\n🧮 MANUAL HF CALCULATION:")
print(f"   BTC: {agent.state.btc_amount} × ${btc_price:,.0f} × 0.80 = ${collateral_value:,.2f}")
print(f"   Debt: ${debt_value:,.2f}")
print(f"   HF: {collateral_value:,.2f} / {debt_value:,.2f} = {calculated_hf:.6f}")
print(f"   Expected: ~1.05")
print(f"   Match: {'✅' if abs(calculated_hf - 1.05) < 0.01 else '❌'}")

# Try buying YT
protocol_state = {
    "current_minute": 0,
    "moet_system": engine.protocol.moet_system
}
action_type, params = agent.decide_action(protocol_state, engine.state.current_prices)
print(f"\n🎯 AGENT DECISION:")
print(f"   Action: {action_type}")
print(f"   Params: {params}")

if action_type.name == "YIELD_TOKEN_PURCHASE":
    success = agent.execute_yield_token_purchase(params["moet_amount"], 0, use_direct_minting=True)
    print(f"   Purchase result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"   YT balance after: {agent.state.yield_token_manager.get_total_yield_tokens()}")
    print(f"   MOET balance after: ${agent.state.token_balances[Asset.MOET]:,.2f}")
    print(f"   HF after purchase: {agent.state.health_factor:.6f}")

