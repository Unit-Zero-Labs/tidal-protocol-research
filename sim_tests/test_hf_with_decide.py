#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tidal_protocol_sim.engine.high_tide_vault_engine import HighTideVaultEngine, HighTideConfig
from tidal_protocol_sim.agents.high_tide_agent import HighTideAgent
from tidal_protocol_sim.core.protocol import Asset

# Create engine
config = HighTideConfig()
config.num_high_tide_agents = 0
config.btc_initial_price = 100_000.0
config.enable_advanced_moet_system = False
engine = HighTideVaultEngine(config)

# Override rate
fixed_rate = 0.056234
engine.protocol.get_moet_borrow_rate = lambda: fixed_rate

# Create agent
agent = HighTideAgent('test', 1.05, 1.015, 1.03, 100_000.0, engine.yield_token_pool)
agent.engine = engine
engine.high_tide_agents = [agent]

print(f'✅ Initial state:')
print(f'   HF (stored): {agent.state.health_factor:.6f}')
print(f'   Debt: ${agent.state.moet_debt:,.2f}')
print(f'   BTC amount: {agent.state.btc_amount}')
print(f'   BTC supplied: {agent.state.supplied_balances[Asset.BTC]}')
print(f'   Engine BTC price: ${engine.state.current_prices[Asset.BTC]:,.0f}')

# Call decide_action like the engine does
protocol_state = engine._get_protocol_state()
protocol_state["current_step"] = 0
action_type, params = agent.decide_action(protocol_state, engine.state.current_prices)

print(f'\n✅ After decide_action (minute 0):')
print(f'   HF: {agent.state.health_factor:.6f}')
print(f'   Action: {action_type}')
print(f'   Expected HF: ~1.05')
print(f'   Match: {"✅" if abs(agent.state.health_factor - 1.05) < 0.01 else "❌"}')

# Simulate 1 minute with interest
engine.protocol.current_block = 1
engine.protocol.accrue_interest()
engine._update_agent_debt_interest(1)

print(f'\n✅ After 1 minute (debt update):')
print(f'   Debt: ${agent.state.moet_debt:,.2f}')
print(f'   HF (stored, not updated yet): {agent.state.health_factor:.6f}')

# Call decide_action again
protocol_state["current_step"] = 1
action_type, params = agent.decide_action(protocol_state, engine.state.current_prices)

print(f'\n✅ After decide_action (minute 1):')
print(f'   HF (now updated): {agent.state.health_factor:.6f}')
print(f'   Action: {action_type}')
print(f'   Expected HF: ~1.05 (slightly lower due to interest)')
print(f'   Match: {"✅" if abs(agent.state.health_factor - 1.05) < 0.02 else "❌"}')

