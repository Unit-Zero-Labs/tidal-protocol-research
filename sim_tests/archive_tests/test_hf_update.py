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
print(f'   HF: {agent.state.health_factor:.6f}')
print(f'   Debt: ${agent.state.moet_debt:,.2f}')
print(f'   BTC: {agent.state.btc_amount}')
print(f'   Rate check: {engine.protocol.get_moet_borrow_rate():.6f}')

# Simulate 1440 minutes (1 day)
for minute in [1, 1440]:
    engine.protocol.current_block = minute
    engine.protocol.accrue_interest()
    engine._update_agent_debt_interest(minute)
    
    print(f'\n✅ After {minute} minutes:')
    print(f'   HF: {agent.state.health_factor:.6f}')
    print(f'   Debt: ${agent.state.moet_debt:,.2f}')
    print(f'   Expected HF: ~1.05 (should stay stable)')
    print(f'   Match: {"✅" if abs(agent.state.health_factor - 1.05) < 0.01 else "❌"}')

