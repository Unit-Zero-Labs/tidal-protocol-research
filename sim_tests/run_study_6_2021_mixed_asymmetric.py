#!/usr/bin/env python3
"""
Study 6: 2021 Mixed Market - Asymmetric Comparison
- High Tide: Advanced MOET (dynamic market-driven rates)
- AAVE: Historical AAVE rates (2021)
- Equal initial health factor: 1.3
- Advanced MOET: ON
- Duration: 365 days (Jan 1 - Dec 31, 2021)
- BTC: $29,001.72 → $46,306.45 (+59.6%)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_tests.full_year_sim import FullYearSimConfig, FullYearSimulation


def main():
    print("=" * 80)
    print("STUDY 6: 2021 Mixed Market - Asymmetric Comparison (Advanced MOET)")
    print("=" * 80)
    print("Configuration:")
    print("  - Market: 2021 (Mixed, +59.6% BTC)")
    print("  - High Tide HF: 1.3 (trigger: 1.1, target: 1.2)")
    print("  - AAVE HF: 1.3 (static)")
    print("  - High Tide Rates: Advanced MOET (dynamic)")
    print("  - AAVE Rates: Historical AAVE 2021")
    print("  - Advanced MOET: ON")
    print("  - Duration: 365 days")
    print("  - Agents: 1 per protocol (clean comparison)")
    print("  - Ecosystem Growth: OFF")
    print("=" * 80)
    
    # Create configuration
    config = FullYearSimConfig()
    
    # Study 6 parameters
    config.test_name = "Full_Year_2021_BTC_Mixed_Market_Advanced_MOET_vs_AAVE_Historical_HT_vs_AAVE_Comparison"
    config.simulation_duration_days = 365
    config.num_agents = 100  # 100 agents per protocol
    config.initial_btc_per_agent = 1.0
    
    # Market data
    config.market_year = 2021
    config.use_historical_btc_data = True
    config.use_historical_aave_rates = True  # AAVE still uses historical
    
    # Health factors (Equal for comparison)
    config.agent_initial_hf = 1.3
    config.agent_rebalancing_hf = 1.1
    config.agent_target_hf = 1.2
    config.aave_initial_hf = 1.3  # Same as High Tide
    
    # Advanced MOET: ON for asymmetric study
    config.use_advanced_moet = True
    
    # Weekly yield harvesting
    config.enable_weekly_yield_harvest = True
    
    # Comparison mode
    config.comparison_mode = True
    config.run_high_tide_only = False
    
    # Ecosystem growth: OFF for clean comparison
    config.ecosystem_growth_enabled = False
    
    print("\nStarting simulation...")
    print("Expected runtime: ~8-15 minutes (Advanced MOET adds complexity)")
    print()
    
    # Run simulation
    sim = FullYearSimulation(config)
    results = sim.run_test()
    
    print("\n" + "=" * 80)
    print("STUDY 6 COMPLETE")
    print("=" * 80)
    print(f"Results saved to: tidal_protocol_sim/results/{config.test_name}/")
    print()


if __name__ == "__main__":
    main()

