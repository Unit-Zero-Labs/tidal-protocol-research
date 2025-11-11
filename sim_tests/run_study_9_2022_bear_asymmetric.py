#!/usr/bin/env python3
"""
Study 9: 2022 Bear Market - Asymmetric Comparison
- High Tide: Advanced MOET (dynamic market-driven rates)
- AAVE: Historical AAVE rates (2022)
- Equal initial health factor: 1.3
- Advanced MOET: ON
- Duration: 365 days (Jan 1 - Dec 31, 2022)
- BTC: $46,320 → $16,604 (-64.2%)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_tests.full_year_sim import FullYearSimConfig, FullYearSimulation


def main():
    print("=" * 80)
    print("STUDY 9: 2022 Bear Market - Asymmetric Comparison (Advanced MOET)")
    print("=" * 80)
    print("Configuration:")
    print("  - Market: 2022 (Bear, -64.2% BTC)")
    print("  - High Tide HF: 1.3 (trigger: 1.1, target: 1.2)")
    print("  - AAVE HF: 1.3 (static)")
    print("  - High Tide Rates: Advanced MOET (dynamic)")
    print("  - AAVE Rates: Historical AAVE 2022")
    print("  - Advanced MOET: ON")
    print("  - Duration: 365 days")
    print("  - Agents: 1 per protocol (clean comparison)")
    print("  - Ecosystem Growth: OFF")
    print("=" * 80)
    print("\n⚠️  CRITICAL STRESS TEST: Bear market capital preservation ⚠️")
    print()
    
    # Create configuration
    config = FullYearSimConfig()
    
    # Study 9 parameters
    config.test_name = "Full_Year_2022_BTC_Bear_Market_Advanced_MOET_vs_AAVE_Historical_HT_vs_AAVE_Comparison"
    config.simulation_duration_days = 365
    config.num_agents = 100  # 100 agents per protocol
    config.initial_btc_per_agent = 1.0
    
    # Market data
    config.market_year = 2022
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
    print("STUDY 9 COMPLETE")
    print("=" * 80)
    print(f"Results saved to: tidal_protocol_sim/results/{config.test_name}/")
    print()


if __name__ == "__main__":
    main()

