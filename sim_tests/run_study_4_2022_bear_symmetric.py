#!/usr/bin/env python3
"""
Study 4: 2022 Bear Market - Symmetric Comparison
- Both protocols use historical AAVE rates (2022)
- Equal initial health factor: 1.3
- Advanced MOET: OFF
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
    print("STUDY 4: 2022 Bear Market - Symmetric Comparison")
    print("=" * 80)
    print("Configuration:")
    print("  - Market: 2022 (Bear, -64.2% BTC)")
    print("  - High Tide HF: 1.3 (trigger: 1.1, target: 1.2)")
    print("  - AAVE HF: 1.3 (static)")
    print("  - Rates: Historical AAVE 2022 (both protocols)")
    print("  - Advanced MOET: OFF")
    print("  - Duration: 365 days")
    print("  - Agents: 1 per protocol (clean comparison)")
    print("  - Ecosystem Growth: OFF")
    print("=" * 80)
    
    # Create configuration
    config = FullYearSimConfig()
    
    # Study 4 parameters
    config.test_name = "Full_Year_2022_BTC_Bear_Market_Equal_HF_Weekly_Yield_Harvest_HT_vs_AAVE_Comparison"
    config.simulation_duration_days = 365
    config.num_agents = 1  # Single agent per protocol
    config.initial_btc_per_agent = 1.0
    
    # Market data
    config.market_year = 2022
    config.use_historical_btc_data = True
    config.use_historical_aave_rates = True
    
    # Health factors (Equal for symmetric comparison)
    config.agent_initial_hf = 1.3
    config.agent_rebalancing_hf = 1.1
    config.agent_target_hf = 1.2
    config.aave_initial_hf = 1.3  # Same as High Tide
    
    # Advanced MOET: OFF for symmetric study
    config.use_advanced_moet = False
    
    # Weekly yield harvesting
    config.enable_weekly_yield_harvest = True
    
    # Comparison mode
    config.comparison_mode = True
    config.run_high_tide_only = False
    
    # Ecosystem growth: OFF for clean comparison
    config.ecosystem_growth_enabled = False
    
    print("\nStarting simulation...")
    print("Expected runtime: ~5-10 minutes")
    print()
    
    # Run simulation
    sim = FullYearSimulation(config)
    results = sim.run_test()
    
    print("\n" + "=" * 80)
    print("STUDY 4 COMPLETE")
    print("=" * 80)
    print(f"Results saved to: tidal_protocol_sim/results/{config.test_name}/")
    print()


if __name__ == "__main__":
    main()

