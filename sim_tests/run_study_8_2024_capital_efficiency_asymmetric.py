#!/usr/bin/env python3
"""
Study 8: 2024 Capital Efficiency - Asymmetric Comparison (Realistic HF)
- High Tide: Advanced MOET (dynamic market-driven rates)
- AAVE: Historical AAVE rates (2024)
- Realistic health factors: High Tide 1.1 vs AAVE 1.95
- Advanced MOET: ON
- Duration: 365 days (Jan 1 - Dec 31, 2024)
- BTC: $42,208 → $92,627 (+119%)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_tests.full_year_sim import FullYearSimConfig, FullYearSimulation


def main():
    print("=" * 80)
    print("STUDY 8: 2024 Capital Efficiency - Asymmetric Comparison (Advanced MOET)")
    print("=" * 80)
    print("Configuration:")
    print("  - Market: 2024 (Bull, +119% BTC)")
    print("  - High Tide HF: 1.1 (trigger: 1.025, target: 1.04)")
    print("  - AAVE HF: 1.95 (realistic median from on-chain data)")
    print("  - High Tide Rates: Advanced MOET (dynamic)")
    print("  - AAVE Rates: Historical AAVE 2024")
    print("  - Advanced MOET: ON")
    print("  - Duration: 365 days")
    print("  - Agents: 1 per protocol (clean comparison)")
    print("  - Ecosystem Growth: OFF")
    print("=" * 80)
    
    # Create configuration
    config = FullYearSimConfig()
    
    # Study 8 parameters
    config.test_name = "Full_Year_2024_BTC_Capital_Efficiency_Advanced_MOET_vs_AAVE_Historical_HT_vs_AAVE_Comparison"
    config.simulation_duration_days = 365
    config.num_agents = 100  # 100 agents per protocol
    config.initial_btc_per_agent = 1.0
    
    # Market data
    config.market_year = 2024
    config.use_historical_btc_data = True
    config.use_historical_aave_rates = True  # AAVE still uses historical
    
    # Health factors (Realistic for capital efficiency comparison)
    config.agent_initial_hf = 1.1  # High Tide: aggressive
    config.agent_rebalancing_hf = 1.025
    config.agent_target_hf = 1.04
    config.aave_initial_hf = 1.95  # AAVE: conservative (median from real users)
    
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
    print("STUDY 8 COMPLETE")
    print("=" * 80)
    print(f"Results saved to: tidal_protocol_sim/results/{config.test_name}/")
    print()


if __name__ == "__main__":
    main()

