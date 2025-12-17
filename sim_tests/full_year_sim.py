#!/usr/bin/env python3
"""
Full Year Simulation with Real 2024 BTC Pricing

Tests the complete Tidal Protocol system over a full year (365 days) using
real 2024 BTC pricing data to validate:

1. ALM Rebalancer: Time-based rebalancing (12-hour intervals, 730 total triggers)
2. Algo Rebalancer: Threshold-based rebalancing (50 bps deviations)
3. Agent Rebalancing: Individual agent position management over full market cycle
4. Agent Leverage Increases: When BTC rises and HF > initial HF
5. Pool State Evolution: MOET:YT pool price accuracy over extended periods
6. Long-term Protocol Stability: Capital efficiency and scaling behavior

This simulation uses real 2024 BTC pricing data ($42k → $93k, +119% over year).
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import csv

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tidal_protocol_sim.engine.high_tide_vault_engine import HighTideVaultEngine, HighTideConfig
from tidal_protocol_sim.engine.aave_protocol_engine import AaveProtocolEngine, AaveConfig
from tidal_protocol_sim.agents.high_tide_agent import HighTideAgent
from tidal_protocol_sim.agents.aave_agent import AaveAgent
from tidal_protocol_sim.agents.pool_rebalancer import PoolRebalancerManager
from tidal_protocol_sim.core.protocol import TidalProtocol, Asset
from tidal_protocol_sim.core.yield_tokens import calculate_true_yield_token_price
from tidal_protocol_sim.core.uniswap_v3_math import create_moet_usdc_pool, create_moet_usdf_pool, UniswapV3SlippageCalculator


class FullYearSimConfig:
    """Configuration for full year simulation with real 2024 BTC pricing (Capital Efficiency Study)"""
    
    def __init__(self):
        # Test scenario parameters
        self.test_name = "Full_Year_2021_BTC_Mixed_Market_Equal_HF_Weekly_Yield_Harvest"
        self.simulation_duration_hours = 24 * 364  # 364 days for 2021
        self.simulation_duration_minutes = 364 * 24 * 60  # 524,160 minutes
        
        # Market year configuration (can be overridden by study scripts)
        self.market_year = 2021  # Default to 2021
        
        # BTC pricing data configuration
        self.btc_csv_path = "btc-usd-max.csv"
        self.btc_data = []  # Will be loaded based on market_year
        # Legacy attribute for backward compatibility
        self.btc_2021_data = []
        
        # AAVE comparison configuration
        self.run_aave_comparison = True  # Enable AAVE vs High Tide comparison
        self.use_historical_rates = True  # Use historical AAVE rates for both protocols
        self.use_historical_btc_data = True  # Use historical BTC price data
        self.use_historical_aave_rates = True  # Use historical AAVE rates
        self.rates_csv_path = "rates_compute.csv"
        self.aave_rates = {}  # Will be loaded based on market_year
        # Legacy attribute for backward compatibility
        self.aave_rates_2021 = {}
        self.enable_advanced_moet_system = False  # Disable Advanced MOET - use historical rates for both
        
        # Alias for backward compatibility with study scripts
        self.use_advanced_moet = False  # Alias for enable_advanced_moet_system
        
        # Agent configuration - Equal HF for 2021 Mixed Market
        self.num_agents = 1  # 1 agent for detailed tracking
        self.use_mixed_risk_profiles = False  # Use uniform profile for all agents
        
        # Both protocols start at equal health factors for baseline comparison
        self.agent_initial_hf = 1.3      # Both start at 1.3 HF (equal positioning)
        self.agent_rebalancing_hf = 1.1  # High Tide rebalancing trigger
        self.agent_target_hf = 1.2       # High Tide rebalancing target
        
        # AAVE: Same initial HF for fair comparison
        self.aave_initial_hf = 1.3       # Equal starting position
        
        # BTC price scenario - Real 2021 data (Mixed Market)
        self.btc_initial_price = 29001.72  # 2021-01-01 price  
        self.btc_final_price = 46306.45    # 2021-12-30 price (+59.6%)
        self.btc_price_pattern = "real_2021_data"  # Use actual historical data
        
        # Pool configurations - Scaled for 120 agents over full year
        self.moet_btc_pool_config = {
            "size": 10_000_000,  # $10M liquidation pool (120 agents × ~$100k each)
            "concentration": 0.80,
            "fee_tier": 0.003,
            "tick_spacing": 60,
            "pool_name": "MOET:BTC"
        }
        
        self.moet_yt_pool_config = {
            "size": 500_000,  # $500K pool with 95% concentration and 75/25 skew
            "concentration": 0.95,  # 95% concentration at 1:1 peg
            "token0_ratio": 0.75,  # 75% MOET, 25% YT
            "fee_tier": 0.0005,  # 0.05% fee tier
            "tick_spacing": 10,
            "pool_name": "MOET:Yield_Token"
        }
        
        # Pool rebalancing configuration - ENABLED for year-long test
        self.enable_pool_arbing = True
        self.alm_rebalance_interval_minutes = 720  # 12 hours (730 triggers over year)
        self.algo_deviation_threshold_bps = 50.0  # 50 basis points
        
        # Arbitrage delay configuration - NEW FEATURE
        self.enable_arb_delay = True  # ENABLED for this test to see delay effects!
        self.arb_delay_description = "1 hour (auto-converted based on simulation time scale)"
        
        # Yield token parameters
        self.yield_apr = 0.10  # 10% APR
        self.use_direct_minting_for_initial = True
        
        # Enhanced logging and data collection - Scaled for year-long simulation
        self.detailed_logging = True
        self.log_every_n_minutes = 1440  # Log daily (every 24 hours)
        self.collect_pool_state_every_n_minutes = 1440  # Daily pool state snapshots
        self.track_all_rebalancing_events = True
        
        # Agent health snapshot frequency (NEW for Study 14 minute-by-minute tracking)
        self.agent_snapshot_frequency_minutes = 1440  # Default: daily (once per day)
        
        # Progress reporting
        self.progress_report_every_n_minutes = 10080  # Weekly progress reports (7 days)
        
        # Advanced MOET system toggle
        # STUDY 1: Symmetric comparison - both protocols use same historical rates (NO Advanced MOET)
        # This isolates the automation advantage by using identical interest rates
        # self.enable_advanced_moet_system = False  # Already set above on line 63
        
        # Note: When enable_advanced_moet_system=True AND use_historical_rates=True:
        # - High Tide agents face dynamic MOET rates (r_floor + r_bond_cost)
        # - AAVE agents face fixed historical rates from CSV
        # This creates an asymmetric but realistic comparison
        
        # Ecosystem Growth Configuration
        # Disable ecosystem growth when comparing with AAVE for clean comparison
        self.enable_ecosystem_growth = False  # Disabled for clean comparison
        self.target_btc_deposits = 150_000_000  # $150M target BTC deposits by year end
        self.max_agents = 500  # Maximum 500 agents to prevent system overload
        self.btc_per_agent_range = (2.0, 5.0)  # 2-5 BTC per agent (whale/institutional behavior)
        self.growth_start_delay_days = 30  # Start adding agents after 30 days
        self.growth_acceleration_factor = 1.2  # Exponential growth factor
        
        # Output configuration
        self.generate_charts = True
        self.save_detailed_csv = True
        
        # Optimization configuration (for binary search studies)
        self.fail_fast_on_liquidation = False  # Exit immediately on first liquidation
        self.suppress_progress_output = False  # Suppress detailed progress for optimization runs
    
    def __setattr__(self, name, value):
        """Override setattr to sync use_advanced_moet with enable_advanced_moet_system"""
        # Sync the two Advanced MOET attributes
        if name == 'use_advanced_moet':
            super().__setattr__('enable_advanced_moet_system', value)
            super().__setattr__('use_advanced_moet', value)
        elif name == 'enable_advanced_moet_system':
            super().__setattr__('use_advanced_moet', value)
            super().__setattr__('enable_advanced_moet_system', value)
        else:
            super().__setattr__(name, value)
    
    def load_market_data(self):
        """Load BTC price data and AAVE rates based on market_year configuration"""
        print(f"\n📅 Loading market data for year: {self.market_year}")
        
        # Load BTC price data
        if self.use_historical_btc_data:
            if self.market_year == 2021:
                self.btc_data = self._load_2021_btc_data()
                self.btc_2021_data = self.btc_data  # For backward compatibility
            elif self.market_year == 2024:
                self.btc_data = self._load_2024_btc_data()
            elif self.market_year == 2022:
                self.btc_data = self._load_2022_btc_data()
            elif self.market_year == 2025:
                self.btc_data = self._load_2025_btc_data()
            else:
                print(f"⚠️ Warning: No loader for year {self.market_year}, using 2021 data")
                self.btc_data = self._load_2021_btc_data()
        
        # Update btc_initial_price and btc_final_price from loaded data
        if self.btc_data and len(self.btc_data) > 0:
            self.btc_initial_price = self.btc_data[0]
            self.btc_final_price = self.btc_data[-1]
        
        # Load AAVE rates
        if self.use_historical_aave_rates:
            if self.market_year == 2021:
                self.aave_rates = self._load_2021_aave_rates()
                self.aave_rates_2021 = self.aave_rates  # For backward compatibility
            elif self.market_year == 2024:
                self.aave_rates = self._load_2024_aave_rates()
            elif self.market_year == 2022:
                self.aave_rates = self._load_2022_aave_rates()
            elif self.market_year == 2025:
                self.aave_rates = self._load_2025_aave_rates()
            else:
                print(f"⚠️ Warning: No rate loader for year {self.market_year}, using 2021 rates")
                self.aave_rates = self._load_2021_aave_rates()
        
        print(f"✅ Market data loaded: {len(self.btc_data)} days of BTC prices\n")
    
    def _load_2024_btc_data(self) -> List[float]:
        """Load 2024 BTC pricing data from CSV file (Bull Market Year)"""
        btc_prices = []
        
        try:
            with open(self.btc_csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Check if this is a 2024 date
                    if '2024-' in row['snapped_at']:
                        price = float(row['price'])
                        btc_prices.append(price)
            
            print(f"📊 Loaded {len(btc_prices)} days of 2024 BTC pricing data")
            print(f"📈 2024 BTC Range: ${btc_prices[0]:,.2f} → ${btc_prices[-1]:,.2f} ({((btc_prices[-1]/btc_prices[0])-1)*100:+.1f}%)")
            
            return btc_prices
            
        except FileNotFoundError:
            print(f"⚠️  Warning: {self.btc_csv_path} not found. Using fallback.")
            return [42208.23] * 365  # Fallback to flat 2024 starting price
        except Exception as e:
            print(f"❌ Error loading BTC data: {e}. Using fallback.")
            return [42208.23] * 365
    
    def _load_2024_aave_rates(self) -> Dict[int, float]:
        """
        Load 2024 AAVE USDC borrow rates from rates_compute.csv (Bull Market Year)
        Returns: Dict mapping day_of_year (0-364) -> daily borrow rate
        """
        from datetime import datetime
        
        rates_by_day = {}
        
        try:
            with open(self.rates_csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Check if this is a 2024 date
                    if '2024-' in row['date']:
                        # Parse date to get day of year
                        date_str = row['date'].split(' ')[0]  # Extract date part before time
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        day_of_year = date_obj.timetuple().tm_yday - 1  # 0-indexed (0-365 for leap year)
                        
                        # Extract borrow rate (avg_variableRate column)
                        borrow_rate = float(row['avg_variableRate'])
                        rates_by_day[day_of_year] = borrow_rate
            
            # Fill in missing days with interpolation (2024 is a leap year - 366 days)
            rates_by_day = self._interpolate_missing_days(rates_by_day, 366)
            
            print(f"📊 Loaded {len(rates_by_day)} days of 2024 AAVE borrow rates")
            rates_list = [rates_by_day[i] for i in sorted(rates_by_day.keys())]
            print(f"📈 2024 Rate Range: {min(rates_list)*100:.2f}% → {max(rates_list)*100:.2f}% APR")
            print(f"📈 Average Rate: {np.mean(rates_list)*100:.2f}% APR")
            
            return rates_by_day
            
        except FileNotFoundError:
            print(f"⚠️  Warning: {self.rates_csv_path} not found. Using fallback 5% APR.")
            return {i: 0.05 for i in range(366)}
        except Exception as e:
            print(f"❌ Error loading AAVE rates: {e}. Using fallback 5% APR.")
            return {i: 0.05 for i in range(366)}
    
    def _load_2025_btc_data(self) -> List[float]:
        """Load 2025 BTC pricing data from CSV file (Low Vol Market - 268 days)"""
        btc_prices = []
        
        try:
            with open(self.btc_csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Check if this is a 2025 date
                    if '2025-' in row['snapped_at']:
                        price = float(row['price'])
                        btc_prices.append(price)
            
            # Only take first 268 days (Jan 1 - Sept 25)
            btc_prices = btc_prices[:268]
            
            print(f"📊 Loaded {len(btc_prices)} days of 2025 BTC pricing data")
            print(f"📈 2025 BTC Range: ${btc_prices[0]:,.2f} → ${btc_prices[-1]:,.2f} ({((btc_prices[-1]/btc_prices[0])-1)*100:+.1f}%)")
            
            return btc_prices
            
        except FileNotFoundError:
            print(f"⚠️  Warning: {self.btc_csv_path} not found. Using fallback.")
            return [93507.86] * 268  # Fallback to flat 2025 starting price
        except Exception as e:
            print(f"❌ Error loading BTC data: {e}. Using fallback.")
            return [93507.86] * 268
    
    def _load_2025_aave_rates(self) -> Dict[int, float]:
        """
        Load 2025 AAVE USDC borrow rates from rates_compute.csv (Low Vol Market - 268 days)
        Returns: Dict mapping day_of_year (0-267) -> daily borrow rate
        """
        from datetime import datetime
        
        rates_by_day = {}
        
        try:
            with open(self.rates_csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Check if this is a 2025 date
                    if '2025-' in row['date']:
                        # Parse date to get day of year
                        date_str = row['date'].split(' ')[0]  # Extract date part before time
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        day_of_year = date_obj.timetuple().tm_yday - 1  # 0-indexed
                        
                        # Only include first 268 days (Jan 1 - Sept 25)
                        if day_of_year < 268:
                            # Extract borrow rate (avg_variableRate column)
                            borrow_rate = float(row['avg_variableRate'])
                            rates_by_day[day_of_year] = borrow_rate
            
            # Fill in missing days with interpolation
            rates_by_day = self._interpolate_missing_days(rates_by_day, 268)
            
            print(f"📊 Loaded {len(rates_by_day)} days of 2025 AAVE borrow rates")
            rates_list = [rates_by_day[i] for i in sorted(rates_by_day.keys())]
            print(f"📈 2025 Rate Range: {min(rates_list)*100:.2f}% → {max(rates_list)*100:.2f}% APR")
            print(f"📈 Average Rate: {np.mean(rates_list)*100:.2f}% APR")
            
            return rates_by_day
            
        except FileNotFoundError:
            print(f"⚠️  Warning: {self.rates_csv_path} not found. Using fallback 5% APR.")
            return {i: 0.05 for i in range(268)}
        except Exception as e:
            print(f"❌ Error loading AAVE rates: {e}. Using fallback 5% APR.")
            return {i: 0.05 for i in range(268)}
    
    def _load_2022_btc_data(self) -> List[float]:
        """Load 2022 BTC pricing data from CSV file (Bear Market Year)"""
        btc_prices = []
        
        try:
            with open(self.btc_csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Check if this is a 2022 date
                    if '2022-' in row['snapped_at']:
                        price = float(row['price'])
                        btc_prices.append(price)
            
            print(f"📊 Loaded {len(btc_prices)} days of 2022 BTC pricing data")
            print(f"📈 2022 BTC Range: ${btc_prices[0]:,.2f} → ${btc_prices[-1]:,.2f} ({((btc_prices[-1]/btc_prices[0])-1)*100:+.1f}%)")
            
            return btc_prices
            
        except FileNotFoundError:
            print(f"⚠️  Warning: {self.btc_csv_path} not found. Using synthetic 2022 data.")
            # Fallback: Generate synthetic 2022-like price progression
            return self._generate_synthetic_2022_data()
        except Exception as e:
            print(f"❌ Error loading BTC data: {e}. Using synthetic data.")
            return self._generate_synthetic_2022_data()
    
    def _generate_synthetic_2022_data(self) -> List[float]:
        """Generate synthetic 2022-like BTC price progression as fallback (bear market)"""
        # Create 365 days of synthetic data (2022 was not a leap year)
        days = 365
        prices = []
        
        # Approximate 2022 progression: $46k → $17k with volatility (bear market)
        start_price = 46319.65
        end_price = 16604.02
        
        for day in range(days):
            # Base progression
            progress = day / (days - 1)
            base_price = start_price + (end_price - start_price) * progress
            
            # Add some realistic volatility (±5% daily)
            volatility = random.uniform(-0.05, 0.05)
            daily_price = base_price * (1 + volatility)
            
            # Ensure price stays positive and reasonable
            daily_price = max(daily_price, 10000.0)
            prices.append(daily_price)
        
        print(f"📊 Generated {len(prices)} days of synthetic 2022 BTC data")
        return prices
    
    def _load_2022_aave_rates(self) -> Dict[int, float]:
        """
        Load 2022 AAVE USDC borrow rates from rates_compute.csv (Bear Market Year)
        Returns: Dict mapping day_of_year (0-364) -> daily borrow rate
        """
        from datetime import datetime
        
        rates_by_day = {}
        
        try:
            with open(self.rates_csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Check if this is a 2022 date
                    if '2022-' in row['date']:
                        # Parse date to get day of year
                        date_str = row['date'].split(' ')[0]  # Extract date part before time
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        day_of_year = date_obj.timetuple().tm_yday - 1  # 0-indexed (0-364)
                        
                        # Extract borrow rate (avg_variableRate column)
                        borrow_rate = float(row['avg_variableRate'])
                        rates_by_day[day_of_year] = borrow_rate
            
            # Fill in missing days with interpolation
            rates_by_day = self._interpolate_missing_days(rates_by_day, 365)  # 2022 is not a leap year
            
            print(f"📊 Loaded {len(rates_by_day)} days of 2022 AAVE borrow rates")
            rates_list = [rates_by_day[i] for i in sorted(rates_by_day.keys())]
            print(f"📈 2022 Rate Range: {min(rates_list)*100:.2f}% → {max(rates_list)*100:.2f}% APR")
            print(f"📈 Average Rate: {np.mean(rates_list)*100:.2f}% APR")
            
            return rates_by_day
            
        except FileNotFoundError:
            print(f"⚠️  Warning: {self.rates_csv_path} not found. Using fallback 5% APR.")
            return {i: 0.05 for i in range(366)}
        except Exception as e:
            print(f"❌ Error loading AAVE rates: {e}. Using fallback 5% APR.")
            return {i: 0.05 for i in range(366)}
    
    def _load_2021_btc_data(self) -> List[float]:
        """Load 2021 BTC pricing data from CSV file (Mixed Market Year)"""
        btc_prices = []
        
        try:
            with open(self.btc_csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Check if this is a 2021 date
                    if '2021-' in row['snapped_at']:
                        price = float(row['price'])
                        btc_prices.append(price)
            
            print(f"📊 Loaded {len(btc_prices)} days of 2021 BTC pricing data")
            print(f"📈 2021 BTC Range: ${btc_prices[0]:,.2f} → ${btc_prices[-1]:,.2f} ({((btc_prices[-1]/btc_prices[0])-1)*100:+.1f}%)")
            
            return btc_prices
            
        except FileNotFoundError:
            print(f"⚠️  Warning: {self.btc_csv_path} not found. Using synthetic 2021 data.")
            # Fallback: Generate synthetic 2021-like price progression
            return self._generate_synthetic_2021_data()
        except Exception as e:
            print(f"❌ Error loading BTC data: {e}. Using synthetic data.")
            return self._generate_synthetic_2021_data()
    
    def _generate_synthetic_2021_data(self) -> List[float]:
        """Generate synthetic 2021-like BTC price progression as fallback (mixed market)"""
        # Create 365 days of synthetic data
        days = 365
        prices = []
        
        # Approximate 2021 progression: $29k → $64k (April) → $46k (December) with volatility
        start_price = 29001.72
        mid_peak = 64000.0  # April peak
        end_price = 46306.45
        
        for day in range(days):
            progress = day / (days - 1)
            
            # Two-phase market: bull run to April, then correction
            if day < 120:  # First 4 months - bull run
                phase_progress = day / 120
                base_price = start_price + (mid_peak - start_price) * phase_progress
            else:  # Rest of year - correction and consolidation
                phase_progress = (day - 120) / (days - 120)
                base_price = mid_peak + (end_price - mid_peak) * phase_progress
            
            # Add daily volatility (~3%)
            daily_volatility = base_price * 0.03 * (random.random() - 0.5) * 2
            final_price = base_price + daily_volatility
            prices.append(max(10000, final_price))  # Floor at $10k
        
        print(f"📊 Generated {len(prices)} days of synthetic 2021 BTC data")
        return prices
    
    def _load_2021_aave_rates(self) -> Dict[int, float]:
        """
        Load 2021 AAVE USDC borrow rates from rates_compute.csv (Mixed Market Year)
        Returns: Dict mapping day_of_year (0-364) -> daily borrow rate
        """
        from datetime import datetime
        
        rates_by_day = {}
        
        try:
            with open(self.rates_csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Check if this is a 2021 date
                    if '2021-' in row['date']:
                        # Parse date to get day of year
                        date_str = row['date'].split(' ')[0]  # Extract date part before time
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        day_of_year = date_obj.timetuple().tm_yday - 1  # 0-indexed (0-364)
                        
                        # Extract borrow rate (avg_variableRate column)
                        borrow_rate = float(row['avg_variableRate'])
                        rates_by_day[day_of_year] = borrow_rate
            
            # Fill in missing days with interpolation
            rates_by_day = self._interpolate_missing_days(rates_by_day, 365)
            
            print(f"📊 Loaded {len(rates_by_day)} days of 2021 AAVE borrow rates")
            rates_list = [rates_by_day[i] for i in sorted(rates_by_day.keys())]
            print(f"📈 2021 Rate Range: {min(rates_list)*100:.2f}% → {max(rates_list)*100:.2f}% APR")
            print(f"📈 Average Rate: {np.mean(rates_list)*100:.2f}% APR")
            
            return rates_by_day
            
        except FileNotFoundError:
            print(f"⚠️  Warning: {self.rates_csv_path} not found. Using fallback 5% APR.")
            return {i: 0.05 for i in range(366)}
        except Exception as e:
            print(f"❌ Error loading AAVE rates: {e}. Using fallback 5% APR.")
            return {i: 0.05 for i in range(366)}
    
    def _interpolate_missing_days(self, rates_dict: Dict[int, float], total_days: int) -> Dict[int, float]:
        """Fill in missing days using linear interpolation"""
        if len(rates_dict) == total_days:
            return rates_dict
        
        # Get sorted day indices and rates
        days = sorted(rates_dict.keys())
        
        # Fill in missing days
        complete_rates = {}
        for day in range(total_days):
            if day in rates_dict:
                complete_rates[day] = rates_dict[day]
            else:
                # Find nearest days with data
                lower_day = max([d for d in days if d < day], default=None)
                upper_day = min([d for d in days if d > day], default=None)
                
                if lower_day is None:
                    # Use first available rate
                    complete_rates[day] = rates_dict[days[0]]
                elif upper_day is None:
                    # Use last available rate
                    complete_rates[day] = rates_dict[days[-1]]
                else:
                    # Linear interpolation
                    weight = (day - lower_day) / (upper_day - lower_day)
                    complete_rates[day] = rates_dict[lower_day] * (1 - weight) + rates_dict[upper_day] * weight
        
        return complete_rates
    
    def get_historical_rate_at_minute(self, minute: int) -> float:
        """
        Get historical AAVE borrow rate for a given simulation minute
        
        Args:
            minute: Simulation minute (0 to 524,160)
        
        Returns:
            Daily borrow rate (APR as decimal, e.g. 0.05 = 5%)
        """
        if not self.use_historical_rates or not self.aave_rates:
            return 0.05  # Fallback to 5% APR
        
        # Convert minute to day
        day = minute // 1440
        
        # Get the max day available (handles different year lengths)
        max_day = max(self.aave_rates.keys()) if self.aave_rates else 363
        day = min(day, max_day)
        
        return self.aave_rates.get(day, 0.05)
        
    def get_btc_price_at_minute(self, minute: int) -> float:
        """Get BTC price at given minute using historical data with interpolation"""
        
        if not self.btc_data:
            # Fallback to linear progression
            progress = minute / self.simulation_duration_minutes
            return self.btc_initial_price + (self.btc_final_price - self.btc_initial_price) * progress
        
        # Calculate which day we're on
        minutes_per_day = 24 * 60  # 1440 minutes per day
        day_of_year = minute // minutes_per_day
        
        # Ensure we don't exceed available data
        if day_of_year >= len(self.btc_data):
            return self.btc_data[-1]  # Use last available price
        
        # Get current day price
        current_day_price = self.btc_data[day_of_year]
        
        # Linear interpolation within the day if we have next day data
        if day_of_year + 1 < len(self.btc_data):
            next_day_price = self.btc_data[day_of_year + 1]
            
            # Calculate progress within the current day (0.0 to 1.0)
            minutes_into_day = minute % minutes_per_day
            daily_progress = minutes_into_day / minutes_per_day
            
            # Linear interpolation between daily prices
            interpolated_price = current_day_price + (next_day_price - current_day_price) * daily_progress
            return interpolated_price
        else:
            # Use current day price if no next day data
            return current_day_price


class FullYearSimulation:
    """Main simulation class for full year BTC protocol testing"""
    
    def __init__(self, config: FullYearSimConfig):
        self.config = config
        self.results = {
            "test_metadata": {
                "test_name": config.test_name,
                "timestamp": datetime.now().isoformat(),
                "duration_hours": config.simulation_duration_hours,
                "num_agents": config.num_agents,
                "btc_decline_percent": ((config.btc_initial_price - config.btc_final_price) / config.btc_initial_price) * 100,
                "pool_arbing_enabled": config.enable_pool_arbing
            },
            "detailed_logs": [],
            "rebalancing_events": {
                "agent_rebalances": [],
                "alm_rebalances": [],
                "algo_rebalances": []
            },
            "pool_state_snapshots": [],
            "agent_performance": {},
            "pool_arbitrage_analysis": {}
        }
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
    def run_test(self) -> Dict[str, Any]:
        """Run the complete full year simulation (High Tide only or High Tide vs AAVE comparison)"""
        
        # Load market data based on configured year (MUST be called after config is set up)
        self.config.load_market_data()
        
        print("🌍 STUDY 1: 2021 MIXED MARKET - EQUAL HEALTH FACTORS")
        print("=" * 70)
        print(f"📅 Duration: {self.config.simulation_duration_hours:,} hours ({self.config.simulation_duration_hours//24} days)")
        print(f"👥 Agents: {self.config.num_agents} agents per system")
        print(f"📊 BTC Market ({self.config.market_year}): ${self.config.btc_initial_price:,.0f} → ${self.config.btc_final_price:,.0f} ({((self.config.btc_final_price/self.config.btc_initial_price)-1)*100:+.1f}%)")
        print(f"🔄 Pool Arbitrage: {'ENABLED' if self.config.enable_pool_arbing else 'DISABLED'}")
        print(f"📊 BTC Data: {len(self.config.btc_data)} daily prices loaded")
        
        if self.config.run_aave_comparison:
            print(f"📊 Mode: SYMMETRIC COMPARISON (Equal HF, Historical AAVE Rates for Both)")
            print(f"💎 High Tide: Automated Rebalancing + Weekly Yield Harvesting")
            print(f"📈 AAVE: Buy & Hold + Historical 2021 rates from CSV")
            print(f"💰 Both start at 1.3 Initial HF (equal positioning)")
            print(f"💎 Weekly yield harvest enabled for High Tide")
            print()
            return self._run_comparison_simulation()
        else:
            print(f"📊 Mode: HIGH TIDE ONLY")
            print(f"⏳ Leverage Increases: ENABLED (agents increase leverage when HF > initial HF)")
            print()
            return self._run_high_tide_only_simulation()
    
    def _run_high_tide_only_simulation(self) -> Dict[str, Any]:
        """Run High Tide simulation only (original behavior)"""
        
        # Create and configure High Tide engine
        engine = self._create_test_engine()
        
        # Run the simulation with detailed tracking
        simulation_results = self._run_simulation_with_detailed_tracking(engine)
        
        # Store simulation results
        self.results["simulation_results"] = simulation_results
        
        # Analyze results
        self._analyze_test_results(engine)
        
        # Save results
        self._save_test_results()
        
        # Generate charts
        if self.config.generate_charts:
            self._generate_test_charts()
        
        print("\n✅ Full year simulation completed!")
        self._print_test_summary()
        
        return self.results
    
    def _run_comparison_simulation(self) -> Dict[str, Any]:
        """Run both High Tide and AAVE simulations for comparison"""
        
        print("\n" + "=" * 70)
        print("RUNNING HIGH TIDE SIMULATION (1/2)")
        print("=" * 70 + "\n")
        
        # Run High Tide simulation
        ht_engine = self._create_test_engine()
        ht_results = self._run_simulation_with_detailed_tracking(ht_engine)
        
        print("\n" + "=" * 70)
        print("RUNNING AAVE SIMULATION (2/2)")
        print("=" * 70 + "\n")
        
        # Run AAVE simulation
        aave_engine = self._create_aave_engine()
        aave_results = self._run_simulation_with_detailed_tracking_aave(aave_engine)
        
        # Store both results
        self.results["high_tide_results"] = ht_results
        self.results["aave_results"] = aave_results
        self.results["ht_engine"] = ht_engine
        self.results["aave_engine"] = aave_engine
        
        # Analyze and compare
        print("\n" + "=" * 70)
        print("ANALYZING COMPARISON RESULTS")
        print("=" * 70 + "\n")
        
        self._analyze_comparison_results(ht_engine, aave_engine)
        
        # Save results
        self._save_comparison_results()
        
        # Generate all charts (detailed + comparison)
        if self.config.generate_charts:
            print("\n📊 Generating comprehensive analysis charts...")
            
            # Generate detailed charts for High Tide
            print("\n📈 Creating High Tide detailed analysis...")
            self._generate_detailed_charts_for_protocol("high_tide", ht_engine)
            
            # Generate detailed charts for AAVE
            print("\n📈 Creating AAVE detailed analysis...")
            self._generate_detailed_charts_for_protocol("aave", aave_engine)
            
            # Generate comparison charts
            self._generate_comparison_charts()
        
        print("\n✅ Full year comparison simulation completed!")
        self._print_comparison_summary()
        
        return self.results
    
    def _create_test_engine(self) -> HighTideVaultEngine:
        """Create and configure the High Tide engine for testing"""
        
        # Create High Tide configuration
        ht_config = HighTideConfig()
        ht_config.num_high_tide_agents = 0  # We'll create custom agents
        # CRITICAL: Ensure BTC decline duration matches full simulation
        ht_config.btc_decline_duration = self.config.simulation_duration_minutes  # 2160 minutes for 36h
        ht_config.btc_initial_price = self.config.btc_initial_price
        ht_config.btc_final_price_range = (self.config.btc_final_price, self.config.btc_final_price)
        
        # Enable advanced MOET system (disabled when using historical AAVE rates for comparison)
        ht_config.enable_advanced_moet_system = self.config.enable_advanced_moet_system
        
        print(f"🔧 Advanced MOET System: {'ENABLED' if ht_config.enable_advanced_moet_system else 'DISABLED (using historical AAVE rates)'}")
        
        # Configure arbitrage agents for the new pool structure
        ht_config.num_arbitrage_agents = 5  # Enable arbitrage agents for peg maintenance
        ht_config.arbitrage_agent_balance = 100_000.0  # $100K per agent
        
        # CRITICAL: Propagate agent snapshot frequency for minute-by-minute tracking (Study 14)
        ht_config.agent_snapshot_frequency_minutes = getattr(self.config, 'agent_snapshot_frequency_minutes', 1440)
        
        print(f"🔧 DEBUG: Configuring BTC decline over {ht_config.btc_decline_duration} minutes")
        print(f"🔧 DEBUG: Price range: ${ht_config.btc_initial_price:,.0f} → ${self.config.btc_final_price:,.0f}")
        
        # Configure pools
        ht_config.moet_btc_pool_size = self.config.moet_btc_pool_config["size"]
        ht_config.moet_btc_concentration = self.config.moet_btc_pool_config["concentration"]
        ht_config.moet_yield_pool_size = self.config.moet_yt_pool_config["size"]
        ht_config.yield_token_concentration = self.config.moet_yt_pool_config["concentration"]
        ht_config.yield_token_ratio = self.config.moet_yt_pool_config["token0_ratio"]
        ht_config.use_direct_minting_for_initial = self.config.use_direct_minting_for_initial
        
        # Create engine
        engine = HighTideVaultEngine(ht_config)
        
        # CRITICAL FIX: Override the default $100k BTC price with actual market data
        ht_config.btc_initial_price = self.config.btc_initial_price
        
        # Override MOET rates with historical AAVE rates ONLY if Advanced MOET is disabled
        # When Advanced MOET is enabled, High Tide uses dynamic bond-based rates
        if self.config.use_historical_rates and not self.config.enable_advanced_moet_system:
            avg_rate = sum(self.config.aave_rates.values()) / len(self.config.aave_rates) * 100 if self.config.aave_rates else 5.0
            print(f"📊 Using historical AAVE rates for High Tide (avg: {avg_rate:.2f}% APR) - Symmetric comparison")
            
            # Create rate override function that uses engine.current_step (simulation minute)
            def get_historical_aave_rate():
                current_minute = getattr(engine, 'current_step', 0)
                return self.config.get_historical_rate_at_minute(current_minute)
            
            # Override the borrow rate function
            engine.protocol.get_moet_borrow_rate = get_historical_aave_rate
        elif self.config.enable_advanced_moet_system:
            print(f"🚀 Advanced MOET System ENABLED - High Tide uses dynamic bond-based rates (Asymmetric comparison)")
        
        # Create uniform profile agents for year-long test
        agents = self._create_uniform_agents(engine)
        engine.high_tide_agents = agents
        
        # Add agents to engine's agent dict
        for agent in agents:
            engine.agents[agent.agent_id] = agent
            agent.engine = engine  # Set engine reference
        
        # Create and configure pool rebalancer
        pool_rebalancer = PoolRebalancerManager(
            alm_interval_minutes=self.config.alm_rebalance_interval_minutes,
            algo_threshold_bps=self.config.algo_deviation_threshold_bps
        )
        pool_rebalancer.set_enabled(self.config.enable_pool_arbing)
        pool_rebalancer.set_yield_token_pool(engine.yield_token_pool)
        
        # Configure arbitrage delay (disabled for year-long test)
        # pool_rebalancer.set_arb_delay_enabled(False)  # Disabled for simplicity in year-long test
        
        # Store rebalancer reference for access during simulation
        engine.pool_rebalancer = pool_rebalancer
        
        # Add stablecoin pools for deleveraging support (when not using advanced MOET system)
        if not self.config.enable_advanced_moet_system:
            self._add_stablecoin_pools_to_engine(engine)
        
        # CRITICAL FIX: Register rebalancer MOET balances with protocol
        # The rebalancers start with MOET that should be counted in total supply
        if engine.protocol.enable_advanced_moet:
            alm_moet_balance = pool_rebalancer.alm_rebalancer.state.moet_balance
            algo_moet_balance = pool_rebalancer.algo_rebalancer.state.moet_balance
            total_rebalancer_moet = alm_moet_balance + algo_moet_balance
            
            # Register this MOET with the protocol system
            engine.protocol.moet_system.mint(total_rebalancer_moet)
            print(f"🔄 Registered rebalancer MOET with protocol: ${total_rebalancer_moet:,.0f}")
            print(f"   ALM: ${alm_moet_balance:,.0f}, Algo: ${algo_moet_balance:,.0f}")
            
            # CRITICAL FIX: Register YT pool's initial MOET balance with protocol
            # The YT pool starts with MOET that should be counted in total supply
            yt_pool_moet_balance = pool_rebalancer.alm_rebalancer.yield_token_pool.moet_reserve
            engine.protocol.moet_system.mint(yt_pool_moet_balance)
            print(f"🏊 Registered YT pool MOET with protocol: ${yt_pool_moet_balance:,.0f}")
            
            # CRITICAL FIX: Initialize reserves based on TOTAL MOET supply (not just agent debt)
            # This ensures the 8% reserve ratio applies to the complete MOET ecosystem
            total_moet_supply = engine.protocol.moet_system.total_supply
            engine.protocol.initialize_moet_reserves(total_moet_supply)
            print(f"🏦 Re-initialized reserves based on total MOET supply: ${total_moet_supply:,.0f}")
            
            # Show the corrected reserve calculation
            actual_reserves = engine.protocol.moet_system.redeemer.reserve_state.total_reserves
            reserve_ratio = actual_reserves / total_moet_supply if total_moet_supply > 0 else 0
            print(f"   Final reserves: ${actual_reserves:,.0f} ({reserve_ratio:.1%} of total supply)")
        
        self._log_event(0, "ENGINE_SETUP", "High Tide engine created with pool rebalancer", {
            "num_agents": len(agents),
            "pool_arbing_enabled": self.config.enable_pool_arbing,
            "alm_interval": self.config.alm_rebalance_interval_minutes,
            "algo_threshold": self.config.algo_deviation_threshold_bps
        })
        
        return engine
    
    def _add_stablecoin_pools_to_engine(self, engine, pool_size: float = 5_000_000.0):
        """
        Manually add MOET:stablecoin pools to engine for deleveraging support
        WITHOUT enabling the full advanced MOET system (no Bonder/Redeemer modules)
        """
        print(f"\n🏊 Adding MOET:stablecoin pools for deleveraging:")
        print(f"   MOET:USDC pool: ${pool_size:,.0f}")
        print(f"   MOET:USDF pool: ${pool_size:,.0f}")
        
        # Create MOET:stablecoin pairs (for step 2 of deleveraging: YT → MOET → USDC/USDF)
        engine.moet_usdc_pool = create_moet_usdc_pool(
            pool_size_usd=pool_size,
            concentration=0.95,  # 95% concentration at 1:1 peg
            token0_ratio=0.5  # 50/50 split
        )
        
        engine.moet_usdf_pool = create_moet_usdf_pool(
            pool_size_usd=pool_size,
            concentration=0.95,  # 95% concentration at 1:1 peg
            token0_ratio=0.5  # 50/50 split
        )
        
        # Create slippage calculators for MOET:stablecoin pools
        engine.moet_usdc_calculator = UniswapV3SlippageCalculator(engine.moet_usdc_pool)
        engine.moet_usdf_calculator = UniswapV3SlippageCalculator(engine.moet_usdf_pool)
        
        print(f"✅ Stablecoin pools created for deleveraging chain")
        print(f"   Path: YT → MOET → USDC/USDF → BTC (market) → Collateral")
    
    def _override_aave_liquidation_with_simple_swap(self, engine):
        """
        Override AAVE liquidation to use simple swap logic instead of Uniswap pool
        
        This uses PARTIAL liquidations (50% of debt) like real AAVE:
        - Liquidates 50% of the debt
        - Takes proportional collateral to cover debt + liquidation bonus (5%)
        - Agent remains active with reduced position
        - Liquidation fee (5 bps) and slippage (5 bps)
        """
        from tidal_protocol_sim.core.protocol import Asset
        
        print(f"\n🔧 Overriding AAVE liquidation to use PARTIAL liquidation (50% of debt)")
        print(f"   Liquidation bonus: 5%, Fees: 5 bps, Slippage: 5 bps")
        
        def simple_swap_liquidate(agent_self, current_minute, current_prices):
            """Partial liquidation - liquidate 50% of debt like real AAVE"""
            btc_price = current_prices[Asset.BTC]
            total_btc = agent_self.state.btc_amount
            total_debt = agent_self.state.moet_debt
            
            # AAVE liquidates up to 50% of the debt in a single transaction
            debt_to_repay = total_debt * 0.5
            
            # Calculate BTC needed to cover debt + liquidation bonus (5%)
            liquidation_bonus = 1.05  # 5% bonus for liquidator
            btc_value_needed = debt_to_repay * liquidation_bonus
            btc_to_liquidate = btc_value_needed / btc_price
            
            # Ensure we don't liquidate more BTC than available
            btc_to_liquidate = min(btc_to_liquidate, total_btc)
            
            # Apply liquidation fee (5 bps) and slippage (5 bps)
            liquidation_fee_factor = 0.9995  # 5 bps fee
            slippage_factor = 0.9995  # 5 bps slippage
            
            # Effective price after fees and slippage
            effective_price = btc_price * liquidation_fee_factor * slippage_factor
            
            # Calculate actual MOET received from selling BTC
            moet_received = btc_to_liquidate * effective_price
            actual_debt_repaid = min(debt_to_repay, moet_received)
            
            # Update agent state - PARTIAL LIQUIDATION
            # CRITICAL: Must update BOTH btc_amount AND supplied_balances[BTC]
            agent_self.state.btc_amount -= btc_to_liquidate
            agent_self.state.supplied_balances[Asset.BTC] -= btc_to_liquidate
            agent_self.state.moet_debt -= actual_debt_repaid
            
            # Agent remains active if they still have collateral
            # Note: Low debt is GOOD, not a reason to mark inactive!
            if agent_self.state.btc_amount < 0.001:
                agent_self.active = False
            
            # Record liquidation event
            liquidation_event = {
                "minute": current_minute,
                "agent_id": agent_self.agent_id,
                "btc_liquidated": btc_to_liquidate,
                "debt_repaid": actual_debt_repaid,
                "bad_debt": max(0, debt_to_repay - moet_received),
                "effective_price": effective_price,
                "remaining_btc": agent_self.state.btc_amount,
                "remaining_debt": agent_self.state.moet_debt,
                "still_active": agent_self.active
            }
            
            # Update protocol pools
            btc_pool = engine.protocol.asset_pools[Asset.BTC]
            btc_pool.total_supplied -= btc_to_liquidate
            
            # Burn the repaid MOET
            engine.protocol.moet_system.burn(actual_debt_repaid)
            
            return liquidation_event
        
        # Override the liquidation method for all AAVE agents
        for agent in engine.aave_agents:
            agent.execute_aave_liquidation = lambda minute, prices, a=agent: simple_swap_liquidate(a, minute, prices)
    
    def _create_aave_engine(self) -> AaveProtocolEngine:
        """Create and configure AAVE engine for comparison"""
        
        # Create AAVE configuration (matching High Tide setup)
        aave_config = AaveConfig()
        aave_config.num_aave_agents = 0  # We'll create custom agents
        aave_config.btc_decline_duration = self.config.simulation_duration_minutes
        aave_config.btc_initial_price = self.config.btc_initial_price
        aave_config.btc_final_price_range = (self.config.btc_final_price, self.config.btc_final_price)
        
        # Match pool configurations (for MOET peg/arbitrage)
        aave_config.moet_btc_pool_size = self.config.moet_btc_pool_config["size"]
        aave_config.moet_btc_concentration = self.config.moet_btc_pool_config["concentration"]
        aave_config.moet_yield_pool_size = self.config.moet_yt_pool_config["size"]
        aave_config.yield_token_concentration = self.config.moet_yt_pool_config["concentration"]
        aave_config.yield_token_ratio = self.config.moet_yt_pool_config["token0_ratio"]
        aave_config.use_direct_minting_for_initial = self.config.use_direct_minting_for_initial
        
        # Configure arbitrage (same as High Tide)
        aave_config.num_arbitrage_agents = 5
        aave_config.arbitrage_agent_balance = 100_000.0
        
        # CRITICAL: Propagate agent snapshot frequency for minute-by-minute tracking (Study 14)
        aave_config.agent_snapshot_frequency_minutes = getattr(self.config, 'agent_snapshot_frequency_minutes', 1440)
        
        # Create engine
        engine = AaveProtocolEngine(aave_config)
        
        # Override MOET rates with historical AAVE rates if enabled
        # AAVE always uses historical rates (even in asymmetric studies)
        if self.config.use_historical_rates:
            avg_rate = sum(self.config.aave_rates.values()) / len(self.config.aave_rates) * 100 if self.config.aave_rates else 5.0
            print(f"📊 AAVE using historical rates from {self.config.market_year} (avg: {avg_rate:.2f}% APR)")
            
            # Create rate override function that uses engine.current_step (simulation minute)
            def get_historical_aave_rate():
                current_minute = getattr(engine, 'current_step', 0)
                return self.config.get_historical_rate_at_minute(current_minute)
            
            # Override the borrow rate function
            engine.protocol.get_moet_borrow_rate = get_historical_aave_rate
        
        # Add stablecoin pools for consistency (AAVE agents don't delever but pools should exist)
        self._add_stablecoin_pools_to_engine(engine)
        
        # Create AAVE agents matching High Tide parameters
        agents = self._create_uniform_aave_agents(engine)
        engine.aave_agents = agents
        
        # Override AAVE liquidation to use simple swap (avoid Uniswap pool liquidity issues)
        self._override_aave_liquidation_with_simple_swap(engine)
        
        # Add agents to engine's agent dict
        for agent in agents:
            engine.agents[agent.agent_id] = agent
            agent.engine = engine
        
        self._log_event(0, "ENGINE_SETUP", "AAVE engine created", {
            "num_agents": len(agents),
            "historical_rates": self.config.use_historical_rates
        })
        
        return engine
    
    def _create_uniform_agents(self, engine) -> List[HighTideAgent]:
        """Create agents with uniform tri-health factor profile for year-long simulation"""
        
        agents = []
        
        for i in range(self.config.num_agents):
            agent_id = f"year_sim_agent_{i:03d}"  # 3-digit padding for 120 agents
            
            # All agents use the same tri-health factor profile
            agent = HighTideAgent(
                agent_id,
                self.config.agent_initial_hf,      # 1.1 Initial HF
                self.config.agent_rebalancing_hf,  # 1.025 Rebalancing HF  
                self.config.agent_target_hf,       # 1.04 Target HF
                initial_balance=self.config.btc_initial_price,  # CRITICAL FIX: Use 2024 BTC price
                yield_token_pool=engine.yield_token_pool
            )
            agents.append(agent)
            
            self._log_event(0, "AGENT_CREATED", f"Created {agent_id}", {
                "initial_hf": self.config.agent_initial_hf,
                "rebalancing_hf": self.config.agent_rebalancing_hf,
                "target_hf": self.config.agent_target_hf
            })
        
        print(f"✅ Created {len(agents)} agents with uniform tri-health factor profile:")
        print(f"   Initial HF: {self.config.agent_initial_hf}")
        print(f"   Rebalancing HF: {self.config.agent_rebalancing_hf} (trigger threshold)")
        print(f"   Target HF: {self.config.agent_target_hf} (rebalancing target)")
        
        return agents
    
    def _create_uniform_aave_agents(self, engine) -> List[AaveAgent]:
        """Create AAVE agents with conservative HF (1.95) - typical of real USDC borrowers"""
        
        agents = []
        
        for i in range(self.config.num_agents):
            agent_id = f"aave_year_sim_agent_{i:03d}"
            
            # Create AAVE agent with conservative 1.95 initial HF (median of real borrowers)
            # Note: rebalancing_hf and target_hf are not used since AAVE is buy-and-hold
            agent = AaveAgent(
                agent_id,
                self.config.aave_initial_hf,       # 1.95 HF (real-world median)
                1.5,                                # Placeholder (not used - AAVE doesn't rebalance)
                1.6,                                # Placeholder (not used - AAVE doesn't rebalance)
                initial_balance=self.config.btc_initial_price
            )
            
            # CRITICAL: Set yield token pool reference for AAVE agents to enable YT purchases
            agent.state.yield_token_manager.yield_token_pool = engine.yield_token_pool
            
            agents.append(agent)
            
            self._log_event(0, "AGENT_CREATED", f"Created {agent_id}", {
                "initial_hf": self.config.aave_initial_hf,
                "type": "AAVE"
            })
        
        print(f"✅ Created {len(agents)} AAVE agents:")
        print(f"   Initial HF: {self.config.aave_initial_hf}")
        print(f"   Strategy: Weekly manual rebalancing (deleverage if HF drops, harvest if HF rises)")
        
        return agents
    
    def _run_simulation_with_detailed_tracking(self, engine):
        """Run simulation with comprehensive tracking of all rebalancing activities"""
        
        print("🚀 Starting full year simulation with detailed tracking...")
        
        if self.config.simulation_duration_minutes >= 100_000:  # Only show for long simulations
            print("⚡ PERFORMANCE OPTIMIZATIONS ENABLED:")
            print("   • Agent health snapshots: Daily (every 1440 minutes)")
            print("   • Protocol state snapshots: Daily (every 1440 minutes)") 
            print("   • BTC price logging: Daily (events still logged immediately)")
            print("   • Expected memory usage: ~50 MB instead of ~16 GB")
            print("   • Expected runtime: ~35 minutes instead of 14+ hours")
            print()
        
        # Run custom simulation loop with pool rebalancing integration
        return self._run_custom_simulation_with_pool_rebalancing(engine)
    
    def _run_simulation_with_detailed_tracking_aave(self, engine):
        """Run AAVE simulation with detailed tracking (no rebalancing)"""
        
        print("🚀 Starting AAVE simulation (buy-and-hold strategy)...")
        
        if self.config.simulation_duration_minutes >= 100_000:
            print("⚡ PERFORMANCE OPTIMIZATIONS ENABLED")
            print("   • Daily snapshots (every 1440 minutes)")
            print()
        
        # AAVE agents don't rebalance, so we just run without pool rebalancing
        return self._run_custom_simulation_aave(engine)
    
    def _process_ecosystem_growth(self, engine, current_minute: int, current_btc_price: float) -> List[Dict]:
        """Process ecosystem growth by adding new agents over time"""
        import random  # Import random at method level
        
        # Don't start growth until after delay period
        days_elapsed = current_minute / 1440
        if days_elapsed < self.config.growth_start_delay_days:
            return []
        
        # FIXED: Balance between growth target and pool liquidity limits
        # Each agent deposits 2-5 BTC (average 3.5), but we need to respect pool liquidity
        max_agents = 400  # Reduced from 500 to 400 to prevent liquidity exhaustion
        avg_btc_per_agent = 3.5  # Average of 2-5 BTC range
        target_total_agents = min(max_agents, int(self.config.target_btc_deposits / (current_btc_price * avg_btc_per_agent)))
        current_total_agents = len(engine.high_tide_agents)
        
        # If we've already reached the target or cap, no more growth needed
        if current_total_agents >= target_total_agents or current_total_agents >= max_agents:
            if current_total_agents >= max_agents:
                # Only log this once when we hit the cap
                if current_total_agents == max_agents and current_minute % 1440 == 0:  # Daily log
                    print(f"🚫 Ecosystem growth capped at {max_agents} agents (increased from 300 to reach $150M target)")
            return []
        
        # Calculate growth rate based on exponential curve
        # We want to reach the target by the end of the year (365 days)
        days_remaining = 365 - days_elapsed
        growth_days = 365 - self.config.growth_start_delay_days  # Total growth period
        
        if days_remaining <= 0:
            return []
        
        # Exponential growth: more agents added as time progresses
        progress = (days_elapsed - self.config.growth_start_delay_days) / growth_days
        agents_needed = target_total_agents - len(engine.high_tide_agents)
        
        # Calculate agents to add this minute using exponential growth
        # Add more agents as we progress through the year
        base_rate = agents_needed / (days_remaining * 1440)  # Base agents per minute
        growth_multiplier = 1 + (progress * self.config.growth_acceleration_factor)
        agents_to_add_float = base_rate * growth_multiplier
        
        # Use probabilistic addition to handle fractional agents
        agents_to_add = int(agents_to_add_float)
        if random.random() < (agents_to_add_float - agents_to_add):
            agents_to_add += 1
        
        # Limit to reasonable batch sizes (max 10 agents per minute)
        agents_to_add = min(agents_to_add, 10)
        
        new_agents = []
        for i in range(agents_to_add):
            # FIXED: Create new agent with 2-5 BTC deposit (whale/institutional behavior)
            agent_id = f"growth_agent_{current_total_agents + i + 1:04d}"
            
            # Create agent using the same logic as the engine
            from tidal_protocol_sim.agents.high_tide_agent import HighTideAgent
            from tidal_protocol_sim.agents.base_agent import AgentAction
            from tidal_protocol_sim.core.protocol import Asset
            
            # Random BTC deposit between 2-5 BTC (more realistic for growth users)
            btc_deposit = random.uniform(2.0, 5.0)
            
            # Check if pool has sufficient liquidity for this deposit size
            moet_amount = btc_deposit * current_btc_price
            if moet_amount > 50000:  # Cap individual deposits to prevent liquidity issues
                btc_deposit = 50000 / current_btc_price
                print(f"⚠️  Capping agent deposit to {btc_deposit:.2f} BTC to prevent pool liquidity exhaustion")
            
            usd_value = current_btc_price * btc_deposit
            
            new_agent = HighTideAgent(
                agent_id=agent_id,
                initial_hf=1.05,  # Same as other agents
                rebalancing_hf=1.015,
                target_hf=1.03,
                initial_balance=current_btc_price,  # Pass BTC price, not total USD value
                yield_token_pool=engine.yield_token_pool
            )
            
            # Set engine reference
            new_agent.engine = engine
            
            # CRITICAL FIX: Override the hardcoded 1 BTC with actual deposit amount
            new_agent.state.btc_amount = btc_deposit
            new_agent.state.supplied_balances[Asset.BTC] = btc_deposit
            
            # Recalculate MOET debt based on actual BTC deposit
            btc_collateral_factor = 0.80
            effective_collateral_value = btc_deposit * current_btc_price * btc_collateral_factor
            moet_to_borrow = effective_collateral_value / 1.05  # initial_hf = 1.05
            
            new_agent.state.moet_debt = moet_to_borrow
            new_agent.state.initial_moet_debt = moet_to_borrow
            new_agent.state.borrowed_balances[Asset.MOET] = moet_to_borrow
            new_agent.state.token_balances[Asset.MOET] = moet_to_borrow
            
            # Add to engine (both collections for proper tracking)
            engine.high_tide_agents.append(new_agent)
            engine.agents[agent_id] = new_agent  # Also add to main agents dict for action recording
            
            # Initialize agent position following the same pattern as _setup_high_tide_positions
            # Update protocol with agent's BTC collateral (use actual deposit amount)
            btc_pool = engine.protocol.asset_pools[Asset.BTC]
            btc_pool.total_supplied += btc_deposit  # Use actual deposit, not hardcoded 1.0
            
            # Update protocol with agent's MOET debt (use recalculated amount)
            engine.protocol.moet_system.mint(moet_to_borrow)
            
            # Initialize agent's health factor
            engine._update_agent_health_factor(new_agent)
            
            # CRITICAL FIX: Execute initial yield token purchase (same as original agents at minute 0)
            if new_agent.state.moet_debt > 0 and len(new_agent.state.yield_token_manager.yield_tokens) == 0:
                # Trigger initial yield token purchase
                action, params = new_agent._initial_yield_token_purchase(current_minute)
                if action == AgentAction.SWAP and params.get("action_type") == "buy_yield_tokens":
                    # Execute the yield token purchase through the engine
                    success = engine._execute_yield_token_purchase(new_agent, params, current_minute)
                    if success:
                        print(f"   ✅ {agent_id}: Initial YT purchase of ${params['moet_amount']:,.0f} MOET successful")
                    else:
                        print(f"   ❌ {agent_id}: Initial YT purchase failed")
            
            # Record the growth event
            growth_event = {
                "minute": current_minute,
                "hour": current_minute / 60,
                "day": days_elapsed,
                "agent_id": agent_id,
                "btc_deposited": btc_deposit,
                "usd_value": usd_value,
                "total_agents": len(engine.high_tide_agents),
                "total_btc_deposits": sum(agent.state.btc_amount for agent in engine.high_tide_agents),
                "total_usd_value": sum(agent.state.btc_amount * current_btc_price for agent in engine.high_tide_agents),
                "target_progress": len(engine.high_tide_agents) / target_total_agents
            }
            new_agents.append(growth_event)
        
        # Log significant growth milestones
        # Very selective logging to prevent console spam with hundreds of agents
        agent_count = len(engine.high_tide_agents)
        
        # Only log at major milestones or weekly intervals
        should_log = False
        if agent_count <= 200:
            should_log = current_minute % 1440 == 0  # Daily for small counts
        elif agent_count <= 500:
            should_log = current_minute % (1440 * 3) == 0  # Every 3 days for medium counts
        else:
            should_log = current_minute % (1440 * 7) == 0  # Weekly for large counts
        
        # Also log at major milestones (every 100 agents)
        if new_agents and agent_count % 100 == 0 and agent_count > (agent_count - len(new_agents)):
            should_log = True
        
        if new_agents and should_log:
            total_deposits = agent_count * current_btc_price
            progress_pct = (total_deposits / self.config.target_btc_deposits) * 100
            print(f"🌱 Day {days_elapsed:.0f}: Added {len(new_agents)} agents. "
                  f"Total: {agent_count} agents, "
                  f"${total_deposits:,.0f} deposits ({progress_pct:.1f}% of target)")
        
        return new_agents
    
    def _custom_btc_price_update(self, minute: int) -> float:
        """Custom BTC price update function that follows our test scenario"""
        new_price = self.config.get_btc_price_at_minute(minute)
        
        # Log price updates
        if minute % self.config.log_every_n_minutes == 0:
            self._log_event(minute, "BTC_PRICE_UPDATE", f"BTC price updated", {
                "minute": minute,
                "hour": minute / 60,
                "btc_price": new_price,
                "change_pct": ((new_price / self.config.btc_initial_price) - 1) * 100
            })
        
        # Progress update - Weekly reports for year-long simulation
        if minute % self.config.progress_report_every_n_minutes == 0:  # Every week
            days = minute / (60 * 24)
            weeks = days / 7
            print(f"⏱️  Week {weeks:.0f}/52 (Day {days:.0f}/365) - BTC: ${new_price:,.0f} ({((new_price/self.config.btc_initial_price)-1)*100:+.1f}%)")
        
        return new_price
    
    def _run_custom_simulation_with_pool_rebalancing(self, engine):
        """Run custom simulation that integrates pool rebalancing with the engine simulation"""
        
        print(f"Starting High Tide simulation with {len(engine.high_tide_agents)} agents")
        print(f"BTC decline from ${self.config.btc_initial_price:,.0f} to ${self.config.btc_final_price:,.0f}")
        print(f"Pool arbitrage: {'ENABLED' if self.config.enable_pool_arbing else 'DISABLED'}")
        
        # Initialize tracking
        engine.btc_price_history = []
        engine.rebalancing_events = []
        engine.yield_token_trades = []
        engine.current_step = 0
        
        # Pool rebalancing tracking
        pool_rebalancing_events = []
        pool_state_snapshots = []
        
        # Ecosystem Growth Tracking
        ecosystem_growth_events = []
        if self.config.enable_ecosystem_growth:
            print(f"🌱 Ecosystem Growth ENABLED: Target ${self.config.target_btc_deposits:,.0f} BTC deposits")
            print(f"   Starting agents: {len(engine.high_tide_agents)}")
            print(f"   Growth starts after day {self.config.growth_start_delay_days}")
            
        for minute in range(self.config.simulation_duration_minutes):
            engine.current_step = minute
            
            # FIXED: Use our custom gradual decline instead of engine's rapid decline manager
            new_btc_price = self.config.get_btc_price_at_minute(minute)
            engine.state.current_prices[Asset.BTC] = new_btc_price
            
            # ECOSYSTEM GROWTH: Add new agents over time to reach target deposits
            if self.config.enable_ecosystem_growth:
                new_agents = self._process_ecosystem_growth(engine, minute, new_btc_price)
                if new_agents:
                    ecosystem_growth_events.extend(new_agents)
            
            # PERFORMANCE OPTIMIZATION: Store BTC price daily instead of every minute
            # This reduces memory usage from 4MB to 3KB for price history
            if minute % 1440 == 0:  # Daily BTC price storage
                engine.btc_price_history.append(new_btc_price)
            
            # Update protocol state
            engine.protocol.current_block = minute
            engine.protocol.accrue_interest()
            
            # Process MOET system updates (bond auctions, interest rate calculations)
            moet_update_results = engine.protocol.process_moet_system_update(minute)
            if moet_update_results.get('advanced_system_enabled') and minute % 60 == 0:  # Log hourly
                if moet_update_results.get('bond_auction_triggered'):
                    print(f"🔔 Bond auction triggered at minute {minute}")
                if moet_update_results.get('bond_auction_completed'):
                    auction = moet_update_results['completed_auction']
                    print(f"✅ Bond auction completed: ${auction['amount_filled']:,.0f} at {auction['final_apr']:.2%} APR")
                if moet_update_results.get('interest_rate_updated'):
                    print(f"📈 MOET rate updated: {moet_update_results['new_interest_rate']:.2%}")
            
            # Update agent debt interest
            engine._update_agent_debt_interest(minute)
            
            # Process pool rebalancing BEFORE agent actions
            if self.config.enable_pool_arbing and hasattr(engine, 'pool_rebalancer'):
                # Calculate current yield token prices and deviations for pool rebalancer
                from tidal_protocol_sim.core.yield_tokens import calculate_true_yield_token_price
                true_yt_price = calculate_true_yield_token_price(minute, 0.10, 1.0)
                pool_yt_price = engine.yield_token_pool.uniswap_pool.get_price()
                deviation_bps = abs((pool_yt_price - true_yt_price) / true_yt_price) * 10000
                
                protocol_state = {
                    "current_minute": minute,
                    "true_yield_token_price": true_yt_price,
                    "pool_yield_token_price": pool_yt_price,
                    "deviation_bps": deviation_bps
                }
                asset_prices = {Asset.BTC: new_btc_price}
                
                # Store pool state before rebalancing
                pool_state_before = {
                    "pool_yt_price": pool_yt_price,
                    "true_yt_price": true_yt_price,
                    "deviation_bps": deviation_bps,
                    "alm_moet_balance": engine.pool_rebalancer.alm_rebalancer.state.moet_balance,
                    "alm_yt_balance": engine.pool_rebalancer.alm_rebalancer.state.yield_token_balance
                }
                
                rebalancing_events = engine.pool_rebalancer.process_rebalancing(protocol_state, asset_prices)
                
                # Debug logging for ALM rebalancer timing
                if minute == 720 or minute == 1440 or minute == 2160:
                    print(f"🔍 DEBUG: Minute {minute} - checking ALM rebalancer")
                    print(f"    ALM next_rebalance_minute: {engine.pool_rebalancer.alm_rebalancer.next_rebalance_minute}")
                    print(f"    ALM enabled: {engine.pool_rebalancer.alm_rebalancer.state.enabled}")
                    print(f"    Pool rebalancer enabled: {engine.pool_rebalancer.enabled}")
                    print(f"    Deviation: {deviation_bps:.1f} bps")
                    
                    # Test ALM rebalancer decision
                    alm_action, alm_params = engine.pool_rebalancer.alm_rebalancer.decide_action(protocol_state, asset_prices)
                    print(f"    ALM decide_action returned: {alm_action}, amount: {alm_params.get('amount', 0)}")
                    print(f"    Min rebalance amount: {engine.pool_rebalancer.alm_rebalancer.state.min_rebalance_amount}")
                    print(f"    True YT price: {true_yt_price:.6f}, Pool YT price: {pool_yt_price:.6f}")
                    print(f"    ALM rebalancer balances: MOET=${engine.pool_rebalancer.alm_rebalancer.state.moet_balance:,.0f}, YT=${engine.pool_rebalancer.alm_rebalancer.state.yield_token_balance:,.0f}")
                    print(f"    ALM params: {alm_params}")
                
                if rebalancing_events:
                    pool_rebalancing_events.extend(rebalancing_events)
                    for event in rebalancing_events:
                        rebalancer_type = event.get("rebalancer", "unknown")
                        
                        # Get pool state after rebalancing
                        pool_yt_price_after = engine.yield_token_pool.uniswap_pool.get_price()
                        deviation_after = abs((pool_yt_price_after - true_yt_price) / true_yt_price) * 10000
                        
                        pool_state_after = {
                            "pool_yt_price": pool_yt_price_after,
                            "deviation_bps": deviation_after,
                            "alm_moet_balance": engine.pool_rebalancer.alm_rebalancer.state.moet_balance,
                            "alm_yt_balance": engine.pool_rebalancer.alm_rebalancer.state.yield_token_balance
                        }
                        
                        print(f"🔄 {rebalancer_type} Rebalancer triggered at minute {minute}")
                        print(f"   📊 Before: Pool=${pool_state_before['pool_yt_price']:.6f}, True=${true_yt_price:.6f}, Dev={pool_state_before['deviation_bps']:.1f} bps")
                        print(f"   📊 After:  Pool=${pool_state_after['pool_yt_price']:.6f}, True=${true_yt_price:.6f}, Dev={pool_state_after['deviation_bps']:.1f} bps")
                        print(f"   💰 ALM Balance Change: MOET ${pool_state_before['alm_moet_balance']:,.0f} → ${pool_state_after['alm_moet_balance']:,.0f}")
                        print(f"   💰 ALM YT Change: YT ${pool_state_before['alm_yt_balance']:.0f} → ${pool_state_after['alm_yt_balance']:.0f}")
                        
                        # Enhanced event logging with before/after states
                        enhanced_event_data = {
                            **event,
                            "pool_state_before": pool_state_before,
                            "pool_state_after": pool_state_after,
                            "true_yt_price": true_yt_price
                        }
                        
                        self._log_event(minute, f"{rebalancer_type.upper()}_REBALANCE", 
                                      f"{rebalancer_type} rebalancer executed", enhanced_event_data)
            
            # Process High Tide agent actions
            swap_data = engine._process_high_tide_agents(minute)
            
            # Process MOET arbitrage agents (if advanced MOET system enabled)
            if hasattr(engine, 'arbitrage_agents') and engine.arbitrage_agents:
                arbitrage_swap_data = engine._process_arbitrage_agents(minute)
                # Merge arbitrage swap data with main swap data
                swap_data.update(arbitrage_swap_data)
            
            # Check for High Tide liquidations
            engine._check_high_tide_liquidations(minute)
            
            # Record position tracking data
            tracked_agent = engine._get_tracked_agent()
            if tracked_agent:
                agent_swap_data = swap_data.get(tracked_agent.agent_id, {})
                engine.position_tracker.record_minute_data(
                    minute, new_btc_price, tracked_agent, engine, agent_swap_data
                )
            
            # Record metrics
            engine._record_high_tide_metrics(minute)
            
            # Progress logging
            if minute % self.config.log_every_n_minutes == 0:
                self._log_event(minute, "BTC_PRICE_UPDATE", f"BTC price updated", {
                    "minute": minute,
                    "hour": minute / 60,
                    "btc_price": new_btc_price,
                    "change_pct": ((new_btc_price / self.config.btc_initial_price) - 1) * 100
                })
            
            # Log pool state snapshots daily for analysis (reduced frequency for year-long sim)
            if minute % self.config.collect_pool_state_every_n_minutes == 0:  # Daily
                from tidal_protocol_sim.core.yield_tokens import calculate_true_yield_token_price
                true_yt_price = calculate_true_yield_token_price(minute, 0.10, 1.0)
                pool_yt_price = engine.yield_token_pool.uniswap_pool.get_price()
                deviation_bps = (pool_yt_price - true_yt_price) / true_yt_price * 10000
                
                pool_state_snapshots.append({
                    "minute": minute,
                    "hour": minute / 60,
                    "btc_price": new_btc_price,
                    "true_yt_price": true_yt_price,
                    "pool_yt_price": pool_yt_price,
                    "deviation_bps": deviation_bps,
                    "active_agents": len([a for a in engine.high_tide_agents if a.active and a.is_healthy()]),
                    "moet_usdc_price": engine.moet_usdc_pool.get_price() if hasattr(engine, 'moet_usdc_pool') else 1.0,
                    "moet_usdf_price": engine.moet_usdf_pool.get_price() if hasattr(engine, 'moet_usdf_pool') else 1.0
                })
            
            if minute % self.config.progress_report_every_n_minutes == 0:  # Every week
                days = minute / (60 * 24)
                weeks = days / 7
                print(f"⏱️  Week {weeks:.0f}/52 (Day {days:.0f}/365) - BTC: ${new_btc_price:,.0f} ({((new_btc_price/self.config.btc_initial_price)-1)*100:+.1f}%)")
            
            # Reduce minute-by-minute logging for year-long simulation
            if minute % 1440 == 0:  # Daily summary instead of every 10 minutes
                day = minute // 1440
                print(f"Day {day}: BTC = ${new_btc_price:,.0f}, Active agents: {engine._count_active_agents()}")
            
            # MEMORY MANAGEMENT: Prevent unbounded growth of data structures
            # Clean up old data every 7 days to prevent crashes
            if minute % (1440 * 7) == 0 and minute > 0:  # Weekly cleanup
                # Keep only recent data to prevent crashes
                max_entries = 5000
                if len(engine.metrics_history) > max_entries:
                    engine.metrics_history = engine.metrics_history[-max_entries//2:]  # Keep recent half
                if len(engine.agent_actions_history) > max_entries:
                    engine.agent_actions_history = engine.agent_actions_history[-max_entries//2:]
                if len(engine.yield_token_trades) > max_entries:
                    engine.yield_token_trades = engine.yield_token_trades[-max_entries//2:]
                
                # Clean up arbitrage tracking data - more aggressive for large agent counts
                if hasattr(engine, 'arbitrage_agents') and engine.arbitrage_agents:
                    cleanup_limit = 100 if len(engine.high_tide_agents) > 300 else max_entries//2
                    for agent in engine.arbitrage_agents:
                        if len(agent.state.arbitrage_attempts) > cleanup_limit:
                            agent.state.arbitrage_attempts = agent.state.arbitrage_attempts[-cleanup_limit:]
                        if len(agent.state.arbitrage_events) > cleanup_limit:
                            agent.state.arbitrage_events = agent.state.arbitrage_events[-cleanup_limit:]
                
                # Clean up pool state snapshots
                if hasattr(engine, 'pool_state_snapshots') and len(engine.pool_state_snapshots) > max_entries:
                    engine.pool_state_snapshots = engine.pool_state_snapshots[-max_entries//2:]
                
                # ECOSYSTEM GROWTH: Clean up growth events to prevent bloat
                if len(ecosystem_growth_events) > 1000:  # Keep last 1000 growth events
                    ecosystem_growth_events = ecosystem_growth_events[-1000:]
                    print(f"   📊 Cleaned ecosystem growth events: kept last 1000")
                
                # Clean up agent tracking data for large agent counts
                if len(engine.high_tide_agents) > 150:
                    print(f"   🧹 Large agent count ({len(engine.high_tide_agents)}): Cleaning agent tracking data...")
                    for agent in engine.high_tide_agents:
                        # Clean up rebalancing events (keep last 2 for very large counts)
                        if hasattr(agent.state, 'rebalancing_events') and len(agent.state.rebalancing_events) > 2:
                            agent.state.rebalancing_events = agent.state.rebalancing_events[-2:]
                        # Clean up deleveraging events (keep last 2 for very large counts)
                        if hasattr(agent.state, 'deleveraging_events') and len(agent.state.deleveraging_events) > 2:
                            agent.state.deleveraging_events = agent.state.deleveraging_events[-2:]
                        # Clean up action history for ecosystem growth agents
                        if hasattr(agent.state, 'action_history') and len(agent.state.action_history) > 10:
                            agent.state.action_history = agent.state.action_history[-10:]
                
                print(f"🧹 Memory cleanup at day {minute//1440}: Trimmed historical data")
        
        # AGGRESSIVE PRE-COMPILATION CLEANUP: Remove memory-wasting event arrays
        print("🧹 AGGRESSIVE CLEANUP: Removing unused event arrays before results compilation...")
        self._aggressive_pre_compilation_cleanup(engine, ecosystem_growth_events, pool_state_snapshots)
        
        # Generate results using the engine's method
        results = engine._generate_high_tide_results()
        
        # Add pool rebalancing data to results
        results["pool_rebalancing_activity"] = {
            "total_rebalances": len(pool_rebalancing_events),
            "alm_rebalances": len([e for e in pool_rebalancing_events if e.get("rebalancer") == "ALM"]),
            "algo_rebalances": len([e for e in pool_rebalancing_events if e.get("rebalancer") == "Algo"]),
            "events": pool_rebalancing_events,
            "alm_profit": sum(e.get("params", {}).get("profit", 0) for e in pool_rebalancing_events if e.get("rebalancer") == "ALM"),
            "algo_profit": sum(e.get("params", {}).get("profit", 0) for e in pool_rebalancing_events if e.get("rebalancer") == "Algo"),
            "total_profit": sum(e.get("params", {}).get("profit", 0) for e in pool_rebalancing_events)
        }
        
        # Add pool state snapshots to results
        results["pool_state_snapshots"] = pool_state_snapshots
        
        # Add ecosystem growth events to results
        if self.config.enable_ecosystem_growth:
            results["ecosystem_growth_events"] = ecosystem_growth_events
            results["ecosystem_growth_summary"] = {
                "total_new_agents": len(ecosystem_growth_events),
                "final_agent_count": len(engine.high_tide_agents),
                "final_btc_deposits": sum(agent.state.btc_amount for agent in engine.high_tide_agents),
                "final_usd_value": len(engine.high_tide_agents) * engine.state.current_prices[Asset.BTC],
                "target_achievement": (len(engine.high_tide_agents) * engine.state.current_prices[Asset.BTC]) / self.config.target_btc_deposits,
                "growth_start_day": self.config.growth_start_delay_days,
                "target_deposits": self.config.target_btc_deposits
            }
        
        # Add MOET system state to results (if advanced system enabled)
        # MOET system data collection (if advanced system enabled)
        # NOTE: Skip this early collection - let the High Tide engine handle it properly
        # in get_simulation_results() to avoid cleanup conflicts
        print("🔧 DEBUG: Skipping early MOET system data collection - will be handled by engine")
        # MOET system summary will be handled by the High Tide engine
        
        return results
    
    def _run_custom_simulation_aave(self, engine):
        """Run simplified AAVE simulation (no agent rebalancing)"""
        
        print(f"Starting AAVE simulation with {len(engine.aave_agents)} agents")
        print(f"BTC progression: ${self.config.btc_initial_price:,.0f} → ${self.config.btc_final_price:,.0f}")
        print(f"Strategy: Weekly manual rebalancing")
        print(f"  • If HF < initial: Sell YT → MOET → Pay down debt (deleverage)")
        print(f"  • If HF > initial: Sell rebased YT → BTC (harvest profits)")
        
        # Initialize tracking
        engine.btc_price_history = []
        engine.current_step = 0
        
        # Agent health factor snapshots for tracking
        agent_snapshots = {}
        
        # Track liquidation events per agent
        liquidation_events = []
        agent_liquidation_counts = {agent.agent_id: 0 for agent in engine.aave_agents}
        
        # Track weekly rebalancing events
        weekly_rebalance_events = []
        
        for minute in range(self.config.simulation_duration_minutes):
            engine.current_step = minute
            
            # Update BTC price using 2024 data
            new_btc_price = self.config.get_btc_price_at_minute(minute)
            engine.state.current_prices[Asset.BTC] = new_btc_price
            
            # Store BTC price daily (memory optimization)
            if minute % 1440 == 0:
                engine.btc_price_history.append(new_btc_price)
            
            # Update protocol state
            engine.protocol.current_block = minute
            engine.protocol.accrue_interest()
            
            # Update agent debt interest (using historical AAVE rates if enabled)
            if hasattr(engine, '_update_agent_debt_interest'):
                engine._update_agent_debt_interest(minute)
            
            # AAVE agents buy YT at minute 0 only (with direct 1:1 minting), then hold
            if minute == 0 and hasattr(engine, '_process_aave_agents'):
                engine._process_aave_agents(minute)  # Initial YT purchase at minute 0
            
            # Check for liquidations and capture events
            if hasattr(engine, '_check_aave_liquidations'):
                # Store liquidation events before checking
                pre_liquidation_count = len(engine.liquidation_events) if hasattr(engine, 'liquidation_events') else 0
                
                engine._check_aave_liquidations(minute)
                
                # Check if new liquidations occurred
                if hasattr(engine, 'liquidation_events'):
                    post_liquidation_count = len(engine.liquidation_events)
                    if post_liquidation_count > pre_liquidation_count:
                        # New liquidation(s) occurred
                        for event in engine.liquidation_events[pre_liquidation_count:]:
                            agent_id = event.get('agent_id')
                            if agent_id in agent_liquidation_counts:
                                agent_liquidation_counts[agent_id] += 1
                            liquidation_events.append(event)
                        
                        # FAIL-FAST MODE: Exit immediately on first liquidation
                        if self.config.fail_fast_on_liquidation:
                            if not self.config.suppress_progress_output:
                                print(f"\n⚠️  LIQUIDATION DETECTED at minute {minute} - Exiting immediately (fail-fast mode)")
                            
                            # Build agent outcomes with snapshots
                            fail_fast_outcomes = []
                            for agent in engine.aave_agents:
                                outcome = {
                                    "agent_id": agent.agent_id,
                                    "survived": False,
                                    "was_liquidated": True,
                                    "liquidation_minute": minute
                                }
                                # Add snapshots collected up to liquidation point
                                if agent.agent_id in agent_snapshots:
                                    outcome["snapshots"] = agent_snapshots[agent.agent_id]
                                fail_fast_outcomes.append(outcome)
                            
                            # Return minimal failure result
                            return {
                                "simulation_metadata": {
                                    "duration_minutes": minute,
                                    "duration_days": minute / 1440,
                                    "btc_initial": self.config.btc_initial_price,
                                    "btc_final": engine.state.current_prices[Asset.BTC],
                                    "terminated_early": True,
                                    "termination_reason": "liquidation_detected"
                                },
                                "agent_outcomes": fail_fast_outcomes,
                                "liquidation_events": liquidation_events,
                                "summary": {
                                    "total_agents": len(engine.aave_agents),
                                    "survived": 0,
                                    "liquidated": len(engine.aave_agents),
                                    "total_liquidation_events": len(liquidation_events)
                                }
                            }
            
            # Record AAVE metrics (includes agent_health_history tracking)
            if hasattr(engine, '_record_aave_metrics'):
                engine._record_aave_metrics(minute)
            
            # Collect agent health factor snapshots (configurable frequency for different study types)
            if minute % self.config.agent_snapshot_frequency_minutes == 0:
                for agent in engine.aave_agents:
                    if agent.active:
                        if agent.agent_id not in agent_snapshots:
                            agent_snapshots[agent.agent_id] = {
                                "timestamps": [],
                                "health_factors": [],
                                "collateral": [],
                                "debt": []
                            }
                        agent_snapshots[agent.agent_id]["timestamps"].append(minute)
                        agent_snapshots[agent.agent_id]["health_factors"].append(agent.state.health_factor)
                        agent_snapshots[agent.agent_id]["collateral"].append(agent.state.btc_amount * new_btc_price)
                        agent_snapshots[agent.agent_id]["debt"].append(agent.state.moet_debt)
            
            # Rebalancing and progress reporting (frequency controlled by config)
            rebalance_frequency = self.config.leverage_frequency_minutes
            if minute > 0 and minute % rebalance_frequency == 0:
                days_elapsed = minute / 1440
                weeks_elapsed = days_elapsed / 7
                active_agents = sum(1 for a in engine.aave_agents if a.active)
                avg_hf = sum(a.state.health_factor for a in engine.aave_agents if a.active) / max(active_agents, 1)
                
                # Execute rebalancing for each active agent (deleverage if HF < initial, harvest if HF > initial)
                rebalance_count = 0
                deleverage_count = 0
                harvest_count = 0
                
                for agent in engine.aave_agents:
                    if agent.active:
                        # Execute rebalancing (method name says 'weekly' but frequency is configurable)
                        rebalance_event = agent.execute_weekly_rebalancing(
                            minute, 
                            engine.state.current_prices,
                            engine
                        )
                        
                        if rebalance_event["action_taken"] != "none":
                            rebalance_count += 1
                            if rebalance_event["action_taken"] == "deleverage":
                                deleverage_count += 1
                            elif rebalance_event["action_taken"] == "harvest_profits":
                                harvest_count += 1
                            
                            # Store the event
                            rebalance_event["agent_id"] = agent.agent_id
                            weekly_rebalance_events.append(rebalance_event)
                
                # Progress reporting (suppress if configured for optimization runs)
                if not self.config.suppress_progress_output:
                    print(f"📅 Week {weeks_elapsed:.0f}/52: BTC ${new_btc_price:,.0f} | "
                          f"Active: {active_agents}/{len(engine.aave_agents)} | "
                          f"Avg HF: {avg_hf:.3f} | "
                          f"Rebalances: {rebalance_count} ({deleverage_count} delev, {harvest_count} harvest)")
        
        # Generate results (suppress if configured for optimization runs)
        if not self.config.suppress_progress_output:
            print("📊 Generating AAVE simulation results...")
            print(f"   Total liquidation events: {len(liquidation_events)}")
            print(f"   Agents liquidated: {sum(1 for count in agent_liquidation_counts.values() if count > 0)}/{len(agent_liquidation_counts)}")
        
        # Build agent_outcomes with liquidation tracking
        agent_outcomes = []
        for agent in engine.aave_agents:
            agent._update_health_factor(engine.state.current_prices)
            portfolio = agent.get_detailed_portfolio_summary(engine.state.current_prices, self.config.simulation_duration_minutes - 1)
            
            liquidation_count = agent_liquidation_counts.get(agent.agent_id, 0)
            was_liquidated = liquidation_count > 0
            
            outcome = {
                "agent_id": agent.agent_id,
                "initial_health_factor": getattr(agent.state, 'initial_health_factor', self.config.aave_initial_hf),
                "final_health_factor": agent.state.health_factor,
                "survived": not was_liquidated,  # Survived = never liquidated
                "was_liquidated": was_liquidated,
                "liquidation_count": liquidation_count,
                "btc_amount": agent.state.btc_amount,
                "current_moet_debt": agent.state.moet_debt,
                "yield_token_value": portfolio["yield_token_portfolio"]["total_current_value"],
                "net_position_value": portfolio["net_position_value"],
                "total_yield_earned": portfolio["yield_token_portfolio"]["total_accrued_yield"],
            }
            
            # Add agent snapshots to outcome
            if agent.agent_id in agent_snapshots:
                outcome["snapshots"] = agent_snapshots[agent.agent_id]
            
            agent_outcomes.append(outcome)
        
        # Build comprehensive results
        results = {
            "simulation_metadata": {
                "duration_minutes": self.config.simulation_duration_minutes,
                "duration_days": self.config.simulation_duration_minutes / 1440,
                "btc_initial": self.config.btc_initial_price,
                "btc_final": engine.state.current_prices[Asset.BTC]
            },
            "agent_outcomes": agent_outcomes,
            "liquidation_events": liquidation_events,
            "weekly_rebalance_events": weekly_rebalance_events,
            "agent_health_history": engine.agent_health_history if hasattr(engine, 'agent_health_history') else [],
            "agent_health_snapshots": agent_snapshots,
            "btc_price_history": engine.btc_price_history,
            "summary": {
                "total_agents": len(engine.aave_agents),
                "survived": sum(1 for a in agent_outcomes if a["survived"]),
                "liquidated": sum(1 for a in agent_outcomes if a["was_liquidated"]),
                "total_liquidation_events": len(liquidation_events),
                "total_weekly_rebalances": len(weekly_rebalance_events),
                "total_deleverages": sum(1 for e in weekly_rebalance_events if e["action_taken"] == "deleverage"),
                "total_harvests": sum(1 for e in weekly_rebalance_events if e["action_taken"] == "harvest_profits")
            }
        }
        
        return results
    
    def _compile_aave_agent_performance(self, agents: List) -> Dict:
        """Compile AAVE agent performance metrics"""
        
        active_agents = [a for a in agents if a.active]
        liquidated_agents = [a for a in agents if not a.active]
        
        return {
            "summary": {
                "total_agents": len(agents),
                "active_agents": len(active_agents),
                "liquidated_agents": len(liquidated_agents),
                "survival_rate": len(active_agents) / len(agents) if agents else 0
            },
            "individual_performance": {
                agent.agent_id: {
                    "active": agent.active,
                    "health_factor": agent.state.health_factor,
                    "btc_amount": agent.state.btc_amount,
                    "moet_debt": agent.state.moet_debt,
                    "net_position_value": agent.calculate_net_position_value() if hasattr(agent, 'calculate_net_position_value') else 0
                }
                for agent in agents
            }
        }
    
    def _log_event(self, minute: int, event_type: str, message: str, data: Dict = None):
        """Log detailed event with timestamp"""
        
        log_entry = {
            "minute": minute,
            "hour": minute / 60,
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "data": data or {}
        }
        
        self.results["detailed_logs"].append(log_entry)
        
        if self.config.detailed_logging:
            hour_str = f"[{minute/60:5.1f}h]"
            print(f"{hour_str} {event_type}: {message}")
    
    def _analyze_test_results(self, engine):
        """Analyze test results and generate comprehensive analysis"""
        
        print("\n🔬 Analyzing test results...")
        
        # Agent performance analysis from simulation results
        simulation_results = self.results["simulation_results"]
        self.results["agent_performance"] = self._analyze_agent_performance_from_results(simulation_results)
        
        # Pool arbitrage analysis (if enabled)
        self.results["pool_arbitrage_analysis"] = self._analyze_pool_arbitrage_from_results(simulation_results)
        
        # Pool state evolution analysis
        self.results["pool_evolution_analysis"] = self._analyze_pool_evolution_from_results(simulation_results)
    
    def _analyze_agent_performance_from_results(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze agent performance from simulation results"""
        
        # Extract agent outcomes from simulation results
        agent_outcomes = simulation_results.get("agent_outcomes", [])
        
        if not agent_outcomes:
            return {
                "agent_details": [],
                "summary": {
                    "total_agents": 0,
                    "survived_agents": 0,
                    "survival_rate": 0.0,
                    "total_rebalances": 0,
                    "total_slippage_costs": 0.0,
                    "avg_final_hf": 0.0
                }
            }
        
        # Process agent data
        agent_data = []
        for outcome in agent_outcomes:
            agent_data.append({
                "agent_id": outcome.get("agent_id", ""),
                "initial_hf": outcome.get("initial_health_factor", 0),
                "final_hf": outcome.get("final_health_factor", 0),
                "target_hf": outcome.get("target_health_factor", 0),
                "survived": outcome.get("survived", False),
                "btc_amount": outcome.get("btc_amount", 0),
                "moet_debt": outcome.get("current_moet_debt", 0),
                "yt_value": outcome.get("yield_token_value", 0),
                "net_position": outcome.get("net_position_value", 0),
                "rebalance_count": outcome.get("rebalancing_events", 0),
                "total_slippage": outcome.get("cost_of_rebalancing", 0),
                "total_yield_sold": outcome.get("total_yield_sold", 0),
                # Enhanced tracking (NEW)
                "total_yield_sold_for_rebalancing": outcome.get("total_yield_sold_for_rebalancing", 0),
                "total_rebalancing_slippage": outcome.get("total_rebalancing_slippage", 0),
                "deleveraging_events_count": outcome.get("deleveraging_events_count", 0),
                "total_deleveraging_sales": outcome.get("total_deleveraging_sales", 0),
                "total_deleveraging_slippage": outcome.get("total_deleveraging_slippage", 0)
            })
        
        return {
            "agent_details": agent_data,
            "summary": {
                "total_agents": len(agent_data),
                "survived_agents": sum(1 for a in agent_data if a["survived"]),
                "survival_rate": sum(1 for a in agent_data if a["survived"]) / len(agent_data) if agent_data else 0,
                "total_rebalances": sum(a.get("rebalance_count", a.get("rebalancing_events", 0)) for a in agent_data),
                "total_slippage_costs": sum(a.get("total_slippage", a.get("total_slippage_costs", a.get("total_rebalancing_slippage", 0))) for a in agent_data),
                "avg_final_hf": np.mean([a.get("final_hf", a.get("final_health_factor", 0)) for a in agent_data if a["survived"]]) if any(a["survived"] for a in agent_data) else 0
            }
        }
    
    def _analyze_pool_arbitrage_from_results(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pool arbitrage activities from simulation results"""
        
        if not self.config.enable_pool_arbing:
            return {"enabled": False}
        
        # Extract pool rebalancing data from simulation results if available
        pool_activity = simulation_results.get("pool_rebalancing_activity", {})
        
        return {
            "enabled": True,
            "alm_rebalances": pool_activity.get("alm_rebalances", 0),
            "algo_rebalances": pool_activity.get("algo_rebalances", 0),
            "total_rebalances": pool_activity.get("total_rebalances", 0),
            "alm_profit": pool_activity.get("alm_profit", 0),
            "algo_profit": pool_activity.get("algo_profit", 0),
            "total_profit": pool_activity.get("total_profit", 0),
            "events": pool_activity.get("events", {})
        }
    
    def _analyze_pool_evolution_from_results(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze pool state evolution from simulation results"""
        
        # Extract pool state data from simulation results if available
        pool_evolution = simulation_results.get("pool_state_evolution", {})
        
        if not pool_evolution:
            return {
                "max_price_deviation_bps": 0,
                "avg_price_deviation_bps": 0,
                "deviation_std_bps": 0,
                "times_above_threshold": 0,
                "pool_accuracy_score": 1.0,
                "snapshots": []
            }
        
        return pool_evolution
    
    def _analyze_comparison_results(self, ht_engine, aave_engine):
        """Analyze and compare High Tide vs AAVE results"""
        
        print("📊 Comparing High Tide vs AAVE performance...")
        
        # Get results from both engines
        ht_results = self.results["high_tide_results"]
        aave_results = self.results["aave_results"]
        
        # DEBUG: Check what's actually in the results
        print(f"\n🔍 DEBUG: High Tide results keys: {ht_results.keys()}")
        print(f"🔍 DEBUG: AAVE results keys: {aave_results.keys()}")
        
        if "agent_outcomes" in ht_results:
            ht_outcomes = ht_results.get("agent_outcomes", [])
            if ht_outcomes:
                print(f"🔍 DEBUG: HT agent_outcomes count: {len(ht_outcomes)}")
                print(f"🔍 DEBUG: Sample HT agent: {ht_outcomes[0]}")
        
        if "agent_performance" in aave_results:
            print(f"🔍 DEBUG: AAVE agent_performance keys: {aave_results['agent_performance'].keys()}")
            summary = aave_results['agent_performance'].get('summary', {})
            print(f"🔍 DEBUG: AAVE summary: {summary}")
            indiv = aave_results['agent_performance'].get('individual_performance', {})
            if indiv:
                first_agent_id = list(indiv.keys())[0]
                print(f"🔍 DEBUG: Sample AAVE agent ({first_agent_id}): {indiv[first_agent_id]}")
        
        # Extract key metrics
        # BOTH use agent_outcomes structure!
        ht_perf = ht_results.get("agent_outcomes", [])
        aave_perf = aave_results.get("agent_outcomes", [])
        
        # Calculate return metrics
        initial_investment_per_agent = self.config.btc_initial_price  # Each agent started with 1 BTC
        
        # High Tide returns
        ht_survivors = [a for a in ht_perf if a.get("survived", False)]
        ht_net_values = [a.get("net_position_value", 0) for a in ht_survivors]
        ht_avg_net_value = np.mean(ht_net_values) if ht_net_values else 0
        ht_avg_return = ((ht_avg_net_value - initial_investment_per_agent) / initial_investment_per_agent) if initial_investment_per_agent > 0 else 0
        
        # AAVE returns
        aave_survivors = [a for a in aave_perf if a.get("survived", False)]
        aave_net_values = [a.get("net_position_value", 0) for a in aave_survivors]
        aave_avg_net_value = np.mean(aave_net_values) if aave_net_values else 0
        aave_avg_return = ((aave_avg_net_value - initial_investment_per_agent) / initial_investment_per_agent) if initial_investment_per_agent > 0 else 0
        
        # Calculate comparison metrics
        comparison = {
            "high_tide": {
                "total_agents": len(ht_perf),
                "survivors": sum(1 for a in ht_perf if a.get("survived", False)),
                "survival_rate": sum(1 for a in ht_perf if a.get("survived", False)) / len(ht_perf) if ht_perf else 0,
                "liquidated_agents": sum(1 for a in ht_perf if not a.get("survived", True)),
                "avg_final_hf": np.mean([a.get("final_health_factor", 0) for a in ht_perf if a.get("survived", False)]) if ht_perf else 0,
                "total_rebalances": sum(a.get("rebalancing_events", 0) for a in ht_perf),
                "total_leverage_increases": sum(a.get("leverage_increase_events", 0) for a in ht_perf),
                "total_position_adjustments": sum(a.get("total_position_adjustments", 0) for a in ht_perf),
                "avg_net_position_value": ht_avg_net_value,
                "avg_return_pct": ht_avg_return * 100,
                "initial_investment": initial_investment_per_agent
            },
            "aave": {
                "total_agents": len(aave_perf),
                "survivors": sum(1 for a in aave_perf if a.get("survived", False)),
                "survival_rate": sum(1 for a in aave_perf if a.get("survived", False)) / len(aave_perf) if aave_perf else 0,
                "liquidated_agents": sum(1 for a in aave_perf if not a.get("survived", True)),
                "total_liquidation_events": sum(a.get("liquidation_count", 0) for a in aave_perf),
                "avg_final_hf": np.mean([a.get("final_health_factor", 0) for a in aave_perf if a.get("survived", False)]) if aave_perf else 0,
                "total_weekly_rebalances": aave_results.get("summary", {}).get("total_weekly_rebalances", 0),
                "total_deleverages": aave_results.get("summary", {}).get("total_deleverages", 0),
                "total_harvests": aave_results.get("summary", {}).get("total_harvests", 0),
                "total_position_adjustments": aave_results.get("summary", {}).get("total_weekly_rebalances", 0),  # Weekly rebalancing counts as position adjustments
                "avg_net_position_value": aave_avg_net_value,
                "avg_return_pct": aave_avg_return * 100,
                "initial_investment": initial_investment_per_agent
            }
        }
        
        # Store comparison
        self.results["comparison_summary"] = comparison
        
        print(f"✅ High Tide Survival Rate: {comparison['high_tide']['survival_rate']:.1%}")
        print(f"✅ AAVE Survival Rate: {comparison['aave']['survival_rate']:.1%}")
        print(f"📈 High Tide Avg Final HF: {comparison['high_tide']['avg_final_hf']:.3f}")
        print(f"📈 AAVE Avg Final HF: {comparison['aave']['avg_final_hf']:.3f}")
    
    def _save_comparison_results(self):
        """Save comparison results"""
        
        # Create results directory for comparison
        test_name = f"{self.config.test_name}_HT_vs_AAVE_Comparison"
        output_dir = Path("tidal_protocol_sim/results") / test_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison results JSON
        results_path = output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert for JSON serialization
        json_results = self._convert_for_json(self.results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"📁 Comparison results saved to: {results_path}")
    
    def _generate_comparison_charts(self):
        """Generate comparison charts for High Tide vs AAVE"""
        
        print("📊 Generating comparison charts...")
        
        # Create output directory
        test_name = f"{self.config.test_name}_HT_vs_AAVE_Comparison"
        output_dir = Path("tidal_protocol_sim/results") / test_name / "charts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Get results
        ht_results = self.results.get("high_tide_results", {})
        aave_results = self.results.get("aave_results", {})
        
        # Chart 1: Survival Comparison
        self._create_survival_comparison_chart(output_dir, ht_results, aave_results)
        
        # Chart 2: Health Factor Evolution (side by side)
        self._create_health_factor_comparison_chart(output_dir, ht_results, aave_results)
        
        # Chart 3: Agent Performance Metrics
        self._create_performance_metrics_chart(output_dir, ht_results, aave_results)
        
        # Chart 4: Net Position and APY Evolution Comparison
        self._create_net_position_apy_comparison_chart(output_dir, ht_results, aave_results)
        
        # Chart 5: BTC Capital Preservation Comparison
        self._create_btc_capital_preservation_chart(output_dir, ht_results, aave_results)
        
        print(f"✅ Generated comparison charts in {output_dir}")
    
    def _generate_detailed_charts_for_protocol(self, protocol_name: str, engine):
        """Generate detailed analysis charts for a specific protocol (high_tide or aave)"""
        
        # Create output directory for this protocol
        test_name = f"{self.config.test_name}_{protocol_name.upper()}_Detailed"
        output_dir = Path("tidal_protocol_sim/results") / test_name / "charts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("darkgrid")
        plt.rcParams['figure.facecolor'] = 'white'
        
        print(f"   Generating charts for {protocol_name.upper()} in {output_dir}")
        
        # Get results for this protocol
        results_key = f"{protocol_name}_results"
        if results_key not in self.results:
            print(f"   ⚠️  No results found for {protocol_name}")
            return
        
        # Temporarily swap results to use existing chart methods
        original_results = self.results.copy()
        protocol_results = self.results[results_key]
        
        # Create a temporary results structure that the chart methods expect
        temp_results = {
            "agent_performance": {
                "agent_details": protocol_results.get("agent_outcomes", []),
                "summary": {
                    "total_agents": len(protocol_results.get("agent_outcomes", [])),
                    "survived": sum(1 for a in protocol_results.get("agent_outcomes", []) if a.get("survived", False))
                }
            },
            "simulation_results": protocol_results,
            "pool_state_snapshots": protocol_results.get("pool_state_snapshots", []),
            "detailed_logs": protocol_results.get("detailed_logs", []),
            "rebalancing_events": protocol_results.get("rebalancing_events", {
                "agent_rebalances": [],
                "pool_rebalances": []
            })
        }
        
        self.results = temp_results
        
        try:
            # Generate core charts (these should work for both protocols)
            print("      Creating time series evolution chart...")
            self._create_time_series_evolution_chart(output_dir)
            
            print("      Creating agent health factor chart...")
            self._create_agent_health_factor_chart(output_dir)
            
            print("      Creating agent performance chart...")
            self._create_agent_performance_chart(output_dir)
            
            print("      Creating net APY analysis chart...")
            self._create_net_apy_analysis_chart(output_dir)
            
            print("      Creating yield strategy comparison chart...")
            self._create_yield_strategy_comparison_chart(output_dir)
            
            # Only create these for High Tide (MOET-specific)
            if protocol_name == "high_tide":
                print("      Creating rebalancing timeline chart...")
                self._create_rebalancing_timeline_chart(output_dir)
                
                if protocol_results.get("pool_state_snapshots"):
                    print("      Creating pool price evolution chart...")
                    self._create_pool_price_evolution_chart(output_dir)
                    
                    print("      Creating pool balance evolution chart...")
                    self._create_pool_balance_evolution_chart(output_dir)
                
                # Only if advanced MOET is enabled
                if self.config.enable_advanced_moet_system:
                    print("      Creating MOET system analysis charts...")
                    self._create_moet_system_analysis_chart(output_dir)
                    self._create_moet_reserve_management_chart(output_dir)
                    self._create_redeemer_analysis_chart(output_dir)
                    self._create_arbitrage_activity_chart(output_dir)
                    self._create_arbitrage_time_series_chart(output_dir)
                    self._create_peg_monitoring_chart(output_dir)
                    self._create_bond_auction_analysis_chart(output_dir)
            
            print(f"   ✅ Generated detailed charts for {protocol_name.upper()} in {output_dir}")
            
        except Exception as e:
            print(f"   ⚠️  Error generating charts for {protocol_name}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Restore original results
            self.results = original_results
    
    def _create_survival_comparison_chart(self, output_dir, ht_results, aave_results):
        """Create survival rate comparison chart"""
        
        # Get survival data
        ht_perf = ht_results.get("agent_outcomes", [])
        aave_perf = aave_results.get("agent_outcomes", [])
        
        ht_survival = sum(1 for a in ht_perf if a.get("survived", False)) / len(ht_perf) if ht_perf else 0
        aave_survival = sum(1 for a in aave_perf if a.get("survived", False)) / len(aave_perf) if aave_perf else 0
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        systems = ['High Tide\n(Active Rebalancing)', 'AAVE\n(Buy & Hold)']
        survival_rates = [ht_survival * 100, aave_survival * 100]
        colors = ['#2ecc71', '#e74c3c']
        
        bars = ax.bar(systems, survival_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, rate in zip(bars, survival_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_ylabel('Survival Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('Agent Survival Rate: High Tide vs AAVE\nFull Year 2024 (BTC: $42k → $93k)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% Survival')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'survival_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Created survival comparison chart")
    
    def _create_health_factor_comparison_chart(self, output_dir, ht_results, aave_results):
        """Create health factor evolution comparison"""
        
        # Get agent snapshots
        ht_snapshots = ht_results.get("agent_health_snapshots", {})
        aave_snapshots = aave_results.get("agent_health_snapshots", {})
        
        if not ht_snapshots and not aave_snapshots:
            print("   ⚠️  No health factor data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # High Tide chart
        for agent_id, data in list(ht_snapshots.items())[:10]:  # Show first 10 agents
            timestamps = np.array(data["timestamps"]) / 1440  # Convert to days
            hfs = data["health_factors"]
            ax1.plot(timestamps, hfs, alpha=0.6, linewidth=1)
        
        ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Liquidation Threshold')
        ax1.axhline(y=1.015, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Rebalancing Trigger')
        ax1.set_xlabel('Days', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Health Factor', fontsize=11, fontweight='bold')
        ax1.set_title('High Tide: Health Factor Evolution\n(10 sample agents)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        ax1.set_ylim(0.9, 1.15)
        
        # AAVE chart
        for agent_id, data in list(aave_snapshots.items())[:10]:  # Show first 10 agents
            timestamps = np.array(data["timestamps"]) / 1440  # Convert to days
            hfs = data["health_factors"]
            ax2.plot(timestamps, hfs, alpha=0.6, linewidth=1)
        
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Liquidation Threshold')
        ax2.set_xlabel('Days', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Health Factor', fontsize=11, fontweight='bold')
        ax2.set_title('AAVE: Health Factor Evolution\n(10 sample agents)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        ax2.set_ylim(0.9, 1.15)
        
        plt.suptitle('Health Factor Evolution Comparison: Full Year 2024', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'health_factor_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Created health factor comparison chart")
    
    def _create_performance_metrics_chart(self, output_dir, ht_results, aave_results):
        """Create performance metrics comparison"""
        
        # Get performance data
        ht_perf = ht_results.get("agent_outcomes", [])
        aave_perf = aave_results.get("agent_outcomes", [])
        
        # Calculate metrics
        ht_active = sum(1 for a in ht_perf if a.get("survived", False))
        ht_total = len(ht_perf)
        ht_rebalances = sum(a.get("rebalancing_events", 0) for a in ht_perf)
        ht_leverage_increases = sum(a.get("leverage_increase_events", 0) for a in ht_perf)
        ht_total_adjustments = sum(a.get("total_position_adjustments", 0) for a in ht_perf)
        
        aave_active = sum(1 for a in aave_perf if a.get("survived", False))
        aave_total = len(aave_perf)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Chart 1: Active vs Liquidated Agents
        ax = axes[0, 0]
        x = np.arange(2)
        width = 0.35
        
        active_counts = [ht_active, aave_active]
        liquidated_counts = [ht_total - ht_active, aave_total - aave_active]
        
        ax.bar(x - width/2, active_counts, width, label='Active', color='#2ecc71', alpha=0.7)
        ax.bar(x + width/2, liquidated_counts, width, label='Liquidated', color='#e74c3c', alpha=0.7)
        
        ax.set_ylabel('Number of Agents', fontsize=11, fontweight='bold')
        ax.set_title('Agent Status at Year End', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['High Tide', 'AAVE'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Chart 2: Survival Rate %
        ax = axes[0, 1]
        survival_rates = [ht_active/ht_total*100 if ht_total > 0 else 0, 
                         aave_active/aave_total*100 if aave_total > 0 else 0]
        colors = ['#2ecc71', '#e74c3c']
        bars = ax.bar(['High Tide', 'AAVE'], survival_rates, color=colors, alpha=0.7)
        
        for bar, rate in zip(bars, survival_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Survival Rate (%)', fontsize=11, fontweight='bold')
        ax.set_title('Survival Rate Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=100, color='green', linestyle='--', alpha=0.5)
        
        # Chart 3: Position Management Activity (Stacked Bar)
        ax = axes[1, 0]
        x_pos = np.arange(2)
        width = 0.5
        
        # Get AAVE rebalancing data from results
        aave_results_summary = aave_results.get("summary", {})
        aave_delev = aave_results_summary.get("total_deleverages", 0)
        aave_harvest = aave_results_summary.get("total_harvests", 0)
        
        # Stacked bars for High Tide showing defensive rebalancing and leverage increases
        defensive_counts = [ht_rebalances, aave_delev]  # AAVE deleverages (sell YT -> pay debt)
        aggressive_counts = [ht_leverage_increases, aave_harvest]  # AAVE harvests (sell YT -> BTC)
        
        bars1 = ax.bar(x_pos, defensive_counts, width, label='Defensive (Sell YT)', color='#e74c3c', alpha=0.7)
        bars2 = ax.bar(x_pos, aggressive_counts, width, bottom=defensive_counts, label='Aggressive (Borrow More)', color='#2ecc71', alpha=0.7)
        
        # Add value labels
        for i, (def_count, agg_count) in enumerate(zip(defensive_counts, aggressive_counts)):
            total = def_count + agg_count
            if total > 0:
                ax.text(i, total, f'{total:,}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Position Adjustments', fontsize=11, fontweight='bold')
        ax.set_title('Active Position Management (Full Year)', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['High Tide\n(Automated)', 'AAVE\n(Weekly Manual)'])
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Chart 4: Strategy Summary (text)
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate survival percentages and returns safely
        ht_survival_pct = (ht_active / ht_total * 100) if ht_total > 0 else 0.0
        aave_survival_pct = (aave_active / aave_total * 100) if aave_total > 0 else 0.0
        
        # Get return data from comparison summary
        comparison = self.results.get("comparison_summary", {})
        ht_return = comparison.get("high_tide", {}).get("avg_return_pct", 0)
        aave_return = comparison.get("aave", {}).get("avg_return_pct", 0)
        ht_avg_pos = comparison.get("high_tide", {}).get("avg_net_position_value", 0)
        aave_avg_pos = comparison.get("aave", {}).get("avg_net_position_value", 0)
        
        summary_text = f"""
COMPARISON SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

High Tide (Active Management):
  • Survived: {ht_active}/{ht_total} ({ht_survival_pct:.1f}%)
  • Avg Return: {ht_return:+.1f}%
  • Avg Position: ${ht_avg_pos:,.0f}
  
Position Adjustments: {ht_total_adjustments:,}
  • Defensive: {ht_rebalances:,}
  • Aggressive: {ht_leverage_increases:,}

AAVE (Weekly Rebalancing):
  • Survived: {aave_active}/{aave_total} ({aave_survival_pct:.1f}%)
  • Avg Return: {aave_return:+.1f}%
  • Avg Position: ${aave_avg_pos:,.0f}
  • Position Adjustments: {aave_delev + aave_harvest:,}
  • Deleverages: {aave_delev:,}
  • Harvests: {aave_harvest:,}

Outperformance: {(ht_return - aave_return):+.1f}%

Market: 2024 Bull (+119% BTC)
        """
        
        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.suptitle('Performance Metrics: High Tide vs AAVE (Full Year 2024)', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Created performance metrics chart")
    
    def _create_net_position_apy_comparison_chart(self, output_dir, ht_results, aave_results):
        """Create 2x2 comparison chart: Net Position Evolution (top) and APY Comparison (bottom left)"""
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('High Tide vs AAVE: Net Position & APY Evolution', fontsize=16, fontweight='bold')
        
        # Get BTC price history
        ht_sim_results = ht_results.get("simulation_results", ht_results)
        aave_sim_results = aave_results.get("simulation_results", aave_results)
        
        btc_history = ht_sim_results.get("btc_price_history", [])
        
        # Extract High Tide time series data from agent_health_history
        # agent_health_history is a list of timestep dictionaries: [{minute, btc_price, agents: [...]}]
        ht_health_history = ht_sim_results.get("agent_health_history", [])
        ht_agent_data = []
        
        if ht_health_history and len(ht_health_history) > 0:
            # Get first agent ID from first timestep
            first_timestep = ht_health_history[0]
            if "agents" in first_timestep and len(first_timestep["agents"]) > 0:
                first_agent_id = first_timestep["agents"][0]["agent_id"]
                
                # Extract data for this agent across all timesteps
                for timestep in ht_health_history:
                    # Find this agent's data in this timestep
                    agent_data_in_timestep = next(
                        (a for a in timestep.get("agents", []) if a.get("agent_id") == first_agent_id),
                        None
                    )
                    if agent_data_in_timestep:
                        hour = timestep.get("minute", 0) / 60.0
                        ht_agent_data.append({
                            "hour": hour,
                            "net_position": agent_data_in_timestep.get("net_position_value", 0),
                            "btc_price": timestep.get("btc_price", 42000)
                        })
        
        # Extract AAVE time series data from agent_health_history
        aave_health_history = aave_sim_results.get("agent_health_history", [])
        aave_agent_data = []
        if aave_health_history and len(aave_health_history) > 0:
            # Get first agent ID from first timestep
            first_timestep = aave_health_history[0]
            if "agents" in first_timestep and len(first_timestep["agents"]) > 0:
                first_agent_id = first_timestep["agents"][0]["agent_id"]
                
                # Extract data for this agent across all timesteps
                for timestep in aave_health_history:
                    # Find this agent's data in this timestep
                    agent_data_in_timestep = next(
                        (a for a in timestep.get("agents", []) if a.get("agent_id") == first_agent_id),
                        None
                    )
                    if agent_data_in_timestep:
                        hour = timestep.get("minute", 0) / 60.0
                        aave_agent_data.append({
                            "hour": hour,
                            "net_position": agent_data_in_timestep.get("net_position_value", 0),
                            "btc_price": timestep.get("btc_price", 42000)
                        })
        
        # Debug: Check data extraction
        print(f"📊 Extracted High Tide data points: {len(ht_agent_data)}")
        print(f"📊 Extracted AAVE data points: {len(aave_agent_data)}")
        if ht_agent_data:
            print(f"   HT first point: hour={ht_agent_data[0]['hour']:.1f}, net_pos=${ht_agent_data[0]['net_position']:,.0f}")
            print(f"   HT last point: hour={ht_agent_data[-1]['hour']:.1f}, net_pos=${ht_agent_data[-1]['net_position']:,.0f}")
        if aave_agent_data:
            print(f"   AAVE first point: hour={aave_agent_data[0]['hour']:.1f}, net_pos=${aave_agent_data[0]['net_position']:,.0f}")
            print(f"   AAVE last point: hour={aave_agent_data[-1]['hour']:.1f}, net_pos=${aave_agent_data[-1]['net_position']:,.0f}")
        
        # Top Left: High Tide Net Position Evolution
        if ht_agent_data:
            hours = [d["hour"] for d in ht_agent_data]
            net_positions = [d["net_position"] for d in ht_agent_data]
            ax1.plot(hours, net_positions, linewidth=2.5, color='#2E86AB', label='High Tide Net Position')
            ax1.fill_between(hours, net_positions, alpha=0.2, color='#2E86AB')
            ax1.set_title('High Tide: Net Position Evolution', fontsize=13, fontweight='bold')
            ax1.set_xlabel('Hours', fontsize=11)
            ax1.set_ylabel('Net Position Value ($)', fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            
            # Add final value annotation
            if net_positions:
                final_val = net_positions[-1]
                ax1.axhline(y=final_val, color='red', linestyle='--', alpha=0.5, linewidth=1)
                ax1.text(hours[-1]*0.95, final_val*1.02, f'Final: ${final_val:,.0f}', 
                        ha='right', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax1.text(0.5, 0.5, 'No High Tide time series data available', 
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('High Tide: Net Position Evolution', fontsize=13, fontweight='bold')
        
        # Top Right: AAVE Net Position Evolution
        if aave_agent_data:
            hours = [d["hour"] for d in aave_agent_data]
            net_positions = [d["net_position"] for d in aave_agent_data]
            ax2.plot(hours, net_positions, linewidth=2.5, color='#A23B72', label='AAVE Net Position')
            ax2.fill_between(hours, net_positions, alpha=0.2, color='#A23B72')
            ax2.set_title('AAVE: Net Position Evolution', fontsize=13, fontweight='bold')
            ax2.set_xlabel('Hours', fontsize=11)
            ax2.set_ylabel('Net Position Value ($)', fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            
            # Add final value annotation
            if net_positions:
                final_val = net_positions[-1]
                ax2.axhline(y=final_val, color='red', linestyle='--', alpha=0.5, linewidth=1)
                ax2.text(hours[-1]*0.95, final_val*1.02, f'Final: ${final_val:,.0f}', 
                        ha='right', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, 'No AAVE time series data available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('AAVE: Net Position Evolution', fontsize=13, fontweight='bold')
        
        # Bottom Left: High Tide APY vs AAVE APY
        if ht_agent_data and aave_agent_data and len(ht_agent_data) > 1 and len(aave_agent_data) > 1:
            # Calculate annualized APYs
            initial_btc_price = ht_agent_data[0]["btc_price"]
            initial_ht_value = ht_agent_data[0]["net_position"]
            initial_aave_value = aave_agent_data[0]["net_position"]
            
            ht_apys = []
            aave_apys = []
            hours = []
            
            for i in range(1, min(len(ht_agent_data), len(aave_agent_data))):
                hour = ht_agent_data[i]["hour"]
                hours.append(hour)
                days_elapsed = hour / 24.0
                
                if days_elapsed > 0:
                    # High Tide APY
                    ht_return = (ht_agent_data[i]["net_position"] / initial_ht_value - 1) * (365 / days_elapsed) * 100
                    ht_apys.append(ht_return)
                    
                    # AAVE APY
                    aave_return = (aave_agent_data[i]["net_position"] / initial_aave_value - 1) * (365 / days_elapsed) * 100
                    aave_apys.append(aave_return)
            
            ax3.plot(hours, ht_apys, linewidth=2.5, color='#2E86AB', label='High Tide APY', alpha=0.9)
            ax3.plot(hours, aave_apys, linewidth=2.5, color='#A23B72', label='AAVE APY', alpha=0.9)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
            ax3.set_title('Annualized APY Comparison', fontsize=13, fontweight='bold')
            ax3.set_xlabel('Hours', fontsize=11)
            ax3.set_ylabel('APY (%)', fontsize=11)
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=11, loc='best')
            
            # Add final APY annotations
            if ht_apys and aave_apys:
                final_ht_apy = ht_apys[-1]
                final_aave_apy = aave_apys[-1]
                outperformance = final_ht_apy - final_aave_apy
                
                ax3.text(0.02, 0.98, f'High Tide Final: {final_ht_apy:.2f}%\nAAVE Final: {final_aave_apy:.2f}%\nOutperformance: +{outperformance:.2f}%',
                        transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        else:
            ax3.text(0.5, 0.5, 'Insufficient data for APY comparison', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Annualized APY Comparison', fontsize=13, fontweight='bold')
        
        # Bottom Right: Summary Statistics
        ax4.axis('off')
        
        # Get final metrics from results
        ht_perf = ht_results.get("agent_outcomes", [])
        aave_perf = aave_results.get("agent_outcomes", [])
        
        ht_survivors = [a for a in ht_perf if a.get("survived", False)]
        aave_survivors = [a for a in aave_perf if a.get("survived", False)]
        
        ht_avg_return = np.mean([a.get("net_position_value", 0) for a in ht_survivors]) / 42208.20 * 100 - 100 if ht_survivors else 0
        aave_avg_return = np.mean([a.get("net_position_value", 0) for a in aave_survivors]) / 42208.20 * 100 - 100 if aave_survivors else 0
        
        # Get market info from test metadata
        test_meta = self.results.get("test_metadata", {})
        year = "2025" if "2025" in test_meta.get("test_name", "") else ("2024" if "2024" in test_meta.get("test_name", "") else ("2021" if "2021" in test_meta.get("test_name", "") else "2022"))
        initial_price = test_meta.get("btc_initial_price", 42208)
        final_price = test_meta.get("btc_final_price", 92627)
        btc_return = ((final_price / initial_price) - 1) * 100
        duration_days = test_meta.get("simulation_duration_hours", 8760) // 24
        
        summary_text = f"""
COMPARISON SUMMARY
━━━━━━━━━━━━━━━━━━━━━━

Market Conditions:
• Duration: {duration_days} days ({year})
• BTC: ${initial_price:,.0f} → ${final_price:,.0f} ({btc_return:+.1f}%)
• Interest: Historical AAVE rates

High Tide (Active Management):
• Avg Return: +{ht_avg_return:.2f}%
• Final HF: {np.mean([a.get("final_health_factor", 0) for a in ht_survivors]) if ht_survivors else 0:.3f}
• Position Adjustments: {sum(a.get("total_position_adjustments", 0) for a in ht_perf):,}

AAVE (Weekly Rebalancing):
• Avg Return: +{aave_avg_return:.2f}%
• Final HF: {np.mean([a.get("final_health_factor", 0) for a in aave_survivors]) if aave_survivors else 0:.3f}
• Position Adjustments: {aave_results.get("summary", {}).get("total_weekly_rebalances", 0):,}

Performance Advantage:
• Return Difference: +{ht_avg_return - aave_avg_return:.2f}%
• Strategy: Active rebalancing outperformed
  passive buy-and-hold by {ht_avg_return/aave_avg_return if aave_avg_return > 0 else 0:.2f}x
        """
        
        ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / "net_position_apy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Created net position and APY comparison chart")
    
    def _create_btc_capital_preservation_chart(self, output_dir, ht_results, aave_results):
        """Create 2x2 BTC Capital Preservation chart showing BTC accumulation vs buy-and-hold"""
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Bear Market Capital Preservation: BTC Accumulation Analysis', fontsize=16, fontweight='bold')
        
        # Get simulation results
        ht_sim_results = ht_results.get("simulation_results", ht_results)
        aave_sim_results = aave_results.get("simulation_results", aave_results)
        
        # Extract Flow Vaults time series data
        ht_health_history = ht_sim_results.get("agent_health_history", [])
        ht_btc_data = []
        if ht_health_history and len(ht_health_history) > 0:
            first_timestep = ht_health_history[0]
            if "agents" in first_timestep and len(first_timestep["agents"]) > 0:
                first_agent_id = first_timestep["agents"][0]["agent_id"]
                print(f"📊 Extracting Flow Vaults BTC data for agent {first_agent_id}")
                for timestep in ht_health_history:
                    agent_data = next((a for a in timestep.get("agents", []) if a.get("agent_id") == first_agent_id), None)
                    if agent_data:
                        hour = timestep.get("minute", 0) / 60.0
                        btc_amount = agent_data.get("btc_amount", 0)
                        btc_price = timestep.get("btc_price", 46320)
                        btc_value = btc_amount * btc_price
                        ht_btc_data.append({
                            "hour": hour,
                            "btc_amount": btc_amount,
                            "btc_value": btc_value,
                            "btc_price": btc_price
                        })
                print(f"   ✅ Extracted {len(ht_btc_data)} Flow Vaults data points")
                if ht_btc_data:
                    print(f"   📈 First BTC amount: {ht_btc_data[0]['btc_amount']:.4f}, Last BTC amount: {ht_btc_data[-1]['btc_amount']:.4f}")
        
        # Extract AAVE time series data
        aave_health_history = aave_sim_results.get("agent_health_history", [])
        aave_btc_data = []
        if aave_health_history and len(aave_health_history) > 0:
            first_timestep = aave_health_history[0]
            if "agents" in first_timestep and len(first_timestep["agents"]) > 0:
                first_agent_id = first_timestep["agents"][0]["agent_id"]
                print(f"📊 Extracting AAVE BTC data for agent {first_agent_id}")
                for timestep in aave_health_history:
                    agent_data = next((a for a in timestep.get("agents", []) if a.get("agent_id") == first_agent_id), None)
                    if agent_data:
                        hour = timestep.get("minute", 0) / 60.0
                        # AAVE records BTC as "remaining_collateral"
                        btc_amount = agent_data.get("remaining_collateral", 0)
                        btc_price = timestep.get("btc_price", 46320)
                        btc_value = btc_amount * btc_price
                        aave_btc_data.append({
                            "hour": hour,
                            "btc_amount": btc_amount,
                            "btc_value": btc_value,
                            "btc_price": btc_price
                        })
                print(f"   ✅ Extracted {len(aave_btc_data)} AAVE data points")
                if aave_btc_data:
                    print(f"   📈 First BTC amount: {aave_btc_data[0]['btc_amount']:.4f}, Last BTC amount: {aave_btc_data[-1]['btc_amount']:.4f}")
        
        # Top Left: Flow Vaults BTC Value vs. BTC Buy and Hold
        if ht_btc_data:
            hours = [d["hour"] for d in ht_btc_data]
            ht_values = [d["btc_value"] for d in ht_btc_data]
            
            # Buy and hold: 1 BTC throughout
            buyhold_values = [1.0 * d["btc_price"] for d in ht_btc_data]
            
            ax1.plot(hours, ht_values, linewidth=2.5, color='#2E86AB', label='Flow Vaults BTC Value', alpha=0.9)
            ax1.plot(hours, buyhold_values, linewidth=2.5, color='#FFB84D', label='BTC Buy & Hold', alpha=0.9, linestyle='--')
            ax1.fill_between(hours, ht_values, buyhold_values, where=[hv >= bh for hv, bh in zip(ht_values, buyhold_values)], 
                            alpha=0.2, color='green', label='Outperformance')
            ax1.fill_between(hours, ht_values, buyhold_values, where=[hv < bh for hv, bh in zip(ht_values, buyhold_values)], 
                            alpha=0.2, color='red', label='Underperformance')
            ax1.set_title('Flow Vaults: BTC Value vs. Buy & Hold', fontsize=13, fontweight='bold')
            ax1.set_xlabel('Hours', fontsize=11)
            ax1.set_ylabel('BTC Value ($)', fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10, loc='upper right')
            
            # Add final comparison annotation
            if ht_values and buyhold_values:
                final_ht = ht_values[-1]
                final_bh = buyhold_values[-1]
                diff_pct = ((final_ht / final_bh) - 1) * 100
                ax1.text(0.02, 0.02, f'Final FV: ${final_ht:,.0f}\nFinal B&H: ${final_bh:,.0f}\nDifference: {diff_pct:+.1f}%',
                        transform=ax1.transAxes, fontsize=10, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax1.text(0.5, 0.5, 'No Flow Vaults data available', ha='center', va='center', transform=ax1.transAxes, fontsize=12)
            ax1.set_title('Flow Vaults: BTC Value vs. Buy & Hold', fontsize=13, fontweight='bold')
        
        # Top Right: AAVE BTC Value vs. BTC Buy and Hold
        if aave_btc_data:
            hours = [d["hour"] for d in aave_btc_data]
            aave_values = [d["btc_value"] for d in aave_btc_data]
            
            # Buy and hold: 1 BTC throughout
            buyhold_values = [1.0 * d["btc_price"] for d in aave_btc_data]
            
            ax2.plot(hours, aave_values, linewidth=2.5, color='#A23B72', label='AAVE BTC Value', alpha=0.9)
            ax2.plot(hours, buyhold_values, linewidth=2.5, color='#FFB84D', label='BTC Buy & Hold', alpha=0.9, linestyle='--')
            ax2.fill_between(hours, aave_values, buyhold_values, where=[av >= bh for av, bh in zip(aave_values, buyhold_values)], 
                            alpha=0.2, color='green', label='Outperformance')
            ax2.fill_between(hours, aave_values, buyhold_values, where=[av < bh for av, bh in zip(aave_values, buyhold_values)], 
                            alpha=0.2, color='red', label='Underperformance')
            ax2.set_title('AAVE: BTC Value vs. Buy & Hold', fontsize=13, fontweight='bold')
            ax2.set_xlabel('Hours', fontsize=11)
            ax2.set_ylabel('BTC Value ($)', fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10, loc='upper right')
            
            # Add final comparison annotation
            if aave_values and buyhold_values:
                final_aave = aave_values[-1]
                final_bh = buyhold_values[-1]
                diff_pct = ((final_aave / final_bh) - 1) * 100
                ax2.text(0.02, 0.02, f'Final AAVE: ${final_aave:,.0f}\nFinal B&H: ${final_bh:,.0f}\nDifference: {diff_pct:+.1f}%',
                        transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, 'No AAVE data available', ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('AAVE: BTC Value vs. Buy & Hold', fontsize=13, fontweight='bold')
        
        # Bottom Left: Flow Vaults BTC Holdings (quantity)
        if ht_btc_data:
            hours = [d["hour"] for d in ht_btc_data]
            btc_amounts = [d["btc_amount"] for d in ht_btc_data]
            
            ax3.plot(hours, btc_amounts, linewidth=2.5, color='#2E86AB', label='Flow Vaults BTC Holdings')
            ax3.fill_between(hours, btc_amounts, alpha=0.2, color='#2E86AB')
            ax3.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Initial (1 BTC)')
            ax3.set_title('Flow Vaults: BTC Holdings Over Time', fontsize=13, fontweight='bold')
            ax3.set_xlabel('Hours', fontsize=11)
            ax3.set_ylabel('BTC Amount', fontsize=11)
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=10)
            
            # Add final BTC amount annotation
            if btc_amounts:
                final_btc = btc_amounts[-1]
                initial_btc = 1.0
                change_pct = ((final_btc / initial_btc) - 1) * 100
                ax3.text(hours[-1]*0.95, final_btc*1.02, f'{final_btc:.4f} BTC\n({change_pct:+.2f}%)', 
                        ha='right', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightgreen' if change_pct > 0 else 'lightcoral', alpha=0.8))
        else:
            ax3.text(0.5, 0.5, 'No Flow Vaults data available', ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Flow Vaults: BTC Holdings Over Time', fontsize=13, fontweight='bold')
        
        # Bottom Right: AAVE BTC Holdings (quantity)
        if aave_btc_data:
            hours = [d["hour"] for d in aave_btc_data]
            btc_amounts = [d["btc_amount"] for d in aave_btc_data]
            
            ax4.plot(hours, btc_amounts, linewidth=2.5, color='#A23B72', label='AAVE BTC Holdings')
            ax4.fill_between(hours, btc_amounts, alpha=0.2, color='#A23B72')
            ax4.axhline(y=1.0, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Initial (1 BTC)')
            ax4.set_title('AAVE: BTC Holdings Over Time', fontsize=13, fontweight='bold')
            ax4.set_xlabel('Hours', fontsize=11)
            ax4.set_ylabel('BTC Amount', fontsize=11)
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=10)
            
            # Add final BTC amount annotation
            if btc_amounts:
                final_btc = btc_amounts[-1]
                initial_btc = 1.0
                change_pct = ((final_btc / initial_btc) - 1) * 100
                ax4.text(hours[-1]*0.95, max(btc_amounts)*0.9, f'{final_btc:.4f} BTC\n({change_pct:+.2f}%)', 
                        ha='right', fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='lightgreen' if change_pct > 0 else 'lightcoral', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No AAVE data available', ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('AAVE: BTC Holdings Over Time', fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "btc_capital_preservation_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Created BTC capital preservation comparison chart")
    
    def _print_comparison_summary(self):
        """Print comparison summary"""
        
        comparison = self.results.get("comparison_summary", {})
        
        if not comparison:
            print("\n⚠️  No comparison data available")
            return
        
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY: HIGH TIDE vs AAVE")
        print("=" * 70)
        
        ht = comparison.get("high_tide", {})
        aave = comparison.get("aave", {})
        
        print(f"\n📊 Liquidation Rates:")
        print(f"   High Tide: {(1 - ht.get('survival_rate', 0)):.1%} liquidated ({ht.get('total_agents', 0) - ht.get('survivors', 0)}/{ht.get('total_agents', 0)} agents)")
        print(f"   AAVE:      {(1 - aave.get('survival_rate', 0)):.1%} liquidated ({aave.get('total_agents', 0) - aave.get('survivors', 0)}/{aave.get('total_agents', 0)} agents)")
        if aave.get('total_liquidation_events'):
            print(f"   AAVE Total Liquidation Events: {aave.get('total_liquidation_events', 0):,} (multiple liquidations per agent)")
        
        print(f"\n📈 Average Final Health Factors:")
        print(f"   High Tide: {ht.get('avg_final_hf', 0):.3f}")
        print(f"   AAVE:      {aave.get('avg_final_hf', 0):.3f}")
        
        print(f"\n💰 Return Performance:")
        print(f"   Initial Investment: ${ht.get('initial_investment', 0):,.2f} per agent (1 BTC)")
        print(f"   High Tide:")
        print(f"      • Avg Final Position: ${ht.get('avg_net_position_value', 0):,.2f}")
        print(f"      • Avg Return: {ht.get('avg_return_pct', 0):+.2f}%")
        print(f"   AAVE:")
        print(f"      • Avg Final Position: ${aave.get('avg_net_position_value', 0):,.2f}")
        print(f"      • Avg Return: {aave.get('avg_return_pct', 0):+.2f}%")
        print(f"   Outperformance: {(ht.get('avg_return_pct', 0) - aave.get('avg_return_pct', 0)):+.2f}% higher returns")
        
        print(f"\n🔄 Active Position Management:")
        print(f"   High Tide (Initial HF: {self.config.agent_initial_hf}):")
        print(f"      • Defensive Rebalancing (sell YT when HF < {self.config.agent_rebalancing_hf}): {ht.get('total_rebalances', 0):,}")
        print(f"      • Leverage Increases (borrow more when HF > {self.config.agent_initial_hf}): {ht.get('total_leverage_increases', 0):,}")
        print(f"      • Total Position Adjustments: {ht.get('total_position_adjustments', 0):,}")
        print(f"   AAVE (Initial HF: {self.config.aave_initial_hf}):")
        print(f"      • Weekly Rebalancing: {aave.get('total_weekly_rebalances', 0):,} ({aave.get('total_deleverages', 0):,} delev, {aave.get('total_harvests', 0):,} harvest)")
        print(f"      • Total Position Adjustments: {aave.get('total_position_adjustments', 0):,}")
        
        # Add detailed accounting breakdown for High Tide
        print(f"\n📋 HIGH TIDE ACCOUNTING BREAKDOWN:")
        ht_agents = self.results.get("high_tide_results", {}).get("agent_outcomes", [])
        if ht_agents:
            agent = ht_agents[0]  # Show first agent as example
            yt_portfolio = agent.get("yield_token_portfolio", {})
            print(f"   Agent: {agent.get('agent_id', 'unknown')}")
            print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"   💰 YT Initial Value: ${yt_portfolio.get('total_initial_value', 0):,.2f}")
            print(f"   📈 Total MOET Borrowed (Principal): ${agent.get('current_moet_debt', 0) - agent.get('total_interest_accrued', 0):,.2f}")
            print(f"   📊 Interest Accrued: ${agent.get('total_interest_accrued', 0):,.2f}")
            print(f"   💵 Total MOET Debt: ${agent.get('current_moet_debt', 0):,.2f}")
            print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"   🔴 YT Sold (Yield Harvest): ${agent.get('total_deleveraging_sales', 0):,.2f}")
            print(f"   💎 Current YT Value: ${yt_portfolio.get('total_current_value', 0):,.2f}")
            print(f"   🪙 BTC Accumulated: {agent.get('btc_amount', 0):.4f} BTC")
            print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            discrepancy = yt_portfolio.get('total_initial_value', 0) - (agent.get('current_moet_debt', 0) - agent.get('total_interest_accrued', 0))
            print(f"   ⚠️  DISCREPANCY CHECK:")
            print(f"       YT Purchased should equal MOET Borrowed (ex-interest)")
            print(f"       Difference: ${abs(discrepancy):,.2f}")
            if abs(discrepancy) > 100:
                print(f"       ❌ ACCOUNTING ERROR DETECTED!")
        
        print("\n" + "=" * 70)
    
    def _save_test_results(self):
        """Save comprehensive test results"""
        
        # Create results directory (use different folder for ecosystem growth)
        if self.config.enable_ecosystem_growth:
            test_name = f"{self.config.test_name}_Ecosystem_Growth"
        else:
            test_name = self.config.test_name
        output_dir = Path("tidal_protocol_sim/results") / test_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results JSON
        results_path = output_dir / f"pool_rebalancer_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert for JSON serialization
        json_results = self._convert_for_json(self.results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"📁 Test results saved to: {results_path}")
        
        # Save detailed CSV if requested
        if self.config.save_detailed_csv:
            self._save_detailed_csv(output_dir)
    
    def _save_detailed_csv(self, output_dir: Path):
        """Save detailed CSV files for analysis"""
        
        # Agent performance CSV
        agent_data = self.results["agent_performance"]["agent_details"]
        agent_df = pd.DataFrame(agent_data)
        agent_csv_path = output_dir / "agent_performance.csv"
        agent_df.to_csv(agent_csv_path, index=False)
        
        # Pool state snapshots CSV
        if self.results["pool_state_snapshots"]:
            pool_data = []
            for snapshot in self.results["pool_state_snapshots"]:
                pool_data.append({
                    "hour": snapshot["hour"],
                    "btc_price": snapshot["btc_price"],
                    "true_yt_price": snapshot["true_yt_price"],
                    "pool_yt_price": snapshot["pool_yt_price"],
                    "price_deviation_bps": snapshot["price_deviation_bps"],
                    "survived_agents": snapshot["agent_summary"]["survived_agents"],
                    "avg_health_factor": snapshot["agent_summary"]["avg_health_factor"]
                })
            
            pool_df = pd.DataFrame(pool_data)
            pool_csv_path = output_dir / "pool_state_evolution.csv"
            pool_df.to_csv(pool_csv_path, index=False)
        
        # Rebalancing events CSV
        all_events = []
        for event in self.results["rebalancing_events"]["agent_rebalances"]:
            all_events.append({
                "type": "agent",
                "hour": event["hour"],
                "agent_id": event["agent_id"],
                "hf_before": event["hf_before"],
                "hf_after": event["hf_after"],
                "moet_raised": event["moet_raised"],
                "slippage_cost": event["slippage_cost"]
            })
        
        for event in self.results["rebalancing_events"]["alm_rebalances"]:
            all_events.append({
                "type": "alm",
                "hour": event["hour"],
                "rebalancer": event["rebalancer"],
                "profit": event.get("result", {}).get("profit", 0)
            })
        
        for event in self.results["rebalancing_events"]["algo_rebalances"]:
            all_events.append({
                "type": "algo",
                "hour": event["hour"],
                "rebalancer": event["rebalancer"],
                "profit": event.get("result", {}).get("profit", 0)
            })
        
        if all_events:
            events_df = pd.DataFrame(all_events)
            events_csv_path = output_dir / "rebalancing_events.csv"
            events_df.to_csv(events_csv_path, index=False)
        
        print(f"📊 Detailed CSV files saved to: {output_dir}")
    
    def _convert_for_json(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {str(key): self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif hasattr(obj, 'dtype') and 'float' in str(obj.dtype):
            return float(obj)
        elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
            return int(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def _generate_test_charts(self):
        """Generate comprehensive test charts"""
        
        # Use same naming logic as results directory
        if self.config.enable_ecosystem_growth:
            test_name = f"{self.config.test_name}_Ecosystem_Growth"
        else:
            test_name = self.config.test_name
        output_dir = Path("tidal_protocol_sim/results") / test_name / "charts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        print("📊 Generating test charts...")
        
        # Chart 1: BTC Price and Pool Price Deviation
        self._create_price_deviation_chart(output_dir)
        
        # Chart 2: Agent Health Factors Over Time
        self._create_agent_health_factor_chart(output_dir)
        
        # Chart 3: Agent Slippage Analysis
        self._create_agent_slippage_analysis_chart(output_dir)
        
        # Chart 4: Pool Price Evolution Analysis
        self._create_pool_price_evolution_chart(output_dir)
        
        # Chart 5: Rebalancer Activity Analysis (2x2 layout)
        self._create_rebalancer_activity_chart(output_dir)
        
        # Chart 6: Time Series Evolution Analysis (2x2 layout)
        self._create_time_series_evolution_chart(output_dir)
        
        # Chart 7: Pool Rebalancer Balance Evolution (2x1 layout)
        self._create_pool_balance_evolution_chart(output_dir)
        
        # Chart 8: Net APY Analysis (Agent Performance vs BTC Hold)
        self._create_net_apy_analysis_chart(output_dir)
        
        # Chart 9: Yield Strategy Comparison (Tidal Protocol vs Base Yield)
        self._create_yield_strategy_comparison_chart(output_dir)
        
        # Chart 10: Ecosystem Growth Analysis (if enabled)
        self._create_ecosystem_growth_chart(output_dir)
        
        # Chart 10: MOET System Analysis (Interest Rates & Bond APRs)
        self._create_moet_system_analysis_chart(output_dir)
        
        # Chart 11: MOET Reserve Management (Target vs Actual Reserves)
        self._create_moet_reserve_management_chart(output_dir)
        
        # Chart 12: Redeemer System Analysis (NEW)
        self._create_redeemer_analysis_chart(output_dir)
        
        # Chart 13: Arbitrage Activity Analysis (ENHANCED)
        self._create_arbitrage_activity_chart(output_dir)
        
        # Chart 14: Arbitrage Time-Series Analysis (NEW)
        self._create_arbitrage_time_series_chart(output_dir)
        
        # Chart 15: MOET Peg Monitoring Analysis (NEW)
        self._create_peg_monitoring_chart(output_dir)
        
        # Chart 16: Pool-Specific Slippage Analysis (NEW)
        self._create_pool_slippage_analysis_chart(output_dir)
        
        # Chart 17: MOET Stablecoin Price Deviations (NEW)
        self._create_moet_stablecoin_price_chart(output_dir)
        
        # Chart 18: Bond Auction Activity Analysis (NEW)
        self._create_bond_auction_analysis_chart(output_dir)
        
        print(f"📊 Charts saved to: {output_dir}")
    
    def _create_price_deviation_chart(self, output_dir: Path):
        """Create BTC price and pool price deviation chart"""
        
        if not self.results["pool_state_snapshots"]:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Pool Rebalancer Test: Price Evolution and Deviations', fontsize=16, fontweight='bold')
        
        snapshots = self.results["pool_state_snapshots"]
        hours = [s["hour"] for s in snapshots]
        btc_prices = [s["btc_price"] for s in snapshots]
        deviations = [s["price_deviation_bps"] for s in snapshots]
        
        # Chart 1: BTC Price Evolution
        ax1.plot(hours, btc_prices, linewidth=2, color='orange', label='BTC Price')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('BTC Price ($)')
        ax1.set_title('BTC Price Evolution (24-Hour Test)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Chart 2: Pool Price Deviations
        ax2.plot(hours, deviations, linewidth=2, color='red', label='Price Deviation')
        ax2.axhline(y=self.config.algo_deviation_threshold_bps, color='red', linestyle='--', alpha=0.5, label='Algo Threshold (50 bps)')
        ax2.axhline(y=-self.config.algo_deviation_threshold_bps, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('Price Deviation (bps)')
        ax2.set_title('MOET:YT Pool Price Deviation from True Price')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "price_deviation_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_agent_health_factor_chart(self, output_dir: Path):
        """Create agent health factor evolution chart"""
        
        # This would require more detailed tracking of agent HF over time
        # For now, create a summary chart
        
        agent_data = self.results["agent_performance"]["agent_details"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Agent Performance Analysis', fontsize=16, fontweight='bold')
        
        # Chart 1: Initial vs Final Health Factors
        agent_ids = [a["agent_id"] for a in agent_data]
        # Handle both field name formats (old and new)
        initial_hfs = [a.get("initial_hf", a.get("initial_health_factor", 0)) for a in agent_data]
        final_hfs = [a.get("final_hf", a.get("final_health_factor", 0)) for a in agent_data]
        survived = [a["survived"] for a in agent_data]
        
        colors = ['green' if s else 'red' for s in survived]
        
        ax1.scatter(initial_hfs, final_hfs, c=colors, alpha=0.7, s=100)
        ax1.plot([1.0, 1.5], [1.0, 1.5], 'k--', alpha=0.5, label='No Change Line')
        ax1.axhline(y=self.config.agent_target_hf, color='blue', linestyle='--', alpha=0.5, label='Target HF')
        ax1.axvline(x=self.config.agent_target_hf, color='blue', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Initial Health Factor')
        ax1.set_ylabel('Final Health Factor')
        ax1.set_title('Initial vs Final Health Factors')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Rebalancing Activity by Agent
        # Handle both field name formats (old and new)
        rebalance_counts = [a.get("rebalance_count", a.get("rebalancing_events", 0)) for a in agent_data]
        slippage_costs = [a.get("total_slippage", a.get("total_slippage_costs", a.get("total_rebalancing_slippage", 0))) for a in agent_data]
        
        bars = ax2.bar(range(len(agent_ids)), rebalance_counts, color=colors, alpha=0.7)
        ax2.set_xlabel('Agent')
        ax2.set_ylabel('Number of Rebalances')
        ax2.set_title('Rebalancing Activity by Agent')
        ax2.set_xticks(range(len(agent_ids)))
        ax2.set_xticklabels([f"A{i}" for i in range(len(agent_ids))], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, rebalance_counts):
            if count > 0:
                ax2.annotate(f'{count}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / "agent_performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_rebalancing_timeline_chart(self, output_dir: Path):
        """Create rebalancing activity timeline chart"""
        
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle('Rebalancing Activity Timeline', fontsize=16, fontweight='bold')
        
        # Handle both dict and list formats for rebalancing_events
        rebalancing_events = self.results.get("rebalancing_events", {})
        if isinstance(rebalancing_events, list):
            # If it's a list, treat it as agent rebalances
            agent_events = rebalancing_events
            alm_events = []
        elif isinstance(rebalancing_events, dict):
            agent_events = rebalancing_events.get("agent_rebalances", [])
            alm_events = rebalancing_events.get("alm_rebalances", [])
        else:
            agent_events = []
            alm_events = []
        
        # Plot agent rebalances
        if agent_events:
            agent_hours = [e.get("hour", e.get("minute", 0) / 60) for e in agent_events]
            ax.scatter(agent_hours, [1] * len(agent_hours), alpha=0.7, s=50, color='blue', label='Agent Rebalances')
        
        # Plot ALM rebalances
        if alm_events:
            alm_hours = [e.get("hour", e.get("minute", 0) / 60) for e in alm_events]
            ax.scatter(alm_hours, [2] * len(alm_hours), alpha=0.7, s=100, color='green', marker='s', label='ALM Rebalances')
        
        # Plot Algo rebalances
        algo_events = rebalancing_events.get("algo_rebalances", []) if isinstance(rebalancing_events, dict) else []
        if algo_events:
            algo_hours = [e.get("hour", e.get("minute", 0) / 60) for e in algo_events]
            ax.scatter(algo_hours, [3] * len(algo_hours), alpha=0.7, s=100, color='red', marker='^', label='Algo Rebalances')
        
        ax.set_xlabel('Hours')
        ax.set_ylabel('Rebalancing Type')
        ax.set_title('Rebalancing Events Over 24-Hour Period')
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(['Agent', 'ALM', 'Algo'])
        ax.set_xlim(0, 24)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "rebalancing_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_agent_performance_chart(self, output_dir: Path):
        """Create agent performance summary chart"""
        
        agent_data = self.results["agent_performance"]["agent_details"]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Agent Performance Summary', fontsize=16, fontweight='bold')
        
        # Chart 1: Survival status
        survived = sum(1 for a in agent_data if a["survived"])
        liquidated = len(agent_data) - survived
        
        ax1.pie([survived, liquidated], labels=['Survived', 'Liquidated'], 
               colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Agent Survival Rate')
        
        # Chart 2: Net position distribution
        # Handle both field name formats (old and new)
        net_positions = [a.get("net_position", a.get("net_position_value", 0)) for a in agent_data if a["survived"]]
        if net_positions:
            ax2.hist(net_positions, bins=10, alpha=0.7, color='blue', edgecolor='black')
            ax2.set_xlabel('Net Position ($)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Net Position Distribution (Survived Agents)')
            ax2.grid(True, alpha=0.3)
        
        # Chart 3: Slippage costs
        # Handle both field name formats (old and new)
        slippage_costs = [a.get("total_slippage", a.get("total_slippage_costs", a.get("total_rebalancing_slippage", 0))) for a in agent_data]
        agent_indices = range(len(agent_data))
        colors = ['green' if a["survived"] else 'red' for a in agent_data]
        
        bars = ax3.bar(agent_indices, slippage_costs, color=colors, alpha=0.7)
        ax3.set_xlabel('Agent')
        ax3.set_ylabel('Total Slippage Costs ($)')
        ax3.set_title('Slippage Costs by Agent')
        ax3.set_xticks(agent_indices)
        ax3.set_xticklabels([f"A{i}" for i in agent_indices], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Health factor final distribution
        # Handle both field name formats (old and new)
        final_hfs = [a.get("final_hf", a.get("final_health_factor", 0)) for a in agent_data if a["survived"]]
        if final_hfs:
            ax4.hist(final_hfs, bins=10, alpha=0.7, color='green', edgecolor='black')
            ax4.axvline(x=self.config.agent_target_hf, color='red', linestyle='--', label='Target HF')
            ax4.set_xlabel('Final Health Factor')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Final Health Factor Distribution')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "agent_performance_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pool_price_evolution_chart(self, output_dir: Path):
        """Create pool price evolution chart showing true vs pool YT prices with ALM interventions"""
        
        # Get pool state snapshots
        sim_results = self.results.get("simulation_results", {})
        pool_snapshots = sim_results.get("pool_state_snapshots", [])
        
        if not pool_snapshots:
            print("⚠️  No pool state snapshots available for price evolution chart")
            return
        
        # Convert to DataFrame for easier handling
        import pandas as pd
        df = pd.DataFrame(pool_snapshots)
        
        # Find ALM rebalancing events
        alm_events = []
        for log_entry in self.results.get("detailed_logs", []):
            if log_entry.get("event_type") == "ALM_REBALANCE":
                data = log_entry.get("data", {})
                alm_events.append({
                    "hour": log_entry.get("hour", 0),
                    "direction": data.get("params", {}).get("direction", "unknown"),
                    "amount": data.get("params", {}).get("amount", 0),
                    "true_price": data.get("true_yt_price", 0)
                })
        
        # Create the chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Pool Price Evolution: True vs Pool YT Prices with ALM Interventions', 
                     fontsize=16, fontweight='bold')
        
        # Chart 1: Price Comparison
        ax1.plot(df['hour'], df['true_yt_price'], linewidth=2, color='blue', 
                 label='True YT Price', alpha=0.8)
        ax1.plot(df['hour'], df['pool_yt_price'], linewidth=2, color='red', 
                 label='Pool YT Price', alpha=0.8)
        
        # Mark ALM rebalancing events (sample every 20th event to reduce clutter)
        sell_yt_labeled = False
        buy_yt_labeled = False
        
        # Sample events to reduce visual clutter - show every 20th event or significant ones
        sampled_events = []
        for i, event in enumerate(alm_events):
            # Show every 20th event OR events with large amounts (>$50k)
            if i % 20 == 0 or event.get('amount', 0) > 50000:
                sampled_events.append(event)
        
        for event in sampled_events:
            color = 'green' if event['direction'] == 'sell_yt_for_moet' else 'orange'
            marker = '^' if event['direction'] == 'sell_yt_for_moet' else 'v'
            
            # Determine label for legend
            label = ""
            if event['direction'] == 'sell_yt_for_moet' and not sell_yt_labeled:
                label = "ALM Sell YT For MOET"
                sell_yt_labeled = True
            elif event['direction'] == 'buy_yt_with_moet' and not buy_yt_labeled:
                label = "ALM Buy YT With MOET"
                buy_yt_labeled = True
            
            ax1.axvline(x=event['hour'], color=color, linestyle='--', alpha=0.5)
            ax1.scatter(event['hour'], event['true_price'], color=color, s=100, 
                       marker=marker, zorder=5, label=label)
        
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('YT Price ($)')
        ax1.set_title('True YT Price vs Pool YT Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, df['hour'].max())
        
        # Format y-axis to show more decimal places
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.6f}'))
        
        # Chart 2: Price Deviation
        ax2.plot(df['hour'], df['deviation_bps'], linewidth=2, color='purple', 
                 label='Pool Price Deviation')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Algo Threshold (+50 bps)')
        ax2.axhline(y=-50, color='red', linestyle='--', alpha=0.5, label='Algo Threshold (-50 bps)')
        
        # Mark ALM events on deviation chart (use same sampled events)
        for event in sampled_events:
            color = 'green' if event['direction'] == 'sell_yt_for_moet' else 'orange'
            ax2.axvline(x=event['hour'], color=color, linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('Deviation (basis points)')
        ax2.set_title('Pool Price Deviation from True Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, df['hour'].max())
        
        plt.tight_layout()
        plt.savefig(output_dir / "pool_price_evolution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print statistics
        max_dev = df['deviation_bps'].abs().max()
        avg_dev = df['deviation_bps'].abs().mean()
        print(f"📊 Pool Price Stats: Max deviation {max_dev:.1f} bps, Avg deviation {avg_dev:.1f} bps, ALM events: {len(alm_events)}")
    
    def _create_rebalancer_activity_chart(self, output_dir: Path):
        """Create 2x2 rebalancer activity chart: ALM (top) and Algo (bottom) with volume and PnL"""
        
        # Extract rebalancer events from detailed logs
        alm_events = []
        algo_events = []
        
        for log_entry in self.results.get("detailed_logs", []):
            if log_entry.get("event_type") == "ALM_REBALANCE":
                data = log_entry.get("data", {})
                params = data.get("params", {})
                pool_before = data.get("pool_state_before", {})
                pool_after = data.get("pool_state_after", {})
                
                # Calculate profit from balance changes
                moet_before = pool_before.get("alm_moet_balance", 0)
                moet_after = pool_after.get("alm_moet_balance", 0)
                profit = moet_after - moet_before
                
                alm_events.append({
                    "hour": log_entry.get("hour", 0),
                    "minute": log_entry.get("minute", 0),
                    "direction": params.get("direction", "unknown"),
                    "amount": params.get("amount", 0),
                    "profit": profit,
                    "cumulative_profit": 0  # Will calculate below
                })
                
            elif log_entry.get("event_type") == "ALGO_REBALANCE":
                data = log_entry.get("data", {})
                params = data.get("params", {})
                pool_before = data.get("pool_state_before", {})
                pool_after = data.get("pool_state_after", {})
                
                # Calculate profit from balance changes
                algo_moet_before = pool_before.get("algo_moet_balance", 0)
                algo_moet_after = pool_after.get("algo_moet_balance", 0)
                profit = algo_moet_after - algo_moet_before
                
                algo_events.append({
                    "hour": log_entry.get("hour", 0),
                    "minute": log_entry.get("minute", 0),
                    "direction": params.get("direction", "unknown"),
                    "amount": params.get("amount", 0),
                    "profit": profit,
                    "cumulative_profit": 0  # Will calculate below
                })
        
        # Calculate cumulative profits
        cumulative_alm = 0
        for event in alm_events:
            cumulative_alm += event["profit"]
            event["cumulative_profit"] = cumulative_alm
            
        cumulative_algo = 0
        for event in algo_events:
            cumulative_algo += event["profit"]
            event["cumulative_profit"] = cumulative_algo
        
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Rebalancer Activity Analysis: Volume & PnL Tracking', 
                     fontsize=16, fontweight='bold')
        
        # ALM Charts (Top Row)
        self._plot_rebalancer_volume(ax1, alm_events, "ALM Rebalancer", "Volume ($)")
        self._plot_rebalancer_pnl(ax2, alm_events, "ALM Rebalancer", "Cumulative PnL ($)")
        
        # Algo Charts (Bottom Row)  
        self._plot_rebalancer_volume(ax3, algo_events, "Algo Rebalancer", "Volume ($)")
        self._plot_rebalancer_pnl(ax4, algo_events, "Algo Rebalancer", "Cumulative PnL ($)")
        
        plt.tight_layout()
        plt.savefig(output_dir / "rebalancer_activity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Rebalancer Activity: ALM events: {len(alm_events)}, Algo events: {len(algo_events)}")
    
    def _plot_rebalancer_volume(self, ax, events, title, ylabel):
        """Plot rebalancer volume as stacked bars (sell vs buy)"""
        # Set consistent x-axis range for both ALM and Algo charts (full year)
        max_hours = 8760  # Full year = 365 days * 24 hours = 8760 hours
        ax.set_xlim(0, max_hours)
        
        if not events:
            ax.set_title(f"{title} - Volume Over Time")
            ax.set_xlabel("Hours")
            ax.set_ylabel(ylabel)
            ax.text(0.5, 0.5, "No rebalancing activity", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, alpha=0.7)
            ax.grid(True, alpha=0.3)
            return
        
        # Separate sell and buy events
        hours = [e["hour"] for e in events]
        sell_amounts = [e["amount"] if e["direction"] == "sell_yt_for_moet" else 0 for e in events]
        buy_amounts = [e["amount"] if e["direction"] == "buy_yt_with_moet" else 0 for e in events]
        
        # Create bar chart
        width = 0.8  # Bar width
        ax.bar(hours, sell_amounts, width, label='Sell YT', color='green', alpha=0.7)
        ax.bar(hours, buy_amounts, width, bottom=sell_amounts, label='Buy YT', color='orange', alpha=0.7)
        
        ax.set_title(f"{title} - Volume Over Time")
        ax.set_xlabel("Hours")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    def _plot_rebalancer_pnl(self, ax, events, title, ylabel):
        """Plot rebalancer cumulative PnL as line chart"""
        # Set consistent x-axis range for both ALM and Algo charts (full year)
        max_hours = 8760  # Full year = 365 days * 24 hours = 8760 hours
        ax.set_xlim(0, max_hours)
        
        if not events:
            ax.set_title(f"{title} - Cumulative PnL")
            ax.set_xlabel("Hours")
            ax.set_ylabel(ylabel)
            ax.text(0.5, 0.5, "No rebalancing activity", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, alpha=0.7)
            ax.grid(True, alpha=0.3)
            return
        
        # Extract data
        hours = [e["hour"] for e in events]
        cumulative_pnl = [e["cumulative_profit"] for e in events]
        
        # Add starting point at hour 0
        if hours and hours[0] > 0:
            hours.insert(0, 0)
            cumulative_pnl.insert(0, 0)
        
        # Plot line
        ax.plot(hours, cumulative_pnl, linewidth=2, color='blue', marker='o', markersize=6)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax.set_title(f"{title} - Cumulative PnL")
        ax.set_xlabel("Hours")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))
        
        # Color the line based on positive/negative PnL
        if cumulative_pnl and cumulative_pnl[-1] >= 0:
            ax.plot(hours, cumulative_pnl, linewidth=2, color='green', marker='o', markersize=6)
        else:
            ax.plot(hours, cumulative_pnl, linewidth=2, color='red', marker='o', markersize=6)
    
    def _create_time_series_evolution_chart(self, output_dir: Path):
        """Create 2x2 time series evolution chart: BTC price, agent HF, net position, and YT holdings"""
        
        # Extract time series data from simulation results
        simulation_results = self.results.get("simulation_results", {})
        
        # Get BTC price history
        btc_history = simulation_results.get("btc_price_history", [])
        btc_data = []
        for i, btc_entry in enumerate(btc_history):
            # BTC data is daily, so convert day to hours (day * 24 hours/day)
            hour = i * 24.0  # Convert day index to hours
            btc_data.append({
                "hour": hour,
                "day": i,
                "btc_price": btc_entry
            })
        
        # Get agent health factor history and other agent data
        agent_health_history = simulation_results.get("agent_health_history", [])
        agent_time_series = []
        net_position_data = []
        yt_holdings_data = []
        
        # Extract data from agent health history snapshots
        for i, health_snapshot in enumerate(agent_health_history):
            # Agent data is also daily, so convert day to hours (day * 24 hours/day)
            hour = i * 24.0  # Convert day index to hours
            if health_snapshot and "agents" in health_snapshot:
                agents_list = health_snapshot["agents"]
                if agents_list:
                    # Use test_agent_03 as representative (they all have same parameters)
                    target_agent = None
                    for agent in agents_list:
                        if agent.get("agent_id") == "test_agent_03":
                            target_agent = agent
                            break
                    
                    # If we didn't find test_agent_03, use the first agent
                    if not target_agent and agents_list:
                        target_agent = agents_list[0]
                    
                    if target_agent:
                        # Extract health factor
                        agent_time_series.append({
                            "hour": hour,
                            "minute": i,
                            "health_factor": target_agent.get("health_factor", 1.25)
                        })
                        
                        # Extract net position value
                        net_position_data.append({
                            "hour": hour,
                            "net_position": target_agent.get("net_position_value", 100000)
                        })
                        
                        # Extract YT holdings (yield token value)
                        yt_holdings_data.append({
                            "hour": hour,
                            "yt_holdings": target_agent.get("yield_token_value", 64000)
                        })
        
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Time Series Evolution Analysis: Agent Behavior & Market Dynamics', 
                     fontsize=16, fontweight='bold')
        
        # Top Left: BTC Price Evolution
        if btc_data:
            hours = [d["hour"] for d in btc_data]
            prices = [d["btc_price"] for d in btc_data]
            ax1.plot(hours, prices, linewidth=2, color='orange', label='BTC Price')
            ax1.set_title('BTC Price Evolution Over Full Year')
            ax1.set_xlabel('Hours')
            ax1.set_ylabel('BTC Price ($)')
            ax1.set_xlim(0, 8760)  # Full year = 365 days * 24 hours = 8760 hours
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            print(f"📊 BTC data range: {min(hours):.1f} to {max(hours):.1f} hours ({len(hours)} points)")
        
        # Top Right: Agent Health Factor Evolution
        if agent_time_series:
            hours = [d["hour"] for d in agent_time_series]
            health_factors = [d["health_factor"] for d in agent_time_series]
            
            ax2.plot(hours, health_factors, linewidth=2, color='blue', label='Agent Health Factor')
            
            # Add threshold lines
            ax2.axhline(y=self.config.agent_initial_hf, color='green', linestyle='-', alpha=0.7, label=f'Initial HF ({self.config.agent_initial_hf})')
            ax2.axhline(y=self.config.agent_target_hf, color='orange', linestyle='--', alpha=0.7, label=f'Target HF ({self.config.agent_target_hf})')
            ax2.axhline(y=self.config.agent_rebalancing_hf, color='red', linestyle=':', alpha=0.7, label=f'Rebalancing HF ({self.config.agent_rebalancing_hf})')
            
            ax2.set_title('Agent Health Factor Evolution Over Full Year')
            ax2.set_xlabel('Hours')
            ax2.set_ylabel('Health Factor')
            ax2.set_xlim(0, 8760)  # Full year
            ax2.set_ylim(1.0, max(1.15, self.config.agent_initial_hf + 0.05))
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            print(f"📊 Agent data range: {min(hours):.1f} to {max(hours):.1f} hours ({len(hours)} points)")
        
        # Bottom Left: Net Position Value Evolution
        if net_position_data:
            hours = [d["hour"] for d in net_position_data]
            net_positions = [d["net_position"] for d in net_position_data]
            ax3.plot(hours, net_positions, linewidth=2, color='purple', label='Net Position Value')
            ax3.set_title('Net Position Value Evolution Over Full Year')
            ax3.set_xlabel('Hours')
            ax3.set_ylabel('Net Position Value ($)')
            ax3.set_xlim(0, 8760)  # Full year
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # Bottom Right: Yield Token Holdings Evolution
        if yt_holdings_data:
            hours = [d["hour"] for d in yt_holdings_data]
            yt_holdings = [d["yt_holdings"] for d in yt_holdings_data]
            ax4.plot(hours, yt_holdings, linewidth=2, color='green', label='YT Holdings')
            ax4.set_title('Yield Token Holdings Evolution Over Full Year')
            ax4.set_xlabel('Hours')
            ax4.set_ylabel('YT Holdings')
            ax4.set_xlim(0, 8760)  # Full year
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        # Handle empty data cases
        for ax, title in [(ax1, "BTC Price"), (ax2, "Agent Health Factor"), 
                         (ax3, "Net Position"), (ax4, "YT Holdings")]:
            if not any([btc_data, agent_time_series, net_position_data, yt_holdings_data]):
                ax.text(0.5, 0.5, f"No {title.lower()} data available", 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_dir / "time_series_evolution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Time Series Analysis: BTC points: {len(btc_data)}, Agent snapshots: {len(agent_time_series)}")
    
    def _create_pool_balance_evolution_chart(self, output_dir: Path):
        """Create 2x1 pool rebalancer balance evolution chart: absolute amounts and percentage composition"""
        
        # Extract rebalancer events from detailed logs
        rebalancer_events = []
        for log_entry in self.results.get("detailed_logs", []):
            if log_entry.get("event_type") in ["ALM_REBALANCE", "ALGO_REBALANCE"]:
                data = log_entry.get("data", {})
                pool_before = data.get("pool_state_before", {})
                pool_after = data.get("pool_state_after", {})
                
                rebalancer_events.append({
                    "hour": log_entry.get("hour", 0),
                    "minute": log_entry.get("minute", 0),
                    "rebalancer_type": data.get("rebalancer", "Unknown"),
                    "moet_balance_before": pool_before.get("alm_moet_balance", 500000),
                    "yt_balance_before": pool_before.get("alm_yt_balance", 0),
                    "moet_balance_after": pool_after.get("alm_moet_balance", 500000),
                    "yt_balance_after": pool_after.get("alm_yt_balance", 0)
                })
        
        # Create time series by carrying balances forward between events
        hours = []
        moet_balances = []
        yt_balances = []
        
        # Initial balances (single shared pool starts with $500K MOET, $0 YT)
        current_moet = 500000
        current_yt = 0
        
        # Create hourly data points for full year
        for hour in range(8761):  # 0 to 8760 hours (full year)
            # Check if there's a rebalancer event at this hour
            for event in rebalancer_events:
                if abs(event["hour"] - hour) < 0.5:  # Within 30 minutes of this hour
                    # Update balances based on event (use after-event balances)
                    current_moet = event["moet_balance_after"]
                    current_yt = event["yt_balance_after"]
                    break
            
            hours.append(hour)
            moet_balances.append(current_moet)
            yt_balances.append(current_yt)
        
        # Create 2x1 subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Pool Rebalancer Balance Evolution: MOET vs YT Holdings', 
                     fontsize=16, fontweight='bold')
        
        # Top: Absolute dollar amounts (line chart)
        ax1.plot(hours, moet_balances, linewidth=3, color='gold', label='MOET Balance', marker='o', markersize=4)
        ax1.plot(hours, yt_balances, linewidth=3, color='green', label='YT Balance', marker='s', markersize=4)
        
        ax1.set_title('Absolute Balance Amounts Over Time')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Balance ($)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add rebalancer event markers (sample every 50th event to reduce clutter)
        sampled_rebalancer_events = []
        for i, event in enumerate(rebalancer_events):
            # Show every 50th event OR significant balance changes (>$100k change)
            balance_change = abs(event["moet_balance_after"] - event["moet_balance_before"])
            if i % 50 == 0 or balance_change > 100000:
                sampled_rebalancer_events.append(event)
        
        for event in sampled_rebalancer_events:
            color = 'blue' if event["rebalancer_type"] == "ALM" else 'red'
            ax1.axvline(x=event["hour"], color=color, linestyle='--', alpha=0.4, linewidth=1)
        
        # Bottom: Percentage composition (100% stacked area chart)
        total_balances = [moet + yt for moet, yt in zip(moet_balances, yt_balances)]
        moet_percentages = [moet / total * 100 if total > 0 else 50 for moet, total in zip(moet_balances, total_balances)]
        yt_percentages = [yt / total * 100 if total > 0 else 50 for yt, total in zip(yt_balances, total_balances)]
        
        # Create stacked area chart
        ax2.fill_between(hours, 0, moet_percentages, color='gold', alpha=0.7, label='MOET %')
        ax2.fill_between(hours, moet_percentages, 100, color='green', alpha=0.7, label='YT %')
        
        ax2.set_title('Pool Composition (Percentage)')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add rebalancer event markers to bottom chart too (use same sampled events)
        for event in sampled_rebalancer_events:
            color = 'blue' if event["rebalancer_type"] == "ALM" else 'red'
            ax2.axvline(x=event["hour"], color=color, linestyle='--', alpha=0.4, linewidth=1)
        
        plt.tight_layout()
        plt.savefig(output_dir / "pool_balance_evolution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Pool Balance Evolution: {len(rebalancer_events)} rebalancer events tracked over {len(hours)} hours")
    
    def _create_agent_slippage_analysis_chart(self, output_dir: Path):
        """Create 2x2 agent slippage and rebalance analysis chart"""
        
        # Extract rebalancing events from simulation results
        simulation_results = self.results.get("simulation_results", {})
        rebalancing_events = simulation_results.get("rebalancing_events", [])
        
        if not rebalancing_events:
            print("⚠️ No rebalancing events found for slippage analysis")
            return
        
        # Extract slippage costs, rebalance amounts, and timestamps
        slippage_costs = []
        rebalance_amounts = []
        rebalance_times = []
        
        for event in rebalancing_events:
            slippage = event.get("slippage_cost", 0)
            moet_raised = event.get("moet_raised", 0)
            minute = event.get("minute", 0)
            hour = minute / 60.0
            
            slippage_costs.append(slippage)
            rebalance_amounts.append(moet_raised)
            rebalance_times.append(hour)
        
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Agent Rebalancing Analysis: Slippage Costs & Activity Patterns', 
                     fontsize=16, fontweight='bold')
        
        # Top Left: Distribution of slippage costs (histogram)
        ax1.hist(slippage_costs, bins=50, color='red', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.set_title('Distribution of Slippage Costs')
        ax1.set_xlabel('Slippage Cost ($)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Add slippage statistics
        mean_slippage = sum(slippage_costs) / len(slippage_costs)
        max_slippage = max(slippage_costs)
        median_slippage = sorted(slippage_costs)[len(slippage_costs)//2]
        
        stats_text = f'Mean: ${mean_slippage:.3f}\\nMax: ${max_slippage:.3f}\\nMedian: ${median_slippage:.3f}'
        ax1.text(0.75, 0.75, stats_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        # Top Right: Average slippage cost over time
        hourly_slippage = {}
        hourly_counts = {}
        
        for hour, slippage in zip(rebalance_times, slippage_costs):
            hour_bucket = int(hour)
            if hour_bucket not in hourly_slippage:
                hourly_slippage[hour_bucket] = 0
                hourly_counts[hour_bucket] = 0
            hourly_slippage[hour_bucket] += slippage
            hourly_counts[hour_bucket] += 1
        
        hours_with_data = sorted(hourly_slippage.keys())
        avg_slippages = [hourly_slippage[h] / hourly_counts[h] for h in hours_with_data]
        
        ax2.plot(hours_with_data, avg_slippages, linewidth=3, color='blue', marker='o', markersize=6)
        ax2.set_title('Average Slippage Cost Over Time')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('Avg Slippage ($)')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.3f}'))
        
        # Bottom Left: Distribution of rebalance amounts
        ax3.hist(rebalance_amounts, bins=50, color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.set_title('Distribution of Rebalance Amounts')
        ax3.set_xlabel('MOET Raised ($)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # Add rebalance statistics
        mean_amount = sum(rebalance_amounts) / len(rebalance_amounts)
        max_amount = max(rebalance_amounts)
        median_amount = sorted(rebalance_amounts)[len(rebalance_amounts)//2]
        
        amount_stats = f'Mean: ${mean_amount:.0f}\\nMax: ${max_amount:.0f}\\nMedian: ${median_amount:.0f}'
        ax3.text(0.75, 0.75, amount_stats, transform=ax3.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        # Bottom Right: Average rebalance amount over time
        hourly_amounts = {}
        
        for hour, amount in zip(rebalance_times, rebalance_amounts):
            hour_bucket = int(hour)
            if hour_bucket not in hourly_amounts:
                hourly_amounts[hour_bucket] = 0
            hourly_amounts[hour_bucket] += amount
        
        # Calculate averages using the same hourly_counts from slippage
        avg_amounts = [hourly_amounts[h] / hourly_counts[h] for h in hours_with_data]
        
        ax4.plot(hours_with_data, avg_amounts, linewidth=3, color='orange', marker='s', markersize=6)
        ax4.set_title('Average Rebalance Amount Over Time')
        ax4.set_xlabel('Hours')
        ax4.set_ylabel('Avg MOET Raised ($)')
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(output_dir / "agent_slippage_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Rebalancing Analysis: {len(slippage_costs):,} events, avg slippage ${mean_slippage:.3f}, avg amount ${mean_amount:.0f}")
    
    def _print_detailed_arbitrage_analysis(self):
        """Print detailed arbitrage analysis including YT sales and pool state changes"""
        
        print(f"\n🔍 DETAILED ARBITRAGE ANALYSIS")
        print("=" * 50)
        
        # Calculate total YT sold by agents
        agent_perf = self.results.get("agent_performance", {}).get("agent_details", [])
        total_yt_sold = 0
        total_slippage = 0
        
        for agent in agent_perf:
            # Use actual rebalancing count from JSON data (no longer triple-counted)
            rebalance_count = agent.get("rebalance_count", 0)
            actual_rebalances = rebalance_count  # Use actual count from simulation results
            slippage = agent.get("total_slippage", 0)
            
            # Use actual YT sold from agent data if available, otherwise estimate
            actual_yt_sold = agent.get("total_yield_sold", 0)
            if actual_yt_sold > 0:
                total_yt_sold += actual_yt_sold
                print(f"   🔍 DEBUG: Agent {agent.get('agent_id', 'unknown')} actual YT sold: ${actual_yt_sold:,.2f}")
            else:
                # Use more accurate estimation based on current system (~$930 per rebalance)
                estimated_yt_per_rebalance = 930  # Updated estimate based on fixed target HF system
                estimated_yt_sold = actual_rebalances * estimated_yt_per_rebalance
                total_yt_sold += estimated_yt_sold
                print(f"   🔍 DEBUG: Agent {agent.get('agent_id', 'unknown')} estimated YT sold: ${estimated_yt_sold:,.2f} ({actual_rebalances} rebalances)")
            
            total_slippage += slippage
        
        # Use actual rebalancing count from simulation data (no triple-counting needed)
        total_actual_rebalances = sum(a.get("rebalance_count", 0) for a in agent_perf)
        
        print(f"💰 Agent YT Sales:")
        print(f"   Total YT Sold for Agent Rebalancing: ${total_yt_sold:,.2f}")
        print(f"   Total Agent Slippage Costs: ${total_slippage:,.2f}")
        print(f"   Total Actual Rebalances: {total_actual_rebalances} (corrected from {sum(a.get('rebalance_count', 0) for a in agent_perf)} events)")
        print(f"   Average Slippage per Rebalance: ${total_slippage / max(1, total_actual_rebalances):.2f}")
        
        # Pool arbitrage details
        arbitrage = self.results.get("pool_arbitrage_analysis", {})
        sim_results = self.results.get("simulation_results", {})
        
        if arbitrage.get("enabled") and arbitrage.get("total_rebalances", 0) > 0:
            print(f"\n🔄 Pool Arbitrage Details:")
            print(f"   ALM Rebalances Executed: {arbitrage.get('alm_rebalances', 0)}")
            print(f"   Algo Rebalances Executed: {arbitrage.get('algo_rebalances', 0)}")
            
            # Look for detailed pool rebalancing events in the logs
            pool_events = []
            for log_entry in self.results.get("detailed_logs", []):
                if log_entry.get("event_type") in ["ALM_REBALANCE", "ALGO_REBALANCE"]:
                    pool_events.append(log_entry)
            
            if pool_events:
                print(f"\n📊 Pool Arbitrage Events:")
                for i, event in enumerate(pool_events, 1):
                    data = event.get("data", {})
                    minute = event.get("minute", 0)
                    hour = minute / 60
                    rebalancer_type = data.get("rebalancer", "unknown")
                    
                    print(f"   Event {i} - {rebalancer_type} Rebalancer at Hour {hour:.1f}:")
                    
                    # Pool state before/after
                    pool_before = data.get("pool_state_before", {})
                    pool_after = data.get("pool_state_after", {})
                    true_price = data.get("true_yt_price", 0)
                    
                    if pool_before and pool_after:
                        print(f"     📊 BEFORE Arbitrage:")
                        print(f"       Pool YT Price: ${pool_before.get('pool_yt_price', 0):.6f}")
                        print(f"       True YT Price: ${true_price:.6f}")
                        print(f"       Deviation: {pool_before.get('deviation_bps', 0):.1f} bps")
                        print(f"       ALM MOET Balance: ${pool_before.get('alm_moet_balance', 0):,.0f}")
                        print(f"       ALM YT Balance: ${pool_before.get('alm_yt_balance', 0):.0f}")
                        
                        print(f"     📊 AFTER Arbitrage:")
                        print(f"       Pool YT Price: ${pool_after.get('pool_yt_price', 0):.6f}")
                        print(f"       True YT Price: ${true_price:.6f}")
                        print(f"       Deviation: {pool_after.get('deviation_bps', 0):.1f} bps")
                        print(f"       ALM MOET Balance: ${pool_after.get('alm_moet_balance', 0):,.0f}")
                        print(f"       ALM YT Balance: ${pool_after.get('alm_yt_balance', 0):.0f}")
                        
                        # Calculate changes
                        moet_change = pool_after.get('alm_moet_balance', 0) - pool_before.get('alm_moet_balance', 0)
                        yt_change = pool_after.get('alm_yt_balance', 0) - pool_before.get('alm_yt_balance', 0)
                        price_change = pool_after.get('pool_yt_price', 0) - pool_before.get('pool_yt_price', 0)
                        
                        print(f"     💰 CHANGES:")
                        print(f"       MOET Balance Change: ${moet_change:+,.2f}")
                        print(f"       YT Balance Change: {yt_change:+.2f}")
                        print(f"       Pool Price Change: ${price_change:+.6f}")
                    
                    # Look for step-by-step arbitrage details in params
                    params = data.get("params", {})
                    if params:
                        print(f"     🔄 ARBITRAGE DETAILS:")
                        print(f"       Amount Traded: ${params.get('amount', 0):.2f}")
                        print(f"       Arbitrage Profit: ${params.get('profit', 0):.2f}")
                    
                    print()  # Empty line between events
            
            # Pool state snapshots analysis
            snapshots = sim_results.get("pool_state_snapshots", [])
            if snapshots:
                # Find snapshots around arbitrage events
                significant_deviations = [s for s in snapshots if abs(s.get("deviation_bps", 0)) > 10]
                
                if significant_deviations:
                    print(f"\n📈 Significant Pool Deviations (>10 bps):")
                    for snapshot in significant_deviations[:5]:  # Show first 5
                        hour = snapshot.get("hour", 0)
                        deviation = snapshot.get("deviation_bps", 0)
                        pool_price = snapshot.get("pool_yt_price", 0)
                        true_price = snapshot.get("true_yt_price", 0)
                        
                        print(f"     Hour {hour:.1f}: Pool=${pool_price:.6f}, True=${true_price:.6f}, Deviation={deviation:.1f} bps")
        
        else:
            print(f"\n🔄 Pool Arbitrage: No arbitrage events occurred")
            
            # Still show pool accuracy
            snapshots = sim_results.get("pool_state_snapshots", [])
            if snapshots:
                max_dev = max(abs(s.get("deviation_bps", 0)) for s in snapshots)
                avg_dev = sum(abs(s.get("deviation_bps", 0)) for s in snapshots) / len(snapshots)
                print(f"     Max Pool Deviation: {max_dev:.1f} bps")
                print(f"     Avg Pool Deviation: {avg_dev:.1f} bps")
    
    def _print_test_summary(self):
        """Print comprehensive test summary"""
        
        print("\n📊 TEST SUMMARY")
        print("=" * 50)
        
        # Agent performance
        agent_perf = self.results.get("agent_performance", {}).get("summary", {})
        if agent_perf:
            print(f"👥 Agent Performance:")
            print(f"   Total Agents: {agent_perf.get('total_agents', 0)}")
            print(f"   Survived: {agent_perf.get('survived_agents', 0)} ({agent_perf.get('survival_rate', 0):.1%})")
            print(f"   Total Rebalances: {agent_perf.get('total_rebalances', 0)}")
            print(f"   Total Slippage Costs: ${agent_perf.get('total_slippage_costs', 0):,.2f}")
            print(f"   Avg Final HF: {agent_perf.get('avg_final_hf', 0):.3f}")
        
        # Pool arbitrage
        arbitrage = self.results.get("pool_arbitrage_analysis", {})
        if arbitrage.get("enabled"):
            print(f"\n🔄 Pool Arbitrage:")
            print(f"   ALM Rebalances: {arbitrage.get('alm_rebalances', 0)}")
            print(f"   Algo Rebalances: {arbitrage.get('algo_rebalances', 0)}")
            print(f"   Total Profit: ${arbitrage.get('total_profit', 0):,.2f}")
        
        # Pool evolution
        pool_evolution = self.results.get("pool_evolution_analysis", {})
        if pool_evolution:
            print(f"\n📈 Pool Price Accuracy:")
            print(f"   Max Deviation: {pool_evolution.get('max_price_deviation_bps', 0):.1f} bps")
            print(f"   Avg Deviation: {pool_evolution.get('avg_price_deviation_bps', 0):.1f} bps")
            print(f"   Times Above Threshold: {pool_evolution.get('times_above_threshold', 0)}")
            print(f"   Pool Accuracy Score: {pool_evolution.get('pool_accuracy_score', 1.0):.3f}")
        
        # Detailed Arbitrage Analysis
        self._print_detailed_arbitrage_analysis()
        
        # Simulation results summary
        sim_results = self.results.get("simulation_results", {})
        if sim_results:
            print(f"\n🎯 Simulation Results:")
            survival_stats = sim_results.get("survival_statistics", {})
            if survival_stats:
                print(f"   Survival Rate: {survival_stats.get('survival_rate', 0):.1%}")
                print(f"   Liquidations: {survival_stats.get('liquidations', 0)}")
                print(f"   Emergency Actions: {survival_stats.get('emergency_actions', 0)}")
    
    def find_agent_limit_for_500k_pool(self) -> Dict[str, Any]:
        """
        Find the number of agents that causes a single rebalance to use the entire $500k pool
        
        This method runs incremental tests with increasing agent counts until we find
        the breaking point where pool rebalancer capacity is exhausted.
        
        Returns:
            Dict with results of the limit-finding test
        """
        
        print("\n🔍 FINDING AGENT LIMIT FOR $500K POOL CAPACITY")
        print("=" * 60)
        print("Testing incremental agent counts to find where single rebalance uses full $500k...")
        print()
        
        limit_results = {
            "test_metadata": {
                "test_name": f"{self.config.test_name}_Agent_Limit_Test",
                "timestamp": datetime.now().isoformat(),
                "max_agents_tested": self.config.max_agents_to_test,
                "pool_capacity": 500_000
            },
            "agent_test_results": [],
            "breaking_point": None,
            "recommendations": {}
        }
        
        # Test with increasing agent counts
        test_agent_counts = [50, 75, 100, 125, 150, 175, 200]  # Incremental testing
        
        for agent_count in test_agent_counts:
            if agent_count > self.config.max_agents_to_test:
                break
                
            print(f"🧪 Testing with {agent_count} agents...")
            
            # Temporarily modify config
            original_num_agents = self.config.num_agents
            original_test_name = self.config.test_name
            
            self.config.num_agents = agent_count
            self.config.test_name = f"Agent_Limit_Test_{agent_count}"
            
            try:
                # Run shortened test (6 hours instead of 36 for speed)
                original_duration_hours = self.config.simulation_duration_hours
                original_duration_minutes = self.config.simulation_duration_minutes
                
                self.config.simulation_duration_hours = 6
                self.config.simulation_duration_minutes = 6 * 60
                
                # Run test
                test_results = self.run_test()
                
                # Restore original duration
                self.config.simulation_duration_hours = original_duration_hours
                self.config.simulation_duration_minutes = original_duration_minutes
                
                # Analyze results for pool exhaustion
                pool_analysis = self._analyze_pool_exhaustion(test_results, agent_count)
                
                limit_results["agent_test_results"].append({
                    "agent_count": agent_count,
                    "pool_exhaustion_detected": pool_analysis["exhaustion_detected"],
                    "max_single_rebalance": pool_analysis["max_single_rebalance"],
                    "pool_utilization_peak": pool_analysis["peak_utilization"],
                    "survival_rate": test_results.get("agent_performance", {}).get("summary", {}).get("survival_rate", 0),
                    "total_rebalances": test_results.get("agent_performance", {}).get("summary", {}).get("total_rebalances", 0)
                })
                
                print(f"   Results: Max single rebalance: ${pool_analysis['max_single_rebalance']:,.0f}")
                print(f"   Pool exhaustion: {'YES' if pool_analysis['exhaustion_detected'] else 'NO'}")
                print(f"   Survival rate: {test_results.get('agent_performance', {}).get('summary', {}).get('survival_rate', 0):.1%}")
                
                # Check if we found the breaking point
                if pool_analysis["exhaustion_detected"] and not limit_results["breaking_point"]:
                    limit_results["breaking_point"] = {
                        "agent_count": agent_count,
                        "max_single_rebalance": pool_analysis["max_single_rebalance"],
                        "estimated_tvl_supported": agent_count * 100_000,  # Assuming $100k per agent
                        "capital_efficiency_ratio": (agent_count * 100_000) / 500_000
                    }
                    print(f"🎯 BREAKING POINT FOUND: {agent_count} agents cause pool exhaustion!")
                    break
                    
            except Exception as e:
                print(f"❌ Test failed with {agent_count} agents: {e}")
                limit_results["agent_test_results"].append({
                    "agent_count": agent_count,
                    "error": str(e),
                    "test_failed": True
                })
            
            finally:
                # Restore original config
                self.config.num_agents = original_num_agents
                self.config.test_name = original_test_name
        
        # Generate recommendations
        if limit_results["breaking_point"]:
            breaking_point = limit_results["breaking_point"]
            limit_results["recommendations"] = {
                "safe_agent_count": max(50, breaking_point["agent_count"] - 25),
                "recommended_pool_size": breaking_point["max_single_rebalance"] * 1.5,
                "capital_efficiency": f"{breaking_point['capital_efficiency_ratio']:.1f}:1",
                "max_tvl_supported": f"${breaking_point['estimated_tvl_supported']:,.0f}"
            }
        
        # Save results
        self._save_limit_test_results(limit_results)
        
        print("\n✅ Agent limit test completed!")
        return limit_results
    
    def _analyze_pool_exhaustion(self, test_results: Dict, agent_count: int) -> Dict[str, Any]:
        """Analyze test results for signs of pool exhaustion"""
        
        analysis = {
            "exhaustion_detected": False,
            "max_single_rebalance": 0.0,
            "peak_utilization": 0.0,
            "exhaustion_indicators": []
        }
        
        # Look for pool rebalancing activity
        pool_activity = test_results.get("simulation_results", {}).get("pool_rebalancing_activity", {})
        
        if pool_activity and pool_activity.get("events"):
            rebalance_amounts = []
            for event in pool_activity["events"]:
                amount = event.get("params", {}).get("amount", 0)
                if amount > 0:
                    rebalance_amounts.append(amount)
            
            if rebalance_amounts:
                analysis["max_single_rebalance"] = max(rebalance_amounts)
                analysis["peak_utilization"] = analysis["max_single_rebalance"] / 500_000
                
                # Check for exhaustion indicators
                if analysis["max_single_rebalance"] > 400_000:  # 80% of pool capacity
                    analysis["exhaustion_detected"] = True
                    analysis["exhaustion_indicators"].append("Single rebalance >80% of pool capacity")
                
                if analysis["peak_utilization"] > 0.9:  # 90% utilization
                    analysis["exhaustion_detected"] = True
                    analysis["exhaustion_indicators"].append("Peak utilization >90%")
        
        return analysis
    
    def _save_limit_test_results(self, results: Dict):
        """Save agent limit test results"""
        output_dir = Path("tidal_protocol_sim/results") / f"{self.config.test_name}_Agent_Limits"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / f"agent_limit_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self._convert_for_json(results), f, indent=2)
        
        print(f"📁 Agent limit test results saved to: {results_path}")
    
    def _create_net_apy_analysis_chart(self, output_dir: Path):
        """Create Net APY analysis chart: Agent performance vs BTC hold strategy"""
        
        # Extract time series data from simulation results
        simulation_results = self.results.get("simulation_results", {})
        
        # Get BTC price history
        btc_history = simulation_results.get("btc_price_history", [])
        btc_data = []
        for i, btc_entry in enumerate(btc_history):
            hour = i * 24.0  # Convert day index to hours
            btc_data.append({
                "hour": hour,
                "day": i,
                "btc_price": btc_entry
            })
        
        # Get agent health factor history and net position data
        agent_health_history = simulation_results.get("agent_health_history", [])
        agent_data = []
        
        # Extract data from agent health history snapshots
        for i, health_snapshot in enumerate(agent_health_history):
            hour = i * 24.0  # Convert day index to hours
            if health_snapshot and "agents" in health_snapshot:
                agents_list = health_snapshot["agents"]
                if agents_list:
                    # Use test_agent_03 as representative
                    target_agent = None
                    for agent in agents_list:
                        if agent.get("agent_id") == "test_agent_03":
                            target_agent = agent
                            break
                    
                    if not target_agent and agents_list:
                        target_agent = agents_list[0]
                    
                    if target_agent:
                        # Get actual initial position from first snapshot or use current as fallback
                        actual_initial_position = 100000  # Default fallback
                        if i == 0 and agent_health_history:
                            # Use the first snapshot's net_position_value as the true initial position
                            actual_initial_position = target_agent.get("net_position_value", 100000)
                        elif agent_data:
                            # Use the initial position from the first data point
                            actual_initial_position = agent_data[0]["initial_position"]
                        
                        agent_data.append({
                            "hour": hour,
                            "day": i,
                            "net_position_value": target_agent.get("net_position_value", actual_initial_position),
                            "initial_position": actual_initial_position
                        })
        
        if not btc_data or not agent_data:
            print("⚠️  No data available for Net APY analysis")
            return
        
        # Calculate performance metrics
        initial_btc_price = btc_data[0]["btc_price"]
        initial_agent_value = agent_data[0]["net_position_value"]
        
        # Create performance time series
        hours = []
        agent_apy = []
        btc_hold_apy = []
        relative_performance = []
        
        for i, (btc_point, agent_point) in enumerate(zip(btc_data, agent_data)):
            if btc_point["hour"] == agent_point["hour"]:  # Ensure alignment
                hour = btc_point["hour"]
                days_elapsed = hour / 24.0
                
                if days_elapsed > 0:
                    # Calculate annualized returns
                    btc_return = (btc_point["btc_price"] / initial_btc_price - 1) * (365 / days_elapsed) * 100
                    agent_return = (agent_point["net_position_value"] / initial_agent_value - 1) * (365 / days_elapsed) * 100
                    
                    hours.append(hour)
                    agent_apy.append(agent_return)
                    btc_hold_apy.append(btc_return)
                    relative_performance.append(agent_return - btc_return)
        
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Net APY Analysis: Agent Performance vs BTC Hold Strategy', 
                     fontsize=16, fontweight='bold')
        
        # Top Left: Absolute Value Comparison
        btc_hold_values = [btc_data[i]["btc_price"] / initial_btc_price * initial_agent_value for i in range(len(hours))]
        agent_values = [agent_data[i]["net_position_value"] for i in range(len(hours))]
        
        ax1.plot(hours, btc_hold_values, linewidth=2, color='orange', label='BTC Hold Value', alpha=0.8)
        ax1.plot(hours, agent_values, linewidth=2, color='blue', label='Agent Net Position', alpha=0.8)
        
        ax1.set_title('Portfolio Value Comparison')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_xlim(0, 8760)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Top Right: Annualized Return Percentage (APY)
        ax2.plot(hours, btc_hold_apy, linewidth=2, color='orange', label='BTC Hold APY', alpha=0.8)
        ax2.plot(hours, agent_apy, linewidth=2, color='blue', label='Agent Strategy APY', alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax2.set_title('Annualized Percentage Yield (APY)')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('APY (%)')
        ax2.set_xlim(0, 8760)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Bottom Left: Relative Performance (Agent APY - BTC Hold APY)
        positive_mask = [x >= 0 for x in relative_performance]
        negative_mask = [x < 0 for x in relative_performance]
        
        # Plot positive and negative separately for color coding
        ax3.fill_between(hours, 0, relative_performance, 
                        where=positive_mask, color='green', alpha=0.7, 
                        interpolate=True, label='Outperformance')
        ax3.fill_between(hours, 0, relative_performance, 
                        where=negative_mask, color='red', alpha=0.7, 
                        interpolate=True, label='Underperformance')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax3.set_title('Relative Performance (Agent APY - BTC Hold APY)')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('APY Difference (%)')
        ax3.set_xlim(0, 8760)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Bottom Right: Average Outperformance
        average_outperformance = []
        running_sum = 0
        for perf in relative_performance:
            running_sum += perf / len(relative_performance)  # Average the performance
            average_outperformance.append(running_sum)
        
        ax4.plot(hours, average_outperformance, linewidth=2, color='purple', 
                 label='Average Outperformance')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.fill_between(hours, 0, average_outperformance, 
                        where=[x >= 0 for x in average_outperformance], 
                        color='green', alpha=0.3)
        ax4.fill_between(hours, 0, average_outperformance, 
                        where=[x < 0 for x in average_outperformance], 
                        color='red', alpha=0.3)
        
        ax4.set_title('Average Outperformance')
        ax4.set_xlabel('Hours')
        ax4.set_ylabel('Average APY Difference (%)')
        ax4.set_xlim(0, 8760)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "net_apy_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        if hours:
            final_agent_apy = agent_apy[-1] if agent_apy else 0
            final_btc_apy = btc_hold_apy[-1] if btc_hold_apy else 0
            final_outperformance = relative_performance[-1] if relative_performance else 0
            avg_outperformance = sum(relative_performance) / len(relative_performance) if relative_performance else 0
            
            print(f"📊 Net APY Analysis:")
            print(f"   Final Agent APY: {final_agent_apy:.2f}%")
            print(f"   Final BTC Hold APY: {final_btc_apy:.2f}%")
            print(f"   Final Outperformance: {final_outperformance:.2f}%")
            print(f"   Average Outperformance: {avg_outperformance:.2f}%")
    
    def _create_yield_strategy_comparison_chart(self, output_dir: Path):
        """Create Yield Strategy Comparison chart: Tidal Protocol vs Base 10% APR yield"""
        
        # Extract time series data from simulation results
        simulation_results = self.results.get("simulation_results", {})
        
        # Get BTC price history
        btc_history = simulation_results.get("btc_price_history", [])
        btc_data = []
        for i, btc_entry in enumerate(btc_history):
            hour = i * 24.0  # Convert day index to hours
            btc_data.append({
                "hour": hour,
                "day": i,
                "btc_price": btc_entry
            })
        
        # Get agent health factor history and net position data
        agent_health_history = simulation_results.get("agent_health_history", [])
        agent_data = []
        
        # Extract data from agent health history snapshots
        for i, health_snapshot in enumerate(agent_health_history):
            hour = i * 24.0  # Convert day index to hours
            if health_snapshot and "agents" in health_snapshot:
                agents_list = health_snapshot["agents"]
                if agents_list:
                    # Use first agent as representative
                    target_agent = agents_list[0]
                    
                    if target_agent:
                        agent_data.append({
                            "hour": hour,
                            "day": i,
                            "net_position_value": target_agent.get("net_position_value", 100000),
                            "btc_price": btc_data[i]["btc_price"] if i < len(btc_data) else btc_data[-1]["btc_price"]
                        })
        
        if not btc_data or not agent_data:
            print("⚠️  No data available for Yield Strategy Comparison")
            return
        
        # Calculate performance metrics
        initial_btc_price = btc_data[0]["btc_price"]
        initial_agent_value = agent_data[0]["net_position_value"]
        base_apr = 0.10  # 10% APR
        
        # Create performance time series
        hours = []
        tidal_yield_adjusted = []  # Net Position Value / BTC Price
        base_yield_value = []      # Just 10% APR compounded
        base_yield_apy = []        # Annualized base yield
        tidal_yield_apy = []       # Annualized Tidal yield
        relative_performance = []   # Tidal vs Base yield difference
        
        for i, (btc_point, agent_point) in enumerate(zip(btc_data, agent_data)):
            if btc_point["hour"] == agent_point["hour"]:  # Ensure alignment
                hour = btc_point["hour"]
                days_elapsed = hour / 24.0
                
                if days_elapsed >= 0:
                    # Tidal Protocol: Net Position Value adjusted for BTC price changes
                    tidal_value_btc_adjusted = agent_point["net_position_value"] / btc_point["btc_price"] * initial_btc_price
                    
                    # Base Yield: Simple 10% APR compounded daily
                    years_elapsed = days_elapsed / 365.0
                    base_value = initial_agent_value * (1 + base_apr) ** years_elapsed
                    
                    hours.append(hour)
                    tidal_yield_adjusted.append(tidal_value_btc_adjusted)
                    base_yield_value.append(base_value)
                    
                    if days_elapsed > 0:
                        # Calculate annualized returns
                        tidal_return = (tidal_value_btc_adjusted / initial_agent_value - 1) * (365 / days_elapsed) * 100
                        base_return = base_apr * 100  # Always 10%
                        
                        tidal_yield_apy.append(tidal_return)
                        base_yield_apy.append(base_return)
                        relative_performance.append(tidal_return - base_return)
                    else:
                        tidal_yield_apy.append(0)
                        base_yield_apy.append(0)
                        relative_performance.append(0)
        
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Yield Strategy Comparison: Tidal Protocol vs Base 10% APR Yield', 
                     fontsize=16, fontweight='bold')
        
        # Top Left: Absolute Value Comparison (BTC-adjusted)
        ax1.plot(hours, base_yield_value, linewidth=2, color='green', label='Base 10% APR Yield', alpha=0.8)
        ax1.plot(hours, tidal_yield_adjusted, linewidth=2, color='blue', label='Tidal Protocol (BTC-adjusted)', alpha=0.8)
        
        ax1.set_title('Portfolio Value Comparison (BTC-Price Adjusted)')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_xlim(0, 8760)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Top Right: Annualized Yield Comparison
        ax2.plot(hours, base_yield_apy, linewidth=2, color='green', label='Base 10% APR', alpha=0.8)
        ax2.plot(hours, tidal_yield_apy, linewidth=2, color='blue', label='Tidal Protocol APY', alpha=0.8)
        ax2.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10% Target')
        
        ax2.set_title('Annualized Yield Comparison')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('APY (%)')
        ax2.set_xlim(0, 8760)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Bottom Left: Relative Performance (Tidal APY - Base APY)
        positive_mask = [x >= 0 for x in relative_performance]
        negative_mask = [x < 0 for x in relative_performance]
        
        # Plot positive and negative separately for color coding
        ax3.fill_between(hours, 0, relative_performance, 
                        where=positive_mask, color='green', alpha=0.7, 
                        interpolate=True, label='Tidal Outperformance')
        ax3.fill_between(hours, 0, relative_performance, 
                        where=negative_mask, color='red', alpha=0.7, 
                        interpolate=True, label='Tidal Underperformance')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax3.set_title('Relative Performance (Tidal APY - Base 10% APR)')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('APY Difference (%)')
        ax3.set_xlim(0, 8760)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Bottom Right: Average Yield Advantage
        average_advantage = []
        running_sum = 0
        for perf in relative_performance:
            running_sum += perf / len(relative_performance)  # Average the performance
            average_advantage.append(running_sum)
        
        ax4.plot(hours, average_advantage, linewidth=2, color='purple', 
                 label='Average Yield Advantage')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.fill_between(hours, 0, average_advantage, 
                        where=[x >= 0 for x in average_advantage], 
                        color='green', alpha=0.3)
        ax4.fill_between(hours, 0, average_advantage, 
                        where=[x < 0 for x in average_advantage], 
                        color='red', alpha=0.3)
        
        ax4.set_title('Average Yield Advantage Over Time')
        ax4.set_xlabel('Hours')
        ax4.set_ylabel('Average APY Advantage (%)')
        ax4.set_xlim(0, 8760)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "yield_strategy_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        if hours:
            final_tidal_apy = tidal_yield_apy[-1] if tidal_yield_apy else 0
            final_base_apy = base_yield_apy[-1] if base_yield_apy else 0
            final_advantage = relative_performance[-1] if relative_performance else 0
            avg_advantage = sum(relative_performance) / len(relative_performance) if relative_performance else 0
            
            # Calculate final values
            final_tidal_value = tidal_yield_adjusted[-1] if tidal_yield_adjusted else 0
            final_base_value = base_yield_value[-1] if base_yield_value else 0
            total_advantage = (final_tidal_value / final_base_value - 1) * 100 if final_base_value > 0 else 0
            
            print(f"📊 Yield Strategy Comparison:")
            print(f"   Final Tidal Protocol APY: {final_tidal_apy:.2f}%")
            print(f"   Base 10% APR Yield: {final_base_apy:.2f}%")
            print(f"   Final APY Advantage: {final_advantage:.2f}%")
            print(f"   Average APY Advantage: {avg_advantage:.2f}%")
            print(f"   Total Value Advantage: {total_advantage:.2f}%")
    
    def _create_moet_system_analysis_chart(self, output_dir: Path):
        """Create MOET System Analysis chart: Interest Rates & Bond APRs over time"""
        
        # Extract MOET system data from simulation results
        simulation_results = self.results.get("simulation_results", {})
        moet_system_state = simulation_results.get("moet_system_state", {})
        
        if not moet_system_state.get("advanced_system_enabled"):
            print("⚠️  Advanced MOET system not enabled - skipping MOET analysis chart")
            return
        
        # Get tracking data
        tracking_data = moet_system_state.get("tracking_data", {})
        moet_rates = tracking_data.get("moet_rate_history", [])
        bond_aprs = tracking_data.get("bond_apr_history", [])
        
        # If no tracking data, create synthetic data based on final state
        if not moet_rates or not bond_aprs:
            print("⚠️  No MOET tracking data available - creating chart with final state data")
            print(f"   Debug: moet_rates length: {len(moet_rates)}, bond_aprs length: {len(bond_aprs)}")
            print(f"   Debug: tracking_data keys: {list(tracking_data.keys())}")
            
            # Try to get bond auction data from bonder system directly
            bonder_system = moet_system_state.get("bonder_system", {})
            recent_auctions = bonder_system.get("recent_auctions", [])
            auction_count = bonder_system.get("auction_history_count", 0)
            
            print(f"   Debug: Found {auction_count} total auctions, {len(recent_auctions)} recent auctions")
            if recent_auctions:
                print(f"   Debug: Recent auction APRs: {[a.get('final_apr', 0) for a in recent_auctions]}")
            
            # Get final state values
            current_rate = moet_system_state.get("current_interest_rate", 0.02)
            components = moet_system_state.get("interest_rate_components", {})
            r_floor = components.get("r_floor", 0.02)
            r_bond_cost = components.get("r_bond_cost", 0.0)
            
            # Calculate current bond APR from reserve state
            reserve_state = moet_system_state.get("reserve_state", {})
            total_supply = moet_system_state.get("total_supply", 1000000)
            total_reserves = reserve_state.get("total_reserves", 0)
            target_ratio = reserve_state.get("target_reserves_ratio", 0.10)
            
            target_reserves = total_supply * target_ratio
            current_deficit_ratio = max(0, (target_reserves - total_reserves) / target_reserves) if target_reserves > 0 else 0
            
            # Create synthetic hourly data for the full year
            hours = list(range(0, 8761, 24))  # Daily data points
            synthetic_moet_rates = [current_rate] * len(hours)
            synthetic_bond_aprs = [current_deficit_ratio] * len(hours)
            synthetic_r_floor = [r_floor] * len(hours)
            synthetic_r_bond_cost = [r_bond_cost] * len(hours)
            
            print(f"📊 MOET System Final State:")
            print(f"   Current MOET Rate: {current_rate:.2%}")
            print(f"   Current Bond APR: {current_deficit_ratio:.2%}")
            print(f"   Reserve Deficit Ratio: {current_deficit_ratio:.2%}")
        else:
            # Use actual tracking data
            hours = [r["minute"] / 60.0 for r in moet_rates]
            synthetic_moet_rates = [r["moet_interest_rate"] for r in moet_rates]
            synthetic_r_floor = [r["r_floor"] for r in moet_rates]
            synthetic_r_bond_cost = [r["r_bond_cost"] for r in moet_rates]
            synthetic_bond_aprs = [b["bond_apr"] for b in bond_aprs]
        
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MOET System Analysis: Interest Rates & Bond Auction Dynamics', 
                     fontsize=16, fontweight='bold')
        
        # Top Left: MOET Interest Rate Components
        ax1.plot(hours, synthetic_moet_rates, linewidth=3, color='blue', label='Total MOET Rate', alpha=0.8)
        ax1.plot(hours, synthetic_r_floor, linewidth=2, color='green', linestyle='--', label='r_floor (Governance)', alpha=0.7)
        ax1.plot(hours, synthetic_r_bond_cost, linewidth=2, color='red', linestyle=':', label='r_bond_cost (EMA)', alpha=0.7)
        
        ax1.set_title('MOET Interest Rate Components')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Interest Rate (%)')
        ax1.set_xlim(0, 8760)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Top Right: Bond APR Evolution
        ax2.plot(hours, synthetic_bond_aprs, linewidth=3, color='orange', label='Bond APR (Deficit-Based)')
        ax2.axhline(y=0.20, color='red', linestyle='--', alpha=0.5, label='Initial 20% Target')
        ax2.set_title('Bond Auction APR Evolution')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('Bond APR (%)')
        ax2.set_xlim(0, 8760)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Bottom Left: Rate Spread Analysis
        rate_spread = [moet - floor for moet, floor in zip(synthetic_moet_rates, synthetic_r_floor)]
        ax3.fill_between(hours, 0, rate_spread, color='purple', alpha=0.6, label='Bond Cost Premium')
        ax3.plot(hours, rate_spread, linewidth=2, color='purple', label='MOET Rate - r_floor')
        ax3.set_title('MOET Rate Premium Over Governance Floor')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('Rate Premium (%)')
        ax3.set_xlim(0, 8760)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Bottom Right: High-Precision EMA Evolution
        # Convert bond cost EMA to basis points for better visibility
        bond_cost_basis_points = [r * 10000 for r in synthetic_r_bond_cost]  # Convert to basis points (0.01%)
        
        ax4.plot(hours, bond_cost_basis_points, linewidth=3, color='darkred', label='r_bond_cost EMA')
        ax4.fill_between(hours, 0, bond_cost_basis_points, alpha=0.3, color='darkred')
        
        ax4.set_title('Bond Cost EMA Evolution (High Precision)')
        ax4.set_xlabel('Hours')
        ax4.set_ylabel('Bond Cost EMA (basis points)')
        ax4.set_xlim(0, 8760)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add precision annotation
        if bond_cost_basis_points:
            final_ema_bp = bond_cost_basis_points[-1]
            final_ema_pct = final_ema_bp / 10000
            max_ema_bp = max(bond_cost_basis_points)
            max_ema_pct = max_ema_bp / 10000
            
            precision_text = f"""EMA Precision:
Final: {final_ema_pct:.6%} ({final_ema_bp:.2f} bp)
Maximum: {max_ema_pct:.6%} ({max_ema_bp:.2f} bp)
7-day half-life smoothing"""
            
            ax4.text(0.02, 0.98, precision_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(output_dir / "moet_system_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate statistics for logging
        avg_moet_rate = sum(synthetic_moet_rates) / len(synthetic_moet_rates) if synthetic_moet_rates else 0
        max_bond_apr = max(synthetic_bond_aprs) if synthetic_bond_aprs else 0
        print(f"📊 MOET System Analysis: Avg rate {avg_moet_rate:.2%}, Max bond APR {max_bond_apr:.2%}")
    
    def _create_moet_reserve_management_chart(self, output_dir: Path):
        """Create MOET Reserve Management chart: Target vs Actual Reserves & Deficit Tracking"""
        
        # Extract MOET system data from simulation results
        simulation_results = self.results.get("simulation_results", {})
        moet_system_state = simulation_results.get("moet_system_state", {})
        
        if not moet_system_state.get("advanced_system_enabled"):
            print("⚠️  Advanced MOET system not enabled - skipping reserve management chart")
            return
        
        # Get tracking data
        tracking_data = moet_system_state.get("tracking_data", {})
        reserve_history = tracking_data.get("reserve_history", [])
        deficit_history = tracking_data.get("deficit_history", [])
        
        # If no tracking data, create synthetic data based on final state
        if not reserve_history or not deficit_history:
            print("⚠️  No reserve tracking data available - creating chart with final state data")
            print(f"   Debug: reserve_history length: {len(reserve_history)}, deficit_history length: {len(deficit_history)}")
            print(f"   Debug: tracking_data keys: {list(tracking_data.keys())}")
            
            # Try to get auction data for bond auction visualization
            bonder_system = moet_system_state.get("bonder_system", {})
            auction_count = bonder_system.get("auction_history_count", 0)
            recent_auctions = bonder_system.get("recent_auctions", [])
            
            print(f"   Debug: Found {auction_count} bond auctions")
            if recent_auctions:
                print(f"   Debug: Recent auctions: {len(recent_auctions)} events")
            
            # Get final state values
            reserve_state = moet_system_state.get("reserve_state", {})
            total_supply = moet_system_state.get("total_supply", 1000000)
            total_reserves = reserve_state.get("total_reserves", 0)
            target_ratio = reserve_state.get("target_reserves_ratio", 0.10)
            usdc_balance = reserve_state.get("usdc_balance", 0)
            usdf_balance = reserve_state.get("usdf_balance", 0)
            
            target_reserves = total_supply * target_ratio
            deficit = max(0, target_reserves - total_reserves)
            actual_ratio = total_reserves / total_supply if total_supply > 0 else 0
            
            # Create synthetic daily data for the full year
            hours = list(range(0, 8761, 24))  # Daily data points
            synthetic_target_reserves = [target_reserves] * len(hours)
            synthetic_actual_reserves = [total_reserves] * len(hours)
            synthetic_deficits = [deficit] * len(hours)
            synthetic_ratios = [actual_ratio] * len(hours)
            synthetic_usdc = [usdc_balance] * len(hours)
            synthetic_usdf = [usdf_balance] * len(hours)
            
            print(f"📊 Reserve Management Final State:")
            print(f"   Target Reserves: ${target_reserves:,.0f}")
            print(f"   Actual Reserves: ${total_reserves:,.0f}")
            print(f"   Current Deficit: ${deficit:,.0f}")
            print(f"   Reserve Ratio: {actual_ratio:.2%}")
        else:
            # Use actual tracking data
            hours = [r["minute"] / 60.0 for r in reserve_history]
            synthetic_target_reserves = [r["target_reserves"] for r in reserve_history]
            synthetic_actual_reserves = [r["actual_reserves"] for r in reserve_history]
            synthetic_ratios = [r["reserve_ratio"] for r in reserve_history]
            synthetic_deficits = [d["deficit"] for d in deficit_history]
            
            # Estimate USDC/USDF split (50/50)
            synthetic_usdc = [r / 2 for r in synthetic_actual_reserves]
            synthetic_usdf = [r / 2 for r in synthetic_actual_reserves]
        
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MOET Reserve Management: Target vs Actual Reserves & Deficit Tracking', 
                     fontsize=16, fontweight='bold')
        
        # Top Left: Target vs Actual Reserves
        ax1.plot(hours, synthetic_target_reserves, linewidth=3, color='green', label='Target Reserves (10%)', alpha=0.8)
        ax1.plot(hours, synthetic_actual_reserves, linewidth=3, color='blue', label='Actual Reserves', alpha=0.8)
        ax1.fill_between(hours, synthetic_actual_reserves, synthetic_target_reserves, 
                        where=[t > a for t, a in zip(synthetic_target_reserves, synthetic_actual_reserves)],
                        color='red', alpha=0.3, label='Reserve Deficit')
        
        ax1.set_title('Target vs Actual Reserves')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Reserves ($)')
        ax1.set_xlim(0, 8760)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Top Right: Reserve Ratio Evolution
        target_ratio_line = [0.10] * len(hours)  # 10% target
        ax2.plot(hours, target_ratio_line, linewidth=2, color='green', linestyle='--', label='Target Ratio (10%)', alpha=0.7)
        ax2.plot(hours, synthetic_ratios, linewidth=3, color='blue', label='Actual Reserve Ratio')
        ax2.fill_between(hours, 0, synthetic_ratios, color='blue', alpha=0.3)
        
        ax2.set_title('Reserve Ratio Evolution')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('Reserve Ratio (%)')
        ax2.set_xlim(0, 8760)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Bottom Left: Reserve Deficit Over Time
        ax3.fill_between(hours, 0, synthetic_deficits, color='red', alpha=0.6, label='Reserve Deficit')
        ax3.plot(hours, synthetic_deficits, linewidth=2, color='darkred', label='Deficit Amount')
        ax3.set_title('Reserve Deficit Over Time')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('Deficit ($)')
        ax3.set_xlim(0, 8760)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Bottom Right: USDC/USDF Reserve Composition
        ax4.stackplot(hours, synthetic_usdc, synthetic_usdf, 
                     labels=['USDC Reserves', 'USDF Reserves'],
                     colors=['gold', 'lightblue'], alpha=0.7)
        ax4.plot(hours, synthetic_actual_reserves, linewidth=2, color='black', 
                linestyle='--', label='Total Reserves', alpha=0.8)
        
        ax4.set_title('Reserve Composition (USDC/USDF)')
        ax4.set_xlabel('Hours')
        ax4.set_ylabel('Reserve Amount ($)')
        ax4.set_xlim(0, 8760)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "moet_reserve_management.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate and print statistics
        if synthetic_deficits:
            max_deficit = max(synthetic_deficits)
            avg_deficit = sum(synthetic_deficits) / len(synthetic_deficits)
            min_ratio = min(synthetic_ratios) if synthetic_ratios else 0
            avg_ratio = sum(synthetic_ratios) / len(synthetic_ratios) if synthetic_ratios else 0
            
            print(f"📊 Reserve Management: Max deficit ${max_deficit:,.0f}, Avg ratio {avg_ratio:.2%}")

    def _create_redeemer_analysis_chart(self, output_dir: Path):
        """Create comprehensive Redeemer system analysis chart (2x2 layout)"""
        
        # Extract MOET system data
        simulation_results = self.results.get("simulation_results", {})
        moet_system_state = simulation_results.get("moet_system_state", {})
        
        if not moet_system_state:
            print("⚠️  No MOET system data available for Redeemer analysis")
            return
            
        # Get redeemer data from tracking
        redeemer_data = moet_system_state.get("redeemer_system", {})
        tracking_data = moet_system_state.get("tracking_data", {})
        
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Enhanced Redeemer System Analysis', fontsize=16, fontweight='bold')
        
        # Top Left: Reserve Balance Evolution (USDC vs USDF)
        reserve_history = tracking_data.get("reserve_history", [])
        if reserve_history:
            hours = [r.get("hour", 0) for r in reserve_history]
            usdc_balances = [r.get("usdc_balance", 0) for r in reserve_history]
            usdf_balances = [r.get("usdf_balance", 0) for r in reserve_history]
            total_reserves = [u + s for u, s in zip(usdc_balances, usdf_balances)]
            
            ax1.plot(hours, usdc_balances, linewidth=2, color='blue', label='USDC Reserves', alpha=0.8)
            ax1.plot(hours, usdf_balances, linewidth=2, color='green', label='USDF Reserves', alpha=0.8)
            ax1.plot(hours, total_reserves, linewidth=2, color='purple', label='Total Reserves', alpha=0.8)
            
            ax1.set_title('Reserve Balance Evolution')
            ax1.set_xlabel('Hours')
            ax1.set_ylabel('Reserve Balance ($)')
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No Reserve History Data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Reserve Balance Evolution')
        
        # Top Right: Pool Weight Deviation (50/50 target)
        if reserve_history:
            weight_deviations = []
            usdc_ratios = []
            usdf_ratios = []
            
            for r in reserve_history:
                total = r.get("usdc_balance", 0) + r.get("usdf_balance", 0)
                if total > 0:
                    usdc_ratio = r.get("usdc_balance", 0) / total
                    usdf_ratio = r.get("usdf_balance", 0) / total
                    deviation = abs(usdc_ratio - 0.5)  # Deviation from 50/50
                    
                    usdc_ratios.append(usdc_ratio * 100)
                    usdf_ratios.append(usdf_ratio * 100)
                    weight_deviations.append(deviation * 100)  # Convert to percentage
                else:
                    usdc_ratios.append(50)
                    usdf_ratios.append(50)
                    weight_deviations.append(0)
            
            ax2.plot(hours, usdc_ratios, linewidth=2, color='blue', label='USDC %', alpha=0.8)
            ax2.plot(hours, usdf_ratios, linewidth=2, color='green', label='USDF %', alpha=0.8)
            ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Target (50%)')
            
            # Add tolerance band (±2%)
            ax2.fill_between(hours, 48, 52, color='gray', alpha=0.2, label='Tolerance Band (±2%)')
            
            ax2.set_title('Pool Weight Distribution')
            ax2.set_xlabel('Hours')
            ax2.set_ylabel('Asset Percentage (%)')
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No Weight Data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Pool Weight Distribution')
        
        # Bottom Left: Fee Collection Analysis
        total_fees_collected = redeemer_data.get("total_fees_collected", 0)
        fee_history_count = redeemer_data.get("fee_history_count", 0)
        
        # Create fee metrics
        fee_metrics = ['Total Fees\nCollected', 'Fee Events\nCount', 'Avg Fee\nper Event']
        fee_values = [
            total_fees_collected,
            fee_history_count,
            total_fees_collected / max(1, fee_history_count)
        ]
        
        bars = ax3.bar(fee_metrics, fee_values, color=['gold', 'orange', 'coral'], alpha=0.8)
        ax3.set_title('Fee Collection Summary')
        ax3.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, fee_values):
            height = bar.get_height()
            if value >= 1000:
                label = f'${value:,.0f}' if 'Fee' in fee_metrics[fee_values.index(value)] else f'{value:,.0f}'
            else:
                label = f'${value:.2f}' if 'Fee' in fee_metrics[fee_values.index(value)] else f'{value:.1f}'
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    label, ha='center', va='bottom', fontweight='bold')
        
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Bottom Right: Fee Parameters Display
        fee_params = redeemer_data.get("fee_parameters", {})
        
        if fee_params:
            param_names = []
            param_values = []
            
            # Convert fee parameters to readable format
            for key, value in fee_params.items():
                if 'fee' in key.lower():
                    param_names.append(key.replace('_', ' ').title())
                    param_values.append(value * 100)  # Convert to percentage
                elif key == 'imbalance_scale_k':
                    param_names.append('Scale Factor K')
                    param_values.append(value * 10000)  # Convert to bps
                elif key == 'imbalance_convexity_gamma':
                    param_names.append('Convexity γ')
                    param_values.append(value)
                elif key == 'tolerance_band':
                    param_names.append('Tolerance Band')
                    param_values.append(value * 100)  # Convert to percentage
            
            # Create horizontal bar chart
            y_pos = range(len(param_names))
            bars = ax4.barh(y_pos, param_values, color='lightblue', alpha=0.8)
            
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(param_names)
            ax4.set_xlabel('Value (%/bps)')
            ax4.set_title('Fee Structure Parameters')
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, param_values)):
                width = bar.get_width()
                if 'γ' in param_names[i]:
                    label = f'{value:.1f}'
                elif value >= 100:
                    label = f'{value:.0f} bps'
                else:
                    label = f'{value:.2f}%'
                ax4.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                        label, ha='left', va='center')
            
            ax4.grid(True, alpha=0.3, axis='x')
        else:
            ax4.text(0.5, 0.5, 'No Fee Parameters Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Fee Structure Parameters')
        
        plt.tight_layout()
        plt.savefig(output_dir / "redeemer_system_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_arbitrage_activity_chart(self, output_dir: Path):
        """Create enhanced arbitrage activity analysis chart (2x2 layout) using new tracking system"""
        
        # Extract arbitrage agent data from new tracking system
        simulation_results = self.results.get("simulation_results", {})
        
        # Try to get arbitrage events from engine tracking first
        arbitrage_events = simulation_results.get("arbitrage_events", [])
        
        # Try to get arbitrage agents from enhanced tracking system
        moet_system_state = simulation_results.get("moet_system_state", {})
        arbitrage_agents = moet_system_state.get("arbitrage_agents_summary", [])
        
        # If we have arbitrage events, create summary from them
        if arbitrage_events and not arbitrage_agents:
            print(f"✅ Using {len(arbitrage_events)} arbitrage events from engine tracking")
            # Group events by agent
            agent_data = {}
            for event in arbitrage_events:
                agent_id = event.get("agent_id", "unknown")
                if agent_id not in agent_data:
                    agent_data[agent_id] = {
                        "total_events": 0,
                        "total_volume": 0,
                        "total_profit": 0,
                        "total_fees": 0
                    }
                agent_data[agent_id]["total_events"] += 1
                agent_data[agent_id]["total_volume"] += event.get("volume", 0)
                agent_data[agent_id]["total_profit"] += event.get("profit", 0)
                agent_data[agent_id]["total_fees"] += event.get("fees_generated", 0)
            
            # Convert to arbitrage_agents format
            arbitrage_agents = []
            for agent_id, data in agent_data.items():
                arbitrage_agents.append({
                    "agent_id": agent_id,
                    "agent_type": "moet_arbitrage_agent",
                    "total_attempts": data["total_events"] * 2,  # Estimate attempts
                    "total_arbitrage_events": data["total_events"],
                    "successful_arbitrages": data["total_events"],
                    "failed_arbitrages": 0,
                    "total_profit": data["total_profit"],
                    "total_volume_traded": data["total_volume"],
                    "total_fees_generated": data["total_fees"],
                    "execution_rate": 100.0,  # All events were executed
                    "success_rate": 100.0,
                    "average_profit": data["total_profit"] / max(1, data["total_events"]),
                    "average_trade_size": data["total_volume"] / max(1, data["total_events"])
                })
        
        # FALLBACK: If enhanced tracking is missing, use agent_outcomes
        elif not arbitrage_agents:
            print("⚠️  Enhanced arbitrage tracking not found, using agent_outcomes fallback")
            agent_outcomes = simulation_results.get("agent_outcomes", [])
            arbitrage_outcomes = [a for a in agent_outcomes if a.get("agent_type") == "moet_arbitrage_agent"]
            
            if not arbitrage_outcomes:
                print("⚠️  No arbitrage agents found in agent_outcomes either")
                return
                
            # Convert agent_outcomes to arbitrage_agents_summary format
            arbitrage_agents = []
            for outcome in arbitrage_outcomes:
                arbitrage_agents.append({
                    "agent_id": outcome.get("agent_id", "unknown"),
                    "agent_type": "moet_arbitrage_agent",
                    "total_attempts": outcome.get("total_arbitrage_events", 0) * 2,  # Estimate attempts
                    "total_mint_attempts": outcome.get("total_arbitrage_events", 0),
                    "total_redeem_attempts": outcome.get("total_arbitrage_events", 0),
                    "total_arbitrage_events": outcome.get("total_arbitrage_events", 0),
                    "successful_arbitrages": outcome.get("successful_arbitrages", 0),
                    "failed_arbitrages": outcome.get("failed_arbitrages", 0),
                    "total_profit": outcome.get("total_profit", 0),
                    "total_volume_traded": outcome.get("total_arbitrage_events", 0) * 1000,  # Estimate
                    "execution_rate": 50.0 if outcome.get("total_arbitrage_events", 0) > 0 else 0.0,
                    "success_rate": outcome.get("success_rate", 0),
                    "average_profit": outcome.get("average_profit", 0),
                    "average_trade_size": 1000.0,  # Default estimate
                })
            
            print(f"✅ Using {len(arbitrage_agents)} arbitrage agents from agent_outcomes fallback")
        else:
            print(f"✅ Using {len(arbitrage_agents)} arbitrage agents from enhanced tracking system")
        
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MOET Arbitrage Activity Analysis', fontsize=16, fontweight='bold')
        
        # Top Left: Attempt vs Execution Analysis (NEW ENHANCED VIEW)
        agent_ids = [agent["agent_id"] for agent in arbitrage_agents]
        total_attempts = [agent.get("total_attempts", 0) for agent in arbitrage_agents]
        total_executed = [agent.get("total_arbitrage_events", 0) for agent in arbitrage_agents]
        execution_rates = [agent.get("execution_rate", 0) for agent in arbitrage_agents]
        
        # Create grouped bar chart
        x = range(len(agent_ids))
        width = 0.35
        
        bars1 = ax1.bar([i - width/2 for i in x], total_attempts, width, 
                       label='Total Attempts', color='lightblue', alpha=0.8)
        bars2 = ax1.bar([i + width/2 for i in x], total_executed, width,
                       label='Executed Trades', color='lightgreen', alpha=0.8)
        
        ax1.set_title('Arbitrage Attempts vs Executions')
        ax1.set_xlabel('Arbitrage Agents')
        ax1.set_ylabel('Count')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'Agent {i+1}' for i in range(len(agent_ids))], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add execution rate labels
        for i, (attempts, executed, rate) in enumerate(zip(total_attempts, total_executed, execution_rates)):
            ax1.text(i, max(attempts, executed) + max(total_attempts) * 0.02,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Top Right: Mint vs Redeem Attempt Breakdown (NEW ENHANCED VIEW)
        mint_attempts = [agent.get("total_mint_attempts", 0) for agent in arbitrage_agents]
        redeem_attempts = [agent.get("total_redeem_attempts", 0) for agent in arbitrage_agents]
        
        # Create stacked bar chart
        bars1 = ax2.bar(x, mint_attempts, label='Mint Attempts', color='coral', alpha=0.8)
        bars2 = ax2.bar(x, redeem_attempts, bottom=mint_attempts, label='Redeem Attempts', color='skyblue', alpha=0.8)
        
        ax2.set_title('Arbitrage Strategy Breakdown')
        ax2.set_xlabel('Arbitrage Agents')
        ax2.set_ylabel('Attempts Count')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'Agent {i+1}' for i in range(len(agent_ids))], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add total attempt labels
        for i, (mint, redeem) in enumerate(zip(mint_attempts, redeem_attempts)):
            total = mint + redeem
            if total > 0:
                ax2.text(i, total + max(total_attempts) * 0.01,
                        f'{total}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Bottom Left: Volume vs Profit Analysis (NEW ENHANCED VIEW)
        total_volumes = [agent.get("total_volume_traded", 0) for agent in arbitrage_agents]
        total_profits = [agent.get("total_profit", 0) for agent in arbitrage_agents]
        
        # Create dual-axis chart
        ax3_twin = ax3.twinx()
        
        # Volume bars
        bars1 = ax3.bar([i - 0.2 for i in x], total_volumes, 0.4, 
                       label='Volume Traded', color='lightcoral', alpha=0.8)
        # Profit bars on secondary axis
        bars2 = ax3_twin.bar([i + 0.2 for i in x], total_profits, 0.4,
                           label='Total Profit', color='gold', alpha=0.8)
        
        ax3.set_title('Volume Traded vs Profit Generated')
        ax3.set_xlabel('Arbitrage Agents')
        ax3.set_ylabel('Volume Traded ($)', color='red')
        ax3_twin.set_ylabel('Total Profit ($)', color='orange')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Agent {i+1}' for i in range(len(agent_ids))], rotation=45)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax3_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}'))
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add combined legend
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Bottom Right: Enhanced System Summary with New Metrics
        total_all_attempts = sum(total_attempts)
        total_all_executed = sum(total_executed)
        total_all_volumes = sum(total_volumes)
        total_all_profits = sum(total_profits)
        overall_execution_rate = (total_all_executed / max(1, total_all_attempts)) * 100
        total_fees_generated = sum(agent.get("total_fees_generated", 0) for agent in arbitrage_agents)
        
        # Create summary with new enhanced metrics
        summary_metrics = ['Total\nAttempts', 'Executed\nTrades', 'Volume\nTraded', 'Fees\nGenerated']
        summary_values = [total_all_attempts, total_all_executed, total_all_volumes, total_fees_generated]
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
        
        bars = ax4.bar(summary_metrics, summary_values, color=colors, alpha=0.8)
        ax4.set_title(f'Enhanced Arbitrage System Summary\n(Execution Rate: {overall_execution_rate:.1f}%)')
        ax4.set_ylabel('Count / Value ($)')
        
        # Add value labels with smart formatting
        for bar, value, metric in zip(bars, summary_values, summary_metrics):
            height = bar.get_height()
            if 'Volume' in metric or 'Fees' in metric:
                if value >= 1000:
                    label = f'${value:,.0f}'
                else:
                    label = f'${value:.0f}'
            else:
                if value >= 1000:
                    label = f'{value:,.0f}'
                else:
                    label = f'{value}'
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Add additional summary text
        ax4.text(0.02, 0.98, f'Avg Profit/Trade: ${total_all_profits/max(1, total_all_executed):.2f}', 
                transform=ax4.transAxes, va='top', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / "arbitrage_activity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_arbitrage_time_series_chart(self, output_dir: Path):
        """Create time-series analysis of arbitrage attempts over time (2x2 layout)"""
        
        # Extract arbitrage history data
        simulation_results = self.results.get("simulation_results", {})
        moet_system_state = simulation_results.get("moet_system_state", {})
        
        if not moet_system_state:
            print("⚠️  No MOET system data available for time-series analysis")
            return
            
        tracking_data = moet_system_state.get("tracking_data", {})
        arbitrage_history = tracking_data.get("arbitrage_history", [])
        
        # If no arbitrage history, try to collect it from arbitrage agents directly
        if not arbitrage_history:
            print("⚠️  No arbitrage history in tracking data, attempting direct collection from agents...")
            
            # Try to collect arbitrage data directly from engine's arbitrage agents
            if hasattr(self, 'engine') and hasattr(self.engine, 'arbitrage_agents'):
                for agent in self.engine.arbitrage_agents:
                    if hasattr(agent.state, 'arbitrage_attempts'):
                        for attempt in agent.state.arbitrage_attempts:
                            arbitrage_history.append({
                                "minute": attempt.get("minute", 0),
                                "hour": attempt.get("minute", 0) / 60,
                                "agent_id": agent.agent_id,
                                "arbitrage_type": attempt.get("type", "unknown"),
                                "executed": attempt.get("executed", False),
                                "expected_profit": attempt.get("expected_profit", 0),
                                "actual_profit": attempt.get("actual_profit", 0),
                                "trade_size": attempt.get("trade_size", 0),
                                "pool_used": attempt.get("pool", "unknown"),
                                "moet_price": attempt.get("moet_price", 1.0),
                                "reason_not_executed": attempt.get("reason_not_executed", None)
                            })
            
            # If still no data, create placeholder
            if not arbitrage_history:
                print("⚠️  No arbitrage data available from any source, creating placeholder chart")
                
                # Create a simple placeholder chart
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                ax.text(0.5, 0.5, 'No Arbitrage Time-Series Data Available\n\nEnhanced tracking system not active\nCheck arbitrage agent configuration', 
                       ha='center', va='center', fontsize=14, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title('Arbitrage Time-Series Analysis', fontsize=16, fontweight='bold')
                ax.axis('off')
                
                plt.tight_layout()
                plt.savefig(output_dir / "arbitrage_time_series_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
                return
            else:
                print(f"✅ Collected {len(arbitrage_history)} arbitrage records directly from agents")
        
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Arbitrage Activity Time-Series Analysis', fontsize=16, fontweight='bold')
        
        # Convert to pandas for easier analysis
        import pandas as pd
        df = pd.DataFrame(arbitrage_history)
        
        # Top Left: Attempts Over Time (Executed vs Not Executed)
        if not df.empty:
            print(f"📊 Processing {len(df)} arbitrage records for time-series analysis")
            
            # CRITICAL FIX: Properly aggregate to true hourly buckets (not minute-level "hours")
            df['hour'] = (df['minute'] // 60).astype(int)  # Convert to integer hour buckets
            hourly_data = df.groupby(['hour', 'executed']).size().unstack(fill_value=0)
            
            print(f"📊 Aggregated to {len(hourly_data)} hourly data points")
            
            if 'executed' in df.columns:
                hours = hourly_data.index
                executed_counts = hourly_data.get(True, pd.Series(0, index=hours))
                not_executed_counts = hourly_data.get(False, pd.Series(0, index=hours))
                
                # CRITICAL FIX: Remove markers to prevent matplotlib overflow
                ax1.plot(hours, executed_counts, linewidth=2, color='green', 
                        label='Executed Attempts')
                ax1.plot(hours, not_executed_counts, linewidth=2, color='red', 
                        label='Not Executed Attempts')
                ax1.fill_between(hours, executed_counts, alpha=0.3, color='green')
                ax1.fill_between(hours, not_executed_counts, alpha=0.3, color='red')
                
                ax1.set_title('Arbitrage Attempts Over Time')
                ax1.set_xlabel('Hours')
                ax1.set_ylabel('Attempts per Hour')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
        
        # Top Right: Mint vs Redeem Attempts Over Time
        if not df.empty and 'arbitrage_type' in df.columns:
            type_hourly = df.groupby(['hour', 'arbitrage_type']).size().unstack(fill_value=0)
            
            hours = type_hourly.index
            mint_counts = type_hourly.get('mint_arbitrage', pd.Series(0, index=hours))
            redeem_counts = type_hourly.get('redeem_arbitrage', pd.Series(0, index=hours))
            
            ax2.plot(hours, mint_counts, linewidth=2, color='coral', 
                    label='Mint Attempts')
            ax2.plot(hours, redeem_counts, linewidth=2, color='skyblue', 
                    label='Redeem Attempts')
            ax2.fill_between(hours, mint_counts, alpha=0.3, color='coral')
            ax2.fill_between(hours, redeem_counts, alpha=0.3, color='skyblue')
            
            ax2.set_title('Arbitrage Strategy Types Over Time')
            ax2.set_xlabel('Hours')
            ax2.set_ylabel('Attempts per Hour')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Bottom Left: Expected Profit Distribution Over Time
        if not df.empty and 'expected_profit' in df.columns:
            # Create profit bins
            profit_bins = [-float('inf'), -1, -0.1, 0, 0.1, 1, float('inf')]
            profit_labels = ['< -$1', '-$1 to -$0.1', '-$0.1 to $0', '$0 to $0.1', '$0.1 to $1', '> $1']
            df['profit_bin'] = pd.cut(df['expected_profit'], bins=profit_bins, labels=profit_labels)
            
            profit_hourly = df.groupby(['hour', 'profit_bin']).size().unstack(fill_value=0)
            
            # Plot stacked area chart
            ax3.stackplot(profit_hourly.index, *[profit_hourly[col] for col in profit_hourly.columns], 
                         labels=profit_hourly.columns, alpha=0.7)
            ax3.set_title('Expected Profit Distribution Over Time')
            ax3.set_xlabel('Hours')
            ax3.set_ylabel('Attempts per Hour')
            ax3.legend(loc='upper right', fontsize=8)
            ax3.grid(True, alpha=0.3)
        
        # Bottom Right: Execution Rate Over Time (Hourly Average)
        if not df.empty and 'executed' in df.columns:
            # CRITICAL FIX: Use proper hourly aggregation to prevent overflow
            hourly_execution = df.groupby('hour')['executed'].mean() * 100
            
            print(f"📊 Execution rate data: {len(hourly_execution)} hourly points")
            
            # Plot without markers to prevent overflow
            ax4.plot(hourly_execution.index, hourly_execution.values, 
                    linewidth=2, color='purple')
            ax4.fill_between(hourly_execution.index, hourly_execution.values, 
                           alpha=0.3, color='purple')
            
            # Add horizontal line for overall execution rate
            overall_rate = df['executed'].mean() * 100
            ax4.axhline(y=overall_rate, color='red', linestyle='--', 
                       label=f'Overall Rate: {overall_rate:.1f}%')
            
            ax4.set_title('Execution Rate Over Time (Hourly Average)')
            ax4.set_xlabel('Hours')
            ax4.set_ylabel('Execution Rate (%)')
            ax4.set_ylim(0, max(5, hourly_execution.max() * 1.1))  # Dynamic y-limit
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "arbitrage_time_series_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_peg_monitoring_chart(self, output_dir: Path):
        """Create MOET peg monitoring analysis chart (2x1 layout)"""
        
        # Try to extract peg monitoring data from engine results
        peg_monitoring = self.results.get("peg_monitoring", {})
        
        if not peg_monitoring or not any(peg_monitoring.values()):
            print("⚠️  No peg monitoring data available")
            return
        
        # Create 2x1 subplot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('MOET Peg Monitoring Analysis', fontsize=16, fontweight='bold')
        
        # Top: MOET Price Evolution in Both Pools
        peg_deviations = peg_monitoring.get("peg_deviations", [])
        
        if peg_deviations:
            steps = [d.get("step", 0) for d in peg_deviations]
            hours = [s / 60.0 for s in steps]  # Convert steps to hours (assuming 1 step = 1 minute)
            usdc_prices = [d.get("usdc_price", 1.0) for d in peg_deviations]
            usdf_prices = [d.get("usdf_price", 1.0) for d in peg_deviations]
            avg_prices = [d.get("avg_price", 1.0) for d in peg_deviations]
            
            ax1.plot(hours, usdc_prices, linewidth=2, color='blue', label='MOET:USDC Pool', alpha=0.8)
            ax1.plot(hours, usdf_prices, linewidth=2, color='green', label='MOET:USDF Pool', alpha=0.8)
            ax1.plot(hours, avg_prices, linewidth=2, color='purple', label='Average Price', alpha=0.8)
            ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Target Peg ($1.00)')
            
            # Add tolerance bands
            ax1.fill_between(hours, 0.99, 1.01, color='gray', alpha=0.2, label='±1% Tolerance')
            
            ax1.set_title('MOET Price Evolution in Stablecoin Pools')
            ax1.set_xlabel('Hours')
            ax1.set_ylabel('MOET Price ($)')
            ax1.set_ylim(0.95, 1.05)  # Focus on relevant price range
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.3f}'))
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, 'No Peg Deviation Data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('MOET Price Evolution in Stablecoin Pools')
        
        # Bottom: Peg Deviation Analysis
        if peg_deviations:
            max_deviations = [d.get("max_deviation", 0) * 100 for d in peg_deviations]  # Convert to percentage
            
            # Create deviation plot
            ax2.fill_between(hours, 0, max_deviations, color='orange', alpha=0.6, label='Max Deviation')
            ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='1% Threshold')
            ax2.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.7, label='0.5% Warning')
            
            ax2.set_title('MOET Peg Deviation from $1.00 Target')
            ax2.set_xlabel('Hours')
            ax2.set_ylabel('Maximum Deviation (%)')
            ax2.set_ylim(0, max(5.0, max(max_deviations) * 1.1))  # Dynamic y-limit
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add statistics box
            avg_deviation = sum(max_deviations) / len(max_deviations)
            max_deviation_value = max(max_deviations)
            violations_1pct = sum(1 for d in max_deviations if d > 1.0)
            
            stats_text = f'Avg Deviation: {avg_deviation:.2f}%\n'
            stats_text += f'Max Deviation: {max_deviation_value:.2f}%\n'
            stats_text += f'>1% Violations: {violations_1pct}'
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, 'No Deviation Data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('MOET Peg Deviation from $1.00 Target')
        
        plt.tight_layout()
        plt.savefig(output_dir / "peg_monitoring_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_pool_slippage_analysis_chart(self, output_dir: Path):
        """Create pool-specific slippage analysis chart for 4-pool structure"""
        
        # Extract deleveraging events with detailed slippage data
        deleveraging_events = []
        for agent_data in self.results.get("simulation_results", {}).get("agent_health_history", []):
            for agent in agent_data.get("agents", []):
                agent_deleveraging = agent.get("deleveraging_events", [])
                deleveraging_events.extend(agent_deleveraging)
        
        if not deleveraging_events:
            print("⚠️  No deleveraging events found for pool slippage analysis")
            return
        
        # Analyze slippage by path (USDC vs USDF)
        usdc_slippage = []
        usdf_slippage = []
        total_slippage = []
        
        for event in deleveraging_events:
            swap_details = event.get("swap_chain_details", {})
            path = swap_details.get("path_chosen", "Unknown")
            total_slip = event.get("total_slippage_cost", 0)
            
            total_slippage.append(total_slip)
            if path == "USDC":
                usdc_slippage.append(total_slip)
            elif path == "USDF":
                usdf_slippage.append(total_slip)
        
        # Create 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Pool-Specific Slippage Analysis: 4-Pool Deleveraging Structure', fontsize=16, fontweight='bold')
        
        # Chart 1: Slippage by Path (USDC vs USDF)
        if usdc_slippage and usdf_slippage:
            paths = ['USDC Path', 'USDF Path']
            avg_slippage = [
                sum(usdc_slippage) / len(usdc_slippage),
                sum(usdf_slippage) / len(usdf_slippage)
            ]
            colors = ['#3498db', '#e74c3c']
            
            bars = ax1.bar(paths, avg_slippage, color=colors, alpha=0.7)
            ax1.set_title('Average Slippage by Deleveraging Path')
            ax1.set_ylabel('Average Slippage Cost ($)')
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Add value labels on bars
            for bar, value in zip(bars, avg_slippage):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_slippage)*0.01,
                        f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
            
            # Add statistics
            stats_text = f'USDC Events: {len(usdc_slippage)}\n'
            stats_text += f'USDF Events: {len(usdf_slippage)}\n'
            stats_text += f'Total Events: {len(deleveraging_events)}'
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax1.text(0.5, 0.5, 'No path-specific data available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Average Slippage by Deleveraging Path')
        
        # Chart 2: Slippage Distribution
        if total_slippage:
            ax2.hist(total_slippage, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
            ax2.set_title('Distribution of Deleveraging Slippage Costs')
            ax2.set_xlabel('Slippage Cost ($)')
            ax2.set_ylabel('Frequency')
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Add statistics
            avg_slip = sum(total_slippage) / len(total_slippage)
            max_slip = max(total_slippage)
            stats_text = f'Avg: ${avg_slip:,.0f}\nMax: ${max_slip:,.0f}\nEvents: {len(total_slippage)}'
            ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax2.text(0.5, 0.5, 'No slippage data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Distribution of Deleveraging Slippage Costs')
        
        # Chart 3: Slippage Over Time
        if deleveraging_events:
            minutes = [event.get("minute", 0) for event in deleveraging_events]
            slippage_costs = [event.get("total_slippage_cost", 0) for event in deleveraging_events]
            hours = [m / 60 for m in minutes]
            
            ax3.scatter(hours, slippage_costs, alpha=0.6, color='#f39c12', s=30)
            ax3.set_title('Slippage Costs Over Time')
            ax3.set_xlabel('Hours')
            ax3.set_ylabel('Slippage Cost ($)')
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax3.grid(True, alpha=0.3)
            
            # Add trend line if enough data points
            if len(hours) > 5:
                z = np.polyfit(hours, slippage_costs, 1)
                p = np.poly1d(z)
                ax3.plot(hours, p(hours), "r--", alpha=0.8, label=f'Trend: ${z[0]:+.1f}/hr')
                ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No time series data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Slippage Costs Over Time')
        
        # Chart 4: Pool Utilization Analysis
        pool_usage = {}
        for event in deleveraging_events:
            swap_details = event.get("swap_chain_details", {})
            path = swap_details.get("path_chosen", "Unknown")
            pool_usage[path] = pool_usage.get(path, 0) + 1
        
        if pool_usage:
            pools = list(pool_usage.keys())
            usage_counts = list(pool_usage.values())
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(pools)]
            
            wedges, texts, autotexts = ax4.pie(usage_counts, labels=pools, colors=colors, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Pool Path Utilization Distribution')
            
            # Enhance text formatting
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax4.text(0.5, 0.5, 'No pool usage data available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Pool Path Utilization Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / "pool_slippage_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_ecosystem_growth_chart(self, output_dir: Path):
        """Create ecosystem growth analysis chart"""
        
        # Check if ecosystem growth is enabled and we have data
        if not self.config.enable_ecosystem_growth:
            return
            
        simulation_results = self.results.get("simulation_results", {})
        growth_events = simulation_results.get("ecosystem_growth_events", [])
        growth_summary = simulation_results.get("ecosystem_growth_summary", {})
        
        if not growth_events:
            print("⚠️  No ecosystem growth events to chart")
            return
        
        # Convert to DataFrame for easier plotting
        import pandas as pd
        df = pd.DataFrame(growth_events)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Ecosystem Growth Analysis: Agent Addition Over Time', fontsize=16, fontweight='bold')
        
        # Chart 1: Total Agents Over Time
        ax1.plot(df['day'], df['total_agents'], linewidth=2, color='#2E86AB', marker='o', markersize=2)
        ax1.set_xlabel('Days', fontsize=12)
        ax1.set_ylabel('Total Agents', fontsize=12)
        ax1.set_title('Total Agents in Ecosystem', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 365)
        
        # Add milestone annotations
        milestones = [100, 500, 1000, 1500]
        for milestone in milestones:
            milestone_data = df[df['total_agents'] >= milestone]
            if not milestone_data.empty:
                first_milestone = milestone_data.iloc[0]
                ax1.annotate(f'{milestone} agents\n(Day {first_milestone["day"]:.0f})', 
                           xy=(first_milestone['day'], first_milestone['total_agents']),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                           fontsize=10)
        
        # Chart 2: Total USD Value Over Time
        ax2.plot(df['day'], df['total_usd_value'] / 1e6, linewidth=2, color='#457B9D', marker='o', markersize=2)
        ax2.axhline(y=self.config.target_btc_deposits / 1e6, color='red', linestyle='--', 
                   label=f'Target: ${self.config.target_btc_deposits/1e6:.0f}M')
        ax2.set_xlabel('Days', fontsize=12)
        ax2.set_ylabel('Total Deposits ($ Millions)', fontsize=12)
        ax2.set_title('Total BTC Deposits Value', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 365)
        
        # Chart 3: Daily Agent Additions
        daily_additions = df.groupby(df['day'].astype(int)).size()
        ax3.bar(daily_additions.index, daily_additions.values, color='#F1C40F', alpha=0.7)
        ax3.set_xlabel('Days', fontsize=12)
        ax3.set_ylabel('New Agents Added', fontsize=12)
        ax3.set_title('Daily Agent Additions', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 365)
        
        # Chart 4: Target Achievement Progress
        ax4.plot(df['day'], df['target_progress'] * 100, linewidth=2, color='#E74C3C', marker='o', markersize=2)
        ax4.axhline(y=100, color='green', linestyle='--', label='100% Target Achievement')
        ax4.set_xlabel('Days', fontsize=12)
        ax4.set_ylabel('Target Achievement (%)', fontsize=12)
        ax4.set_title('Progress Toward $150M Target', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 365)
        ax4.set_ylim(0, max(110, df['target_progress'].max() * 100 + 10))
        
        # Add summary statistics
        final_agents = growth_summary.get('final_agent_count', 0)
        final_value = growth_summary.get('final_usd_value', 0)
        target_achievement = growth_summary.get('target_achievement', 0) * 100
        
        summary_text = f"""Ecosystem Growth Summary:
        • Final Agents: {final_agents:,}
        • Final Deposits: ${final_value/1e6:.1f}M
        • Target Achievement: {target_achievement:.1f}%
        • Growth Period: {self.config.growth_start_delay_days} - 365 days
        • New Agents Added: {len(growth_events):,}"""
        
        fig.text(0.02, 0.02, summary_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / "ecosystem_growth_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Ecosystem Growth Analysis:")
        print(f"   Final agent count: {final_agents:,}")
        print(f"   Final deposits: ${final_value:,.0f} (${final_value/1e6:.1f}M)")
        print(f"   Target achievement: {target_achievement:.1f}%")
        print(f"   Growth events: {len(growth_events):,}")
    
    def _create_moet_stablecoin_price_chart(self, output_dir: Path):
        """Create focused chart for MOET:USDC and MOET:USDF pool price deviations"""
        
        # Extract pool state snapshots from simulation results
        simulation_results = self.results.get("simulation_results", {})
        pool_snapshots = simulation_results.get("pool_state_snapshots", [])
        
        # Check if we have MOET stablecoin price data
        if not pool_snapshots:
            print("⚠️  No pool state snapshots available for MOET stablecoin price analysis")
            return
        
        # Check if the snapshots contain MOET stablecoin price data
        sample_snapshot = pool_snapshots[0] if pool_snapshots else {}
        if 'moet_usdc_price' not in sample_snapshot or 'moet_usdf_price' not in sample_snapshot:
            print("⚠️  MOET stablecoin price data not found in pool snapshots")
            print(f"Available keys: {list(sample_snapshot.keys())}")
            print("💡 This data will be available after running a simulation with the advanced MOET system enabled")
            
            # Create a placeholder chart
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.text(0.5, 0.5, 'MOET Stablecoin Price Data Not Available\n\nRun a new simulation with advanced MOET system\nto generate MOET:USDC and MOET:USDF price tracking', 
                   ha='center', va='center', fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('MOET Stablecoin Pool Price Deviations', fontsize=16, fontweight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / "moet_stablecoin_price_deviations.png", dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        # Convert to DataFrame for easier handling
        import pandas as pd
        df = pd.DataFrame(pool_snapshots)
        
        # Convert minute to hours for better visualization
        df['hour'] = df['minute'] / 60.0
        
        # Calculate deviations from $1.00 peg in basis points
        df['usdc_deviation_bps'] = (df['moet_usdc_price'] - 1.0) * 10000
        df['usdf_deviation_bps'] = (df['moet_usdf_price'] - 1.0) * 10000
        
        # Create the chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle('MOET Stablecoin Pool Price Deviations: Full Year Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Top panel: Actual prices
        ax1.plot(df['hour'], df['moet_usdc_price'], label='MOET:USDC Price', color='blue', alpha=0.8, linewidth=1)
        ax1.plot(df['hour'], df['moet_usdf_price'], label='MOET:USDF Price', color='red', alpha=0.8, linewidth=1)
        ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='$1.00 Peg')
        
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.set_title('MOET Stablecoin Pool Prices: Full Year Evolution', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 8760)  # Full year = 8760 hours
        
        # Add price range info
        usdc_min, usdc_max = df['moet_usdc_price'].min(), df['moet_usdc_price'].max()
        usdf_min, usdf_max = df['moet_usdf_price'].min(), df['moet_usdf_price'].max()
        
        ax1.text(0.02, 0.98, f'USDC Range: ${usdc_min:.6f} - ${usdc_max:.6f}', 
                 transform=ax1.transAxes, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax1.text(0.02, 0.88, f'USDF Range: ${usdf_min:.6f} - ${usdf_max:.6f}', 
                 transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Bottom panel: Deviations in basis points
        ax2.plot(df['hour'], df['usdc_deviation_bps'], label='MOET:USDC Deviation', color='blue', alpha=0.8, linewidth=1)
        ax2.plot(df['hour'], df['usdf_deviation_bps'], label='MOET:USDF Deviation', color='red', alpha=0.8, linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Perfect Peg')
        
        # Add threshold lines for reference
        ax2.axhline(y=50, color='orange', linestyle=':', alpha=0.7, label='±50 bps')
        ax2.axhline(y=-50, color='orange', linestyle=':', alpha=0.7)
        ax2.axhline(y=100, color='red', linestyle=':', alpha=0.7, label='±100 bps')
        ax2.axhline(y=-100, color='red', linestyle=':', alpha=0.7)
        
        ax2.set_xlabel('Time (Hours)', fontsize=12)
        ax2.set_ylabel('Deviation (Basis Points)', fontsize=12)
        ax2.set_title('MOET Stablecoin Pool Price Deviations from $1.00 Peg', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 8760)  # Full year = 8760 hours
        
        # Add deviation statistics
        usdc_dev_stats = f'USDC: μ={df["usdc_deviation_bps"].mean():.1f} bps, σ={df["usdc_deviation_bps"].std():.1f} bps, max=±{abs(df["usdc_deviation_bps"]).max():.1f} bps'
        usdf_dev_stats = f'USDF: μ={df["usdf_deviation_bps"].mean():.1f} bps, σ={df["usdf_deviation_bps"].std():.1f} bps, max=±{abs(df["usdf_deviation_bps"]).max():.1f} bps'
        
        ax2.text(0.02, 0.98, usdc_dev_stats, transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax2.text(0.02, 0.88, usdf_dev_stats, transform=ax2.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the chart
        plt.savefig(output_dir / "moet_stablecoin_price_deviations.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary statistics
        print(f"\n📊 MOET Stablecoin Pool Price Analysis:")
        print(f"   Simulation duration: {df['hour'].max():.0f} hours ({df['hour'].max()/24:.1f} days)")
        print(f"   Data points: {len(df):,}")
        
        print(f"\n📈 MOET:USDC Pool:")
        print(f"   Price range: ${usdc_min:.6f} - ${usdc_max:.6f}")
        print(f"   Average deviation: {df['usdc_deviation_bps'].mean():+.1f} bps")
        print(f"   Max deviation: ±{abs(df['usdc_deviation_bps']).max():.1f} bps")
        print(f"   Std deviation: {df['usdc_deviation_bps'].std():.1f} bps")
        
        print(f"\n📈 MOET:USDF Pool:")
        print(f"   Price range: ${usdf_min:.6f} - ${usdf_max:.6f}")
        print(f"   Average deviation: {df['usdf_deviation_bps'].mean():+.1f} bps")
        print(f"   Max deviation: ±{abs(df['usdf_deviation_bps']).max():.1f} bps")
        print(f"   Std deviation: {df['usdf_deviation_bps'].std():.1f} bps")
        
        # Check for significant deviations
        significant_usdc = df[abs(df['usdc_deviation_bps']) > 50]
        significant_usdf = df[abs(df['usdf_deviation_bps']) > 50]
        
        print(f"\n🚨 Significant Deviations (>50 bps):")
        print(f"   MOET:USDC: {len(significant_usdc)} instances ({len(significant_usdc)/len(df)*100:.1f}%)")
        print(f"   MOET:USDF: {len(significant_usdf)} instances ({len(significant_usdf)/len(df)*100:.1f}%)")
        
        if len(significant_usdc) > 0:
            print(f"   USDC max deviation time: Hour {significant_usdc.loc[significant_usdc['usdc_deviation_bps'].abs().idxmax(), 'hour']:.1f}")
        if len(significant_usdf) > 0:
            print(f"   USDF max deviation time: Hour {significant_usdf.loc[significant_usdf['usdf_deviation_bps'].abs().idxmax(), 'hour']:.1f}")

    def _aggressive_pre_compilation_cleanup(self, engine, ecosystem_growth_events, pool_state_snapshots):
        """Remove massive event arrays that are never used by charts but consume huge memory"""
        
        cleanup_stats = {
            "engine_events_cleared": 0,
            "agent_events_cleared": 0,
            "agents_processed": 0
        }
        
        # 1. CLEAR ENGINE-LEVEL EVENT ARRAYS (the main memory bombs)
        if hasattr(engine, 'rebalancing_events'):
            cleanup_stats["engine_events_cleared"] += len(engine.rebalancing_events)
            engine.rebalancing_events = []
            
        if hasattr(engine, 'yield_token_trades'):
            cleanup_stats["engine_events_cleared"] += len(engine.yield_token_trades)
            engine.yield_token_trades = []
            
        if hasattr(engine, 'arbitrage_events'):
            cleanup_stats["engine_events_cleared"] += len(engine.arbitrage_events)
            engine.arbitrage_events = []
            
        # 2. CLEAR AGENT-LEVEL EVENT ARRAYS (stored but never read by charts)
        for agent in engine.high_tide_agents:
            cleanup_stats["agents_processed"] += 1
            
            # Clear rebalancing event arrays
            if hasattr(agent.state, 'rebalancing_events'):
                cleanup_stats["agent_events_cleared"] += len(agent.state.rebalancing_events)
                agent.state.rebalancing_events = []
                
            # Clear deleveraging event arrays  
            if hasattr(agent.state, 'deleveraging_events'):
                cleanup_stats["agent_events_cleared"] += len(agent.state.deleveraging_events)
                agent.state.deleveraging_events = []
                
            # Clear arbitrage event arrays
            if hasattr(agent.state, 'arbitrage_events'):
                cleanup_stats["agent_events_cleared"] += len(agent.state.arbitrage_events)
                agent.state.arbitrage_events = []
                
            # Clear action history arrays
            if hasattr(agent.state, 'action_history'):
                cleanup_stats["agent_events_cleared"] += len(agent.state.action_history)
                agent.state.action_history = []
        
        # 3. CLEAR ARBITRAGE AGENT EVENT ARRAYS (if they exist)
        if hasattr(engine, 'arbitrage_agents'):
            for agent in engine.arbitrage_agents:
                cleanup_stats["agents_processed"] += 1
                
                if hasattr(agent.state, 'arbitrage_attempts'):
                    cleanup_stats["agent_events_cleared"] += len(agent.state.arbitrage_attempts)
                    agent.state.arbitrage_attempts = []
                    
                if hasattr(agent.state, 'arbitrage_events'):
                    cleanup_stats["agent_events_cleared"] += len(agent.state.arbitrage_events)
                    agent.state.arbitrage_events = []
        
        # 4. KEEP CRITICAL DATA (charts need these)
        # - pool_state_snapshots: PRESERVED (charts use this)
        # - ecosystem_growth_events: PRESERVED (charts use this)
        # - agent summary stats: PRESERVED (in agent.state, not arrays)
        
        print(f"✅ Cleanup complete:")
        print(f"   🗑️  Cleared {cleanup_stats['engine_events_cleared']:,} engine events")
        print(f"   🗑️  Cleared {cleanup_stats['agent_events_cleared']:,} agent events") 
        print(f"   👥 Processed {cleanup_stats['agents_processed']} agents")
        print(f"   📊 Preserved pool snapshots: {len(pool_state_snapshots)}")
        print(f"   🌱 Preserved growth events: {len(ecosystem_growth_events)}")

    def enable_ecosystem_growth(self, target_btc_deposits: float = 150_000_000, 
                                growth_start_delay_days: int = 30,
                                growth_acceleration_factor: float = 1.2):
        """Enable ecosystem growth simulation with custom parameters"""
        self.config.enable_ecosystem_growth = True
        self.config.target_btc_deposits = target_btc_deposits
        self.config.growth_start_delay_days = growth_start_delay_days
        self.config.growth_acceleration_factor = growth_acceleration_factor
        
        print(f"🌱 Ecosystem Growth ENABLED:")
        print(f"   Target BTC deposits: ${target_btc_deposits:,.0f}")
        print(f"   Growth starts after: {growth_start_delay_days} days")
        print(f"   Growth acceleration: {growth_acceleration_factor}x")
        print(f"   Results will be saved in: {self.config.test_name}_Ecosystem_Growth/")
    
    def _create_bond_auction_analysis_chart(self, output_dir: Path):
        """Create Bond Auction Activity Analysis chart (2x2 layout)"""
        
        # Extract MOET system data from simulation results
        simulation_results = self.results.get("simulation_results", {})
        moet_system_state = simulation_results.get("moet_system_state", {})
        
        # Check if advanced MOET system is enabled - if not, create a "system disabled" chart
        if not moet_system_state.get("advanced_system_enabled"):
            print("⚠️  Advanced MOET system not enabled - creating 'system disabled' bond auction chart")
            print(f"   Debug: moet_system_state keys: {list(moet_system_state.keys())}")
            print(f"   Debug: advanced_system_enabled value: {moet_system_state.get('advanced_system_enabled')}")
            
            # Create a chart showing that the bond auction system is disabled
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.text(0.5, 0.5, 'Bond Auction System Not Enabled\n\nThe Advanced MOET system is disabled in this simulation.\nBond auctions require the advanced MOET system to be active.\n\nTo enable: set enable_advanced_moet_system = True in config', 
                   ha='center', va='center', fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.7))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Bond Auction Activity Analysis - System Disabled', fontsize=16, fontweight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_dir / "bond_auction_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Bond Auction Analysis: System disabled - placeholder chart created")
            return
        
        # Get bonder system data
        bonder_system = moet_system_state.get("bonder_system", {})
        recent_auctions = bonder_system.get("recent_auctions", [])
        auction_count = bonder_system.get("auction_history_count", 0)
        current_bond_cost = bonder_system.get("current_bond_cost_ema", 0)
        
        # Get tracking data for time series
        tracking_data = moet_system_state.get("tracking_data", {})
        bond_aprs = tracking_data.get("bond_apr_history", [])
        deficit_history = tracking_data.get("deficit_history", [])
        
        print(f"📊 Bond Auction Analysis: {auction_count} total auctions, {len(recent_auctions)} recent")
        
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Bond Auction Activity Analysis', fontsize=16, fontweight='bold')
        
        # Top Left: Bond APR Evolution Over Time
        if bond_aprs:
            hours = [entry["minute"] / 60.0 for entry in bond_aprs]
            apr_values = [entry["bond_apr"] * 100 for entry in bond_aprs]  # Convert to percentage
            
            ax1.plot(hours, apr_values, linewidth=2, color='red', label='Bond APR', alpha=0.8)
            ax1.fill_between(hours, 0, apr_values, color='red', alpha=0.3)
            ax1.axhline(y=current_bond_cost * 100, color='darkred', linestyle='--', 
                       label=f'Current EMA: {current_bond_cost:.2%}')
        else:
            # Synthetic data if no tracking available
            hours = list(range(0, 8761, 24))  # Daily points
            synthetic_aprs = [current_bond_cost * 100] * len(hours)
            ax1.plot(hours, synthetic_aprs, linewidth=2, color='red', label='Bond APR (Final State)')
            ax1.fill_between(hours, 0, synthetic_aprs, color='red', alpha=0.3)
        
        ax1.set_title('Bond APR Evolution')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Bond APR (%)')
        ax1.set_xlim(0, 8760)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Top Right: Auction Frequency and Success Rate
        if recent_auctions:
            auction_times = [a.get('timestamp', 0) / 60 for a in recent_auctions]  # Convert to hours
            auction_aprs = [a.get('final_apr', 0) * 100 for a in recent_auctions]
            auction_filled = [a.get('filled_completely', False) for a in recent_auctions]
            
            # Scatter plot of auctions
            filled_auctions = [apr for apr, filled in zip(auction_aprs, auction_filled) if filled]
            unfilled_auctions = [apr for apr, filled in zip(auction_aprs, auction_filled) if not filled]
            filled_times = [time for time, filled in zip(auction_times, auction_filled) if filled]
            unfilled_times = [time for time, filled in zip(auction_times, auction_filled) if not filled]
            
            if filled_auctions:
                ax2.scatter(filled_times, filled_auctions, color='green', s=50, alpha=0.7, label='Filled Auctions')
            if unfilled_auctions:
                ax2.scatter(unfilled_times, unfilled_auctions, color='red', s=50, alpha=0.7, label='Unfilled Auctions')
            
            fill_rate = sum(auction_filled) / len(auction_filled) * 100 if auction_filled else 0
            ax2.text(0.02, 0.98, f'Fill Rate: {fill_rate:.1f}%', transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        else:
            ax2.text(0.5, 0.5, f'Total Auctions: {auction_count}\nNo Recent Auction Data Available', 
                    ha='center', va='center', transform=ax2.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        ax2.set_title('Auction Activity & Success Rate')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('Auction APR (%)')
        ax2.set_xlim(0, 8760)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Bottom Left: Reserve Deficit Triggering Auctions
        if deficit_history:
            hours = [entry["minute"] / 60.0 for entry in deficit_history]
            deficits = [entry["deficit"] for entry in deficit_history]
            
            ax3.fill_between(hours, 0, deficits, color='orange', alpha=0.6, label='Reserve Deficit')
            ax3.plot(hours, deficits, linewidth=2, color='darkorange')
            ax3.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Auction Trigger ($100)')
        else:
            # Use synthetic data
            reserve_state = moet_system_state.get("reserve_state", {})
            total_supply = moet_system_state.get("total_supply", 1000000)
            total_reserves = reserve_state.get("total_reserves", 0)
            target_ratio = reserve_state.get("target_reserves_ratio", 0.10)
            
            target_reserves = total_supply * target_ratio
            current_deficit = max(0, target_reserves - total_reserves)
            
            hours = list(range(0, 8761, 24))
            synthetic_deficits = [current_deficit] * len(hours)
            ax3.fill_between(hours, 0, synthetic_deficits, color='orange', alpha=0.6, label='Reserve Deficit')
            ax3.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Auction Trigger ($100)')
        
        ax3.set_title('Reserve Deficit (Auction Trigger)')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('Deficit ($)')
        ax3.set_xlim(0, 8760)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Bottom Right: Auction Summary Statistics
        ax4.axis('off')
        
        # Calculate summary statistics
        if recent_auctions:
            total_raised = sum(a.get('amount_filled', 0) for a in recent_auctions)
            avg_apr = sum(a.get('final_apr', 0) for a in recent_auctions) / len(recent_auctions) * 100
            fill_rate = sum(1 for a in recent_auctions if a.get('filled_completely', False)) / len(recent_auctions) * 100
        else:
            total_raised = 0
            avg_apr = current_bond_cost * 100
            fill_rate = 0
        
        summary_text = f"""Bond Auction Summary
        
Total Auctions: {auction_count:,}
Recent Auctions: {len(recent_auctions)}
Total Funds Raised: ${total_raised:,.0f}
Average APR: {avg_apr:.2f}%
Fill Rate: {fill_rate:.1f}%
Current Bond Cost EMA: {current_bond_cost:.2%}

System Status: {'Active' if auction_count > 0 else 'No Auctions'}
Auction Trigger: $100 deficit minimum
Frequency: Hourly (when deficit exists)"""
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / "bond_auction_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Bond Auction Analysis: {auction_count} auctions, avg APR {avg_apr:.2f}%")


def main():
    """Main execution function"""
    
    print("STUDY 1: 2021 Mixed Market - Equal Health Factors (Historical AAVE Rates for Both)")
    print("=" * 50)
    print()
    
    # Create configuration
    config = FullYearSimConfig()
    
    # Load market data (required before accessing btc_data or aave_rates)
    config.load_market_data()
    
    if config.run_aave_comparison:
        print("COMPARISON MODE: High Tide vs AAVE")
        print("This simulation compares High Tide's active rebalancing vs AAVE's buy-and-hold strategy.")
        print()
        print("Features:")
        print("• 1 agent per system (High Tide & AAVE)")
        print("• High Tide: Historical 2021 AAVE rates (symmetric comparison)")
        print("• AAVE: Historical 2021 rates from CSV")
        print("• High Tide: Active rebalancing (HF: 1.3 → 1.1 → 1.2) - Equal HF")
        print("• AAVE: Buy-and-hold (HF: 1.3) - Equal HF")
        print("• Real 2021 BTC: $29k → $46k (+59.6%) - 364 days")
        print("• 💎 Weekly yield harvest (sell only YT rebasing yield)")
    else:
        print("This simulation will run for 268 days using real 2025 BTC pricing data.")
        print("Features:")
        print("• High Tide agents with uniform tri-health factor profile")
        print("  - Initial HF: 1.3, Rebalancing HF: 1.1, Target HF: 1.2")
        print("• Leverage increases when BTC rises (HF > initial HF)")
        print("• Pool rebalancing with ALM (12h intervals) and Algo (50 bps threshold)")
        print("• Real 2025 BTC price progression: $93k → $96k (+2.9%)")
        print("• Ecosystem growth to $150M deposits")
    
    print()
    print(f"Configuration:")
    print(f"• Duration: {config.simulation_duration_hours:,} hours ({config.simulation_duration_hours//24} days)")
    print(f"• BTC Data: {len(config.btc_data)} daily prices from {config.market_year}")
    print(f"• Pool Sizes: MOET:BTC ${config.moet_btc_pool_config['size']:,}, MOET:YT ${config.moet_yt_pool_config['size']:,}")
    print()
    
    # Auto-confirm for automated runs
    print("✅ Auto-confirming simulation start...")
    confirm = 'y'
    
    # Run the full year simulation
    print("\n🌍 FULL YEAR SIMULATION STARTING")
    if config.run_aave_comparison:
        print("Mode: High Tide vs AAVE Comparison (120 agents each)")
        print("This will take significant time - running both simulations...")
    else:
        print("Mode: High Tide with Ecosystem Growth")
        print("This will take significant time - progress reports every week...")
    
    try:
        simulation = FullYearSimulation(config)
        
        # ECOSYSTEM GROWTH: Enable scaling to $150M deposits with optimized parameters
        # Disabled when running AAVE comparison for clean apples-to-apples comparison
        if not config.run_aave_comparison:
            simulation.enable_ecosystem_growth(target_btc_deposits=150_000_000, growth_start_delay_days=30)
        
        results = simulation.run_test()
        
        # Print final summary
        agent_perf = results.get("agent_performance", {}).get("summary", {})
        pool_arb = results.get("pool_arbitrage_analysis", {})
        
        print(f"\n🎯 YEAR-LONG SIMULATION COMPLETED!")
        print(f"📊 Final Results:")
        if agent_perf:
            print(f"   Agents: {agent_perf.get('survived_agents', 0)}/{agent_perf.get('total_agents', 0)} survived ({agent_perf.get('survival_rate', 0):.1%})")
            print(f"   Total Rebalances: {agent_perf.get('total_rebalances', 0):,}")
            print(f"   Total Slippage: ${agent_perf.get('total_slippage_costs', 0):,.2f}")
        if pool_arb.get("enabled"):
            print(f"   Pool Rebalances: ALM {pool_arb.get('alm_rebalances', 0)}, Algo {pool_arb.get('algo_rebalances', 0)}")
            print(f"   Pool Arbitrage Profit: ${pool_arb.get('total_profit', 0):,.2f}")
        
        return results
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Simulation interrupted by user.")
        return None
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
