#!/usr/bin/env python3
"""
Base Case Scenario: High Tide vs AAVE Comparison
Technical Whitepaper Analysis - 3 Month Test

Compares High Tide's automated rebalancing system against AAVE's traditional
liquidation mechanism under controlled conditions:

- Duration: 3 months (90 days)
- BTC Range: ±15% ($85k - $115k from $100k base)
- Fixed Borrow Rate: 5.6234% (AAVE's historical 90-day average)
- Agents: 100 identical agents for both systems
- Initial Deposit: 1 BTC per agent at $100k
- Both systems: Initial YT purchase
- High Tide: Full rebalancing features (no Bonder system)
- AAVE: Static positions with liquidations only

This test isolates the impact of active risk management from interest rate dynamics.
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
from dataclasses import dataclass

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
from tidal_protocol_sim.core.uniswap_v3_math import (
    create_moet_usdc_pool, create_moet_usdf_pool, 
    UniswapV3SlippageCalculator
)

# Import comprehensive charting module
import base_case_charts


@dataclass
class BaseCaseConfig:
    """Configuration for base case 3-month comparison"""
    
    # Test scenario parameters
    test_name: str = "Base_Case_HT_vs_AAVE_3mo"
    simulation_duration_days: int = 90  # 3 months
    simulation_duration_hours: int = 90 * 24  # 2160 hours
    simulation_duration_minutes: int = 90 * 24 * 60  # 129,600 minutes
    
    # Monte Carlo parameters
    num_monte_carlo_runs: int = 1  # Set to 1 for single run, >1 for MC simulation
    random_seed_base: int = 42  # Base seed (run i uses seed base + i)
    save_individual_runs: bool = False  # Save each run's JSON or just aggregate?
    
    # Fixed borrow rate (no Bonder system)
    fixed_borrow_rate: float = 0.056234  # 5.6234% APR (AAVE historical 90-day)
    use_fixed_rate: bool = True  # Disable advanced MOET system
    
    # Agent configuration - IDENTICAL for both systems
    num_agents: int = 20  # Reduced for initial testing
    initial_btc_per_agent: float = 1.0  # 1 BTC per agent
    
    # Tri-health factor parameters
    agent_initial_hf: float = 1.2  # Starting HF (updated for safety)
    agent_rebalancing_hf: float = 1.05  # Rebalancing trigger (High Tide only)
    agent_target_hf: float = 1.1  # Rebalancing target (High Tide only)
    
    # BTC price scenario - ±15% range over 3 months
    btc_initial_price: float = 100_000.0  # $100k starting price
    btc_price_volatility: float = 0.15  # ±15% range
    btc_min_price: float = 85_000.0  # -15%
    btc_max_price: float = 115_000.0  # +15%
    
    # Pool configurations (same for both systems)
    moet_btc_pool_size: float = 10_000_000.0  # $10M liquidation pool
    moet_btc_concentration: float = 0.80  # 80% concentration
    
    moet_yt_pool_size: float = 500_000.0  # $500K YT pool
    moet_yt_concentration: float = 0.95  # 95% concentration
    moet_yt_token0_ratio: float = 0.75  # 75/25 MOET:YT split
    
    # Stablecoin pools for deleveraging swap chain (CRITICAL for weekly deleveraging)
    moet_stablecoin_pool_size: float = 5_000_000.0  # $5M for MOET:USDC and MOET:USDF each
    enable_stablecoin_pools: bool = True  # Enable MOET:USDC/USDF pools for deleveraging
    
    # Pool rebalancing configuration (High Tide only)
    enable_pool_rebalancing: bool = True
    alm_rebalance_interval_minutes: int = 720  # 12 hours
    algo_deviation_threshold_bps: float = 50.0  # 50 bps
    
    # Yield parameters
    yield_apr: float = 0.10  # 10% APR
    use_direct_minting_for_initial: bool = True  # 1:1 minting at t=0
    
    # AAVE liquidation configuration
    use_simple_liquidation_swap: bool = True  # Bypass Uniswap pool for AAVE liquidations
    liquidation_fee_bps: float = 5.0  # Liquidation fee (5 bps = 0.05%)
    liquidation_slippage_bps: float = 5.0  # Slippage (5 bps = 0.05%)
    
    # Logging and output
    detailed_logging: bool = True
    log_every_n_minutes: int = 1440  # Daily logging
    collect_pool_state_every_n_minutes: int = 1440  # Daily snapshots
    generate_charts: bool = True
    save_detailed_csv: bool = True
    
    def generate_btc_price_path(self, seed: int = None) -> List[float]:
        """
        Generate 3-month BTC price path using Geometric Brownian Motion (GBM)
        
        GBM is the standard model for asset prices:
        S(t+1) = S(t) * exp((μ - σ²/2) * dt + σ * sqrt(dt) * Z)
        
        Where:
        - μ = drift (expected annual return) 
        - σ = volatility (annualized)
        - dt = time step (1/365 for daily)
        - Z ~ N(0,1) random normal
        
        Args:
            seed: Random seed for reproducibility (uses random_seed_base if None)
        """
        if seed is None:
            seed = self.random_seed_base
        np.random.seed(seed)  # Reproducibility
        
        # GBM parameters with stronger bullish bias
        mu = 0.60  # 60% annual drift (strong positive skew for 3-month period)
        sigma = 0.15  # 15% annualized volatility (moderate noise)
        dt = 1.0 / 365.0  # Daily time step
        
        # Add upward bias to ensure more time above starting price
        upward_bias = 0.0005  # Small daily upward nudge (~18% annualized additional drift)
        
        prices = []
        current_price = self.btc_initial_price
        prices.append(current_price)
        
        # Generate daily prices using GBM (90 days)
        for day in range(self.simulation_duration_days):
            # Generate random shock from normal distribution
            Z = np.random.standard_normal()
            
            # GBM formula for next price with upward bias
            drift_term = (mu - 0.5 * sigma**2) * dt
            diffusion_term = sigma * np.sqrt(dt) * Z
            bias_term = upward_bias  # Additional upward nudge
            
            next_price = current_price * np.exp(drift_term + diffusion_term + bias_term)
            
            # Enforce soft bounds with dampening near edges (realistic)
            # This prevents extreme outliers while allowing natural movement
            if next_price < self.btc_min_price:
                next_price = self.btc_min_price + (next_price - self.btc_min_price) * 0.1
            elif next_price > self.btc_max_price:
                next_price = self.btc_max_price + (next_price - self.btc_max_price) * 0.1
            
            prices.append(next_price)
            current_price = next_price
        
        print(f"📊 Generated {len(prices)} days of BTC pricing using GBM")
        print(f"📈 Parameters: μ={mu:.1%}, σ={sigma:.1%} annualized")
        print(f"📈 Price range: ${min(prices):,.0f} - ${max(prices):,.0f}")
        print(f"📈 Start: ${prices[0]:,.0f}, End: ${prices[-1]:,.0f} ({((prices[-1]/prices[0])-1)*100:+.1f}%)")
        
        return prices
    
    def get_btc_price_at_minute(self, minute: int, price_path: List[float]) -> float:
        """Get BTC price at given minute with linear interpolation"""
        minutes_per_day = 24 * 60  # 1440 minutes per day
        day_of_sim = minute // minutes_per_day
        
        # Ensure we don't exceed available data
        if day_of_sim >= len(price_path) - 1:
            return price_path[-1]
        
        # Linear interpolation within the day
        current_day_price = price_path[day_of_sim]
        next_day_price = price_path[day_of_sim + 1]
        
        minutes_into_day = minute % minutes_per_day
        daily_progress = minutes_into_day / minutes_per_day
        
        interpolated_price = current_day_price + (next_day_price - current_day_price) * daily_progress
        return interpolated_price


class BaseCaseComparison:
    """Main comparison class for High Tide vs AAVE base case analysis"""
    
    def __init__(self, config: BaseCaseConfig):
        self.config = config
        
        # Monte Carlo run storage
        self.all_btc_paths = []
        self.all_ht_results = []
        self.all_aave_results = []
        
        # Single run storage (for backwards compatibility)
        self.btc_price_path = None
        
        # Results storage
        self.results = {
            "test_metadata": {
                "test_name": config.test_name,
                "timestamp": datetime.now().isoformat(),
                "duration_days": config.simulation_duration_days,
                "num_agents": config.num_agents,
                "fixed_borrow_rate": config.fixed_borrow_rate,
                "btc_price_range": f"±{config.btc_price_volatility*100}%",
                "initial_btc_price": config.btc_initial_price,
                "monte_carlo_runs": config.num_monte_carlo_runs,
                "is_monte_carlo": config.num_monte_carlo_runs > 1
            },
            "high_tide_results": {},
            "aave_results": {},
            "comparative_analysis": {}
        }
        
        # Aggregated MC results (if MC mode)
        self.aggregated_results = None
    
    def run_comparison(self) -> Dict[str, Any]:
        """Run the complete comparison test (supports Monte Carlo mode)"""
        
        is_monte_carlo = self.config.num_monte_carlo_runs > 1
        
        print("🎯 BASE CASE SCENARIO: HIGH TIDE VS AAVE")
        print("=" * 80)
        print(f"📅 Duration: {self.config.simulation_duration_days} days")
        print(f"👥 Agents: {self.config.num_agents} (identical parameters for both systems)")
        print(f"💰 Initial Deposit: {self.config.initial_btc_per_agent} BTC @ ${self.config.btc_initial_price:,.0f}")
        print(f"📈 BTC Range: ${self.config.btc_min_price:,.0f} - ${self.config.btc_max_price:,.0f} (±15%)")
        print(f"💸 Fixed Borrow Rate: {self.config.fixed_borrow_rate*100:.4f}% APR")
        print(f"⚙️  High Tide: Full rebalancing features (NO Bonder system)")
        print(f"⚙️  AAVE: Static positions with liquidations only")
        
        if is_monte_carlo:
            print(f"🎲 Monte Carlo Mode: {self.config.num_monte_carlo_runs} runs")
        print()
        
        # Monte Carlo loop (or single run if num_monte_carlo_runs = 1)
        for run_idx in range(self.config.num_monte_carlo_runs):
            if is_monte_carlo:
                print(f"\n{'='*80}")
                print(f"🎲 Monte Carlo Run {run_idx + 1}/{self.config.num_monte_carlo_runs}")
                print(f"{'='*80}\n")
            
            # Generate BTC price path for this run
            seed = self.config.random_seed_base + run_idx
            self.btc_price_path = self.config.generate_btc_price_path(seed=seed)
            self.all_btc_paths.append(self.btc_price_path)
            
            # Set random seed for agent/engine randomness
            random.seed(seed)
            np.random.seed(seed)
            
            # Run High Tide simulation
            print("🌊 RUNNING HIGH TIDE SIMULATION...")
            print("-" * 80)
            ht_engine = self._create_high_tide_engine()
            ht_results = self._run_simulation(ht_engine, "high_tide")
            self.all_ht_results.append(ht_results)
            
            print("\n\n")
            
            # Run AAVE simulation
            print("🏦 RUNNING AAVE SIMULATION...")
            print("-" * 80)
            aave_engine = self._create_aave_engine()
            aave_results = self._run_simulation(aave_engine, "aave")
            self.all_aave_results.append(aave_results)
            
            print("\n\n")
        
        # Process results based on mode
        if is_monte_carlo:
            # Aggregate results across all runs
            print("📊 AGGREGATING MONTE CARLO RESULTS...")
            print("-" * 80)
            self.aggregated_results = self._aggregate_monte_carlo_results()
            
            # Store aggregated results in main results dict
            self.results["high_tide_results"] = self.aggregated_results["high_tide"]
            self.results["aave_results"] = self.aggregated_results["aave"]
            
            # Comparative analysis on aggregated data
            self._perform_comparative_analysis_mc()
        else:
            # Single run - use existing logic
            self.results["high_tide_results"] = self.all_ht_results[0]
            self.results["aave_results"] = self.all_aave_results[0]
            self.btc_price_path = self.all_btc_paths[0]
            
            print("📊 PERFORMING COMPARATIVE ANALYSIS...")
            print("-" * 80)
            self._perform_comparative_analysis()
        
        # Save results
        self._save_results()
        
        # Generate charts
        if self.config.generate_charts:
            self._generate_all_charts()
        
        print("\n✅ Base case comparison completed!")
        self._print_summary()
        
        return self.results
    
    
    def _add_stablecoin_pools_to_engine(self, engine, pool_size: float):
        """
        Manually add MOET:stablecoin pools to engine for deleveraging support
        WITHOUT enabling the full advanced MOET system (no Bonder/Redeemer modules)
        
        Uses simplified market execution for final stablecoin → BTC swap (like full year sim)
        """
        print(f"\n🏊 Manually creating MOET:stablecoin pools for deleveraging:")
        print(f"   MOET:USDC pool: ${pool_size:,.0f}")
        print(f"   MOET:USDF pool: ${pool_size:,.0f}")
        print(f"   Stablecoin → BTC: Market execution (0.1% fee + 0.05% slippage)")
        
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
        
        # NOTE: NO USDC:BTC or USDF:BTC pools created
        # The agent's _execute_stablecoin_to_btc_market_order() handles final swap with:
        # - 0.1% trading fee (realistic for CEX)
        # - 0.05% slippage (realistic for large orders)
        
        print(f"✅ Stablecoin pools created successfully for deleveraging chain")
        print(f"   Deleveraging path: YT → MOET → USDC/USDF → BTC (market) → Collateral")
    
    def _create_high_tide_engine(self) -> HighTideVaultEngine:
        """Create High Tide engine with fixed borrow rate"""
        
        # Create High Tide configuration
        ht_config = HighTideConfig()
        ht_config.num_high_tide_agents = 0  # We'll create custom agents
        ht_config.btc_decline_duration = self.config.simulation_duration_minutes
        ht_config.btc_initial_price = self.config.btc_initial_price
        ht_config.btc_final_price_range = (self.btc_price_path[-1], self.btc_price_path[-1])
        
        # CRITICAL: Disable advanced MOET system to use fixed rate (no Bonder/Redeemer modules)
        ht_config.enable_advanced_moet_system = False
        
        # Configure pools
        ht_config.moet_btc_pool_size = self.config.moet_btc_pool_size
        ht_config.moet_btc_concentration = self.config.moet_btc_concentration
        ht_config.moet_yield_pool_size = self.config.moet_yt_pool_size
        ht_config.yield_token_concentration = self.config.moet_yt_concentration
        ht_config.yield_token_ratio = self.config.moet_yt_token0_ratio
        ht_config.use_direct_minting_for_initial = self.config.use_direct_minting_for_initial
        
        # Create engine
        engine = HighTideVaultEngine(ht_config)
        
        # CRITICAL FIX: Manually add stablecoin pools for deleveraging (without enabling full advanced MOET)
        self._add_stablecoin_pools_to_engine(engine, self.config.moet_stablecoin_pool_size)
        
        # Override protocol to return fixed borrow rate
        fixed_rate = self.config.fixed_borrow_rate
        original_get_rate = engine.protocol.get_moet_borrow_rate
        engine.protocol.get_moet_borrow_rate = lambda: fixed_rate
        
        # Create uniform profile agents
        agents = self._create_uniform_agents(engine, "high_tide")
        engine.high_tide_agents = agents
        
        # Add agents to engine's agent dict
        for agent in agents:
            engine.agents[agent.agent_id] = agent
            agent.engine = engine
        
        # CRITICAL FIX: Initialize agent positions in protocol
        self._initialize_agent_positions(engine, agents, "high_tide")
        
        # Create and configure pool rebalancer (High Tide only)
        if self.config.enable_pool_rebalancing:
            pool_rebalancer = PoolRebalancerManager(
                alm_interval_minutes=self.config.alm_rebalance_interval_minutes,
                algo_threshold_bps=self.config.algo_deviation_threshold_bps
            )
            pool_rebalancer.set_enabled(True)
            pool_rebalancer.set_yield_token_pool(engine.yield_token_pool)
            engine.pool_rebalancer = pool_rebalancer
            
            print(f"✅ Pool rebalancer configured:")
            print(f"   ALM interval: {self.config.alm_rebalance_interval_minutes} min")
            print(f"   Algo threshold: {self.config.algo_deviation_threshold_bps} bps")
        
        print(f"✅ High Tide engine created:")
        print(f"   Agents: {len(agents)}")
        print(f"   Fixed borrow rate: {self.config.fixed_borrow_rate*100:.4f}% APR")
        print(f"   Duration: {self.config.simulation_duration_days} days")
        
        return engine
    
    def _create_aave_engine(self) -> AaveProtocolEngine:
        """Create AAVE engine with fixed borrow rate"""
        
        # Create AAVE configuration
        aave_config = AaveConfig()
        aave_config.num_aave_agents = 0  # We'll create custom agents
        aave_config.btc_decline_duration = self.config.simulation_duration_minutes
        aave_config.btc_initial_price = self.config.btc_initial_price
        aave_config.btc_final_price_range = (self.btc_price_path[-1], self.btc_price_path[-1])
        
        # CRITICAL: Disable advanced MOET system to use fixed rate (no Bonder/Redeemer modules)
        aave_config.enable_advanced_moet_system = False
        
        # Configure pools (same as High Tide)
        aave_config.moet_btc_pool_size = self.config.moet_btc_pool_size
        aave_config.moet_btc_concentration = self.config.moet_btc_concentration
        aave_config.moet_yield_pool_size = self.config.moet_yt_pool_size
        aave_config.yield_token_concentration = self.config.moet_yt_concentration
        aave_config.yield_token_ratio = self.config.moet_yt_token0_ratio
        aave_config.use_direct_minting_for_initial = self.config.use_direct_minting_for_initial
        
        # Create engine
        engine = AaveProtocolEngine(aave_config)
        
        # CRITICAL FIX: Manually add stablecoin pools for consistency (AAVE agents don't delever but pools should exist)
        self._add_stablecoin_pools_to_engine(engine, self.config.moet_stablecoin_pool_size)
        
        # Override protocol to return fixed borrow rate
        fixed_rate = self.config.fixed_borrow_rate
        # For AAVE, we need to override the BTC pool's borrow rate calculation
        btc_pool = engine.protocol.asset_pools.get(Asset.BTC)
        if btc_pool:
            btc_pool.calculate_borrow_rate = lambda: fixed_rate
        
        # Create uniform profile agents (same parameters as High Tide)
        agents = self._create_uniform_agents(engine, "aave")
        engine.aave_agents = agents
        
        # Add agents to engine's agent dict
        engine.agents = {}
        for agent in agents:
            engine.agents[agent.agent_id] = agent
        
        # CRITICAL FIX: Initialize agent positions in protocol
        self._initialize_agent_positions(engine, agents, "aave")
        
        # Override liquidation logic to use simple swap (if enabled)
        if self.config.use_simple_liquidation_swap:
            self._override_aave_liquidation_with_simple_swap(engine)
            print(f"✅ Simple liquidation swap enabled:")
            print(f"   Liquidation fee: {self.config.liquidation_fee_bps} bps")
            print(f"   Slippage: {self.config.liquidation_slippage_bps} bps")
        
        print(f"✅ AAVE engine created:")
        print(f"   Agents: {len(agents)}")
        print(f"   Fixed borrow rate: {self.config.fixed_borrow_rate*100:.4f}% APR")
        print(f"   Duration: {self.config.simulation_duration_days} days")
        
        return engine
    
    def _create_uniform_agents(self, engine, system_type: str) -> List:
        """Create agents with uniform tri-health factor profile"""
        
        agents = []
        
        for i in range(self.config.num_agents):
            agent_id = f"{system_type}_agent_{i:03d}"
            
            if system_type == "high_tide":
                # High Tide agent
                agent = HighTideAgent(
                    agent_id,
                    self.config.agent_initial_hf,
                    self.config.agent_rebalancing_hf,
                    self.config.agent_target_hf,
                    initial_balance=self.config.btc_initial_price,
                    yield_token_pool=engine.yield_token_pool
                )
            else:
                # AAVE agent (same parameters, different behavior)
                agent = AaveAgent(
                    agent_id,
                    self.config.agent_initial_hf,
                    self.config.agent_rebalancing_hf,
                    self.config.agent_target_hf,
                    initial_balance=self.config.btc_initial_price
                )
                # CRITICAL: Set yield token pool reference for AAVE agents too
                agent.state.yield_token_manager.yield_token_pool = engine.yield_token_pool
            
            agents.append(agent)
        
        return agents
    
    def _override_aave_liquidation_with_simple_swap(self, engine: AaveProtocolEngine):
        """
        Override AAVE liquidation to use simple swap logic instead of Uniswap pool
        
        This bypasses the concentrated liquidity pool and does a direct swap with:
        - Liquidation fee (5 bps)
        - Slippage (5 bps)
        
        Total cost: ~10 bps (0.1%) on the swap
        """
        
        # Create simple swap liquidation method
        def simple_swap_liquidate(agent_self, current_minute: int, asset_prices: Dict):
            """Liquidate agent using simple swap instead of Uniswap pool"""
            
            if not agent_self.active:
                return None
            
            # Get current BTC price
            btc_price = asset_prices.get(Asset.BTC, self.config.btc_initial_price)
            
            # Calculate liquidation amounts
            btc_to_liquidate = agent_self.state.btc_amount
            debt_to_repay = agent_self.state.moet_debt
            
            # Simple swap: BTC → MOET at current price with fees
            liquidation_fee_factor = 1.0 - (self.config.liquidation_fee_bps / 10000)
            slippage_factor = 1.0 - (self.config.liquidation_slippage_bps / 10000)
            
            # Effective price after fees and slippage
            effective_price = btc_price * liquidation_fee_factor * slippage_factor
            
            # Calculate MOET received from selling BTC
            moet_received = btc_to_liquidate * effective_price
            
            # Print liquidation details
            print(f"           Simple swap: {btc_to_liquidate:.4f} BTC @ ${effective_price:,.2f} = ${moet_received:,.2f} MOET")
            print(f"           Debt: ${debt_to_repay:,.2f}, Coverage: {(moet_received/debt_to_repay)*100:.1f}%")
            
            # Update agent state
            agent_self.state.btc_amount = 0.0
            agent_self.state.moet_debt = max(0, debt_to_repay - moet_received)
            agent_self.active = False
            
            # Record liquidation event
            liquidation_event = {
                "minute": current_minute,
                "agent_id": agent_self.agent_id,
                "btc_liquidated": btc_to_liquidate,
                "debt_repaid": min(debt_to_repay, moet_received),
                "bad_debt": max(0, debt_to_repay - moet_received),
                "effective_price": effective_price
            }
            
            # Update protocol pools
            btc_pool = engine.protocol.asset_pools[Asset.BTC]
            btc_pool.total_supplied -= btc_to_liquidate
            
            if moet_received >= debt_to_repay:
                # Full repayment
                engine.protocol.moet_system.burn(debt_to_repay)
            else:
                # Partial repayment (bad debt)
                engine.protocol.moet_system.burn(moet_received)
                bad_debt = debt_to_repay - moet_received
                print(f"           ⚠️  Bad debt: ${bad_debt:,.2f}")
            
            return liquidation_event
        
        # Override the liquidation method for all AAVE agents
        for agent in engine.aave_agents:
            agent.execute_aave_liquidation = lambda minute, prices, a=agent: simple_swap_liquidate(a, minute, prices)
    
    def _initialize_agent_positions(self, engine, agents: List, system_type: str):
        """
        Initialize agent positions in protocol pools
        
        This syncs agent state (which has BTC/debt) with the protocol pools.
        Without this, agents have balances but the protocol doesn't know about them.
        """
        
        print(f"\n📊 Initializing {system_type.upper()} agent positions in protocol...")
        
        total_btc_supplied = 0.0
        total_moet_debt = 0.0
        
        for agent in agents:
            # Agent state already has BTC and debt from HighTideAgentState.__init__
            # We just need to update the protocol pools to match
            
            btc_amount = agent.state.btc_amount
            moet_debt = agent.state.moet_debt
            
            # Update protocol BTC pool
            btc_pool = engine.protocol.asset_pools[Asset.BTC]
            btc_pool.total_supplied += btc_amount
            
            # Update protocol MOET system
            engine.protocol.moet_system.mint(moet_debt)
            
            # CRITICAL: Put MOET into borrowed_balances so initial YT purchase can happen
            # This is what enables agents to call _initial_yield_token_purchase()
            agent.state.borrowed_balances[Asset.MOET] = moet_debt
            
            # Update agent health factor
            engine._update_agent_health_factor(agent)
            
            total_btc_supplied += btc_amount
            total_moet_debt += moet_debt
            
            print(f"   ✅ {agent.agent_id}: {btc_amount:.2f} BTC → ${moet_debt:,.0f} MOET debt (HF: {agent.state.health_factor:.3f})")
        
        print(f"\n✅ Protocol initialized:")
        print(f"   Total BTC supplied: {total_btc_supplied:.2f} BTC")
        print(f"   Total MOET debt: ${total_moet_debt:,.0f}")
        print(f"   Average HF: {sum(a.state.health_factor for a in agents) / len(agents):.3f}")
        print()
    
    def _run_simulation(self, engine, system_type: str) -> Dict:
        """Run simulation for either High Tide or AAVE using existing architecture"""
        
        results = {
            "system_type": system_type,
            "time_series_data": {
                "timestamps": [],
                "btc_prices": [],
                "agent_health_factors": [],
                "pool_states": [],
                "rebalancing_events": [] if system_type == "high_tide" else [],
                "liquidation_events": []
            },
            "agent_performance": {},
            "final_metrics": {}
        }
        
        # Get agents list
        if system_type == "high_tide":
            agents = engine.high_tide_agents
        else:  # aave
            agents = engine.aave_agents
        
        print(f"🚀 Starting {system_type.upper()} simulation...")
        print(f"   Duration: {self.config.simulation_duration_minutes:,} minutes")
        print(f"   Agents: {len(agents)}")
        
        # Initialize tracking
        progress_interval = 7 * 1440  # Weekly progress reports
        snapshot_interval = self.config.collect_pool_state_every_n_minutes
        
        # Run simulation minute-by-minute
        for minute in range(self.config.simulation_duration_minutes):
            # Update BTC price
            new_btc_price = self.config.get_btc_price_at_minute(minute, self.btc_price_path)
            engine.state.current_prices[Asset.BTC] = new_btc_price
            
            # Update protocol state
            engine.protocol.current_block = minute
            engine.protocol.accrue_interest()
            
            # Update agent debt interest using ENGINE's method (respects our fixed rate override)
            engine._update_agent_debt_interest(minute)
            
            # Process agents using ENGINE methods (like full_year_sim)
            # NOTE: High Tide agents have built-in weekly deleveraging (1% of YT position)
            # This is automatically triggered in agent.decide_action() -> _check_deleveraging()
            if system_type == "high_tide":
                swap_data = engine._process_high_tide_agents(minute)
                # Check for liquidations
                engine._check_high_tide_liquidations(minute)
            else:
                swap_data = engine._process_aave_agents(minute)
                # Check for AAVE liquidations
                engine._check_aave_liquidations(minute)
            
            # Process pool rebalancing (High Tide only)
            if system_type == "high_tide" and hasattr(engine, 'pool_rebalancer'):
                rebalancing_events = self._process_pool_rebalancing(engine, minute, new_btc_price)
                if rebalancing_events:
                    results["time_series_data"]["rebalancing_events"].extend(rebalancing_events)
            
            # Collect snapshots
            if minute % snapshot_interval == 0:
                self._collect_snapshot(engine, agents, minute, new_btc_price, results)
            
            # Progress reporting
            if minute % progress_interval == 0 and minute > 0:
                days = minute / 1440
                weeks = days / 7
                active_count = sum(1 for a in agents if a.active)
                print(f"   Week {weeks:.0f}/13 (Day {days:.0f}/{self.config.simulation_duration_days}) - "
                      f"BTC: ${new_btc_price:,.0f}, Active agents: {active_count}/{len(agents)}")
        
        # Final analysis
        print(f"✅ {system_type.upper()} simulation complete!")
        self._analyze_simulation_results(engine, agents, results, system_type)
        
        return results
    
    def _process_pool_rebalancing(self, engine: HighTideVaultEngine, minute: int, btc_price: float) -> List[Dict]:
        """Process pool rebalancing for High Tide (following full_year_sim pattern)"""
        if not hasattr(engine, 'pool_rebalancer'):
            print(f"   ⚠️  MIN {minute}: No pool_rebalancer found!")
            return []
        
        # DEBUG: Log rebalancer calls
        if minute == 0 or minute == 10 or minute == 60:
            print(f"   🔍 MIN {minute}: Calling pool rebalancer...")
        
        # Build protocol state (like full_year_sim)
        from tidal_protocol_sim.core.yield_tokens import calculate_true_yield_token_price
        true_yt_price = calculate_true_yield_token_price(minute, 0.10, 1.0)
        pool_yt_price = engine.yield_token_pool.uniswap_pool.get_price()
        deviation_bps = abs((pool_yt_price - true_yt_price) / true_yt_price) * 10000
        
        # DEBUG: Log every 60 minutes to see rebalancer status
        if minute > 0 and minute % 60 == 0:
            print(f"   🔍 MIN {minute}: Pool price=${pool_yt_price:.4f}, True=${true_yt_price:.4f}, Dev={deviation_bps:.0f}bps")
        
        protocol_state = {
            "current_minute": minute,
            "yield_token_true_price": true_yt_price,
            "yield_token_pool_price": pool_yt_price,
            "deviation_bps": deviation_bps,
            "alm_moet_balance": engine.pool_rebalancer.alm_rebalancer.state.moet_balance,
            "alm_yt_balance": engine.pool_rebalancer.alm_rebalancer.state.yield_token_balance
        }
        
        asset_prices = {"BTC": btc_price}
        
        # Process rebalancing (returns list of events)
        rebalancing_events = engine.pool_rebalancer.process_rebalancing(protocol_state, asset_prices)
        
        return rebalancing_events if rebalancing_events else []
    
    def _collect_snapshot(self, engine, agents, minute: int, btc_price: float, results: Dict):
        """Collect snapshot data for analysis"""
        
        # Collect agent health factors
        health_factors = []
        for agent in agents:
            if agent.active:
                health_factors.append({
                    "agent_id": agent.agent_id,
                    "minute": minute,
                    "health_factor": agent.state.health_factor,
                    "btc_amount": agent.state.btc_amount,
                    "moet_debt": agent.state.moet_debt
                })
        
        results["time_series_data"]["timestamps"].append(minute)
        results["time_series_data"]["btc_prices"].append(btc_price)
        results["time_series_data"]["agent_health_factors"].append(health_factors)
        
        # Collect pool state
        pool_state = {
            "minute": minute,
            "btc_price": btc_price,
            "yt_pool_moet_reserve": engine.yield_token_pool.moet_reserve,
            "yt_pool_yt_reserve": engine.yield_token_pool.yield_token_reserve,
            "yt_pool_price": engine.yield_token_pool.uniswap_pool.get_price()
        }
        results["time_series_data"]["pool_states"].append(pool_state)
    
    def _analyze_simulation_results(self, engine, agents, results: Dict, system_type: str):
        """Analyze final simulation results"""
        
        final_metrics = {}
        
        # Count active vs liquidated
        active_agents = [a for a in agents if a.active]
        liquidated_agents = [a for a in agents if not a.active]
        
        final_metrics["total_agents"] = len(agents)
        final_metrics["active_agents"] = len(active_agents)
        final_metrics["liquidated_agents"] = len(liquidated_agents)
        final_metrics["survival_rate"] = len(active_agents) / len(agents)
        
        # Agent performance
        for agent in agents:
            final_btc_price = engine.state.current_prices.get(Asset.BTC, self.config.btc_initial_price)
            asset_prices = {Asset.BTC: final_btc_price, Asset.MOET: 1.0}
            
            portfolio = agent.get_detailed_portfolio_summary(asset_prices, self.config.simulation_duration_minutes)
            results["agent_performance"][agent.agent_id] = portfolio
        
        # Calculate aggregate metrics (matching full_year_sim methodology)
        # Use NET POSITION VALUE approach: (BTC Value + YT Value - Debt) - Initial Investment
        total_interest_paid = 0
        total_net_position_value = 0
        
        initial_btc_price = self.config.btc_initial_price
        final_btc_price = engine.state.current_prices.get(Asset.BTC, self.config.btc_initial_price)
        initial_investment_per_agent = self.config.initial_btc_per_agent * initial_btc_price
        
        for agent_id, perf in results["agent_performance"].items():
            # Interest paid
            if "total_interest_accrued" in perf:
                total_interest_paid += perf["total_interest_accrued"]
            
            # Use NET POSITION VALUE methodology (matches full_year_sim)
            # Net position = BTC value + YT value - Debt
            if "net_position_value" in perf:
                net_position = perf["net_position_value"]
            else:
                # Fallback: calculate net position manually
                btc_value = perf.get("btc_amount", self.config.initial_btc_per_agent) * final_btc_price
                debt = perf.get("current_moet_debt", 0)
                
                # Get YT value
                yt_value = 0
                if "yield_token_portfolio" in perf:
                    yt_value = perf["yield_token_portfolio"].get("total_current_value", 0)
                
                net_position = btc_value + yt_value - debt
            
            # Net position gain = Current net position - Initial investment
            net_position_gain = net_position - initial_investment_per_agent
            total_net_position_value += net_position_gain
        
        # Total yield = Sum of all net position gains
        # This captures: BTC appreciation + YT yield - Debt costs
        total_yield_earned = total_net_position_value
        
        # Calculate net APY
        # Net return = Total yield / Initial Investment
        # Annualized APY = (Net Return / Days) * 365
        days_elapsed = self.config.simulation_duration_days
        if days_elapsed > 0:
            # Initial investment = num_agents * initial_btc_per_agent * btc_price
            initial_investment = len(agents) * initial_investment_per_agent
            net_return_pct = (total_yield_earned / initial_investment) if initial_investment > 0 else 0
            avg_net_apy = (net_return_pct / days_elapsed) * 365  # Annualize
        else:
            avg_net_apy = 0
        
        final_metrics["avg_net_apy"] = avg_net_apy
        final_metrics["total_interest_paid"] = total_interest_paid
        final_metrics["total_yield_earned"] = total_yield_earned
        
        results["final_metrics"] = final_metrics
        
        print(f"\n📊 Final Metrics:")
        print(f"   Active agents: {final_metrics['active_agents']}/{final_metrics['total_agents']} ({final_metrics['survival_rate']:.1%})")
        print(f"   Avg Net APY: {final_metrics['avg_net_apy']:.2%}")
        print(f"   Total interest paid: ${final_metrics['total_interest_paid']:,.0f}")
        print(f"   Total yield earned: ${final_metrics['total_yield_earned']:,.0f}")
    
    def _perform_comparative_analysis(self):
        """Perform comparative analysis between High Tide and AAVE"""
        
        ht_metrics = self.results["high_tide_results"]["final_metrics"]
        aave_metrics = self.results["aave_results"]["final_metrics"]
        
        comparison = {
            "survival_rate_delta": ht_metrics["survival_rate"] - aave_metrics["survival_rate"],
            "net_apy_delta": ht_metrics["avg_net_apy"] - aave_metrics["avg_net_apy"],
            "interest_paid_delta": ht_metrics["total_interest_paid"] - aave_metrics["total_interest_paid"],
            "yield_earned_delta": ht_metrics["total_yield_earned"] - aave_metrics["total_yield_earned"]
        }
        
        self.results["comparative_analysis"] = comparison
        
        print(f"\n🔬 COMPARATIVE ANALYSIS:")
        print(f"   Survival Rate: HT {ht_metrics['survival_rate']:.1%} vs AAVE {aave_metrics['survival_rate']:.1%} "                                                          
              f"(Δ {comparison['survival_rate_delta']:+.1%})")
        print(f"   Avg Net APY: HT {ht_metrics['avg_net_apy']:.2%} vs AAVE {aave_metrics['avg_net_apy']:.2%} "                                                                
              f"(Δ {comparison['net_apy_delta']:+.2%})")
    
    def _save_results(self):
        """Save results to JSON file"""
        
        output_dir = Path("tidal_protocol_sim/results") / self.config.test_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert results to JSON-serializable format
        json_results = self._convert_for_json(self.results)
        
        # Save to file
        output_file = output_dir / f"{self.config.test_name}_results.json"
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
    
    def _convert_for_json(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            # Convert Asset enum keys to strings
            return {str(k) if isinstance(k, Asset) else k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, Asset):
            # Convert Asset enum to string
            return str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return self._convert_for_json(obj.__dict__)
        else:
            return obj
    
    def _generate_all_charts(self):
        """Generate comprehensive chart suite using external charting module"""
        print("\n📊 Generating comprehensive chart suite...")
        
        if self.config.num_monte_carlo_runs > 1:
            # Use Monte Carlo charting functions
            base_case_charts.create_monte_carlo_charts(self)
        else:
            # Use single-run charting functions
            base_case_charts.create_all_charts(self)
    
    def _print_summary(self):
        """Print final summary"""
        
        ht_metrics = self.results["high_tide_results"]["final_metrics"]
        aave_metrics = self.results["aave_results"]["final_metrics"]
        comparison = self.results["comparative_analysis"]
        
        print("\n" + "=" * 80)
        print("📊 FINAL SUMMARY: HIGH TIDE VS AAVE BASE CASE")
        print("=" * 80)
        print(f"\n🌊 HIGH TIDE RESULTS:")
        print(f"   Survival Rate: {ht_metrics['survival_rate']:.1%}")
        print(f"   Avg Net APY: {ht_metrics['avg_net_apy']:.2%}")
        print(f"   Total Interest Paid: ${ht_metrics['total_interest_paid']:,.0f}")
        print(f"   Total Yield Earned: ${ht_metrics['total_yield_earned']:,.0f}")
        
        print(f"\n🏦 AAVE RESULTS:")
        print(f"   Survival Rate: {aave_metrics['survival_rate']:.1%}")
        print(f"   Avg Net APY: {aave_metrics['avg_net_apy']:.2%}")
        print(f"   Total Interest Paid: ${aave_metrics['total_interest_paid']:,.0f}")
        print(f"   Total Yield Earned: ${aave_metrics['total_yield_earned']:,.0f}")
        
        print(f"\n🔬 DELTA (HIGH TIDE - AAVE):")
        print(f"   Survival Rate: {comparison['survival_rate_delta']:+.1%}")
        print(f"   Avg Net APY: {comparison['net_apy_delta']:+.2%}")
        print(f"   Interest Paid: ${comparison['interest_paid_delta']:+,.0f}")
        print(f"   Yield Earned: ${comparison['yield_earned_delta']:+,.0f}")
        print("=" * 80)
    
    def _aggregate_monte_carlo_results(self) -> Dict[str, Any]:
        """Aggregate time-series and metrics across MC runs"""
        
        print(f"   Aggregating {len(self.all_ht_results)} runs...")
        
        num_timesteps = self.config.simulation_duration_days + 1
        
        # Helper function to extract time series data safely
        def extract_time_series(results_list, key):
            """Extract time series, handling missing data"""
            series_list = []
            for r in results_list:
                if key in r.get("time_series_data", {}):
                    series_list.append(r["time_series_data"][key])
            return series_list if series_list else [np.zeros(num_timesteps) for _ in results_list]
        
        # Aggregate High Tide results
        ht_health_factors = extract_time_series(self.all_ht_results, "health_factors")
        ht_collateral = extract_time_series(self.all_ht_results, "total_collateral")
        ht_debt = extract_time_series(self.all_ht_results, "total_debt")
        ht_yt_value = extract_time_series(self.all_ht_results, "total_yt_value")
        ht_net_position = extract_time_series(self.all_ht_results, "net_position_value")
        
        ht_aggregated = {
            "time_series": {
                "avg_health_factors": np.mean(ht_health_factors, axis=0).tolist(),
                "std_health_factors": np.std(ht_health_factors, axis=0).tolist(),
                "p10_health_factors": np.percentile(ht_health_factors, 10, axis=0).tolist(),
                "p90_health_factors": np.percentile(ht_health_factors, 90, axis=0).tolist(),
                
                "avg_collateral": np.mean(ht_collateral, axis=0).tolist(),
                "std_collateral": np.std(ht_collateral, axis=0).tolist(),
                
                "avg_debt": np.mean(ht_debt, axis=0).tolist(),
                "std_debt": np.std(ht_debt, axis=0).tolist(),
                
                "avg_yt_value": np.mean(ht_yt_value, axis=0).tolist(),
                "std_yt_value": np.std(ht_yt_value, axis=0).tolist(),
                
                "avg_net_position": np.mean(ht_net_position, axis=0).tolist(),
                "std_net_position": np.std(ht_net_position, axis=0).tolist(),
            },
            "final_metrics": {
                "avg_survival_rate": float(np.mean([r["final_metrics"]["survival_rate"] for r in self.all_ht_results])),
                "std_survival_rate": float(np.std([r["final_metrics"]["survival_rate"] for r in self.all_ht_results])),
                
                "avg_net_apy": float(np.mean([r["final_metrics"]["avg_net_apy"] for r in self.all_ht_results])),
                "std_net_apy": float(np.std([r["final_metrics"]["avg_net_apy"] for r in self.all_ht_results])),
                "min_net_apy": float(np.min([r["final_metrics"]["avg_net_apy"] for r in self.all_ht_results])),
                "max_net_apy": float(np.max([r["final_metrics"]["avg_net_apy"] for r in self.all_ht_results])),
                
                "avg_total_interest_paid": float(np.mean([r["final_metrics"]["total_interest_paid"] for r in self.all_ht_results])),
                "avg_total_yield_earned": float(np.mean([r["final_metrics"]["total_yield_earned"] for r in self.all_ht_results])),
                
                "distribution_net_apy": [r["final_metrics"]["avg_net_apy"] for r in self.all_ht_results],
                "distribution_survival_rate": [r["final_metrics"]["survival_rate"] for r in self.all_ht_results],
            },
            "all_runs": self.all_ht_results  # Keep individual run data for detailed analysis
        }
        
        # Aggregate AAVE results (same structure)
        aave_health_factors = extract_time_series(self.all_aave_results, "health_factors")
        aave_collateral = extract_time_series(self.all_aave_results, "total_collateral")
        aave_debt = extract_time_series(self.all_aave_results, "total_debt")
        aave_yt_value = extract_time_series(self.all_aave_results, "total_yt_value")
        aave_net_position = extract_time_series(self.all_aave_results, "net_position_value")
        
        aave_aggregated = {
            "time_series": {
                "avg_health_factors": np.mean(aave_health_factors, axis=0).tolist(),
                "std_health_factors": np.std(aave_health_factors, axis=0).tolist(),
                "p10_health_factors": np.percentile(aave_health_factors, 10, axis=0).tolist(),
                "p90_health_factors": np.percentile(aave_health_factors, 90, axis=0).tolist(),
                
                "avg_collateral": np.mean(aave_collateral, axis=0).tolist(),
                "std_collateral": np.std(aave_collateral, axis=0).tolist(),
                
                "avg_debt": np.mean(aave_debt, axis=0).tolist(),
                "std_debt": np.std(aave_debt, axis=0).tolist(),
                
                "avg_yt_value": np.mean(aave_yt_value, axis=0).tolist(),
                "std_yt_value": np.std(aave_yt_value, axis=0).tolist(),
                
                "avg_net_position": np.mean(aave_net_position, axis=0).tolist(),
                "std_net_position": np.std(aave_net_position, axis=0).tolist(),
            },
            "final_metrics": {
                "avg_survival_rate": float(np.mean([r["final_metrics"]["survival_rate"] for r in self.all_aave_results])),
                "std_survival_rate": float(np.std([r["final_metrics"]["survival_rate"] for r in self.all_aave_results])),
                
                "avg_net_apy": float(np.mean([r["final_metrics"]["avg_net_apy"] for r in self.all_aave_results])),
                "std_net_apy": float(np.std([r["final_metrics"]["avg_net_apy"] for r in self.all_aave_results])),
                "min_net_apy": float(np.min([r["final_metrics"]["avg_net_apy"] for r in self.all_aave_results])),
                "max_net_apy": float(np.max([r["final_metrics"]["avg_net_apy"] for r in self.all_aave_results])),
                
                "avg_total_interest_paid": float(np.mean([r["final_metrics"]["total_interest_paid"] for r in self.all_aave_results])),
                "avg_total_yield_earned": float(np.mean([r["final_metrics"]["total_yield_earned"] for r in self.all_aave_results])),
                
                "distribution_net_apy": [r["final_metrics"]["avg_net_apy"] for r in self.all_aave_results],
                "distribution_survival_rate": [r["final_metrics"]["survival_rate"] for r in self.all_aave_results],
            },
            "all_runs": self.all_aave_results
        }
        
        print(f"   ✅ Aggregation complete")
        print(f"   HT Avg Net APY: {ht_aggregated['final_metrics']['avg_net_apy']:.2%} ± {ht_aggregated['final_metrics']['std_net_apy']:.2%}")
        print(f"   AAVE Avg Net APY: {aave_aggregated['final_metrics']['avg_net_apy']:.2%} ± {aave_aggregated['final_metrics']['std_net_apy']:.2%}")
        
        return {
            "num_runs": self.config.num_monte_carlo_runs,
            "btc_price_paths": self.all_btc_paths,
            "high_tide": ht_aggregated,
            "aave": aave_aggregated
        }
    
    def _perform_comparative_analysis_mc(self):
        """Comparative analysis for Monte Carlo results"""
        
        ht_metrics = self.aggregated_results["high_tide"]["final_metrics"]
        aave_metrics = self.aggregated_results["aave"]["final_metrics"]
        
        comparison = {
            "survival_rate_delta": ht_metrics["avg_survival_rate"] - aave_metrics["avg_survival_rate"],
            "net_apy_delta": ht_metrics["avg_net_apy"] - aave_metrics["avg_net_apy"],
            "interest_paid_delta": ht_metrics["avg_total_interest_paid"] - aave_metrics["avg_total_interest_paid"],
            "yield_earned_delta": ht_metrics["avg_total_yield_earned"] - aave_metrics["avg_total_yield_earned"]
        }
        
        self.results["comparative_analysis"] = comparison
        
        print(f"\n🔬 COMPARATIVE ANALYSIS (Monte Carlo Averages):")
        print(f"   Survival Rate: HT {ht_metrics['avg_survival_rate']:.1%} ± {ht_metrics['std_survival_rate']:.1%} vs "
              f"AAVE {aave_metrics['avg_survival_rate']:.1%} ± {aave_metrics['std_survival_rate']:.1%} "
              f"(Δ {comparison['survival_rate_delta']:+.1%})")
        print(f"   Avg Net APY: HT {ht_metrics['avg_net_apy']:.2%} ± {ht_metrics['std_net_apy']:.2%} vs "
              f"AAVE {aave_metrics['avg_net_apy']:.2%} ± {aave_metrics['std_net_apy']:.2%} "
              f"(Δ {comparison['net_apy_delta']:+.2%})")


def main():
    """Main execution function"""
    
    # Create configuration
    config = BaseCaseConfig()
    
    # Run comparison
    comparison = BaseCaseComparison(config)
    results = comparison.run_comparison()
    
    return results


if __name__ == "__main__":
    main()
