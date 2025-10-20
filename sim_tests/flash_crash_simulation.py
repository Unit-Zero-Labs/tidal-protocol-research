#!/usr/bin/env python3
"""
Flash Crash Simulation - YT and BTC Simultaneous Crash

Simulates a mechanical flash crash of YT (Yield Token) with cascading market-structure effects:
- 25-minute crash window with liquidity evaporation
- Oracle/index mispricing and forced liquidations
- Temporary MOET stablecoin depeg
- YT continues rebasing at 10% APR throughout crash
- Recovery dynamics over 2+ days

Timeline:
- Day 1, 00:00-15:00: Normal operations (900 minutes)
- Day 1, 15:00-15:25: FLASH CRASH (25 minutes)
- Day 1, 15:25-17:25: Recovery phase (120 minutes)
- Day 1-3: Long-term stability analysis (remaining time)
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
from typing import Dict, List, Any, Tuple, Optional

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tidal_protocol_sim.engine.high_tide_vault_engine import HighTideVaultEngine, HighTideConfig
from tidal_protocol_sim.agents.high_tide_agent import HighTideAgent
from tidal_protocol_sim.agents.pool_rebalancer import PoolRebalancerManager
from tidal_protocol_sim.agents.moet_arbitrage_agent import MoetArbitrageAgent, create_moet_arbitrage_agents
from tidal_protocol_sim.core.protocol import TidalProtocol, Asset
from tidal_protocol_sim.core.yield_tokens import calculate_true_yield_token_price, YieldTokenPool


class FlashCrashSimConfig:
    """Configuration for Flash Crash Simulation with YT and BTC simultaneous crash"""
    
    def __init__(self, scenario: str = "moderate"):
        # Base simulation parameters - 3 DAYS TOTAL
        self.test_name = f"Flash_Crash_YT_BTC_{scenario.title()}_Scenario"
        self.simulation_duration_minutes = 3 * 24 * 60  # 3 days = 4,320 minutes
        
        # Agent configuration - 150 agents with ~$100M total debt
        self.num_agents = 150
        self.target_total_debt = 100_000_000  # $100M system debt
        self.avg_debt_per_agent = self.target_total_debt / self.num_agents  # ~$667k per agent
        
        # Flash crash timing - WITHIN FIRST DAY (15:00 on Day 1)
        self.crash_start_minute = 15 * 60 + 0  # 15:00 on Day 1 (900 minutes from start)
        self.crash_duration_minutes = 25      # 25-minute crash window (15:00-15:25)
        self.crash_end_minute = self.crash_start_minute + self.crash_duration_minutes  # 925 minutes
        self.recovery_duration_minutes = 120  # 2-hour recovery phase (15:25-17:25)
        
        # Scenario-based crash parameters
        self.scenario = scenario
        if scenario == "mild":
            self.yt_crash_magnitude = 0.35      # 35% drop
            self.btc_crash_magnitude = 0.12     # 12% drop
            self.yt_wick_magnitude = 0.25       # Wicks to 75 cents
            self.liquidity_reduction_peak = 0.60 # 60% liquidity reduction
            self.oracle_outlier_magnitude = 0.15 # 15% outliers
        elif scenario == "moderate":
            self.yt_crash_magnitude = 0.50      # 50% drop
            self.btc_crash_magnitude = 0.20     # 20% drop
            self.yt_wick_magnitude = 0.40       # Wicks to 60 cents
            self.liquidity_reduction_peak = 0.70 # 70% liquidity reduction
            self.oracle_outlier_magnitude = 0.25 # 25% outliers
        elif scenario == "severe":
            self.yt_crash_magnitude = 0.70      # 70% drop (maximum)
            self.btc_crash_magnitude = 0.25     # 25% drop
            self.yt_wick_magnitude = 0.50       # Wicks to 50 cents
            self.liquidity_reduction_peak = 0.80 # 80% liquidity reduction
            self.oracle_outlier_magnitude = 0.30 # 30% outliers
        
        # Liquidity evaporation parameters
        self.liquidity_reduction_start = 0.30   # Start with 30% reduction
        self.one_sided_selling_pressure = True  # Everyone wants to sell YT
        self.rebalancer_effectiveness_factor = 0.2  # Rebalancers 20% as effective during crash
        
        # Oracle mispricing parameters
        self.oracle_outlier_duration_minutes = 5  # 5-minute outlier periods
        self.extreme_tick_magnitude = 0.45     # Single 45% off-price tick
        self.outlier_frequency_minutes = 3     # Outlier every 3 minutes during crash
        
        # Recovery parameters (gradual over 2 hours)
        self.liquidity_recovery_50pct_minute = 60  # 50% liquidity back by t+60min
        self.liquidity_recovery_90pct_minute = 120 # 90% liquidity back by t+120min
        self.yt_rebound_percentage = 0.70      # YT rebounds to 70% of pre-crash
        
        # Pool configurations - KEEP EXISTING MOET:YT POOL SIZE
        self.moet_btc_pool_config = {
            "size": 25_000_000,  # $25M liquidation pool (scaled for $100M system)
            "concentration": 0.80,
            "fee_tier": 0.003,
            "tick_spacing": 60,
            "pool_name": "MOET:BTC"
        }
        
        # KEEP EXISTING YT POOL CONFIGURATION
        self.moet_yt_pool_config = {
            "size": 500_000,     # $500K pool (UNCHANGED)
            "concentration": 0.95, # 95% concentration at 1:1 peg
            "token0_ratio": 0.75,  # 75% MOET, 25% YT
            "fee_tier": 0.0005,    # 0.05% fee tier
            "tick_spacing": 10,
            "pool_name": "MOET:Yield_Token"
        }
        
        # Rebalancer configuration - matches existing system
        self.enable_pool_arbing = True
        self.alm_rebalance_interval_minutes = 720  # 12 hours (matches existing)
        self.algo_deviation_threshold_bps = 50.0   # 50 bps (matches existing)
        
        # Advanced system features
        self.enable_advanced_moet_system = True
        self.enable_forced_liquidations = True
        self.enable_adl_mechanism = True
        self.insurance_fund_multiplier = 3.0  # Test with 3x larger insurance fund
        
        # Yield token parameters - UNCHANGED
        self.yield_apr = 0.10  # 10% APR (continues during crash)
        self.use_direct_minting_for_initial = True
        
        # Logging configuration - more frequent during crash
        self.detailed_logging = True
        self.log_every_n_minutes = 60  # Log hourly during 3-day period
        self.collect_pool_state_every_n_minutes = 15  # Every 15 minutes for detailed crash analysis
        self.track_all_rebalancing_events = True
        
        # Progress reporting
        self.progress_report_every_n_minutes = 360  # Every 6 hours during 3-day test
        
        # Chart generation
        self.generate_charts = True
        self.save_detailed_csv = True


class LiquidityEvaporationManager:
    """Manages liquidity reduction during flash crash by controlling rebalancer effectiveness"""
    
    def __init__(self, config: FlashCrashSimConfig):
        self.config = config
        self.original_alm_balance = None
        self.original_algo_balance = None
        self.original_alm_max_rebalance = None
        self.original_algo_max_rebalance = None
        self.current_liquidity_factor = 1.0
        self.initialized = False
        
    def initialize_original_settings(self, pool_rebalancer_manager):
        """Store original rebalancer settings before crash"""
        if self.initialized:
            return
            
        self.original_alm_balance = pool_rebalancer_manager.alm_rebalancer.state.moet_balance
        self.original_algo_balance = pool_rebalancer_manager.algo_rebalancer.state.moet_balance
        self.original_alm_max_rebalance = pool_rebalancer_manager.alm_rebalancer.state.max_single_rebalance
        self.original_algo_max_rebalance = pool_rebalancer_manager.algo_rebalancer.state.max_single_rebalance
        self.initialized = True
        
        print(f"🔧 Liquidity Manager initialized:")
        print(f"   ALM Balance: ${self.original_alm_balance:,.0f}, Max Rebalance: ${self.original_alm_max_rebalance:,.0f}")
        print(f"   Algo Balance: ${self.original_algo_balance:,.0f}, Max Rebalance: ${self.original_algo_max_rebalance:,.0f}")
        
    def update_liquidity_during_crash(self, current_minute: int, pool_rebalancer_manager):
        """Reduce rebalancer effectiveness during crash to simulate liquidity evaporation"""
        
        if not self.initialized:
            self.initialize_original_settings(pool_rebalancer_manager)
        
        if not self._is_crash_active(current_minute):
            # During recovery, gradually restore liquidity
            if current_minute > self.config.crash_end_minute:
                self._process_liquidity_recovery(current_minute, pool_rebalancer_manager)
            return
            
        # Calculate liquidity reduction factor during crash
        crash_progress = (current_minute - self.config.crash_start_minute) / self.config.crash_duration_minutes
        
        if crash_progress <= 0.5:
            # Linear reduction first half of crash
            reduction = self.config.liquidity_reduction_start + (
                (self.config.liquidity_reduction_peak - self.config.liquidity_reduction_start) * (crash_progress * 2)
            )
        else:
            # Exponential reduction second half (liquidity evaporates faster)
            exponential_factor = ((crash_progress - 0.5) * 2) ** 2
            reduction = self.config.liquidity_reduction_peak * (1 + exponential_factor * 0.2)
            reduction = min(reduction, 0.95)  # Cap at 95% reduction
        
        self.current_liquidity_factor = 1.0 - reduction
        
        # Reduce rebalancer capital and effectiveness
        self._apply_liquidity_reduction(pool_rebalancer_manager, self.current_liquidity_factor)
        
        # Apply one-sided selling pressure
        if self.config.one_sided_selling_pressure:
            self._apply_one_sided_pressure(pool_rebalancer_manager, reduction)
    
    def _apply_liquidity_reduction(self, pool_rebalancer_manager, liquidity_factor):
        """Reduce rebalancer capital and maximum rebalance sizes"""
        
        # Reduce available MOET balances (simulates capital flight)
        pool_rebalancer_manager.alm_rebalancer.state.moet_balance = (
            self.original_alm_balance * liquidity_factor
        )
        pool_rebalancer_manager.algo_rebalancer.state.moet_balance = (
            self.original_algo_balance * liquidity_factor
        )
        
        # Reduce maximum single rebalance amounts (simulates reduced market depth)
        pool_rebalancer_manager.alm_rebalancer.state.max_single_rebalance = (
            self.original_alm_max_rebalance * liquidity_factor
        )
        pool_rebalancer_manager.algo_rebalancer.state.max_single_rebalance = (
            self.original_algo_max_rebalance * liquidity_factor
        )
        
        # Increase minimum rebalance thresholds (harder to trigger rebalancing)
        base_min_rebalance = 1000.0
        pool_rebalancer_manager.alm_rebalancer.state.min_rebalance_amount = (
            base_min_rebalance / liquidity_factor  # Higher threshold when liquidity is low
        )
        pool_rebalancer_manager.algo_rebalancer.state.min_rebalance_amount = (
            base_min_rebalance / liquidity_factor
        )
    
    def _apply_one_sided_pressure(self, pool_rebalancer_manager, reduction_factor):
        """Simulate one-sided selling pressure - everyone wants to sell YT"""
        
        # During crash, make it much harder for rebalancers to buy YT (simulate lack of buyers)
        # This is implemented by reducing their willingness to execute "buy_yt_with_moet" operations
        
        # Increase the minimum profit threshold for YT purchases during crash
        original_profit_threshold = 0.001  # 0.1% normally
        crash_profit_threshold = original_profit_threshold * (1 + reduction_factor * 10)  # Much higher threshold
        
        # Apply to both rebalancers (if they have this attribute)
        if hasattr(pool_rebalancer_manager.alm_rebalancer.state, 'min_profit_threshold'):
            pool_rebalancer_manager.alm_rebalancer.state.min_profit_threshold = crash_profit_threshold
        if hasattr(pool_rebalancer_manager.algo_rebalancer.state, 'min_profit_threshold'):
            pool_rebalancer_manager.algo_rebalancer.state.min_profit_threshold = crash_profit_threshold
    
    def _is_crash_active(self, current_minute: int) -> bool:
        """Check if we're currently in the crash window"""
        return self.config.crash_start_minute <= current_minute <= self.config.crash_end_minute
    
    def _process_liquidity_recovery(self, current_minute: int, pool_rebalancer_manager):
        """Gradually restore liquidity during recovery phase"""
        
        recovery_elapsed = current_minute - self.config.crash_end_minute
        
        if recovery_elapsed <= 60:  # First hour: 50% recovery
            recovery_progress = recovery_elapsed / 60 * 0.5
        elif recovery_elapsed <= 120:  # Second hour: up to 90% recovery
            recovery_progress = 0.5 + ((recovery_elapsed - 60) / 60) * 0.4
        else:  # After 2 hours: full recovery
            recovery_progress = 1.0
        
        # Restore liquidity factor
        restored_factor = self.current_liquidity_factor + (1.0 - self.current_liquidity_factor) * recovery_progress
        self._apply_liquidity_reduction(pool_rebalancer_manager, restored_factor)


class OracleMispricingEngine:
    """Injects oracle outliers and extreme price ticks during crash"""
    
    def __init__(self, config: FlashCrashSimConfig):
        self.config = config
        self.outlier_events = []
        self.extreme_tick_injected = False
        self.last_outlier_minute = 0
        
    def get_manipulated_yt_price(self, current_minute: int, true_yt_price: float) -> float:
        """Return manipulated YT price during crash periods"""
        if not self._is_crash_active(current_minute):
            return true_yt_price
            
        # Inject extreme tick once during crash (at minute 10 of crash)
        if not self.extreme_tick_injected and current_minute == self.config.crash_start_minute + 10:
            self.extreme_tick_injected = True
            manipulated_price = true_yt_price * (1 - self.config.extreme_tick_magnitude)
            self._log_outlier_event(current_minute, "extreme_tick", manipulated_price, true_yt_price)
            return manipulated_price
            
        # Regular outlier periods every N minutes during crash
        if (current_minute - self.last_outlier_minute >= self.config.outlier_frequency_minutes and 
            current_minute <= self.config.crash_start_minute + self.config.oracle_outlier_duration_minutes):
            
            self.last_outlier_minute = current_minute
            manipulated_price = true_yt_price * (1 - self.config.oracle_outlier_magnitude)
            self._log_outlier_event(current_minute, "outlier", manipulated_price, true_yt_price)
            return manipulated_price
            
        return true_yt_price
    
    def _is_crash_active(self, current_minute: int) -> bool:
        """Check if we're currently in the crash window"""
        return self.config.crash_start_minute <= current_minute <= self.config.crash_end_minute
    
    def _log_outlier_event(self, minute: int, event_type: str, manipulated_price: float, true_price: float):
        """Log oracle manipulation events"""
        deviation_pct = abs(manipulated_price - true_price) / true_price
        event = {
            "minute": minute,
            "type": event_type,
            "true_price": true_price,
            "manipulated_price": manipulated_price,
            "deviation_pct": deviation_pct,
            "deviation_bps": deviation_pct * 10000
        }
        self.outlier_events.append(event)
        print(f"🚨 Oracle {event_type}: YT price ${manipulated_price:.4f} (true: ${true_price:.4f}, -{deviation_pct:.1%})")


class ForcedLiquidationEngine:
    """Handles forced liquidations during crash periods"""
    
    def __init__(self, config: FlashCrashSimConfig):
        self.config = config
        self.liquidation_events = []
        
    def process_crash_liquidations(self, current_minute: int, agents: List[HighTideAgent], asset_prices: Dict[Asset, float]) -> List[Dict]:
        """Process forced liquidations during crash"""
        if not self._is_crash_active(current_minute):
            return []
        
        liquidations = []
        
        for agent in agents:
            if not agent.active:
                continue
                
            # Check if agent needs liquidation (HF < 1.0)
            if agent.state.health_factor < 1.0:
                liquidation_result = self._execute_forced_liquidation(agent, current_minute, asset_prices)
                if liquidation_result:
                    liquidations.append(liquidation_result)
                    
        return liquidations
    
    def _execute_forced_liquidation(self, agent: HighTideAgent, current_minute: int, asset_prices: Dict[Asset, float]) -> Optional[Dict]:
        """Execute forced liquidation with crash conditions"""
        
        btc_price = asset_prices.get(Asset.BTC, 100_000.0)
        btc_collateral = agent.state.supplied_balances.get(Asset.BTC, 0.0)
        moet_debt = agent.state.moet_debt
        
        # Calculate liquidation amounts (50% of collateral, 5% bonus)
        btc_to_liquidate = btc_collateral * 0.5
        liquidation_bonus = 0.05
        
        # During crash, liquidation slippage is much higher
        crash_slippage_multiplier = 2.0  # 2x normal slippage during crash
        base_slippage = 0.02  # 2% base slippage
        total_slippage = base_slippage * crash_slippage_multiplier
        
        # Calculate effective liquidation value after slippage
        btc_value_gross = btc_to_liquidate * btc_price
        btc_value_net = btc_value_gross * (1 - total_slippage)
        
        # Update agent state
        agent.state.supplied_balances[Asset.BTC] -= btc_to_liquidate
        debt_reduction = min(btc_value_net, moet_debt)
        agent.state.moet_debt -= debt_reduction
        
        # Mark agent as liquidated if debt is fully repaid
        if agent.state.moet_debt <= 100:  # Small threshold
            agent.active = False
        
        liquidation_event = {
            "minute": current_minute,
            "agent_id": agent.agent_id,
            "btc_liquidated": btc_to_liquidate,
            "btc_value_gross": btc_value_gross,
            "btc_value_net": btc_value_net,
            "debt_reduction": debt_reduction,
            "slippage_pct": total_slippage,
            "health_factor_before": agent.state.health_factor,
            "agent_liquidated": not agent.active
        }
        
        self.liquidation_events.append(liquidation_event)
        
        print(f"⚡ Forced liquidation: {agent.agent_id} - {btc_to_liquidate:.4f} BTC (${btc_value_gross:,.0f} → ${btc_value_net:,.0f})")
        
        return liquidation_event
    
    def _is_crash_active(self, current_minute: int) -> bool:
        """Check if we're currently in the crash window"""
        return self.config.crash_start_minute <= current_minute <= self.config.crash_end_minute


class FlashCrashSimulation:
    """Main flash crash simulation extending full_year_sim architecture"""
    
    def __init__(self, config: FlashCrashSimConfig):
        self.config = config
        self.results = {
            "test_metadata": {
                "test_name": config.test_name,
                "timestamp": datetime.now().isoformat(),
                "scenario": config.scenario,
                "crash_start_minute": config.crash_start_minute,
                "crash_duration": config.crash_duration_minutes,
                "target_system_debt": config.target_total_debt,
                "num_agents": config.num_agents,
                "yt_crash_magnitude": config.yt_crash_magnitude,
                "btc_crash_magnitude": config.btc_crash_magnitude
            },
            "crash_events": [],
            "liquidation_events": [],
            "oracle_events": [],
            "recovery_metrics": {},
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
        
        # Initialize specialized managers
        self.liquidity_manager = LiquidityEvaporationManager(config)
        self.oracle_engine = OracleMispricingEngine(config)
        self.forced_liquidation_engine = ForcedLiquidationEngine(config)
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
    def run_flash_crash_test(self) -> Dict[str, Any]:
        """Run the complete flash crash simulation"""
        
        print("⚡ FLASH CRASH SIMULATION")
        print("=" * 70)
        print(f"📊 Scenario: {self.config.scenario.upper()}")
        print(f"💥 Crash Window: Day 1, 15:00-15:25 (minutes {self.config.crash_start_minute}-{self.config.crash_end_minute})")
        print(f"📉 YT Drop: {self.config.yt_crash_magnitude:.0%}, BTC Drop: {self.config.btc_crash_magnitude:.0%}")
        print(f"👥 Agents: {self.config.num_agents} with ~${self.config.target_total_debt:,.0f} total debt")
        print(f"💧 Liquidity Evaporation: {self.config.liquidity_reduction_start:.0%} → {self.config.liquidity_reduction_peak:.0%}")
        print(f"⏱️  Duration: {self.config.simulation_duration_minutes:,} minutes ({self.config.simulation_duration_minutes//1440} days)")
        print()
        
        # Create engine with 150 large agents
        engine = self._create_flash_crash_engine()
        
        # Run simulation with crash mechanics
        simulation_results = self._run_crash_simulation_with_tracking(engine)
        
        # Store simulation results
        self.results["simulation_results"] = simulation_results
        
        # Analyze crash results
        self._analyze_crash_results(engine)
        
        # Save results
        self._save_crash_results()
        
        # Generate crash-specific charts
        if self.config.generate_charts:
            self._generate_crash_charts()
        
        print("\n✅ Flash crash simulation completed!")
        self._print_crash_summary()
        
        return self.results
        
    def _create_flash_crash_engine(self) -> HighTideVaultEngine:
        """Create engine with 150 agents targeting $100M total debt"""
        
        # Create High Tide config for large system
        ht_config = HighTideConfig()
        ht_config.num_agents = self.config.num_agents
        ht_config.btc_initial_price = 100_000.0  # Start at $100k BTC
        ht_config.btc_decline_duration = self.config.simulation_duration_minutes
        ht_config.btc_decline_percent = 0.0  # No decline initially - we'll control BTC price manually
        
        # Pool configurations
        ht_config.moet_btc_pool_size = self.config.moet_btc_pool_config["size"]
        ht_config.yield_token_pool_size = self.config.moet_yt_pool_config["size"]
        
        # CRITICAL: Enable advanced MOET system
        ht_config.enable_advanced_moet = self.config.enable_advanced_moet_system
        
        # Create engine
        engine = HighTideVaultEngine(ht_config)
        
        # CRITICAL: Set up the engine's protocol with advanced MOET system
        if self.config.enable_advanced_moet_system:
            engine.protocol.enable_advanced_moet = True
            # Initialize MOET system properly
            if hasattr(engine.protocol, 'moet_stablecoin'):
                engine.protocol.moet_stablecoin.enable_advanced_system = True
        
        # Create 150 large agents with ~$667k debt each
        large_agents = self._create_large_debt_agents(engine)
        engine.high_tide_agents = large_agents
        
        # CRITICAL: Set engine reference for all agents AND register them in engine.agents
        for agent in large_agents:
            agent.engine = engine
            # Register agent in engine's tracking system
            engine.agents[agent.agent_id] = agent
        
        # Setup pool rebalancing
        pool_rebalancer = PoolRebalancerManager(
            alm_interval_minutes=self.config.alm_rebalance_interval_minutes,
            algo_threshold_bps=self.config.algo_deviation_threshold_bps,
            enable_pool_replenishment=True,
            target_pool_size=self.config.moet_yt_pool_config["size"]
        )
        pool_rebalancer.set_enabled(self.config.enable_pool_arbing)
        
        # Create yield token pool with specified configuration
        yield_token_pool = YieldTokenPool(
            total_pool_size=self.config.moet_yt_pool_config["size"],
            token0_ratio=self.config.moet_yt_pool_config["token0_ratio"],
            concentration=self.config.moet_yt_pool_config["concentration"]
        )
        
        # Set pool references
        pool_rebalancer.set_yield_token_pool(yield_token_pool)
        engine.yield_token_pool = yield_token_pool
        engine.pool_rebalancer = pool_rebalancer
        
        # Create arbitrage agents for MOET peg maintenance
        if self.config.enable_advanced_moet_system:
            arbitrage_agents = create_moet_arbitrage_agents(10, 50_000.0)  # 10 agents with $50k each
            engine.arbitrage_agents = arbitrage_agents
            
            # Set engine reference for arbitrage agents
            for agent in arbitrage_agents:
                agent.engine = engine
        
        # Initialize positions to reach $100M total debt
        self._setup_large_system_positions(engine)
        
        return engine
        
    def _create_large_debt_agents(self, engine) -> List[HighTideAgent]:
        """Create 150 agents with large BTC deposits for ~$100M system debt"""
        
        # Use the existing agent creation system but with custom parameters
        from tidal_protocol_sim.agents.high_tide_agent import create_high_tide_agents
        
        # Create agents using the existing system (but override the count)
        agents = []
        
        for i in range(self.config.num_agents):
            # Create agent with aggressive but safe profile for large positions
            agent = HighTideAgent(
                f"flash_crash_agent_{i+1}",
                initial_hf=1.15,      # Slightly higher initial HF for large positions
                rebalancing_hf=1.05,  # Aggressive rebalancing threshold
                target_hf=1.08,       # Tight target for maximum leverage
                initial_balance=100_000.0,  # Standard initial balance
                yield_token_pool=engine.yield_token_pool if hasattr(engine, 'yield_token_pool') else None
            )
            
            agents.append(agent)
            
        return agents
    
    def _setup_large_system_positions(self, engine):
        """Initialize agent positions to reach target debt levels"""
        
        print(f"🏗️  Setting up large system positions...")
        
        total_debt_target = self.config.target_total_debt
        debt_per_agent = total_debt_target / len(engine.high_tide_agents)
        
        for agent in engine.high_tide_agents:
            # Calculate required debt based on BTC collateral and target HF
            btc_collateral = agent.state.supplied_balances.get(Asset.BTC, 0.0)
            btc_value = btc_collateral * 100_000.0  # $100k BTC price
            
            # Target debt based on initial health factor
            target_debt = btc_value * 0.8 / agent.state.initial_health_factor  # 80% collateral factor
            target_debt = min(target_debt, debt_per_agent * 1.2)  # Cap at 120% of average
            
            # Set debt and MOET balance
            agent.state.moet_debt = target_debt
            agent.state.borrowed_balances[Asset.MOET] = target_debt
            agent.state.token_balances[Asset.MOET] = target_debt
            
            # Purchase yield tokens with MOET (at minute 0, use direct minting)
            if engine.yield_token_pool:
                try:
                    yt_tokens = agent.state.yield_token_manager.mint_yield_tokens(
                        target_debt, 0, use_direct_minting=True
                    )
                    if yt_tokens:
                        print(f"   {agent.agent_id}: ${btc_value:,.0f} BTC → ${target_debt:,.0f} debt → YT tokens")
                except Exception as e:
                    print(f"   Warning: YT minting failed for {agent.agent_id}: {e}")
            
            # Update health factor
            engine._update_agent_health_factor(agent)
        
        # Calculate actual total debt
        actual_total_debt = sum(agent.state.moet_debt for agent in engine.high_tide_agents)
        avg_debt = actual_total_debt / len(engine.high_tide_agents)
        
        print(f"✅ System initialized:")
        print(f"   Total debt: ${actual_total_debt:,.0f} (target: ${total_debt_target:,.0f})")
        print(f"   Average debt per agent: ${avg_debt:,.0f}")
        print(f"   Active agents: {len([a for a in engine.high_tide_agents if a.active])}")
        
    def _run_crash_simulation_with_tracking(self, engine):
        """Run simulation with detailed crash event tracking using engine's built-in loop"""
        
        print(f"🚀 Starting {self.config.simulation_duration_minutes:,}-minute simulation...")
        
        # Initialize tracking variables
        btc_price_history = []
        yt_price_history = []
        crash_events = []
        
        # Override the engine's BTC price manager to use our crash dynamics
        original_update_method = engine.btc_price_manager.update_btc_price
        
        def crash_btc_price_update(minute: int) -> float:
            """Override BTC price update with crash dynamics"""
            btc_price = self._calculate_btc_price_during_crash(minute)
            engine.btc_price_manager.current_price = btc_price
            return btc_price
        
        # Replace the BTC price update method
        engine.btc_price_manager.update_btc_price = crash_btc_price_update
        
        # Run the simulation using the engine's built-in loop
        for minute in range(self.config.simulation_duration_minutes):
            engine.current_step = minute
            
            # Update BTC price using our crash dynamics
            btc_price = crash_btc_price_update(minute)
            engine.state.current_prices[Asset.BTC] = btc_price
            btc_price_history.append(btc_price)
            
            # Update YT price with oracle manipulation
            true_yt_price = calculate_true_yield_token_price(minute, self.config.yield_apr, 1.0)
            manipulated_yt_price = self.oracle_engine.get_manipulated_yt_price(minute, true_yt_price)
            yt_price_history.append({"minute": minute, "true": true_yt_price, "manipulated": manipulated_yt_price})
            
            # Apply liquidity evaporation to rebalancers
            if hasattr(engine, 'pool_rebalancer'):
                self.liquidity_manager.update_liquidity_during_crash(minute, engine.pool_rebalancer)
            
            # Update protocol state
            engine.protocol.current_block = minute
            engine.protocol.accrue_interest()
            
            # Process MOET system updates
            if hasattr(engine.protocol, 'process_moet_system_update'):
                engine.protocol.process_moet_system_update(minute)
            
            # Update agent debt interest
            if hasattr(engine, '_update_agent_debt_interest'):
                engine._update_agent_debt_interest(minute)
            
            # Process High Tide agents using engine's method
            if hasattr(engine, '_process_high_tide_agents'):
                swap_data = engine._process_high_tide_agents(minute)
            
            # Process forced liquidations during crash
            if self._is_crash_active(minute):
                liquidation_events = self.forced_liquidation_engine.process_crash_liquidations(
                    minute, engine.high_tide_agents, engine.state.current_prices
                )
                self.results["liquidation_events"].extend(liquidation_events)
            
            # Process pool rebalancing
            if hasattr(engine, 'pool_rebalancer') and engine.pool_rebalancer and engine.pool_rebalancer.enabled:
                try:
                    rebalancing_events = engine.pool_rebalancer.process_rebalancing(
                        {"current_minute": minute}, engine.state.current_prices
                    )
                    if rebalancing_events:
                        for event in rebalancing_events:
                            event["minute"] = minute
                        self.results["rebalancing_events"]["alm_rebalances"].extend(
                            [e for e in rebalancing_events if e.get("rebalancer") == "ALM"]
                        )
                        self.results["rebalancing_events"]["algo_rebalances"].extend(
                            [e for e in rebalancing_events if e.get("rebalancer") == "ALGO"]
                        )
                except Exception as e:
                    if minute % 100 == 0:  # Log occasionally
                        print(f"   Warning: Pool rebalancing failed at minute {minute}: {e}")
            
            # Process arbitrage agents (MOET peg maintenance)
            if hasattr(engine, 'arbitrage_agents') and engine.arbitrage_agents:
                self._process_arbitrage_agents(minute, engine)
            
            # Record detailed crash metrics
            if minute % self.config.collect_pool_state_every_n_minutes == 0:
                self._record_crash_metrics(minute, engine, btc_price, true_yt_price, manipulated_yt_price)
            
            # Progress reporting
            if minute % self.config.progress_report_every_n_minutes == 0:
                self._print_progress_report(minute, engine, btc_price)
        
        # Restore original BTC price update method
        engine.btc_price_manager.update_btc_price = original_update_method
        
        # Compile simulation results
        simulation_results = {
            "btc_price_history": btc_price_history,
            "yt_price_history": yt_price_history,
            "crash_events": crash_events,
            "oracle_events": self.oracle_engine.outlier_events,
            "liquidity_events": [],  # Could add liquidity change events
            "final_agent_count": len([a for a in engine.high_tide_agents if a.active]),
            "total_liquidations": len(self.results["liquidation_events"])
        }
        
        return simulation_results
    
    def _calculate_btc_price_during_crash(self, minute: int) -> float:
        """Calculate BTC price with crash dynamics"""
        base_price = 100_000.0  # Starting BTC price
        
        if not self._is_crash_active(minute):
            return base_price
        
        # During crash, apply crash magnitude
        crash_progress = (minute - self.config.crash_start_minute) / self.config.crash_duration_minutes
        
        # Smooth crash curve (not instant)
        if crash_progress <= 0.3:
            # First 30% of crash: gradual decline
            price_reduction = self.config.btc_crash_magnitude * (crash_progress / 0.3) * 0.5
        elif crash_progress <= 0.7:
            # Middle 40%: steep decline
            price_reduction = self.config.btc_crash_magnitude * (0.5 + (crash_progress - 0.3) / 0.4 * 0.4)
        else:
            # Final 30%: complete crash
            price_reduction = self.config.btc_crash_magnitude
        
        return base_price * (1 - price_reduction)
    
    def _process_yt_rebasing(self, minute: int, agents: List[HighTideAgent]):
        """Ensure YT continues rebasing at 10% APR throughout crash"""
        
        # YT rebasing is handled automatically by the YieldToken.get_current_value() method
        # which uses calculate_true_yield_token_price() - no additional processing needed
        
        # The key is that rebasing continues even during price crashes
        # Agents' YT balances grow due to rebasing, but market price can still crash
        pass
    
    def _process_agents_during_crash(self, minute: int, engine) -> Dict:
        """Process agent actions with crash-specific conditions"""
        
        agent_actions = {}
        
        for agent in engine.high_tide_agents:
            if not agent.active:
                continue
            
            # Update agent health factor
            engine._update_agent_health_factor(agent)
            
            # During crash, agents may have different behavior
            if self._is_crash_active(minute):
                # Agents are more likely to panic and try to reduce positions
                # But this is handled by their normal rebalancing logic based on HF
                pass
            
            # Normal agent decision making
            try:
                action_type, params = agent.decide_action(
                    {"current_minute": minute, "current_step": minute}, engine.state.current_prices
                )
                
                if action_type != "hold" and action_type != AgentAction.HOLD:
                    # Use the engine's proper action execution method
                    if hasattr(engine, '_process_high_tide_agents'):
                        # Let the engine handle the action properly
                        pass  # The engine will process this in its normal flow
                    else:
                        # Fallback to base engine method
                        success = engine._execute_agent_action(agent, action_type, params)
                        if success:
                            agent_actions[agent.agent_id] = {
                                "action": str(action_type),
                                "params": params,
                                "health_factor": agent.state.health_factor
                            }
            except Exception as e:
                # Don't spam errors - just log occasionally
                if minute % 100 == 0:  # Log every 100 minutes
                    print(f"   Warning: Agent {agent.agent_id} action failed: {e}")
        
        return agent_actions
    
    def _process_arbitrage_agents(self, minute: int, engine):
        """Process MOET arbitrage agents for peg maintenance"""
        
        for agent in engine.arbitrage_agents:
            if not agent.active:
                continue
            
            try:
                action_type, params = agent.decide_action(
                    {"current_minute": minute}, engine.state.current_prices
                )
                
                if action_type != "hold":
                    # Execute arbitrage action
                    if hasattr(agent, 'execute_arbitrage'):
                        result = agent.execute_arbitrage(params, minute)
                        if result:
                            # Track arbitrage activity
                            pass
            except Exception as e:
                print(f"   Warning: Arbitrage agent {agent.agent_id} failed: {e}")
    
    def _record_crash_metrics(self, minute: int, engine, btc_price: float, 
                             true_yt_price: float, manipulated_yt_price: float):
        """Record comprehensive crash metrics for analysis"""
        
        # Calculate system-wide metrics
        active_agents = [a for a in engine.high_tide_agents if a.active]
        total_debt = sum(agent.state.moet_debt for agent in active_agents)
        avg_health_factor = np.mean([agent.state.health_factor for agent in active_agents]) if active_agents else 0
        liquidatable_count = sum(1 for agent in active_agents if agent.state.health_factor < 1.0)
        
        # Pool state metrics
        pool_yt_price = 1.0  # Default
        if engine.yield_token_pool:
            try:
                pool_state = engine.yield_token_pool.get_pool_state()
                pool_yt_price = pool_state.get("yield_token_price", 1.0)
            except:
                pass
        
        # MOET peg metrics
        moet_price = 1.0  # Default
        if hasattr(engine.protocol, 'moet_stablecoin'):
            moet_price = engine.protocol.moet_stablecoin.current_price
        peg_deviation = abs(moet_price - 1.0) / 1.0
        
        # Determine current phase
        phase = self._get_crash_phase(minute)
        
        crash_snapshot = {
            "minute": minute,
            "hour": minute / 60,
            "phase": phase,
            "btc_price": btc_price,
            "true_yt_price": true_yt_price,
            "manipulated_yt_price": manipulated_yt_price,
            "pool_yt_price": pool_yt_price,
            "moet_price": moet_price,
            "peg_deviation_bps": peg_deviation * 10000,
            "total_system_debt": total_debt,
            "active_agents": len(active_agents),
            "avg_health_factor": avg_health_factor,
            "liquidatable_agents": liquidatable_count,
            "liquidity_factor": self.liquidity_manager.current_liquidity_factor,
            "oracle_manipulation_active": self.oracle_engine._is_crash_active(minute)
        }
        
        self.results["detailed_logs"].append(crash_snapshot)
        self.results["pool_state_snapshots"].append(crash_snapshot)
    
    def _get_crash_phase(self, minute: int) -> str:
        """Determine current phase of the simulation"""
        if minute < self.config.crash_start_minute:
            return "pre_crash"
        elif minute <= self.config.crash_end_minute:
            return "crash"
        elif minute <= self.config.crash_end_minute + self.config.recovery_duration_minutes:
            return "recovery"
        else:
            return "post_recovery"
    
    def _is_crash_active(self, minute: int) -> bool:
        """Check if we're currently in the crash window"""
        return self.config.crash_start_minute <= minute <= self.config.crash_end_minute
    
    def _print_progress_report(self, minute: int, engine, btc_price: float):
        """Print progress report"""
        active_agents = len([a for a in engine.high_tide_agents if a.active])
        liquidatable = len([a for a in engine.high_tide_agents if a.active and a.state.health_factor < 1.0])
        phase = self._get_crash_phase(minute)
        
        print(f"⏱️  Minute {minute:,} ({minute/60:.1f}h) - {phase.upper()}: "
              f"BTC=${btc_price:,.0f}, Active={active_agents}, Liquidatable={liquidatable}")
    
    def _analyze_crash_results(self, engine):
        """Analyze crash simulation results"""
        
        print(f"\n📊 Analyzing crash results...")
        
        # Agent performance analysis
        active_agents = [a for a in engine.high_tide_agents if a.active]
        liquidated_agents = [a for a in engine.high_tide_agents if not a.active]
        
        agent_performance = {
            "total_agents": len(engine.high_tide_agents),
            "survived_agents": len(active_agents),
            "liquidated_agents": len(liquidated_agents),
            "survival_rate": len(active_agents) / len(engine.high_tide_agents),
            "total_liquidation_events": len(self.results["liquidation_events"]),
            "oracle_manipulation_events": len(self.oracle_engine.outlier_events)
        }
        
        self.results["agent_performance"] = agent_performance
        
        # Pool arbitrage analysis
        alm_rebalances = len(self.results["rebalancing_events"]["alm_rebalances"])
        algo_rebalances = len(self.results["rebalancing_events"]["algo_rebalances"])
        
        pool_analysis = {
            "enabled": self.config.enable_pool_arbing,
            "alm_rebalances": alm_rebalances,
            "algo_rebalances": algo_rebalances,
            "total_rebalances": alm_rebalances + algo_rebalances
        }
        
        self.results["pool_arbitrage_analysis"] = pool_analysis
        
        print(f"✅ Analysis complete:")
        print(f"   Survival rate: {agent_performance['survival_rate']:.1%}")
        print(f"   Liquidation events: {agent_performance['total_liquidation_events']}")
        print(f"   Oracle events: {agent_performance['oracle_manipulation_events']}")
        print(f"   Pool rebalances: {pool_analysis['total_rebalances']}")
    
    def _save_crash_results(self):
        """Save comprehensive crash test results"""
        
        # Create results directory
        output_dir = Path("tidal_protocol_sim/results") / self.config.test_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results JSON
        results_path = output_dir / f"flash_crash_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert for JSON serialization
        json_results = self._convert_for_json(self.results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"📁 Flash crash results saved to: {results_path}")
        
        # Save detailed CSV if requested
        if self.config.save_detailed_csv:
            self._save_detailed_csv(output_dir)
    
    def _save_detailed_csv(self, output_dir: Path):
        """Save detailed CSV files for analysis"""
        
        # Crash timeline CSV
        if self.results["detailed_logs"]:
            crash_df = pd.DataFrame(self.results["detailed_logs"])
            crash_csv_path = output_dir / "crash_timeline.csv"
            crash_df.to_csv(crash_csv_path, index=False)
        
        # Liquidation events CSV
        if self.results["liquidation_events"]:
            liquidation_df = pd.DataFrame(self.results["liquidation_events"])
            liquidation_csv_path = output_dir / "liquidation_events.csv"
            liquidation_df.to_csv(liquidation_csv_path, index=False)
        
        # Oracle events CSV
        if self.oracle_engine.outlier_events:
            oracle_df = pd.DataFrame(self.oracle_engine.outlier_events)
            oracle_csv_path = output_dir / "oracle_events.csv"
            oracle_df.to_csv(oracle_csv_path, index=False)
        
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
    
    def _generate_crash_charts(self):
        """Generate flash crash specific charts"""
        
        output_dir = Path("tidal_protocol_sim/results") / self.config.test_name / "charts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        print("📊 Generating flash crash charts...")
        
        # Chart 1: Flash Crash Timeline (YT + BTC prices)
        self._create_flash_crash_timeline_chart(output_dir)
        
        # Chart 2: Liquidation Cascade Analysis
        self._create_liquidation_cascade_chart(output_dir)
        
        # Chart 3: MOET Depeg Analysis
        self._create_moet_depeg_analysis_chart(output_dir)
        
        # Chart 4: Liquidity Evaporation Impact
        self._create_liquidity_evaporation_chart(output_dir)
        
        # Chart 5: System Health Evolution
        self._create_system_health_chart(output_dir)
        
        print(f"📊 Flash crash charts saved to: {output_dir}")
    
    def _create_flash_crash_timeline_chart(self, output_dir: Path):
        """Create comprehensive flash crash timeline chart"""
        
        if not self.results["detailed_logs"]:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Flash Crash Timeline - {self.config.scenario.title()} Scenario', fontsize=16, fontweight='bold')
        
        df = pd.DataFrame(self.results["detailed_logs"])
        
        # Chart 1: BTC Price Evolution
        ax1.plot(df["hour"], df["btc_price"], linewidth=2, color='orange', label='BTC Price')
        ax1.axvspan(self.config.crash_start_minute/60, self.config.crash_end_minute/60, alpha=0.3, color='red', label='Crash Window')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('BTC Price ($)')
        ax1.set_title('BTC Price During Flash Crash')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Chart 2: YT Price Evolution
        ax2.plot(df["hour"], df["true_yt_price"], linewidth=2, color='green', label='True YT Price (10% APR)')
        ax2.plot(df["hour"], df["pool_yt_price"], linewidth=2, color='blue', label='Pool YT Price')
        if "manipulated_yt_price" in df.columns:
            manipulated_data = df[df["manipulated_yt_price"] != df["true_yt_price"]]
            if not manipulated_data.empty:
                ax2.scatter(manipulated_data["hour"], manipulated_data["manipulated_yt_price"], 
                           color='red', s=50, label='Oracle Manipulation', zorder=5)
        ax2.axvspan(self.config.crash_start_minute/60, self.config.crash_end_minute/60, alpha=0.3, color='red')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('YT Price ($)')
        ax2.set_title('Yield Token Price Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Chart 3: MOET Peg Deviation
        ax3.plot(df["hour"], df["peg_deviation_bps"], linewidth=2, color='purple', label='MOET Peg Deviation')
        ax3.axhline(y=200, color='red', linestyle='--', alpha=0.7, label='2% Stability Band')
        ax3.axvspan(self.config.crash_start_minute/60, self.config.crash_end_minute/60, alpha=0.3, color='red')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('Deviation (bps)')
        ax3.set_title('MOET Stablecoin Peg Deviation')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Chart 4: System Health Metrics
        ax4.plot(df["hour"], df["avg_health_factor"], linewidth=2, color='darkgreen', label='Avg Health Factor')
        ax4.plot(df["hour"], df["liquidatable_agents"], linewidth=2, color='red', label='Liquidatable Agents')
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Liquidation Threshold')
        ax4.axvspan(self.config.crash_start_minute/60, self.config.crash_end_minute/60, alpha=0.3, color='red')
        ax4.set_xlabel('Hours')
        ax4.set_ylabel('Health Factor / Agent Count')
        ax4.set_title('System Health Evolution')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "flash_crash_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_liquidation_cascade_chart(self, output_dir: Path):
        """Create liquidation cascade analysis chart"""
        
        if not self.results["liquidation_events"]:
            print("   No liquidation events to chart")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Liquidation Cascade Analysis', fontsize=16, fontweight='bold')
        
        df = pd.DataFrame(self.results["liquidation_events"])
        
        # Chart 1: Liquidations over time
        df["hour"] = df["minute"] / 60
        liquidations_per_minute = df.groupby("minute").size()
        
        ax1.bar(liquidations_per_minute.index / 60, liquidations_per_minute.values, 
                width=0.02, color='red', alpha=0.7, label='Liquidations per Minute')
        ax1.axvspan(self.config.crash_start_minute/60, self.config.crash_end_minute/60, alpha=0.3, color='red')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Liquidations per Minute')
        ax1.set_title('Liquidation Timeline')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Chart 2: Liquidation value distribution
        ax2.hist(df["btc_value_gross"], bins=20, alpha=0.7, color='orange', label='Gross Value')
        ax2.hist(df["btc_value_net"], bins=20, alpha=0.7, color='red', label='Net Value (after slippage)')
        ax2.set_xlabel('Liquidation Value ($)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Liquidation Value Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "liquidation_cascade_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_moet_depeg_analysis_chart(self, output_dir: Path):
        """Create MOET depeg analysis chart"""
        
        if not self.results["detailed_logs"]:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle('MOET Stablecoin Depeg Analysis', fontsize=16, fontweight='bold')
        
        df = pd.DataFrame(self.results["detailed_logs"])
        
        # MOET price and deviation
        ax.plot(df["hour"], df["moet_price"], linewidth=2, color='blue', label='MOET Price')
        ax.axhline(y=1.0, color='green', linestyle='-', alpha=0.7, label='$1.00 Peg')
        ax.axhline(y=0.98, color='orange', linestyle='--', alpha=0.7, label='2% Stability Band')
        ax.axhline(y=1.02, color='orange', linestyle='--', alpha=0.7)
        ax.axvspan(self.config.crash_start_minute/60, self.config.crash_end_minute/60, alpha=0.3, color='red', label='Crash Window')
        
        ax.set_xlabel('Hours')
        ax.set_ylabel('MOET Price ($)')
        ax.set_title('MOET Price Stability During Flash Crash')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "moet_depeg_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_liquidity_evaporation_chart(self, output_dir: Path):
        """Create liquidity evaporation impact chart"""
        
        if not self.results["detailed_logs"]:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle('Liquidity Evaporation Impact', fontsize=16, fontweight='bold')
        
        df = pd.DataFrame(self.results["detailed_logs"])
        
        # Liquidity factor over time
        ax.plot(df["hour"], df["liquidity_factor"], linewidth=3, color='red', label='Liquidity Factor')
        ax.axhline(y=1.0, color='green', linestyle='-', alpha=0.7, label='Normal Liquidity')
        ax.axvspan(self.config.crash_start_minute/60, self.config.crash_end_minute/60, alpha=0.3, color='red', label='Crash Window')
        ax.axvspan(self.config.crash_end_minute/60, (self.config.crash_end_minute + self.config.recovery_duration_minutes)/60, 
                   alpha=0.2, color='blue', label='Recovery Phase')
        
        ax.set_xlabel('Hours')
        ax.set_ylabel('Liquidity Factor (1.0 = Normal)')
        ax.set_title('Rebalancer Liquidity During Flash Crash')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "liquidity_evaporation_impact.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_system_health_chart(self, output_dir: Path):
        """Create system health evolution chart"""
        
        if not self.results["detailed_logs"]:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle('System Health Evolution', fontsize=16, fontweight='bold')
        
        df = pd.DataFrame(self.results["detailed_logs"])
        
        # Chart 1: Agent health metrics
        ax1.plot(df["hour"], df["active_agents"], linewidth=2, color='green', label='Active Agents')
        ax1.plot(df["hour"], df["liquidatable_agents"], linewidth=2, color='red', label='Liquidatable Agents')
        ax1.axvspan(self.config.crash_start_minute/60, self.config.crash_end_minute/60, alpha=0.3, color='red')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Agent Count')
        ax1.set_title('Agent Health Status')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Chart 2: System debt and health factor
        ax2.plot(df["hour"], df["total_system_debt"] / 1e6, linewidth=2, color='blue', label='Total Debt ($M)')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(df["hour"], df["avg_health_factor"], linewidth=2, color='orange', label='Avg Health Factor')
        ax2_twin.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Liquidation Threshold')
        
        ax2.axvspan(self.config.crash_start_minute/60, self.config.crash_end_minute/60, alpha=0.3, color='red')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('Total Debt ($M)', color='blue')
        ax2_twin.set_ylabel('Health Factor', color='orange')
        ax2.set_title('System Debt and Health Factor')
        ax2.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_dir / "system_health_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_crash_summary(self):
        """Print comprehensive crash test summary"""
        
        print(f"\n🎯 FLASH CRASH SIMULATION SUMMARY")
        print(f"=" * 50)
        print(f"Scenario: {self.config.scenario.upper()}")
        print(f"Duration: {self.config.simulation_duration_minutes:,} minutes ({self.config.simulation_duration_minutes//1440} days)")
        print()
        
        # Agent performance
        agent_perf = self.results.get("agent_performance", {})
        if agent_perf:
            print(f"👥 Agent Performance:")
            print(f"   Total agents: {agent_perf.get('total_agents', 0)}")
            print(f"   Survived: {agent_perf.get('survived_agents', 0)} ({agent_perf.get('survival_rate', 0):.1%})")
            print(f"   Liquidated: {agent_perf.get('liquidated_agents', 0)}")
            print(f"   Liquidation events: {agent_perf.get('total_liquidation_events', 0)}")
            print()
        
        # Oracle manipulation
        print(f"🔮 Oracle Manipulation:")
        print(f"   Outlier events: {len(self.oracle_engine.outlier_events)}")
        print(f"   Extreme tick injected: {self.oracle_engine.extreme_tick_injected}")
        print()
        
        # Pool rebalancing
        pool_arb = self.results.get("pool_arbitrage_analysis", {})
        if pool_arb.get("enabled"):
            print(f"🔄 Pool Rebalancing:")
            print(f"   ALM rebalances: {pool_arb.get('alm_rebalances', 0)}")
            print(f"   Algo rebalances: {pool_arb.get('algo_rebalances', 0)}")
            print(f"   Total rebalances: {pool_arb.get('total_rebalances', 0)}")
            print()
        
        # Crash parameters achieved
        print(f"💥 Crash Parameters:")
        print(f"   YT crash magnitude: {self.config.yt_crash_magnitude:.0%}")
        print(f"   BTC crash magnitude: {self.config.btc_crash_magnitude:.0%}")
        print(f"   Liquidity reduction: {self.config.liquidity_reduction_start:.0%} → {self.config.liquidity_reduction_peak:.0%}")
        print(f"   Oracle outliers: {self.config.oracle_outlier_magnitude:.0%}")


def main():
    """Main execution function"""
    
    print("Flash Crash Simulation - YT and BTC Simultaneous Crash")
    print("=" * 60)
    print()
    print("Available scenarios:")
    print("• mild: 35% YT drop, 12% BTC drop")
    print("• moderate: 50% YT drop, 20% BTC drop") 
    print("• severe: 70% YT drop, 25% BTC drop")
    print()
    
    # Default to moderate scenario
    scenario = "moderate"
    
    print(f"Running {scenario} scenario...")
    print()
    
    # Create configuration
    config = FlashCrashSimConfig(scenario)
    
    print(f"Configuration:")
    print(f"• Scenario: {config.scenario}")
    print(f"• Duration: {config.simulation_duration_minutes:,} minutes ({config.simulation_duration_minutes//1440} days)")
    print(f"• Agents: {config.num_agents} with ${config.target_total_debt:,.0f} target debt")
    print(f"• Crash window: Day 1, 15:00-15:25 (minutes {config.crash_start_minute}-{config.crash_end_minute})")
    print(f"• YT crash: {config.yt_crash_magnitude:.0%}, BTC crash: {config.btc_crash_magnitude:.0%}")
    print()
    
    try:
        # Run the flash crash simulation
        print("⚡ FLASH CRASH SIMULATION STARTING")
        print("This will simulate 3 days with detailed crash analysis...")
        
        simulation = FlashCrashSimulation(config)
        results = simulation.run_flash_crash_test()
        
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
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
