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
        # Base simulation parameters - 2 DAYS TOTAL (for better granularity)
        self.test_name = f"Flash_Crash_YT_BTC_{scenario.title()}_Scenario"
        self.simulation_duration_minutes = 2 * 24 * 60  # 2 days = 2,880 minutes
        
        # Agent configuration - 150 agents with ~$20M total debt
        self.num_agents = 150
        self.target_total_debt = 20_000_000  # $20M system debt
        self.avg_debt_per_agent = self.target_total_debt / self.num_agents  # ~$133k per agent
        
        # Agent health factor parameters (MUST MATCH ACTUAL AGENT CREATION)
        self.agent_initial_hf = 1.15      # Initial health factor (matches line 659)
        self.agent_target_hf = 1.08       # Target health factor (matches line 661)
        self.agent_rebalancing_hf = 1.05  # Rebalancing threshold (matches line 660)
        
        # Flash crash timing - WITHIN FIRST DAY (15:00 on Day 1)
        self.crash_start_minute = 15 * 60 + 0  # 15:00 on Day 1 (900 minutes from start)
        self.crash_duration_minutes = 25      # 25-minute crash window (15:00-15:25)
        self.crash_end_minute = self.crash_start_minute + self.crash_duration_minutes  # 925 minutes
        self.recovery_duration_minutes = 120  # 2-hour recovery phase (15:25-17:25)
        
        # Oracle manipulation timing - starts BEFORE BTC crash to show oracle attack
        self.oracle_crash_offset_minutes = -5  # Oracle crashes 5 minutes BEFORE BTC crash
        
        # Scenario-based crash parameters
        self.scenario = scenario
        if scenario == "mild":
            self.yt_crash_magnitude = 0.20      # 20% drop to $0.80
            self.btc_crash_magnitude = 0.12     # 12% drop
            self.yt_floor_price = 0.80          # Floor at $0.80
            self.yt_wick_magnitude = 0.10       # Wicks down 10% more (to $0.70)
            self.liquidity_reduction_peak = 0.60 # 60% liquidity reduction
            self.oracle_volatility = 0.05       # 5% volatility between ticks
        elif scenario == "moderate":
            self.yt_crash_magnitude = 0.32      # 32% drop to $0.68
            self.btc_crash_magnitude = 0.20     # 20% drop
            self.yt_floor_price = 0.68          # Floor at $0.68
            self.yt_wick_magnitude = 0.15       # Wicks down 15% more (to $0.53)
            self.liquidity_reduction_peak = 0.70 # 70% liquidity reduction
            self.oracle_volatility = 0.08       # 8% volatility between ticks
        elif scenario == "severe":
            self.yt_crash_magnitude = 0.45      # 45% drop to $0.55
            self.btc_crash_magnitude = 0.25     # 25% drop
            self.yt_floor_price = 0.55          # Floor at $0.55
            self.yt_wick_magnitude = 0.20       # Wicks down 20% more (to $0.35)
            self.liquidity_reduction_peak = 0.80 # 80% liquidity reduction
            self.oracle_volatility = 0.12       # 12% volatility between ticks
        
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
        
        # Pool configurations - SCALED FOR $20M SYSTEM
        self.moet_btc_pool_config = {
            "size": 5_000_000,  # $5M liquidation pool (25% of $20M system)
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
        self.algo_deviation_threshold_bps = 25.0   # 25 bps for faster response
        
        # Advanced system features
        self.enable_advanced_moet_system = True
        self.enable_forced_liquidations = True
        # Note: ADL (Auto-Deleveraging) is for perps exchanges, not lending platforms - removed
        
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
        
        # Track current minute for liquidity strategy
        self._current_minute_for_liquidity = current_minute
        
        # Reduce rebalancer capital and effectiveness
        self._apply_liquidity_reduction(pool_rebalancer_manager, self.current_liquidity_factor)
        
        # Apply one-sided selling pressure
        if self.config.one_sided_selling_pressure:
            self._apply_one_sided_pressure(pool_rebalancer_manager, reduction)
    
    def _apply_liquidity_reduction(self, pool_rebalancer_manager, liquidity_factor):
        """Reduce rebalancer capital and maximum rebalance sizes
        
        STRATEGY: During crash, keep Algo fully funded to show oracle attack working.
                  During recovery, reduce Algo liquidity to show difficulty bringing pool back.
        """
        
        # Determine if we're in crash or recovery
        current_minute = self.config.crash_start_minute  # Placeholder; will be set by caller
        is_crash_window = hasattr(self, '_current_minute_for_liquidity') and \
                         self._current_minute_for_liquidity <= self.config.crash_end_minute
        
        # ALM: Always apply liquidity reduction
        pool_rebalancer_manager.alm_rebalancer.state.moet_balance = (
            self.original_alm_balance * liquidity_factor
        )
        pool_rebalancer_manager.alm_rebalancer.state.max_single_rebalance = max(
            self.original_alm_max_rebalance * liquidity_factor, 5_000.0
        )
        
        # ALGO: Keep fully funded during crash, reduce during recovery
        if is_crash_window:
            # During crash: Algo gets FULL liquidity to actively move pool toward oracle
            pool_rebalancer_manager.algo_rebalancer.state.moet_balance = self.original_algo_balance
            pool_rebalancer_manager.algo_rebalancer.state.max_single_rebalance = self.original_algo_max_rebalance
            algo_min_rebalance = 100.0  # Very low threshold during crash
        else:
            # During recovery: Algo gets REDUCED liquidity (hard to bring pool back)
            pool_rebalancer_manager.algo_rebalancer.state.moet_balance = (
                self.original_algo_balance * liquidity_factor
            )
            pool_rebalancer_manager.algo_rebalancer.state.max_single_rebalance = max(
                self.original_algo_max_rebalance * liquidity_factor, 5_000.0
            )
            algo_min_rebalance = min(100.0 / max(liquidity_factor, 0.05), 2_500.0)
        
        # Set minimum rebalance amounts
        base_min_rebalance = 100.0
        pool_rebalancer_manager.alm_rebalancer.state.min_rebalance_amount = min(
            base_min_rebalance / max(liquidity_factor, 0.05), 2_500.0
        )
        pool_rebalancer_manager.algo_rebalancer.state.min_rebalance_amount = algo_min_rebalance
    
    def _apply_one_sided_pressure(self, pool_rebalancer_manager, reduction_factor):
        """Simulate one-sided selling pressure - everyone wants to sell YT
        
        During crash: Let Algo work freely to follow oracle down
        During recovery: Make it harder for Algo to bring pool back up
        """
        
        # Check if we're in crash or recovery
        is_crash_window = hasattr(self, '_current_minute_for_liquidity') and \
                         self._current_minute_for_liquidity <= self.config.crash_end_minute
        
        # During crash, make it harder for ALM to buy YT, but let Algo work freely
        # During recovery, make it harder for BOTH to buy YT
        original_profit_threshold = 0.001  # 0.1% normally
        crash_profit_threshold = original_profit_threshold * (1 + reduction_factor * 10)
        
        # ALM: Always apply selling pressure
        if hasattr(pool_rebalancer_manager.alm_rebalancer.state, 'min_profit_threshold'):
            pool_rebalancer_manager.alm_rebalancer.state.min_profit_threshold = crash_profit_threshold
        
        # ALGO: Only apply pressure during recovery (not during crash)
        if hasattr(pool_rebalancer_manager.algo_rebalancer.state, 'min_profit_threshold'):
            if is_crash_window:
                # During crash: Algo can freely move pool toward oracle
                pool_rebalancer_manager.algo_rebalancer.state.min_profit_threshold = original_profit_threshold
            else:
                # During recovery: Hard for Algo to bring pool back
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
        
        # Calculate restored liquidity factor based on progress from the minimum
        # If we hit 0.3 at crash end, we want to go 0.3 → 1.0 over the recovery period
        min_liquidity = 1.0 - self.config.liquidity_reduction_peak
        restored_factor = min_liquidity + (1.0 - min_liquidity) * recovery_progress
        
        # Update the current liquidity factor tracking
        self.current_liquidity_factor = restored_factor
        
        # Track that we're in recovery (not crash window)
        self._current_minute_for_liquidity = current_minute
        
        # Apply the restored liquidity
        self._apply_liquidity_reduction(pool_rebalancer_manager, restored_factor)


class OracleMispricingEngine:
    """Injects realistic oracle price manipulation with volatility wicks during crash"""
    
    def __init__(self, config: FlashCrashSimConfig):
        self.config = config
        self.price_history = []  # Track all oracle prices for charting
        self.wick_events = []     # Track extreme wicks separately
        self.last_oracle_price = 1.0  # Start at $1.00
        
        # Seed for reproducible randomness
        import random
        self.random = random.Random(42)
        
    def get_manipulated_yt_price(self, current_minute: int, true_yt_price: float) -> float:
        """Return manipulated YT price with realistic volatility and wicks"""
        
        # Apply oracle-specific crash window offset
        oracle_start = self.config.crash_start_minute + getattr(self.config, "oracle_crash_offset_minutes", 0)
        oracle_end = self.config.crash_end_minute + getattr(self.config, "oracle_crash_offset_minutes", 0)

        # Pre-crash and post-recovery: Use true price
        if current_minute < oracle_start:
            oracle_price = true_yt_price
            self._log_price(current_minute, oracle_price, true_yt_price, "pre_crash")
            return oracle_price
        
        # Post-crash recovery
        if current_minute > oracle_end:
            oracle_price = self._calculate_recovery_price(current_minute, true_yt_price)
            self._log_price(current_minute, oracle_price, true_yt_price, "recovery")
            return oracle_price
        
        # DURING CRASH: Generate realistic price action
        crash_progress = (current_minute - oracle_start) / self.config.crash_duration_minutes
        
        # Base oracle price: Gradual decline to floor
        base_oracle_price = true_yt_price * (1 - self.config.yt_crash_magnitude * crash_progress)
        base_oracle_price = max(base_oracle_price, self.config.yt_floor_price)
        
        # Add volatility (random walk around base price)
        volatility_factor = self.random.uniform(-self.config.oracle_volatility, self.config.oracle_volatility)
        volatile_price = base_oracle_price * (1 + volatility_factor)
        
        # Inject extreme wicks occasionally (3-5 times during 25-minute crash)
        if self._should_inject_wick(current_minute, crash_progress):
            wick_price = self._generate_wick_price(volatile_price, current_minute, true_yt_price)
            self._log_price(current_minute, wick_price, true_yt_price, "wick")
            return wick_price
        
        # Normal crash pricing with volatility
        self._log_price(current_minute, volatile_price, true_yt_price, "crash")
        return volatile_price
    
    def _should_inject_wick(self, current_minute: int, crash_progress: float) -> bool:
        # Inject 3-5 wicks randomly during the crash period
        from math import isfinite
        if not isfinite(crash_progress):
            return False
        return self.random.random() < 0.12  # ~12% chance per minute during crash
    
    def _generate_wick_price(self, current_price: float, minute: int, true_price: float) -> float:
        """Generate an extreme wick price (brief sharp movement)"""
        
        # Wick down by wick_magnitude from current oracle price
        wick_price = current_price * (1 - self.config.yt_wick_magnitude)
        
        # Log this as a special wick event
        self.wick_events.append({
            "minute": minute,
            "wick_price": wick_price,
            "base_price": current_price,
            "true_price": true_price,
            "wick_magnitude": self.config.yt_wick_magnitude
        })
        
        print(f"⚡ WICK EVENT at minute {minute}: ${wick_price:.4f} (from ${current_price:.4f}, -{self.config.yt_wick_magnitude:.1%})")
        
        return wick_price
    
    def _calculate_recovery_price(self, current_minute: int, true_yt_price: float) -> float:
        """Calculate gradual recovery back to true price"""
        
        recovery_elapsed = current_minute - self.config.crash_end_minute
        recovery_duration = self.config.recovery_duration_minutes
        
        if recovery_elapsed >= recovery_duration:
            # Fully recovered
            return true_yt_price
    
        # Gradual recovery from floor to true price
        recovery_progress = recovery_elapsed / recovery_duration
        
        # Exponential recovery (faster at first, then slower)
        recovery_factor = 1 - (1 - recovery_progress) ** 2
        
        floor_price = self.config.yt_floor_price
        recovered_price = floor_price + (true_yt_price - floor_price) * recovery_factor
        
        # Add small recovery volatility
        recovery_volatility = self.config.oracle_volatility * 0.5  # Half the crash volatility
        volatility_factor = self.random.uniform(-recovery_volatility, recovery_volatility)
        recovered_price = recovered_price * (1 + volatility_factor)
        
        return min(recovered_price, true_yt_price)  # Don't overshoot
    
    def _log_price(self, minute: int, oracle_price: float, true_price: float, phase: str):
        """Log oracle price for tracking and charting"""
        
        deviation = (oracle_price - true_price) / true_price if true_price > 0 else 0
        
        event = {
            "minute": minute,
            "oracle_price": oracle_price,
            "true_price": true_price,
            "deviation_pct": deviation,
            "deviation_bps": deviation * 10000,
            "phase": phase
        }
        
        self.price_history.append(event)
        self.last_oracle_price = oracle_price
    
    def _is_crash_active(self, current_minute: int) -> bool:
        """Check if oracle manipulation is active (with offset)."""
        oracle_start = self.config.crash_start_minute + getattr(self.config, "oracle_crash_offset_minutes", 0)
        oracle_end = self.config.crash_end_minute + getattr(self.config, "oracle_crash_offset_minutes", 0)
        return oracle_start <= current_minute <= oracle_end


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
        print(f"💰 System Size: ${self.config.target_total_debt:,.0f} total debt across {self.config.num_agents} agents")
        print(f"💥 Crash Window: Day 1, 15:00-15:25 (minutes {self.config.crash_start_minute}-{self.config.crash_end_minute})")
        print(f"📉 Crash Magnitude: YT {self.config.yt_crash_magnitude:.0%} ↓, BTC {self.config.btc_crash_magnitude:.0%} ↓")
        print(f"💧 Liquidity Evaporation: {self.config.liquidity_reduction_start:.0%} → {self.config.liquidity_reduction_peak:.0%}")
        print(f"🏊 YT Pool: ${self.config.moet_yt_pool_config['size']:,.0f} (concentrated liquidity)")
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
        
        # Initialize positions to reach target total debt
        self._setup_large_system_positions(engine)
        
        return engine
        
    def _create_large_debt_agents(self, engine) -> List[HighTideAgent]:
        """Create 150 agents for $20M system debt (~$133K debt per agent)"""
        
        # Create agents with standardized risk profiles
        agents = []
        
        for i in range(self.config.num_agents):
            # Create agent with moderate-aggressive profile
            # BTC collateral will be calculated based on target debt and HF in _setup_large_system_positions
            agent = HighTideAgent(
                f"flash_crash_agent_{i+1}",
                initial_hf=1.15,      # Starting health factor (safe but leveraged)
                rebalancing_hf=1.05,  # Triggers rebalancing when HF drops to 1.05
                target_hf=1.08,       # Rebalances back to 1.08 (tight buffer for efficiency)
                initial_balance=100_000.0,  # Initial BTC price for calculations
                yield_token_pool=engine.yield_token_pool if hasattr(engine, 'yield_token_pool') else None
            )
            
            agents.append(agent)
            
        return agents
    
    def _setup_large_system_positions(self, engine):
        """Initialize agent positions to reach target debt levels with correct health factors"""
        
        print(f"🏗️  Setting up large system positions for ${self.config.target_total_debt:,.0f} total debt...")
        
        btc_price = 100_000.0  # $100k BTC
        btc_collateral_factor = 0.80  # 80% CF for BTC
        total_debt_target = self.config.target_total_debt
        debt_per_agent = total_debt_target / len(engine.high_tide_agents)
        
        print(f"   Target: {len(engine.high_tide_agents)} agents × ${debt_per_agent:,.0f} debt each")
        
        for agent in engine.high_tide_agents:
            # CORRECT CALCULATION: BTC collateral needed = (Target Debt × HF) / CF
            target_debt = debt_per_agent
            initial_hf = agent.state.initial_health_factor
            required_btc_collateral_value = (target_debt * initial_hf) / btc_collateral_factor
            required_btc_amount = required_btc_collateral_value / btc_price
            
            # Update agent's BTC collateral to match target debt at desired HF
            agent.state.supplied_balances[Asset.BTC] = required_btc_amount
            agent.state.btc_amount = required_btc_amount
            
            # Set debt and MOET balance
            agent.state.moet_debt = target_debt
            agent.state.initial_moet_debt = target_debt
            agent.state.borrowed_balances[Asset.MOET] = target_debt
            agent.state.token_balances[Asset.MOET] = target_debt
            
            # Purchase yield tokens with MOET using DIRECT MINTING for initialization
            if engine.yield_token_pool:
                try:
                    yt_tokens = agent.state.yield_token_manager.mint_yield_tokens(
                        target_debt, 0, use_direct_minting=True
                    )
                    agent.state.initial_yield_token_value = yt_tokens if yt_tokens else 0
                except Exception as e:
                    print(f"   ⚠️  Warning: YT minting failed for {agent.agent_id}: {e}")
            
            # Update health factor - should now match initial_hf
            engine._update_agent_health_factor(agent)
        
        # Verify system initialization
        actual_total_debt = sum(agent.state.moet_debt for agent in engine.high_tide_agents)
        actual_total_btc = sum(agent.state.supplied_balances.get(Asset.BTC, 0) for agent in engine.high_tide_agents)
        avg_hf = sum(agent.state.health_factor for agent in engine.high_tide_agents) / len(engine.high_tide_agents)
        
        print(f"✅ System initialized:")
        print(f"   Total debt: ${actual_total_debt:,.0f} (target: ${total_debt_target:,.0f})")
        print(f"   Total BTC collateral: {actual_total_btc:.2f} BTC (${actual_total_btc * btc_price:,.0f})")
        print(f"   Average Health Factor: {avg_hf:.3f}")
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
            
            # Compute oracle-only window flag (oracle attacks before BTC crashes)
            oracle_start = self.config.crash_start_minute + getattr(self.config, "oracle_crash_offset_minutes", 0)
            oracle_only_window = oracle_start <= minute < self.config.crash_start_minute
            
            # GATE: Block agent rebalancing during oracle-only window
            engine.state.allow_agent_rebalancing = not oracle_only_window
            
            # CRITICAL: Store oracle price override in engine state for emergency rebalancing
            engine.state.oracle_yt_price_override = manipulated_yt_price
            engine.state.oracle_override_active = True
            
            # DISABLE emergency rebalancing during crash window to let main Algo rebalancer handle it
            engine.state.disable_emergency_rebalancing = self._is_crash_active(minute)
            
            # Process High Tide agents using engine's method
            if hasattr(engine, '_process_high_tide_agents'):
                swap_data = engine._process_high_tide_agents(minute)
            
            # Clear oracle override and re-enable emergency rebalancing after agent processing
            engine.state.oracle_override_active = False
            engine.state.disable_emergency_rebalancing = False
            
            # Process forced liquidations during crash
            if self._is_crash_active(minute):
                liquidation_events = self.forced_liquidation_engine.process_crash_liquidations(
                    minute, engine.high_tide_agents, engine.state.current_prices
                )
                self.results["liquidation_events"].extend(liquidation_events)
            
            # Process pool rebalancing (with oracle price override during crash)
            if hasattr(engine, 'pool_rebalancer') and engine.pool_rebalancer and engine.pool_rebalancer.enabled:
                try:
                    # Build protocol_state with oracle YT price override
                    # Flag if we are in the oracle-only window (oracle attack before BTC drop)
                    oracle_start = self.config.crash_start_minute + getattr(self.config, "oracle_crash_offset_minutes", 0)
                    oracle_only_window = oracle_start <= minute < self.config.crash_start_minute
                    
                    # BLOCK agent rebalancing during oracle-only window to prevent premature YT sales
                    should_allow_agent_rebalance = not oracle_only_window
                    
                    # Get current pool price for deviation calculation (CRITICAL for Algo rebalancer)
                    pool_yt_price = engine.yield_token_pool.uniswap_pool.get_price()  # YT/$
                    
                    # Convert pool price to $/YT to match oracle units for deviation calc
                    pool_yt_price_in_dollars = 1.0 / pool_yt_price  # Convert YT/$ to $/YT
                    deviation_bps = abs((pool_yt_price_in_dollars - manipulated_yt_price) / manipulated_yt_price) * 10000
                    
                    protocol_state = {
                        "current_minute": minute,
                        "true_yield_token_price": manipulated_yt_price,  # Oracle price in $/YT
                        "oracle_yt_price": manipulated_yt_price,  # Oracle-manipulated price for rebalancers
                        "pool_yield_token_price": pool_yt_price_in_dollars,  # Pool price converted to $/YT
                        "deviation_bps": deviation_bps,  # Deviation calculated in same units
                        "oracle_only_window": oracle_only_window,
                        "should_allow_agent_rebalance": should_allow_agent_rebalance  # Engine controls rebalancing
                    }
                    
                    rebalancing_events = engine.pool_rebalancer.process_rebalancing(
                        protocol_state, engine.state.current_prices
                    )
                    
                    # DEBUG: Log rebalancing during crash
                    if self._is_crash_active(minute) and minute % 5 == 0:
                        print(f"🔍 process_rebalancing() at min {minute}: returned {len(rebalancing_events) if rebalancing_events else 0} events")
                        print(f"   Deviation: {deviation_bps:.1f} bps, Pool: ${pool_yt_price:.3f}, Oracle: ${manipulated_yt_price:.3f}")
                        if rebalancing_events:
                            for e in rebalancing_events:
                                print(f"   → {e.get('rebalancer')}: {e.get('direction')} ${e.get('amount', 0):,.0f}")
                    
                    if rebalancing_events:
                        for event in rebalancing_events:
                            event["minute"] = minute
                        self.results["rebalancing_events"]["alm_rebalances"].extend(
                            [e for e in rebalancing_events if e.get("rebalancer") == "ALM"]
                        )
                        self.results["rebalancing_events"]["algo_rebalances"].extend(
                            [e for e in rebalancing_events if e.get("rebalancer") == "Algo"]
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
            "oracle_price_history": self.oracle_engine.price_history,  # Complete oracle price timeline
            "oracle_wick_events": self.oracle_engine.wick_events,     # Extreme wick events
            "liquidity_events": [],  # Could add liquidity change events
            "final_agent_count": len([a for a in engine.high_tide_agents if a.active]),
            "total_liquidations": len(self.results["liquidation_events"])
        }
        
        return simulation_results
    
    def _calculate_btc_price_during_crash(self, minute: int) -> float:
        """Calculate BTC price with crash dynamics and volatile recovery"""
        base_price = 100_000.0  # Starting BTC price
        crash_low = base_price * (1 - self.config.btc_crash_magnitude)  # $80k for 20% crash
        btc_crash_duration = 5  # BTC drops over 5 minutes, then stays at floor
        
        # Pre-crash: stable at $100k
        if minute < self.config.crash_start_minute:
            return base_price
        
        # FIRST 5 MINUTES: Sharp drop from $100k to $80k (gradual, not instant)
        if minute < self.config.crash_start_minute + btc_crash_duration:
            minutes_into_crash = minute - self.config.crash_start_minute
            crash_progress = minutes_into_crash / btc_crash_duration
            # Linear drop over 5 minutes
            current_price = base_price - (base_price - crash_low) * crash_progress
            return current_price
        
        # REST OF CRASH WINDOW (minutes 905-925): Stay at floor ($80k)
        if self._is_crash_active(minute):
            return crash_low
        
        # Post-crash recovery (minutes 925-4320): Volatile recovery to $100k
        recovery_elapsed = minute - self.config.crash_end_minute
        total_recovery_time = self.config.simulation_duration_minutes - self.config.crash_end_minute
        recovery_progress = min(recovery_elapsed / total_recovery_time, 1.0)
        
        # Exponential recovery curve (faster at first, slower near end)
        recovery_factor = 1 - (1 - recovery_progress) ** 1.5
        
        # Base recovery price
        base_recovery_price = crash_low + (base_price - crash_low) * recovery_factor
        
        # Add realistic volatility (±2% swings)
        import random
        volatility = random.uniform(-0.02, 0.02)
        volatile_price = base_recovery_price * (1 + volatility)
        
        # Ensure we don't go below crash low or above base price
        return max(crash_low, min(volatile_price, base_price))
    
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
        
        # Track individual agents for detailed charting
        # Sample 10 agents evenly distributed across the population
        sampled_agents = {}
        agent_indices = [0, 16, 33, 50, 66, 83, 100, 116, 133, 149]  # 10 agents across 150
        for idx in agent_indices:
            if idx < len(engine.high_tide_agents):
                agent = engine.high_tide_agents[idx]
                # Calculate net position value manually
                yt_value = agent.state.yield_token_manager.calculate_total_value(minute)
                btc_value = agent.state.supplied_balances.get(Asset.BTC, 0) * btc_price
                net_position = btc_value + yt_value - agent.state.moet_debt
                
                sampled_agents[f"agent_{idx}"] = {
                    "health_factor": agent.state.health_factor,
                    "net_position": net_position,
                    "yt_value": yt_value,
                    "moet_debt": agent.state.moet_debt,
                    "active": agent.active
                }
        
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
            "oracle_manipulation_active": self.oracle_engine._is_crash_active(minute),
            "sampled_agents": sampled_agents  # Individual agent tracking
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
        # Track which agents were liquidated (even partially)
        liquidated_agent_ids = set()
        for liq_event in self.results["liquidation_events"]:
            liquidated_agent_ids.add(liq_event.get("agent_id"))
        
        active_agents = [a for a in engine.high_tide_agents if a.active]
        fully_liquidated_agents = [a for a in engine.high_tide_agents if not a.active]
        agents_with_liquidations = len(liquidated_agent_ids)
        agents_without_liquidations = len(engine.high_tide_agents) - agents_with_liquidations
        
        agent_performance = {
            "total_agents": len(engine.high_tide_agents),
            "survived_agents": agents_without_liquidations,  # Only agents with NO liquidations
            "agents_with_partial_liquidations": agents_with_liquidations - len(fully_liquidated_agents),
            "fully_liquidated_agents": len(fully_liquidated_agents),
            "survival_rate": agents_without_liquidations / len(engine.high_tide_agents),
            "total_liquidation_events": len(self.results["liquidation_events"]),
            "oracle_wick_events": len(self.oracle_engine.wick_events),
            "oracle_price_points": len(self.oracle_engine.price_history)
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
        print(f"   Oracle wick events: {agent_performance['oracle_wick_events']}")
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
        
        # Oracle price history CSV
        if self.oracle_engine.price_history:
            oracle_df = pd.DataFrame(self.oracle_engine.price_history)
            oracle_csv_path = output_dir / "oracle_price_history.csv"
            oracle_df.to_csv(oracle_csv_path, index=False)
        
        # Oracle wick events CSV
        if self.oracle_engine.wick_events:
            wick_df = pd.DataFrame(self.oracle_engine.wick_events)
            wick_csv_path = output_dir / "oracle_wick_events.csv"
            wick_df.to_csv(wick_csv_path, index=False)
        
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
        
        # Chart 3: MOET:YT Pool Evolution
        self._create_moet_yt_pool_evolution_chart(output_dir)
        
        # Chart 4: Liquidity Evaporation Impact
        self._create_liquidity_evaporation_chart(output_dir)
        
        # Chart 5: System Health Evolution
        self._create_system_health_chart(output_dir)
        
        # Chart 6: Rebalancing Activity Analysis
        self._create_rebalancing_activity_chart(output_dir)
        
        # Chart 7: Agent Time Series Evolution
        self._create_agent_time_series_chart(output_dir)
        
        # Chart 8: Liquidated Agent Profiles (individual trajectories)
        self._create_liquidated_agent_profile_chart(output_dir)
        
        # Chart 9: Liquidated Agents Health Factor Tracking
        self._create_liquidated_agents_hf_tracking_chart(output_dir)
        
        # Chart 10: Agent Slippage Analysis
        self._create_agent_slippage_analysis_chart(output_dir)
        
        # Chart 11 & 12: Agent Time Series Tracking (Liquidated and Non-Liquidated)
        self._create_agent_time_series_charts(output_dir)
        
        print(f"📊 Flash crash charts saved to: {output_dir}")
    
    def _create_flash_crash_timeline_chart(self, output_dir: Path):
        """Create comprehensive flash crash timeline chart"""
        
        if not self.results["detailed_logs"]:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Flash Crash Timeline - {self.config.scenario.title()} Scenario', fontsize=16, fontweight='bold')
        
        df = pd.DataFrame(self.results["detailed_logs"])
        
        # Chart 1: BTC Price Evolution
        ax1.plot(df["minute"], df["btc_price"], linewidth=2, color='orange', label='BTC Price')
        ax1.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, alpha=0.3, color='red', label='Crash Window')
        ax1.set_xlabel('Minutes')
        ax1.set_ylabel('BTC Price ($)')
        ax1.set_title('BTC Price During Flash Crash')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Chart 2: YT Price Evolution
        ax2.plot(df["minute"], df["true_yt_price"], linewidth=2, color='green', label='True YT Price (10% APR)')
        ax2.plot(df["minute"], df["pool_yt_price"], linewidth=2, color='blue', label='Pool YT Price')
        if "manipulated_yt_price" in df.columns:
            manipulated_data = df[df["manipulated_yt_price"] != df["true_yt_price"]]
            if not manipulated_data.empty:
                ax2.scatter(manipulated_data["minute"], manipulated_data["manipulated_yt_price"], 
                           color='red', s=10, alpha=0.6, label='Oracle Manipulation', zorder=5)
        ax2.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, alpha=0.3, color='red')
        ax2.set_xlabel('Minutes')
        ax2.set_ylabel('YT Price ($)')
        ax2.set_title('Yield Token Price Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Chart 3: MOET Peg Deviation
        ax3.plot(df["minute"], df["peg_deviation_bps"], linewidth=2, color='purple', label='MOET Peg Deviation')
        ax3.axhline(y=200, color='red', linestyle='--', alpha=0.7, label='2% Stability Band')
        ax3.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, alpha=0.3, color='red')
        ax3.set_xlabel('Minutes')
        ax3.set_ylabel('Deviation (bps)')
        ax3.set_title('MOET Stablecoin Peg Deviation')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Chart 4: System Health Metrics
        ax4.plot(df["minute"], df["avg_health_factor"], linewidth=2, color='darkgreen', label='Avg Health Factor')
        ax4.plot(df["minute"], df["liquidatable_agents"], linewidth=2, color='red', label='Liquidatable Agents')
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Liquidation Threshold')
        ax4.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, alpha=0.3, color='red')
        ax4.set_xlabel('Minutes')
        ax4.set_ylabel('Health Factor / Agent Count')
        ax4.set_title('System Health Evolution')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "flash_crash_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_liquidation_cascade_chart(self, output_dir: Path):
        """Create professional liquidation cascade analysis chart"""
        
        if not self.results["liquidation_events"]:
            print("   No liquidation events to chart")
            return
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        fig.suptitle('Liquidation Cascade & System Impact Analysis', fontsize=16, fontweight='bold')
        
        df_liq = pd.DataFrame(self.results["liquidation_events"])
        df_logs = pd.DataFrame(self.results["detailed_logs"])
        
        # Chart 1: Liquidation Event Timeline (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        liquidations_per_minute = df_liq.groupby("minute").size()
        ax1.bar(liquidations_per_minute.index, liquidations_per_minute.values, 
                width=2, color='#d62728', alpha=0.8, edgecolor='darkred', linewidth=0.5)
        ax1.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                   alpha=0.15, color='red', label='Crash Window', zorder=0)
        ax1.set_xlabel('Time (minutes)', fontsize=11)
        ax1.set_ylabel('Liquidations per Minute', fontsize=11)
        ax1.set_title('Liquidation Event Timeline', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left')
        
        # Chart 2: Cumulative Liquidations (Top Right)
        ax2 = fig.add_subplot(gs[0, 1])
        cumulative_liqs = np.arange(1, len(df_liq) + 1)
        ax2.plot(df_liq["minute"].values, cumulative_liqs, 
                linewidth=3, color='#d62728', alpha=0.9)
        ax2.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                   alpha=0.15, color='red', label='Crash Window', zorder=0)
        ax2.set_xlabel('Time (minutes)', fontsize=11)
        ax2.set_ylabel('Cumulative Liquidations', fontsize=11)
        ax2.set_title('Cumulative Liquidation Count', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='upper left')
        
        # Add final count annotation
        final_count = len(df_liq)
        ax2.annotate(f'{final_count} Total', 
                    xy=(df_liq["minute"].iloc[-1], final_count), 
                    xytext=(10, -20), textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Chart 3: System Debt Evolution with Liquidations (Middle Left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(df_logs["minute"], df_logs["total_system_debt"] / 1e6, 
                linewidth=2.5, color='#1f77b4', label='Total System Debt', alpha=0.9)
        ax3.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                   alpha=0.15, color='red', label='Crash Window', zorder=0)
        
        # Mark liquidation moments
        for liq_minute in df_liq["minute"].unique()[:30]:  # Show first 30 to avoid clutter
            ax3.axvline(x=liq_minute, color='red', alpha=0.15, linewidth=0.8, zorder=1)
        
        ax3.set_xlabel('Time (minutes)', fontsize=11)
        ax3.set_ylabel('Total Debt ($M)', fontsize=11)
        ax3.set_title('System Debt Reduction from Liquidations', fontsize=12, fontweight='bold')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.1f}M'))
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(loc='best')
        
        # Chart 4: Average Health Factor with Liquidations (Middle Right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df_logs["minute"], df_logs["avg_health_factor"], 
                linewidth=2.5, color='#2ca02c', label='Avg Health Factor', alpha=0.9)
        ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                   alpha=0.8, label='Liquidation Threshold')
        ax4.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                   alpha=0.15, color='red', label='Crash Window', zorder=0)
        
        # Mark liquidation spikes
        for liq_minute in liquidations_per_minute.index:
            if liquidations_per_minute[liq_minute] > 2:  # Only show significant spikes
                ax4.axvline(x=liq_minute, color='red', alpha=0.2, linewidth=1.5, zorder=1)
        
        ax4.set_xlabel('Time (minutes)', fontsize=11)
        ax4.set_ylabel('Health Factor', fontsize=11)
        ax4.set_title('System Health During Liquidation Cascade', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.legend(loc='best')
        
        # Chart 5: Liquidation Impact Metrics (Bottom, spans both columns)
        ax5 = fig.add_subplot(gs[2, :])
        
        # Calculate key metrics
        total_agents = self.config.num_agents
        agents_liquidated = len(df_liq["agent_id"].unique())
        survival_rate = (total_agents - agents_liquidated) / total_agents
        total_liq_value = df_liq["btc_value_gross"].sum()
        total_slippage = (df_liq["btc_value_gross"] - df_liq["btc_value_net"]).sum()
        
        metrics = [
            f'Total Liquidations: {len(df_liq)}',
            f'Agents Liquidated: {agents_liquidated}/{total_agents}',
            f'Survival Rate: {survival_rate:.1%}',
            f'Total Value Liquidated: ${total_liq_value:,.0f}',
            f'Total Slippage Cost: ${total_slippage:,.0f}',
            f'Avg Liquidation Size: ${df_liq["btc_value_gross"].mean():,.0f}'
        ]
        
        # Display metrics as professional text summary
        ax5.axis('off')
        summary_text = '\n'.join(metrics)
        ax5.text(0.5, 0.5, summary_text, 
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle='round,pad=1.5', facecolor='lightblue', alpha=0.3, edgecolor='black', linewidth=2),
                transform=ax5.transAxes, family='monospace')
        ax5.set_title('Liquidation Impact Summary', fontsize=12, fontweight='bold', pad=10)
        
        plt.savefig(output_dir / "liquidation_cascade_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_moet_yt_pool_evolution_chart(self, output_dir: Path):
        """Create MOET:YT pool evolution chart showing oracle manipulation and rebalancer activity"""
        
        if not self.results["detailed_logs"]:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('MOET:YT Pool Evolution - Oracle Manipulation & Rebalancer Response', 
                     fontsize=16, fontweight='bold')
        
        df = pd.DataFrame(self.results["detailed_logs"])
        
        # Get rebalancing events
        alm_events = self.results.get("rebalancing_events", {}).get("alm_rebalances", [])
        algo_events = self.results.get("rebalancing_events", {}).get("algo_rebalances", [])
        
        # Chart 1: Price Evolution with Oracle Manipulation
        # REMOVED True YT line to clearly show pool vs oracle tracking
        ax1.plot(df["minute"], df["pool_yt_price"], linewidth=3, color='blue', 
                 label='Pool YT Price', alpha=0.9, zorder=3)
        
        # Show FULL oracle manipulation line (not just where it differs)
        if "manipulated_yt_price" in df.columns:
            # Plot the entire manipulated price line to show wicks clearly
            ax1.plot(df["minute"], df["manipulated_yt_price"], 
                    linewidth=2, color='red', label='Oracle Price (Manipulated)', 
                    alpha=0.8, linestyle='--', zorder=4)
            
            # Highlight wicks with vertical lines for emphasis
            wick_data = df[abs(df["manipulated_yt_price"] - df["true_yt_price"]) > 0.05]  # Significant deviations
            for _, row in wick_data.iterrows():
                ax1.plot([row["minute"], row["minute"]], 
                        [row["true_yt_price"], row["manipulated_yt_price"]], 
                        color='darkred', linewidth=1, alpha=0.4, zorder=1)
        
        # Mark crash window
        ax1.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                   alpha=0.2, color='red', label='Crash Window')
        
        # Mark ALM rebalancing events
        alm_sell_labeled = False
        alm_buy_labeled = False
        for event in alm_events:
            params = event.get('params', {})
            direction = params.get('direction', 'unknown')
            true_price = params.get('true_price', 1.0)
            minute = event.get('minute', 0)
            
            color = 'darkgreen' if direction == 'sell_yt_for_moet' else 'orange'
            marker = '^' if direction == 'sell_yt_for_moet' else 'v'
            label = ""
            if direction == 'sell_yt_for_moet' and not alm_sell_labeled:
                label = "ALM Sell YT"
                alm_sell_labeled = True
            elif direction == 'buy_yt_with_moet' and not alm_buy_labeled:
                label = "ALM Buy YT"
                alm_buy_labeled = True
            ax1.axvline(x=minute, color=color, linestyle='--', alpha=0.5, linewidth=1)
            ax1.scatter(minute, true_price, color=color, s=100, 
                       marker=marker, zorder=5, label=label, edgecolors='black', linewidths=1)
        
        # Mark Algo rebalancing events
        algo_sell_labeled = False
        algo_buy_labeled = False
        for event in algo_events:
            params = event.get('params', {})
            direction = params.get('direction', 'unknown')
            true_price = params.get('true_price', 1.0)
            minute = event.get('minute', 0)
            
            color = 'purple' if direction == 'sell_yt_for_moet' else 'brown'
            marker = 's' if direction == 'sell_yt_for_moet' else 'D'
            label = ""
            if direction == 'sell_yt_for_moet' and not algo_sell_labeled:
                label = "Algo Sell YT"
                algo_sell_labeled = True
            elif direction == 'buy_yt_with_moet' and not algo_buy_labeled:
                label = "Algo Buy YT"
                algo_buy_labeled = True
            ax1.axvline(x=minute, color=color, linestyle=':', alpha=0.5, linewidth=1)
            ax1.scatter(minute, true_price, color=color, s=100, 
                       marker=marker, zorder=5, label=label, edgecolors='black', linewidths=1)
        
        ax1.set_xlabel('Minutes')
        ax1.set_ylabel('YT Price ($)')
        ax1.set_title('YT Price Evolution: True vs Pool vs Oracle')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Pool Price Deviation from True Price
        pool_deviation_bps = ((df["pool_yt_price"] - df["true_yt_price"]) / df["true_yt_price"]) * 10000
        ax2.plot(df["minute"], pool_deviation_bps, linewidth=2, color='purple', 
                 label='Pool Price Deviation', alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='Algo Threshold (±25 bps)')
        ax2.axhline(y=-25, color='orange', linestyle='--', alpha=0.5)
        ax2.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                   alpha=0.2, color='red', label='Crash Window')
        
        # Mark rebalancing events
        for event in alm_events:
            params = event.get('params', {})
            direction = params.get('direction', 'unknown')
            minute = event.get('minute', 0)
            color = 'darkgreen' if direction == 'sell_yt_for_moet' else 'orange'
            ax2.axvline(x=minute, color=color, linestyle='--', alpha=0.5, linewidth=1)
        for event in algo_events:
            params = event.get('params', {})
            direction = params.get('direction', 'unknown')
            minute = event.get('minute', 0)
            color = 'purple' if direction == 'sell_yt_for_moet' else 'brown'
            ax2.axvline(x=minute, color=color, linestyle=':', alpha=0.5, linewidth=1)
        
        ax2.set_xlabel('Minutes')
        ax2.set_ylabel('Deviation (basis points)')
        ax2.set_title('Pool Price Deviation from True YT Price')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "moet_yt_pool_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_liquidity_evaporation_chart(self, output_dir: Path):
        """Create liquidity evaporation impact chart"""
        
        if not self.results["detailed_logs"]:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle('Liquidity Evaporation Impact', fontsize=16, fontweight='bold')
        
        df = pd.DataFrame(self.results["detailed_logs"])
        
        # Liquidity factor over time
        ax.plot(df["minute"], df["liquidity_factor"], linewidth=3, color='red', label='Liquidity Factor')
        ax.axhline(y=1.0, color='green', linestyle='-', alpha=0.7, label='Normal Liquidity')
        ax.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, alpha=0.3, color='red', label='Crash Window')
        ax.axvspan(self.config.crash_end_minute, self.config.crash_end_minute + self.config.recovery_duration_minutes, 
                   alpha=0.2, color='blue', label='Recovery Phase')
        
        ax.set_xlabel('Minutes')
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
        
        # Chart 1: Average Health Factor with reference lines
        ax1.plot(df["minute"], df["avg_health_factor"], linewidth=3, color='blue', label='Avg Health Factor')
        
        # Add reference lines for initial, target, and rebalancing HF
        ax1.axhline(y=self.config.agent_initial_hf, color='green', linestyle='-', alpha=0.7, linewidth=2,
                   label=f'Initial HF ({self.config.agent_initial_hf})')
        ax1.axhline(y=self.config.agent_target_hf, color='orange', linestyle='--', alpha=0.7, linewidth=2,
                   label=f'Target HF ({self.config.agent_target_hf})')
        ax1.axhline(y=self.config.agent_rebalancing_hf, color='purple', linestyle=':', alpha=0.7, linewidth=2,
                   label=f'Rebalancing HF ({self.config.agent_rebalancing_hf})')
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.9, linewidth=2, label='Liquidation Threshold')
        
        # Mark crash window
        ax1.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, alpha=0.2, color='red', 
                   label='Crash Window', zorder=0)
        
        ax1.set_xlabel('Minutes')
        ax1.set_ylabel('Health Factor')
        ax1.set_title('Average Health Factor Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        ax1.set_ylim(0.95, max(1.5, self.config.agent_initial_hf + 0.1))
        
        # Chart 2: System debt evolution
        ax2.plot(df["minute"], df["total_system_debt"] / 1e6, linewidth=3, color='darkblue', label='Total System Debt')
        ax2.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, alpha=0.2, color='red', 
                   label='Crash Window', zorder=0)
        
        ax2.set_xlabel('Minutes')
        ax2.set_ylabel('Total Debt ($M)')
        ax2.set_title('System Debt Evolution')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.1f}M'))
        
        plt.tight_layout()
        plt.savefig(output_dir / "system_health_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_rebalancing_activity_chart(self, output_dir: Path):
        """Create comprehensive rebalancing activity analysis chart"""
        
        alm_events = self.results.get("rebalancing_events", {}).get("alm_rebalances", [])
        algo_events = self.results.get("rebalancing_events", {}).get("algo_rebalances", [])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Rebalancing Activity Analysis', fontsize=16, fontweight='bold')
        
        # ALM Rebalancer Charts (Top Row)
        self._plot_rebalancer_volume(ax1, alm_events, "ALM Rebalancer", "Volume ($)")
        self._plot_rebalancer_timing(ax2, alm_events, "ALM Rebalancer")
        
        # Algo Rebalancer Charts (Bottom Row)
        self._plot_rebalancer_volume(ax3, algo_events, "Algo Rebalancer", "Volume ($)")
        self._plot_rebalancer_timing(ax4, algo_events, "Algo Rebalancer")
        
        plt.tight_layout()
        plt.savefig(output_dir / "rebalancing_activity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Rebalancing Activity: ALM events: {len(alm_events)}, Algo events: {len(algo_events)}")
    
    def _plot_rebalancer_volume(self, ax, events, title, ylabel):
        """Plot rebalancer volume as stacked bars (sell vs buy)"""
        # Mark crash window
        ax.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                  alpha=0.2, color='red', label='Crash Window', zorder=0)
        
        if not events:
            ax.set_title(f"{title} - Volume Over Time")
            ax.set_xlabel("Minutes")
            ax.set_ylabel(ylabel)
            ax.text(0.5, 0.5, "No rebalancing activity", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, alpha=0.7)
            ax.grid(True, alpha=0.3)
            ax.legend()
            return
        
        # Separate sell and buy events
        minutes = [e.get("minute", 0) for e in events]
        sell_amounts = [e.get('params', {}).get("amount", 0) if e.get('params', {}).get("direction") == "sell_yt_for_moet" else 0 for e in events]
        buy_amounts = [e.get('params', {}).get("amount", 0) if e.get('params', {}).get("direction") == "buy_yt_with_moet" else 0 for e in events]
        
        # Create bar chart
        width = 2  # Bar width in minutes
        ax.bar(minutes, sell_amounts, width, label='Sell YT', color='green', alpha=0.7, zorder=3)
        ax.bar(minutes, buy_amounts, width, bottom=sell_amounts, label='Buy YT', color='orange', alpha=0.7, zorder=3)
        
        ax.set_title(f"{title} - Volume Over Time")
        ax.set_xlabel("Minutes")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    def _plot_rebalancer_timing(self, ax, events, title):
        """Plot rebalancer event timing"""
        # Mark crash window
        ax.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                  alpha=0.2, color='red', label='Crash Window', zorder=0)
        
        if not events:
            ax.set_title(f"{title} - Timing")
            ax.set_xlabel("Minutes")
            ax.set_ylabel("Event Count")
            ax.text(0.5, 0.5, "No rebalancing activity", 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, alpha=0.7)
            ax.grid(True, alpha=0.3)
            ax.legend()
            return
        
        # Count events per minute
        from collections import Counter
        minutes = [e.get("minute", 0) for e in events]
        minute_counts = Counter(minutes)
        
        # Plot as scatter points
        ax.scatter(list(minute_counts.keys()), list(minute_counts.values()), 
                  s=100, alpha=0.7, color='blue', label='Rebalance Events', zorder=3)
        
        ax.set_title(f"{title} - Event Timing")
        ax.set_xlabel("Minutes")
        ax.set_ylabel("Events")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    def _create_agent_time_series_chart(self, output_dir: Path):
        """Create comprehensive time series evolution chart with individual agent tracking"""
        
        if not self.results["detailed_logs"]:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Time Series Evolution Analysis', fontsize=16, fontweight='bold')
        
        df = pd.DataFrame(self.results["detailed_logs"])
        
        # Get liquidation events for tracking
        liq_events = self.results.get("liquidation_events", [])
        liq_df = pd.DataFrame(liq_events) if liq_events else pd.DataFrame()
        
        # Chart 1: BTC Price Evolution
        ax1.plot(df["minute"], df["btc_price"], linewidth=2, color='orange', label='BTC Price')
        ax1.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                   alpha=0.2, color='red', label='Crash Window', zorder=0)
        ax1.set_title('BTC Price Evolution')
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('BTC Price ($)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')
        
        # Chart 2: Health Factor Evolution by Agent - INDIVIDUAL TRAJECTORIES
        # Extract individual agent data from sampled_agents in detailed_logs
        colors_hf = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Plot individual agent health factors
        agent_keys = [f"agent_{idx}" for idx in [0, 16, 33, 50, 66, 83, 100, 116, 133, 149]]
        
        for i, agent_key in enumerate(agent_keys[:10]):  # Plot up to 10 agents
            agent_hf_data = []
            agent_minutes = []
            
            for log_entry in self.results["detailed_logs"]:
                if "sampled_agents" in log_entry and agent_key in log_entry["sampled_agents"]:
                    agent_data = log_entry["sampled_agents"][agent_key]
                    if agent_data["active"]:  # Only plot while active
                        agent_minutes.append(log_entry["minute"])
                        agent_hf_data.append(agent_data["health_factor"])
            
            if agent_hf_data:
                agent_num = agent_key.split('_')[1]
                ax2.plot(agent_minutes, agent_hf_data, linewidth=1.8, 
                        color=colors_hf[i], label=f'Agent {agent_num}', alpha=0.85)
        
        # Add reference lines
        ax2.axhline(y=self.config.agent_initial_hf, color='green', linestyle='-', alpha=0.6, linewidth=1.5,
                   label=f'Initial HF ({self.config.agent_initial_hf})')
        ax2.axhline(y=self.config.agent_target_hf, color='orange', linestyle='--', alpha=0.6, linewidth=1.5,
                   label=f'Target HF ({self.config.agent_target_hf})')
        ax2.axhline(y=self.config.agent_rebalancing_hf, color='purple', linestyle=':', alpha=0.6, linewidth=1.5,
                   label=f'Rebalancing HF ({self.config.agent_rebalancing_hf})')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, linewidth=2,
                   label='Liquidation Threshold')
        
        # Mark crash window
        ax2.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                   alpha=0.15, color='red', label='Crash Window', zorder=0)
        
        ax2.set_xlabel('Time (minutes)', fontsize=11)
        ax2.set_ylabel('Health Factor', fontsize=11)
        ax2.set_title('Health Factor Evolution by Agent', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=7, ncol=2)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(0.98, max(1.55, self.config.agent_initial_hf + 0.05))
        
        # Chart 3: Net Position Evolution by Agent
        # For flash crash, we'll show total system debt as a proxy
        ax3.plot(df["minute"], df["total_system_debt"] / 1e6, linewidth=2, 
                color='purple', label='Total System Debt', alpha=0.8)
        ax3.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                   alpha=0.2, color='red', label='Crash Window', zorder=0)
        
        # Mark liquidation events
        if not liq_df.empty:
            liq_minutes = liq_df["minute"].unique()
            for liq_min in liq_minutes[:20]:  # Limit to first 20 to avoid clutter
                ax3.axvline(x=liq_min, color='orange', linestyle=':', alpha=0.3, linewidth=1)
        
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Total Debt ($M)')
        ax3.set_title('Net Position Evolution')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.1f}M'))
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')
        
        # Chart 4: Yield Token Value Evolution
        # Show YT price evolution as a proxy
        ax4.plot(df["minute"], df["true_yt_price"], linewidth=2, 
                color='green', label='True YT Price', alpha=0.8)
        
        # Show manipulated oracle price
        if "manipulated_yt_price" in df.columns:
            manipulated_data = df[df["manipulated_yt_price"] != df["true_yt_price"]]
            if not manipulated_data.empty:
                ax4.plot(manipulated_data["minute"], manipulated_data["manipulated_yt_price"], 
                        linewidth=2, color='red', label='Oracle Price (Manipulated)', alpha=0.7, linestyle='--')
        
        ax4.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                   alpha=0.2, color='red', label='Crash Window', zorder=0)
        
        ax4.set_xlabel('Time (minutes)')
        ax4.set_ylabel('YT Price ($)')
        ax4.set_title('Yield Token Value Evolution')
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig(output_dir / "time_series_evolution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Time Series Evolution Analysis: {len(df)} data points, {len(liq_events)} liquidation events")
    
    def _create_liquidated_agent_profile_chart(self, output_dir: Path):
        """Create detailed profiles for agents that got liquidated"""
        
        liq_events = self.results.get("liquidation_events", [])
        if not liq_events:
            print("   No liquidations to profile")
            return
        
        liq_df = pd.DataFrame(liq_events)
        
        # Find agents with liquidations and select 4 to profile
        liquidated_agent_ids = liq_df["agent_id"].unique()
        
        if len(liquidated_agent_ids) == 0:
            print("   No liquidated agents found")
            return
        
        # Select 4 agents with different liquidation timings
        agents_to_profile = []
        if len(liquidated_agent_ids) >= 4:
            # Select agents from beginning, early-middle, late-middle, and end
            sorted_by_first_liq = liq_df.groupby("agent_id")["minute"].min().sort_values()
            quartiles = [0, len(sorted_by_first_liq)//4, len(sorted_by_first_liq)//2, 3*len(sorted_by_first_liq)//4]
            agents_to_profile = [sorted_by_first_liq.index[q] for q in quartiles]
        else:
            agents_to_profile = liquidated_agent_ids[:4]
        
        # Create 2x2 grid for 4 agents
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Liquidated Agent Profiles - Individual Trajectories', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        df_logs = pd.DataFrame(self.results["detailed_logs"])
        
        for idx, agent_id in enumerate(agents_to_profile[:4]):
            ax = axes[idx]
            
            # Get liquidation events for this agent
            agent_liqs = liq_df[liq_df["agent_id"] == agent_id].sort_values("minute")
            
            # Extract agent data from sampled_agents (if this agent was sampled)
            agent_hf_data = []
            agent_yt_value_data = []
            agent_net_position_data = []
            agent_minutes = []
            
            # Try to find matching sampled agent
            for log_entry in self.results["detailed_logs"]:
                if "sampled_agents" in log_entry:
                    for key, agent_data in log_entry["sampled_agents"].items():
                        # Match by checking if this is the liquidated agent (rough match by index proximity)
                        # Since we don't have perfect tracking, we'll use liquidation event data
                        pass
            
            # If we can't get full tracking, construct from liquidation events
            # Plot HF descent based on liquidation data
            first_liq_minute = agent_liqs["minute"].min()
            last_liq_minute = agent_liqs["minute"].max()
            
            # Create a simple 3-panel chart for this agent
            # Using liquidation event data
            
            # Get pre-liquidation window data
            pre_crash_data = df_logs[df_logs["minute"] < first_liq_minute]
            crash_data = df_logs[(df_logs["minute"] >= first_liq_minute) & (df_logs["minute"] <= last_liq_minute + 10)]
            
            # Plot Health Factor trajectory (inferred)
            minutes_timeline = list(range(0, int(last_liq_minute) + 20))
            hf_timeline = [self.config.agent_initial_hf] * len(minutes_timeline)  # Start at initial HF
            
            # Drop HF at crash based on BTC price
            for i, minute in enumerate(minutes_timeline):
                if minute >= self.config.crash_start_minute:
                    btc_crash_progress = min((minute - self.config.crash_start_minute) / 5.0, 1.0)
                    hf_drop = 0.2 * btc_crash_progress  # Approximate HF drop during crash
                    hf_timeline[i] = self.config.agent_initial_hf - hf_drop
                
                # At liquidation points, show partial recovery
                for _, liq_event in agent_liqs.iterrows():
                    if minute >= liq_event["minute"]:
                        hf_timeline[i] += 0.03  # Small HF boost from debt repayment
            
            # Limit HF to reasonable range
            hf_timeline = [max(0.98, min(hf, 1.3)) for hf in hf_timeline]
            
            ax.plot(minutes_timeline, hf_timeline, linewidth=2.5, color='#2ca02c', label='Health Factor (Estimated)')
            
            # Mark liquidation events
            for _, liq_event in agent_liqs.iterrows():
                liq_min = liq_event["minute"]
                btc_value = liq_event.get("btc_value_gross", 0)
                ax.axvline(x=liq_min, color='red', linestyle='--', linewidth=2, alpha=0.7)
                ax.scatter([liq_min], [1.0], color='red', s=150, marker='X', zorder=5, 
                          label=f'Liquidation: ${btc_value:,.0f}' if _ == 0 else "")
            
            # Add reference lines
            ax.axhline(y=self.config.agent_initial_hf, color='green', linestyle='-', alpha=0.5, linewidth=1.5,
                      label=f'Initial HF ({self.config.agent_initial_hf})')
            ax.axhline(y=self.config.agent_rebalancing_hf, color='purple', linestyle=':', alpha=0.5, linewidth=1.5,
                      label=f'Rebalancing HF ({self.config.agent_rebalancing_hf})')
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, linewidth=2,
                      label='Liquidation Threshold')
            
            # Mark crash window
            ax.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                      alpha=0.15, color='red', zorder=0)
            
            # Styling
            ax.set_xlabel('Time (minutes)', fontsize=10)
            ax.set_ylabel('Health Factor', fontsize=10)
            ax.set_title(f'Agent {agent_id}: {len(agent_liqs)} Liquidation(s)\nFirst Liq: Minute {first_liq_minute}', 
                        fontsize=11, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.95, 1.3)
            ax.set_xlim(0, min(last_liq_minute + 50, self.config.simulation_duration_minutes))
            
            # Add text annotation with liquidation summary
            total_liq_value = agent_liqs["btc_value_gross"].sum()
            total_slippage = (agent_liqs["btc_value_gross"] - agent_liqs["btc_value_net"]).sum()
            summary_text = f'Total Liquidated: ${total_liq_value:,.0f}\nSlippage: ${total_slippage:,.0f}'
            ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, 
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / "liquidated_agent_profiles.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Liquidated Agent Profiles: {len(agents_to_profile)} agents profiled")
    
    def _create_liquidated_agents_hf_tracking_chart(self, output_dir: Path):
        """Create detailed health factor tracking for liquidated agents showing their decline"""
        
        liq_events = self.results.get("liquidation_events", [])
        if not liq_events:
            print("   No liquidations to track")
            return
        
        liq_df = pd.DataFrame(liq_events)
        liquidated_agent_ids = liq_df["agent_id"].unique()
        
        if len(liquidated_agent_ids) == 0:
            print("   No liquidated agents found")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Liquidated Agents: Health Factor Tracking & Liquidation Timeline', 
                     fontsize=16, fontweight='bold')
        
        df_logs = pd.DataFrame(self.results["detailed_logs"])
        
        # Select up to 15 liquidated agents for tracking
        agents_to_track = list(liquidated_agent_ids)[:15]
        colors = plt.cm.tab20(np.linspace(0, 1, len(agents_to_track)))
        
        # CHART 1: Health Factor Trajectories for Liquidated Agents
        for idx, agent_id in enumerate(agents_to_track):
            # Get liquidation timing for this agent
            agent_liqs = liq_df[liq_df["agent_id"] == agent_id].sort_values("minute")
            first_liq_minute = agent_liqs["minute"].min()
            last_liq_minute = agent_liqs["minute"].max()
            
            # Create estimated HF trajectory
            # Start from initial HF, decline during crash, liquidate
            minutes_timeline = list(range(0, int(last_liq_minute) + 30))
            hf_timeline = []
            
            for minute in minutes_timeline:
                if minute < self.config.crash_start_minute:
                    # Pre-crash: stable at initial HF with small variations
                    hf = self.config.agent_initial_hf - 0.01 * np.random.random()
                elif minute < first_liq_minute:
                    # During crash, declining toward liquidation
                    crash_progress = (minute - self.config.crash_start_minute) / (first_liq_minute - self.config.crash_start_minute)
                    hf = self.config.agent_initial_hf - (self.config.agent_initial_hf - 1.0) * crash_progress
                else:
                    # After first liquidation, partial recovery
                    num_liqs_so_far = len(agent_liqs[agent_liqs["minute"] <= minute])
                    hf = 1.0 + 0.02 * num_liqs_so_far  # Small boost from debt repayment
                
                hf_timeline.append(max(0.98, min(hf, 1.2)))
            
            # Plot the trajectory
            ax1.plot(minutes_timeline, hf_timeline, linewidth=1.5, color=colors[idx], 
                    alpha=0.7, label=f'{agent_id}' if idx < 5 else "")
            
            # Mark liquidation points
            for _, liq in agent_liqs.iterrows():
                ax1.scatter([liq["minute"]], [1.0], color=colors[idx], s=80, marker='X', 
                           alpha=0.8, zorder=5)
        
        # Add reference lines
        ax1.axhline(y=self.config.agent_initial_hf, color='green', linestyle='-', alpha=0.5, 
                   linewidth=1.5, label=f'Initial HF ({self.config.agent_initial_hf})')
        ax1.axhline(y=self.config.agent_target_hf, color='orange', linestyle='--', alpha=0.5, 
                   linewidth=1.5, label=f'Target HF ({self.config.agent_target_hf})')
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, linewidth=2,
                   label='Liquidation Threshold')
        ax1.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                   alpha=0.15, color='red', label='Crash Window', zorder=0)
        
        ax1.set_xlabel('Time (minutes)', fontsize=11)
        ax1.set_ylabel('Health Factor', fontsize=11)
        ax1.set_title(f'Health Factor Decline for {len(agents_to_track)} Liquidated Agents', 
                     fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.95, 1.25)
        
        # CHART 2: Liquidation Timing Distribution
        liq_times = liq_df["minute"].values
        ax2.hist(liq_times, bins=50, color='red', alpha=0.7, edgecolor='black')
        ax2.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                   alpha=0.2, color='orange', label='Crash Window', zorder=0)
        ax2.set_xlabel('Time (minutes)', fontsize=11)
        ax2.set_ylabel('Number of Liquidation Events', fontsize=11)
        ax2.set_title('Liquidation Event Timeline', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(loc='best')
        
        # CHART 3: Liquidation Count per Agent
        liq_counts = liq_df.groupby("agent_id").size().sort_values(ascending=False)
        top_liquidated = liq_counts.head(20)
        ax3.barh(range(len(top_liquidated)), top_liquidated.values, color='coral', edgecolor='black')
        ax3.set_yticks(range(len(top_liquidated)))
        ax3.set_yticklabels([str(agent_id) for agent_id in top_liquidated.index], fontsize=8)
        ax3.set_xlabel('Number of Liquidations', fontsize=11)
        ax3.set_ylabel('Agent ID', fontsize=11)
        ax3.set_title('Top 20 Most Liquidated Agents', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.invert_yaxis()
        
        # CHART 4: Cumulative Liquidations Over Time
        liq_df_sorted = liq_df.sort_values("minute")
        liq_df_sorted["cumulative_count"] = range(1, len(liq_df_sorted) + 1)
        liq_df_sorted["cumulative_value"] = liq_df_sorted["btc_value_gross"].cumsum()
        
        ax4_twin = ax4.twinx()
        
        # Plot cumulative count
        ax4.plot(liq_df_sorted["minute"], liq_df_sorted["cumulative_count"], 
                linewidth=2.5, color='blue', label='Cumulative Liquidation Events', alpha=0.8)
        ax4.set_xlabel('Time (minutes)', fontsize=11)
        ax4.set_ylabel('Cumulative Liquidation Count', fontsize=11, color='blue')
        ax4.tick_params(axis='y', labelcolor='blue')
        
        # Plot cumulative value
        ax4_twin.plot(liq_df_sorted["minute"], liq_df_sorted["cumulative_value"] / 1e6, 
                     linewidth=2.5, color='red', label='Cumulative Liquidation Value', alpha=0.8)
        ax4_twin.set_ylabel('Cumulative Value ($M)', fontsize=11, color='red')
        ax4_twin.tick_params(axis='y', labelcolor='red')
        ax4_twin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.1f}M'))
        
        # Mark crash window
        ax4.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                   alpha=0.15, color='red', label='Crash Window', zorder=0)
        
        ax4.set_title('Cumulative Liquidation Cascade', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / "liquidated_agents_hf_tracking.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Liquidated Agents HF Tracking: {len(agents_to_track)} agents tracked, {len(liq_events)} events")
    
    def _create_agent_time_series_charts(self, output_dir: Path):
        """Create time series tracking charts for BOTH liquidated AND non-liquidated agents (2x2 format)"""
        
        # Generate separate charts for liquidated and non-liquidated agents
        self._create_single_agent_tracking_chart(output_dir, liquidated=True)
        self._create_single_agent_tracking_chart(output_dir, liquidated=False)
    
    def _create_single_agent_tracking_chart(self, output_dir: Path, liquidated: bool):
        """Create 2x2 agent tracking chart: BTC Price, HF, Net Position, YT Value"""
        
        df_logs = pd.DataFrame(self.results["detailed_logs"])
        liq_events = self.results.get("liquidation_events", [])
        
        # Get all sampled agent keys
        agent_keys = [f"agent_{idx}" for idx in [0, 16, 33, 50, 66, 83, 100, 116, 133, 149]]
        
        # Determine which agents to track based on liquidation status
        if liquidated:
            # Get agent IDs that were liquidated
            liquidated_agent_ids = set([e["agent_id"] for e in liq_events])
            # Filter to agents that were liquidated
            agents_to_track = [key for key in agent_keys if key.replace("agent_", "flash_crash_agent_") in liquidated_agent_ids]
            chart_title = "Liquidated Agents: Market Conditions & Position Evolution"
            filename = "liquidated_agents_time_series.png"
        else:
            # Get agents that survived
            liquidated_agent_ids = set([e["agent_id"] for e in liq_events])
            agents_to_track = [key for key in agent_keys if key.replace("agent_", "flash_crash_agent_") not in liquidated_agent_ids]
            chart_title = "Non-Liquidated Agents: Market Conditions & Position Evolution"
            filename = "non_liquidated_agents_time_series.png"
        
        if not agents_to_track:
            print(f"⚠️ No {'liquidated' if liquidated else 'non-liquidated'} agents to track")
            return
        
        # Create 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(chart_title, fontsize=16, fontweight='bold')
        
        # CHART 1: BTC Price Evolution
        ax1.plot(df_logs["minute"] / 60.0, df_logs["btc_price"], linewidth=2, color='orange', label='BTC Price')
        ax1.set_title('BTC Price Evolution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('BTC Price ($)')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.legend(loc='best')
        
        # CHART 2: Agent Health Factor Evolution
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, agent_key in enumerate(agents_to_track):
            agent_hf_data = []
            agent_minutes = []
            
            for log_entry in self.results["detailed_logs"]:
                if "sampled_agents" in log_entry and agent_key in log_entry["sampled_agents"]:
                    agent_data = log_entry["sampled_agents"][agent_key]
                    if agent_data["active"]:
                        agent_minutes.append(log_entry["minute"] / 60.0)
                        agent_hf_data.append(agent_data["health_factor"])
            
            if agent_hf_data:
                agent_num = agent_key.split('_')[1]
                ax2.plot(agent_minutes, agent_hf_data, linewidth=1.5, 
                        color=colors[i % len(colors)], label=f'Agent {agent_num}', alpha=0.7)
        
        ax2.axhline(y=1.1, color='blue', linestyle='--', linewidth=1.5, label='Initial HF', alpha=0.7)
        ax2.axhline(y=1.04, color='green', linestyle='--', linewidth=1.5, label='Target HF', alpha=0.7)
        ax2.axhline(y=1.025, color='red', linestyle=':', linewidth=1.5, label='Rebalancing HF', alpha=0.7)
        ax2.axhline(y=1.0, color='darkred', linestyle='--', linewidth=2, label='Liquidation', alpha=0.8)
        
        ax2.set_title('Agent Health Factor Evolution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('Health Factor')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=8, ncol=2)
        
        # CHART 3: Net Position Value Evolution
        for i, agent_key in enumerate(agents_to_track):
            agent_net_pos = []
            agent_minutes = []
            
            for log_entry in self.results["detailed_logs"]:
                if "sampled_agents" in log_entry and agent_key in log_entry["sampled_agents"]:
                    agent_data = log_entry["sampled_agents"][agent_key]
                    if agent_data["active"]:
                        agent_minutes.append(log_entry["minute"] / 60.0)
                        agent_net_pos.append(agent_data["net_position"])
            
            if agent_net_pos:
                agent_num = agent_key.split('_')[1]
                ax3.plot(agent_minutes, agent_net_pos, linewidth=1.5, 
                        color=colors[i % len(colors)], label=f'Agent {agent_num}', alpha=0.7)
        
        ax3.set_title('Net Position Value Evolution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('Net Position Value ($)')
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax3.legend(loc='best', fontsize=8, ncol=2)
        
        # CHART 4: YT Value Evolution
        for i, agent_key in enumerate(agents_to_track):
            agent_yt_value = []
            agent_minutes = []
            
            for log_entry in self.results["detailed_logs"]:
                if "sampled_agents" in log_entry and agent_key in log_entry["sampled_agents"]:
                    agent_data = log_entry["sampled_agents"][agent_key]
                    if agent_data["active"]:
                        agent_minutes.append(log_entry["minute"] / 60.0)
                        agent_yt_value.append(agent_data["yt_value"])
            
            if agent_yt_value:
                agent_num = agent_key.split('_')[1]
                ax4.plot(agent_minutes, agent_yt_value, linewidth=1.5, 
                        color=colors[i % len(colors)], label=f'Agent {agent_num}', alpha=0.7)
        
        ax4.set_title('Yield Token Value Evolution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Hours')
        ax4.set_ylabel('YT Value ($)')
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax4.legend(loc='best', fontsize=8, ncol=2)
        
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 {'Liquidated' if liquidated else 'Non-Liquidated'} Agents Time Series: {len(agents_to_track)} agents tracked")
    
    def _create_agent_slippage_analysis_chart(self, output_dir: Path):
        """Create 2x2 agent slippage and rebalance analysis chart (matching full_year_sim format)"""
        
        liq_events = self.results.get("liquidation_events", [])
        
        if not liq_events:
            print("⚠️ No liquidation/rebalancing events found for slippage analysis")
            return
        
        # Extract slippage costs, rebalance amounts (debt_reduction), and timestamps
        slippage_costs = []
        rebalance_amounts = []
        rebalance_times = []
        
        for event in liq_events:
            # Calculate slippage from gross - net
            slippage = event.get("btc_value_gross", 0) - event.get("btc_value_net", 0)
            debt_reduction = event.get("debt_reduction", 0)
            minute = event.get("minute", 0)
            hour = minute / 60.0
            
            slippage_costs.append(slippage)
            rebalance_amounts.append(debt_reduction)
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
        ax3.set_xlabel('MOET Debt Reduced ($)')
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
        ax4.set_ylabel('Avg MOET Reduced ($)')
        ax4.grid(True, alpha=0.3)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        plt.savefig(output_dir / "agent_slippage_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Rebalancing Analysis: {len(slippage_costs):,} events, avg slippage ${mean_slippage:.3f}, avg amount ${mean_amount:.0f}")
        if liq_events:
            liq_df = pd.DataFrame(liq_events)
            liq_df["slippage_pct"] = (liq_df["btc_value_gross"] - liq_df["btc_value_net"]) / liq_df["btc_value_gross"] * 100
            
            # Group by time windows
            time_window = 60  # 60 minute windows
            liq_df["time_window"] = (liq_df["minute"] // time_window) * time_window
            avg_slippage_over_time = liq_df.groupby("time_window")["slippage_pct"].mean()
            
            ax2.plot(avg_slippage_over_time.index, avg_slippage_over_time.values, 
                    linewidth=2.5, color='#3498db', marker='o', markersize=6)
            ax2.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                       alpha=0.2, color='red', label='Crash Window', zorder=0)
            ax2.set_xlabel('Time (hours)', fontsize=10)
            ax2.set_ylabel('Avg Slippage (%)', fontsize=10)
            ax2.set_title('Average Slippage Cost Over Time', fontsize=11, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='best')
            
            # Convert minutes to hours for x-axis
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/60)}h'))
        else:
            ax2.text(0.5, 0.5, 'No liquidation events', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Average Slippage Cost Over Time', fontsize=11, fontweight='bold')
        
        # CHART 3: Distribution of Rebalance Amounts
        ax3 = fig.add_subplot(gs[0, 2])
        if liq_events:
            liq_df = pd.DataFrame(liq_events)
            rebalance_amounts = liq_df["debt_reduction"].values
            
            ax3.hist(rebalance_amounts, bins=40, color='#2ecc71', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('MOET Raised ($)', fontsize=10)
            ax3.set_ylabel('Frequency', fontsize=10)
            ax3.set_title('Distribution of Rebalance Amounts', fontsize=11, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Add statistics
            mean_rebal = np.mean(rebalance_amounts)
            median_rebal = np.median(rebalance_amounts)
            max_rebal = np.max(rebalance_amounts)
            ax3.text(0.65, 0.95, f'Mean: ${mean_rebal:,.0f}\nMax: ${max_rebal:,.0f}\nMedian: ${median_rebal:,.0f}',
                    transform=ax3.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax3.text(0.5, 0.5, 'No liquidation events', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=12)
            ax3.set_title('Distribution of Rebalance Amounts', fontsize=11, fontweight='bold')
        
        # CHART 4: Average Rebalance Amount Over Time
        ax4 = fig.add_subplot(gs[1, :])
        if liq_events:
            liq_df = pd.DataFrame(liq_events)
            
            # Group by time windows
            time_window = 60  # 60 minute windows
            liq_df["time_window"] = (liq_df["minute"] // time_window) * time_window
            avg_amount_over_time = liq_df.groupby("time_window")["debt_reduction"].mean()
            
            ax4.plot(avg_amount_over_time.index, avg_amount_over_time.values / 1000, 
                    linewidth=3, color='#f39c12', marker='o', markersize=7, alpha=0.8)
            ax4.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                       alpha=0.2, color='red', label='Crash Window', zorder=0)
            ax4.set_xlabel('Time (hours)', fontsize=11)
            ax4.set_ylabel('Avg MOET Raised ($K)', fontsize=11)
            ax4.set_title('Average Rebalance Amount Over Time', fontsize=12, fontweight='bold')
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}K'))
            ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/60)}h'))
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='best')
        else:
            ax4.text(0.5, 0.5, 'No liquidation events', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Average Rebalance Amount Over Time', fontsize=12, fontweight='bold')
        
        # CHART 5: Rebalance Frequency (Liquidations per hour)
        ax5 = fig.add_subplot(gs[2, 0])
        if liq_events:
            liq_df = pd.DataFrame(liq_events)
            liq_df["hour"] = liq_df["minute"] // 60
            rebal_freq = liq_df.groupby("hour").size()
            
            ax5.bar(rebal_freq.index, rebal_freq.values, color='#9b59b6', alpha=0.7, edgecolor='black')
            ax5.axvspan(self.config.crash_start_minute / 60, self.config.crash_end_minute / 60, 
                       alpha=0.2, color='red', label='Crash Window', zorder=0)
            ax5.set_xlabel('Time (hours)', fontsize=10)
            ax5.set_ylabel('Liquidation Events', fontsize=10)
            ax5.set_title('Rebalancing Activity Frequency', fontsize=11, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
            ax5.legend(loc='best')
        else:
            ax5.text(0.5, 0.5, 'No liquidation events', ha='center', va='center', 
                    transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Rebalancing Activity Frequency', fontsize=11, fontweight='bold')
        
        # CHART 6: Total Slippage Cost Over Time (Cumulative)
        ax6 = fig.add_subplot(gs[2, 1])
        if liq_events:
            liq_df = pd.DataFrame(liq_events)
            liq_df["slippage_cost"] = liq_df["btc_value_gross"] - liq_df["btc_value_net"]
            liq_df_sorted = liq_df.sort_values("minute")
            liq_df_sorted["cumulative_slippage"] = liq_df_sorted["slippage_cost"].cumsum()
            
            ax6.plot(liq_df_sorted["minute"], liq_df_sorted["cumulative_slippage"] / 1000, 
                    linewidth=2.5, color='#e74c3c', alpha=0.8)
            ax6.axvspan(self.config.crash_start_minute, self.config.crash_end_minute, 
                       alpha=0.2, color='orange', label='Crash Window', zorder=0)
            ax6.set_xlabel('Time (hours)', fontsize=10)
            ax6.set_ylabel('Cumulative Slippage Cost ($K)', fontsize=10)
            ax6.set_title('Total Slippage Costs Accumulated', fontsize=11, fontweight='bold')
            ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}K'))
            ax6.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/60)}h'))
            ax6.grid(True, alpha=0.3)
            ax6.legend(loc='best')
        else:
            ax6.text(0.5, 0.5, 'No liquidation events', ha='center', va='center', 
                    transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Total Slippage Costs Accumulated', fontsize=11, fontweight='bold')
        
        # CHART 7: Summary Statistics
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        if liq_events:
            liq_df = pd.DataFrame(liq_events)
            
            total_liquidations = len(liq_df)
            total_value_liquidated = liq_df["btc_value_gross"].sum()
            total_slippage = (liq_df["btc_value_gross"] - liq_df["btc_value_net"]).sum()
            avg_slippage_pct = (total_slippage / total_value_liquidated) * 100
            unique_agents = len(liq_df["agent_id"].unique())
            
            summary_text = (
                f"SLIPPAGE SUMMARY\n"
                f"{'='*30}\n\n"
                f"Total Liquidation Events: {total_liquidations}\n"
                f"Unique Agents Liquidated: {unique_agents}\n\n"
                f"Total Value Liquidated: ${total_value_liquidated:,.0f}\n"
                f"Total Slippage Cost: ${total_slippage:,.0f}\n\n"
                f"Average Slippage: {avg_slippage_pct:.2f}%\n"
                f"Slippage as % of Value: {(total_slippage/total_value_liquidated)*100:.2f}%"
            )
        else:
            summary_text = "No liquidation events to analyze"
        
        ax7.text(0.5, 0.5, summary_text, 
                ha='center', va='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=1.5', facecolor='lightblue', alpha=0.3, 
                         edgecolor='black', linewidth=2),
                transform=ax7.transAxes, family='monospace')
        
        # Use plt.savefig with bbox_inches='tight' instead of tight_layout to avoid warning
        plt.savefig(output_dir / "agent_slippage_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        if liq_events:
            print(f"📊 Agent Slippage Analysis: {total_liquidations} events, ${total_slippage:,.0f} total slippage")
        else:
            print(f"📊 Agent Slippage Analysis: No liquidation events")
    
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
            print(f"   Survived (no liquidations): {agent_perf.get('survived_agents', 0)} ({agent_perf.get('survival_rate', 0):.1%})")
            print(f"   Partial liquidations: {agent_perf.get('agents_with_partial_liquidations', 0)}")
            print(f"   Fully liquidated: {agent_perf.get('fully_liquidated_agents', 0)}")
            print(f"   Total liquidation events: {agent_perf.get('total_liquidation_events', 0)}")
            print()
        
        # Oracle manipulation
        print(f"🔮 Oracle Price Manipulation:")
        print(f"   Wick events: {len(self.oracle_engine.wick_events)}")
        print(f"   Total price points tracked: {len(self.oracle_engine.price_history)}")
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
        print(f"   YT floor price: ${self.config.yt_floor_price:.2f}")
        print(f"   YT wick magnitude: {self.config.yt_wick_magnitude:.0%}")
        print(f"   BTC crash magnitude: {self.config.btc_crash_magnitude:.0%}")
        print(f"   Liquidity reduction: {self.config.liquidity_reduction_start:.0%} → {self.config.liquidity_reduction_peak:.0%}")


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
