#!/usr/bin/env python3
"""
Pool Rebalancer 24-Hour Test Script

Tests the pool rebalancing (arbitrage) functionality over a focused 24-hour period
with 10 High Tide agents and a 25% BTC price drawdown to validate:

1. ALM Rebalancer: Time-based rebalancing (12-hour intervals)
2. Algo Rebalancer: Threshold-based rebalancing (50 bps deviations)
3. Agent Rebalancing: Individual agent position management
4. Pool State Evolution: MOET:YT pool price accuracy maintenance

This script provides detailed logging and analysis of all rebalancing activities.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import random

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tidal_protocol_sim.engine.high_tide_vault_engine import HighTideVaultEngine, HighTideConfig
from tidal_protocol_sim.agents.high_tide_agent import HighTideAgent
from tidal_protocol_sim.agents.pool_rebalancer import PoolRebalancerManager
from tidal_protocol_sim.core.protocol import TidalProtocol, Asset
from tidal_protocol_sim.core.yield_tokens import calculate_true_yield_token_price


class PoolRebalancer24HConfig:
    """Configuration for 24-hour pool rebalancer test"""
    
    def __init__(self):
        # Test scenario parameters
        self.test_name = "Pool_Rebalancer_36H_Test"
        self.simulation_duration_hours = 36
        self.simulation_duration_minutes = 36 * 60  # 2160 minutes
        
        # Agent configuration - Tri-Health Factor Profile
        self.num_agents = 120  # Start with 50 agents, can be increased to test limits
        self.agent_initial_hf = 1.1  # All agents start with same HF
        self.agent_rebalancing_hf = 1.025  # Trigger rebalancing at this HF
        self.agent_target_hf = 1.04  # Rebalance to this target HF
        
        # BTC price scenario - 25% drawdown over 36 hours
        self.btc_initial_price = 100_000.0
        self.btc_final_price = 50_000.0  # 50% decline over 36 hours (slower pace)
        self.btc_decline_pattern = "gradual"  # "gradual" or "sudden" or "volatile"
        
        # Pool configurations
        self.moet_btc_pool_config = {
            "size": 2_000_000,  # $2M liquidation pool (smaller for focused test)
            "concentration": 0.80,
            "fee_tier": 0.003,
            "tick_spacing": 60,
            "pool_name": "MOET:BTC"
        }
        
        self.moet_yt_pool_config = {
            "size": 500_000,  # $500k pool ($250k each side)
            "concentration": 0.95,  # 95% concentration at 1:1 peg
            "token0_ratio": 0.75,  # 75% MOET, 25% YT
            "fee_tier": 0.0005,  # 0.05% fee tier
            "tick_spacing": 10,
            "pool_name": "MOET:Yield_Token"
        }
        
        # Pool rebalancing configuration - ENABLED for this test
        self.enable_pool_arbing = True
        self.alm_rebalance_interval_minutes = 720  # 12 hours (should trigger 3 times in 36h: 12h, 24h, 36h)
        self.algo_deviation_threshold_bps = 50.0  # 50 basis points
        
        # Arbitrage delay configuration - NEW FEATURE
        self.enable_arb_delay = True  # ENABLED for this test to see delay effects!
        self.arb_delay_description = "1 hour (auto-converted based on simulation time scale)"
        
        # Yield token parameters
        self.yield_apr = 0.10  # 10% APR
        self.use_direct_minting_for_initial = True
        
        # Enhanced logging and data collection
        self.detailed_logging = True
        self.log_every_n_minutes = 30  # Log every 30 minutes
        self.collect_pool_state_every_n_minutes = 60  # Hourly pool state snapshots
        self.track_all_rebalancing_events = True
        
        # Output configuration
        self.generate_charts = True
        self.save_detailed_csv = True
        
    def get_btc_price_at_minute(self, minute: int) -> float:
        """Calculate BTC price at given minute based on decline pattern"""
        
        progress = minute / self.simulation_duration_minutes
        total_decline = self.btc_initial_price - self.btc_final_price
        
        if self.btc_decline_pattern == "gradual":
            # Linear decline
            price_decline = total_decline * progress
            return self.btc_initial_price - price_decline
            
        elif self.btc_decline_pattern == "sudden":
            # Sharp drop in first 6 hours, then stabilize
            if minute <= 360:  # First 6 hours
                drop_progress = minute / 360
                price_decline = total_decline * drop_progress
            else:
                price_decline = total_decline
            return self.btc_initial_price - price_decline
            
        elif self.btc_decline_pattern == "volatile":
            # Add volatility around the main trend
            base_decline = total_decline * progress
            # Add some random volatility (±2% around trend)
            volatility = 0.02 * self.btc_initial_price * (random.random() - 0.5) * 2
            return max(self.btc_initial_price - base_decline + volatility, 10_000.0)
        
        else:
            return self.btc_initial_price - (total_decline * progress)


class PoolRebalancer24HTest:
    """Main test class for 24-hour pool rebalancer validation"""
    
    def __init__(self, config: PoolRebalancer24HConfig):
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
        """Run the complete 24-hour pool rebalancer test"""
        
        print("🧪 POOL REBALANCER 36-HOUR TEST")
        print("=" * 60)
        print(f"📅 Duration: {self.config.simulation_duration_hours} hours")
        print(f"👥 Agents: {self.config.num_agents} High Tide agents (Tri-HF Profile)")
        print(f"📊 Agent Profile: Initial HF={self.config.agent_initial_hf}, Rebalance HF={self.config.agent_rebalancing_hf}, Target HF={self.config.agent_target_hf}")
        print(f"📉 BTC Decline: ${self.config.btc_initial_price:,.0f} → ${self.config.btc_final_price:,.0f} ({self.results['test_metadata']['btc_decline_percent']:.1f}%)")
        print(f"🔄 Pool Arbitrage: {'ENABLED' if self.config.enable_pool_arbing else 'DISABLED'}")
        print(f"⏱️  ALM Interval: {self.config.alm_rebalance_interval_minutes} minutes (expect 3 triggers: 12h, 24h, 36h)")
        print(f"📊 Algo Threshold: {self.config.algo_deviation_threshold_bps} bps")
        print(f"⏳ Arbitrage Delay: {'ENABLED' if self.config.enable_arb_delay else 'DISABLED'} ({self.config.arb_delay_description})")
        print()
        
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
        
        print("\n✅ 36-hour pool rebalancer test completed!")
        self._print_test_summary()
        
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
        
        # Create custom agents
        agents = self._create_test_agents(engine)
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
        
        # Configure arbitrage delay
        pool_rebalancer.set_arb_delay_enabled(self.config.enable_arb_delay)
        
        # Store rebalancer reference for access during simulation
        engine.pool_rebalancer = pool_rebalancer
        
        self._log_event(0, "ENGINE_SETUP", "High Tide engine created with pool rebalancer", {
            "num_agents": len(agents),
            "pool_arbing_enabled": self.config.enable_pool_arbing,
            "alm_interval": self.config.alm_rebalance_interval_minutes,
            "algo_threshold": self.config.algo_deviation_threshold_bps
        })
        
        return engine
    
    def _create_test_agents(self, engine) -> List[HighTideAgent]:
        """Create test agents with tri-health factor profile"""
        
        agents = []
        
        for i in range(self.config.num_agents):
            agent_id = f"test_agent_{i:02d}"
            
            # All agents use the same tri-health factor profile
            agent = HighTideAgent(
                agent_id,
                self.config.agent_initial_hf,  # All start at 1.25
                self.config.agent_rebalancing_hf,  # Rebalance at 1.025
                self.config.agent_target_hf,  # Target 1.04
                yield_token_pool=engine.yield_token_pool
            )
            
            agents.append(agent)
            
            self._log_event(0, "AGENT_CREATED", f"Created {agent_id}", {
                "initial_hf": self.config.agent_initial_hf,
                "rebalancing_hf": self.config.agent_rebalancing_hf,
                "target_hf": self.config.agent_target_hf
            })
        
        return agents
    
    def _run_simulation_with_detailed_tracking(self, engine):
        """Run simulation with comprehensive tracking of all rebalancing activities"""
        
        print("🚀 Starting 24-hour simulation with detailed tracking...")
        
        # Run custom simulation loop with pool rebalancing integration
        return self._run_custom_simulation_with_pool_rebalancing(engine)
    
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
        
        # Progress update
        if minute % 240 == 0:  # Every 4 hours
            hours = minute / 60
            print(f"⏱️  Hour {hours:.0f}/36 - BTC: ${new_price:,.0f} ({((new_price/self.config.btc_initial_price)-1)*100:+.1f}%)")
        
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
        
        for minute in range(self.config.simulation_duration_minutes):
            engine.current_step = minute
            
            # FIXED: Use our custom gradual decline instead of engine's rapid decline manager
            new_btc_price = self.config.get_btc_price_at_minute(minute)
            engine.state.current_prices[Asset.BTC] = new_btc_price
            engine.btc_price_history.append(new_btc_price)
            
            # Update protocol state
            engine.protocol.current_block = minute
            engine.protocol.accrue_interest()
            
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
            
            # Log pool state snapshots every hour for analysis
            if minute % 60 == 0:  # Every hour
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
                    "active_agents": len([a for a in engine.high_tide_agents if a.active and a.is_healthy()])
                })
            
            if minute % 240 == 0:  # Every 4 hours
                hours = minute / 60
                print(f"⏱️  Hour {hours:.0f}/36 - BTC: ${new_btc_price:,.0f} ({((new_btc_price/self.config.btc_initial_price)-1)*100:+.1f}%)")
            
            if minute % 10 == 0:
                print(f"Minute {minute}: BTC = ${new_btc_price:,.0f}, Active agents: {engine._count_active_agents()}")
        
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
        
        return results
    
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
                "total_yield_sold": outcome.get("total_yield_sold", 0)  # FIXED: Add missing field
            })
        
        return {
            "agent_details": agent_data,
            "summary": {
                "total_agents": len(agent_data),
                "survived_agents": sum(1 for a in agent_data if a["survived"]),
                "survival_rate": sum(1 for a in agent_data if a["survived"]) / len(agent_data) if agent_data else 0,
                "total_rebalances": sum(a["rebalance_count"] for a in agent_data),
                "total_slippage_costs": sum(a["total_slippage"] for a in agent_data),
                "avg_final_hf": np.mean([a["final_hf"] for a in agent_data if a["survived"]]) if any(a["survived"] for a in agent_data) else 0
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
    
    def _save_test_results(self):
        """Save comprehensive test results"""
        
        # Create results directory
        output_dir = Path("tidal_protocol_sim/results") / self.config.test_name
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
        
        output_dir = Path("tidal_protocol_sim/results") / self.config.test_name / "charts"
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
        initial_hfs = [a["initial_hf"] for a in agent_data]
        final_hfs = [a["final_hf"] for a in agent_data]
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
        rebalance_counts = [a["rebalance_count"] for a in agent_data]
        slippage_costs = [a["total_slippage"] for a in agent_data]
        
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
        
        # Plot agent rebalances
        agent_events = self.results["rebalancing_events"]["agent_rebalances"]
        if agent_events:
            agent_hours = [e["hour"] for e in agent_events]
            ax.scatter(agent_hours, [1] * len(agent_hours), alpha=0.7, s=50, color='blue', label='Agent Rebalances')
        
        # Plot ALM rebalances
        alm_events = self.results["rebalancing_events"]["alm_rebalances"]
        if alm_events:
            alm_hours = [e["hour"] for e in alm_events]
            ax.scatter(alm_hours, [2] * len(alm_hours), alpha=0.7, s=100, color='green', marker='s', label='ALM Rebalances')
        
        # Plot Algo rebalances
        algo_events = self.results["rebalancing_events"]["algo_rebalances"]
        if algo_events:
            algo_hours = [e["hour"] for e in algo_events]
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
        net_positions = [a["net_position"] for a in agent_data if a["survived"]]
        if net_positions:
            ax2.hist(net_positions, bins=10, alpha=0.7, color='blue', edgecolor='black')
            ax2.set_xlabel('Net Position ($)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Net Position Distribution (Survived Agents)')
            ax2.grid(True, alpha=0.3)
        
        # Chart 3: Slippage costs
        slippage_costs = [a["total_slippage"] for a in agent_data]
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
        final_hfs = [a["final_hf"] for a in agent_data if a["survived"]]
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
        
        # Mark ALM rebalancing events
        sell_yt_labeled = False
        buy_yt_labeled = False
        
        for event in alm_events:
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
            
            ax1.axvline(x=event['hour'], color=color, linestyle='--', alpha=0.7)
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
        
        # Mark ALM events on deviation chart
        for event in alm_events:
            color = 'green' if event['direction'] == 'sell_yt_for_moet' else 'orange'
            ax2.axvline(x=event['hour'], color=color, linestyle='--', alpha=0.7)
        
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
        # Set consistent x-axis range for both ALM and Algo charts (0-36 hours)
        max_hours = 36
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
        # Set consistent x-axis range for both ALM and Algo charts (0-36 hours)
        max_hours = 36
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
            hour = i / 60.0  # Convert minutes to hours
            btc_data.append({
                "hour": hour,
                "minute": i,
                "btc_price": btc_entry
            })
        
        # Get agent health factor history and other agent data
        agent_health_history = simulation_results.get("agent_health_history", [])
        agent_time_series = []
        net_position_data = []
        yt_holdings_data = []
        
        # Extract data from agent health history snapshots
        for i, health_snapshot in enumerate(agent_health_history):
            hour = i / 60.0  # Convert minutes to hours
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
            ax1.set_title('BTC Price Decline Over Time')
            ax1.set_xlabel('Hours')
            ax1.set_ylabel('BTC Price ($)')
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # Top Right: Agent Health Factor Evolution
        if agent_time_series:
            hours = [d["hour"] for d in agent_time_series]
            health_factors = [d["health_factor"] for d in agent_time_series]
            
            ax2.plot(hours, health_factors, linewidth=2, color='blue', label='Agent Health Factor')
            
            # Add threshold lines
            ax2.axhline(y=self.config.agent_initial_hf, color='green', linestyle='-', alpha=0.7, label=f'Initial HF ({self.config.agent_initial_hf})')
            ax2.axhline(y=self.config.agent_target_hf, color='orange', linestyle='--', alpha=0.7, label=f'Target HF ({self.config.agent_target_hf})')
            ax2.axhline(y=self.config.agent_rebalancing_hf, color='red', linestyle=':', alpha=0.7, label=f'Rebalancing HF ({self.config.agent_rebalancing_hf})')
            
            ax2.set_title('Agent Health Factor Evolution')
            ax2.set_xlabel('Hours')
            ax2.set_ylabel('Health Factor')
            ax2.set_ylim(1.0, max(1.15, self.config.agent_initial_hf + 0.05))
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        # Bottom Left: Net Position Value Evolution
        if net_position_data:
            hours = [d["hour"] for d in net_position_data]
            net_positions = [d["net_position"] for d in net_position_data]
            ax3.plot(hours, net_positions, linewidth=2, color='purple', label='Net Position Value')
            ax3.set_title('Net Position Value Over Time')
            ax3.set_xlabel('Hours')
            ax3.set_ylabel('Net Position Value ($)')
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # Bottom Right: Yield Token Holdings Evolution
        if yt_holdings_data:
            hours = [d["hour"] for d in yt_holdings_data]
            yt_holdings = [d["yt_holdings"] for d in yt_holdings_data]
            ax4.plot(hours, yt_holdings, linewidth=2, color='green', label='YT Holdings')
            ax4.set_title('Yield Token Holdings Over Time')
            ax4.set_xlabel('Hours')
            ax4.set_ylabel('YT Holdings')
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
        
        # Create hourly data points
        for hour in range(37):  # 0 to 36 hours
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
        
        # Add rebalancer event markers
        for event in rebalancer_events:
            color = 'blue' if event["rebalancer_type"] == "ALM" else 'red'
            ax1.axvline(x=event["hour"], color=color, linestyle='--', alpha=0.7, linewidth=2)
            ax1.text(event["hour"], max(max(moet_balances), max(yt_balances)) * 0.9, 
                    f'{event["rebalancer_type"]}', rotation=90, ha='right', va='top', 
                    color=color, fontweight='bold')
        
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
        
        # Add rebalancer event markers to bottom chart too
        for event in rebalancer_events:
            color = 'blue' if event["rebalancer_type"] == "ALM" else 'red'
            ax2.axvline(x=event["hour"], color=color, linestyle='--', alpha=0.7, linewidth=2)
        
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


def main():
    """Main execution function"""
    
    print("Pool Rebalancer 36-Hour Test Script")
    print("=" * 50)
    print()
    print("Available test modes:")
    print("1. Standard Test - Test pool rebalancer with current agent count")
    print("2. Agent Limit Test - Find agent count that exhausts $500k pool")
    print("3. Arbitrage Delay Test - Test with 1-hour arbitrage delay enabled")
    print()
    
    # Get user choice
    mode = input("Select test mode (1/2/3) [default: 1]: ").strip() or "1"
    
    # Create configuration
    config = PoolRebalancer24HConfig()
    
    if mode == "2":
        print("\n🔍 AGENT LIMIT FINDING MODE SELECTED")
        print("This will test incremental agent counts to find $500k pool capacity limit...")
        
        # Run agent limit test
        test = PoolRebalancer24HTest(config)
        results = test.find_agent_limit_for_500k_pool()
        
        # Print summary
        if results.get("breaking_point"):
            bp = results["breaking_point"]
            print(f"\n🎯 BREAKING POINT FOUND:")
            print(f"   Agent Count: {bp['agent_count']}")
            print(f"   Max Single Rebalance: ${bp['max_single_rebalance']:,.0f}")
            print(f"   Estimated TVL Supported: ${bp['estimated_tvl_supported']:,.0f}")
            print(f"   Capital Efficiency: {bp['capital_efficiency_ratio']:.1f}:1")
            
            recommendations = results.get("recommendations", {})
            if recommendations:
                print(f"\n📋 RECOMMENDATIONS:")
                print(f"   Safe Agent Count: {recommendations['safe_agent_count']}")
                print(f"   Recommended Pool Size: ${recommendations['recommended_pool_size']:,.0f}")
                print(f"   Max TVL Supported: {recommendations['max_tvl_supported']}")
        
        print("\n✅ Agent limit test completed!")
        return results
        
    elif mode == "3":
        print("\n⏳ ARBITRAGE DELAY TEST MODE SELECTED")
        print("This will test the system with 1-hour arbitrage delay enabled...")
        
        # Enable arbitrage delay
        config.enable_arb_delay = True
        print(f"🔄 Arbitrage delay enabled: {config.arb_delay_description}")
        
        # Run test with arbitrage delay
        test = PoolRebalancer24HTest(config)
        results = test.run_test()
        
        print("\n✅ Arbitrage delay test completed!")
        return results
        
    else:
        print("\n📊 STANDARD TEST MODE SELECTED")
        print("This test will:")
        print(f"• Create {config.num_agents} High Tide agents with tri-health factor profile")
        print(f"  - Initial HF: {config.agent_initial_hf}, Rebalancing HF: {config.agent_rebalancing_hf}, Target HF: {config.agent_target_hf}")
        print("• Simulate 36-hour period with 50% BTC price decline")
        print("• Test ALM rebalancer (12-hour intervals - expect 3 triggers)")
        print("• Test Algo rebalancer (50 bps threshold)")
        print("• Track all rebalancing activities in detail")
        print("• Generate comprehensive analysis and charts")
        print()
        
        # Optional: Allow user to customize configuration
        enable_delay = input("Enable arbitrage delay? (y/N): ").strip().lower()
        if enable_delay == 'y':
            config.enable_arb_delay = True
            print(f"🔄 Arbitrage delay enabled: {config.arb_delay_description}")
        
        # Run standard test
        test = PoolRebalancer24HTest(config)
        results = test.run_test()
        
        print("\n✅ Pool rebalancer 36-hour test completed!")
        return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
