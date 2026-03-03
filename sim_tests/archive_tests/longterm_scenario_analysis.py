#!/usr/bin/env python3
"""
Long-Term Scenario Analysis for Tidal Protocol

Runs extended simulations (up to 12 months) with hourly BTC price updates and 
comprehensive pool arbitrage testing. This script allows for realistic long-term 
analysis of High Tide protocol performance with automated pool rebalancing.

Key Features:
- Hourly BTC price updates (configurable volatility patterns)
- ALM rebalancer operates on 12-hour intervals  
- Algo rebalancer triggers on 50bps+ deviations
- Agent rebalancing occurs on price updates
- Comprehensive yield token performance tracking
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import random
import math

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tidal_protocol_sim.engine.high_tide_vault_engine import HighTideVaultEngine, HighTideConfig
from tidal_protocol_sim.agents.high_tide_agent import HighTideAgent
from tidal_protocol_sim.agents.pool_rebalancer import PoolRebalancerManager
from tidal_protocol_sim.core.protocol import TidalProtocol, Asset
from tidal_protocol_sim.core.yield_tokens import YieldTokenPool


class LongTermSimulationConfig:
    """Configuration for long-term scenario analysis"""
    
    def __init__(self):
        # Simulation duration parameters
        self.simulation_duration_months = 12  # 12 month simulation
        self.simulation_duration_hours = self.simulation_duration_months * 30 * 24  # Convert to hours
        self.price_update_interval_minutes = 60  # Hourly price updates
        
        # Agent population
        self.num_high_tide_agents = 10
        self.agent_initial_hf_range = (1.25, 1.45)
        self.agent_target_hf = 1.1
        
        # BTC price evolution parameters
        self.btc_initial_price = 100_000.0
        self.btc_annual_drift = 0.15  # 15% annual expected return
        self.btc_annual_volatility = 0.80  # 80% annual volatility
        self.btc_price_path_type = "geometric_brownian_motion"  # or "historical_simulation"
        
        # Pool configurations
        self.moet_btc_pool_config = {
            "size": 5_000_000,  # $5M liquidation pool
            "concentration": 0.80,  # 80% concentration around BTC price
            "fee_tier": 0.003,  # 0.3% fee tier for volatile pairs
            "tick_spacing": 60,
            "pool_name": "MOET:BTC"
        }
        
        self.moet_yt_pool_config = {
            "size": 500_000,  # $500k pool ($250k each side)
            "concentration": 0.95,  # 95% concentration at 1:1 peg
            "token0_ratio": 0.75,  # 75% MOET, 25% YT
            "fee_tier": 0.0005,  # 0.05% fee tier for stable pairs
            "tick_spacing": 10,
            "pool_name": "MOET:Yield_Token"
        }
        
        # Pool arbitrage/rebalancing configuration
        self.enable_pool_arbing = True  # Enable pool arbitrage for long-term analysis
        self.alm_rebalance_interval_minutes = 720  # 12 hours (720 minutes)
        self.algo_deviation_threshold_bps = 50.0  # 50 basis points
        
        # Yield token parameters
        self.yield_apr = 0.10  # 10% APR for yield tokens
        self.use_direct_minting_for_initial = True
        
        # Data collection and output
        self.scenario_name = "Long_Term_Analysis_12M"
        self.collect_detailed_data = True
        self.generate_charts = True
        self.save_hourly_snapshots = True
        
        # Risk scenarios (optional overlays)
        self.enable_flash_crashes = False  # Enable random flash crash events
        self.flash_crash_probability_per_month = 0.1  # 10% chance per month
        self.flash_crash_magnitude_range = (0.15, 0.35)  # 15-35% crashes
        
        self.enable_bull_bear_cycles = True  # Enable cyclical market behavior
        self.bull_bear_cycle_months = 6  # 6-month cycles
        
    def get_simulation_metadata(self) -> Dict[str, Any]:
        """Get simulation metadata for results"""
        return {
            "simulation_type": "Long_Term_Analysis",
            "duration_months": self.simulation_duration_months,
            "duration_hours": self.simulation_duration_hours,
            "price_update_interval_minutes": self.price_update_interval_minutes,
            "btc_initial_price": self.btc_initial_price,
            "btc_annual_drift": self.btc_annual_drift,
            "btc_annual_volatility": self.btc_annual_volatility,
            "pool_arbing_enabled": self.enable_pool_arbing,
            "alm_interval_minutes": self.alm_rebalance_interval_minutes,
            "algo_threshold_bps": self.algo_deviation_threshold_bps,
            "num_agents": self.num_high_tide_agents
        }


class BTCPriceGenerator:
    """Generate realistic BTC price paths for long-term simulation"""
    
    def __init__(self, config: LongTermSimulationConfig):
        self.config = config
        self.current_price = config.btc_initial_price
        self.price_history = [config.btc_initial_price]
        self.time_history = [0]  # Minutes from simulation start
        
    def generate_next_price(self, current_minute: int) -> float:
        """Generate next hourly BTC price"""
        
        if self.config.btc_price_path_type == "geometric_brownian_motion":
            return self._generate_gbm_price(current_minute)
        else:
            # Could add historical simulation or other methods
            return self._generate_gbm_price(current_minute)
    
    def _generate_gbm_price(self, current_minute: int) -> float:
        """Generate price using Geometric Brownian Motion"""
        
        # Time step in years (1 hour = 1/8760 years)
        dt = self.config.price_update_interval_minutes / (365 * 24 * 60)
        
        # Apply bull/bear cycle adjustment if enabled
        drift = self.config.btc_annual_drift
        if self.config.enable_bull_bear_cycles:
            cycle_position = (current_minute / (self.config.bull_bear_cycle_months * 30 * 24 * 60)) % 1
            # Sine wave adjustment: +50% drift in bull, -50% in bear
            cycle_adjustment = 0.5 * math.sin(2 * math.pi * cycle_position)
            drift = drift * (1 + cycle_adjustment)
        
        # Standard GBM formula: dS = μS*dt + σS*dW
        mu = drift
        sigma = self.config.btc_annual_volatility
        
        # Random shock (normally distributed)
        dW = np.random.normal(0, math.sqrt(dt))
        
        # Calculate price change
        price_change = self.current_price * (mu * dt + sigma * dW)
        new_price = self.current_price + price_change
        
        # Ensure price doesn't go negative (floor at $1,000)
        new_price = max(new_price, 1_000.0)
        
        # Apply flash crashes if enabled
        if self.config.enable_flash_crashes:
            if self._should_flash_crash(current_minute):
                crash_magnitude = random.uniform(*self.config.flash_crash_magnitude_range)
                new_price = new_price * (1 - crash_magnitude)
                print(f"💥 Flash crash at minute {current_minute}: -{crash_magnitude:.1%} to ${new_price:,.0f}")
        
        # Update state
        self.current_price = new_price
        self.price_history.append(new_price)
        self.time_history.append(current_minute)
        
        return new_price
    
    def _should_flash_crash(self, current_minute: int) -> bool:
        """Determine if a flash crash should occur"""
        # Convert monthly probability to per-hour probability
        hours_per_month = 30 * 24
        hourly_probability = self.config.flash_crash_probability_per_month / hours_per_month
        
        return random.random() < hourly_probability
    
    def get_price_statistics(self) -> Dict[str, float]:
        """Get price path statistics"""
        if len(self.price_history) < 2:
            return {}
            
        prices = np.array(self.price_history)
        returns = np.diff(np.log(prices))
        
        return {
            "initial_price": self.price_history[0],
            "final_price": self.price_history[-1],
            "min_price": np.min(prices),
            "max_price": np.max(prices),
            "total_return": (self.price_history[-1] / self.price_history[0]) - 1,
            "volatility_realized": np.std(returns) * math.sqrt(365 * 24),  # Annualized
            "max_drawdown": self._calculate_max_drawdown(prices),
            "num_price_updates": len(self.price_history)
        }
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        return np.min(drawdown)


class LongTermAnalysisEngine:
    """Enhanced High Tide engine for long-term analysis"""
    
    def __init__(self, config: LongTermSimulationConfig):
        self.config = config
        
        # Create High Tide configuration
        self.ht_config = HighTideConfig()
        self.ht_config.num_high_tide_agents = 0  # We'll create custom agents
        self.ht_config.btc_decline_duration = config.simulation_duration_hours * 60  # Full simulation
        self.ht_config.btc_initial_price = config.btc_initial_price
        self.ht_config.btc_final_price_range = (config.btc_initial_price, config.btc_initial_price)  # Will be updated
        
        # Pool configurations
        self.ht_config.moet_btc_pool_size = config.moet_btc_pool_config["size"]
        self.ht_config.moet_btc_concentration = config.moet_btc_pool_config["concentration"]
        self.ht_config.moet_yield_pool_size = config.moet_yt_pool_config["size"]
        self.ht_config.yield_token_concentration = config.moet_yt_pool_config["concentration"]
        self.ht_config.yield_token_ratio = config.moet_yt_pool_config["token0_ratio"]
        self.ht_config.use_direct_minting_for_initial = config.use_direct_minting_for_initial
        
        # Create engine
        self.engine = HighTideVaultEngine(self.ht_config)
        
        # Create pool rebalancer manager
        self.pool_rebalancer = PoolRebalancerManager(
            alm_interval_minutes=config.alm_rebalance_interval_minutes,
            algo_threshold_bps=config.algo_deviation_threshold_bps
        )
        self.pool_rebalancer.set_enabled(config.enable_pool_arbing)
        self.pool_rebalancer.set_yield_token_pool(self.engine.yield_token_pool)
        
        # Create BTC price generator
        self.price_generator = BTCPriceGenerator(config)
        
        # Create agents
        self.agents = self._create_longterm_agents()
        self.engine.high_tide_agents = self.agents
        
        # Data collection
        self.hourly_snapshots = []
        self.rebalancing_events = []
        self.agent_performance_history = []
        
    def _create_longterm_agents(self) -> List[HighTideAgent]:
        """Create High Tide agents for long-term analysis"""
        agents = []
        
        for i in range(self.config.num_high_tide_agents):
            # Randomize initial health factor within range
            initial_hf = random.uniform(*self.config.agent_initial_hf_range)
            
            agent_id = f"longterm_agent_{i}"
            agent = HighTideAgent(
                agent_id,
                initial_hf,
                self.config.agent_target_hf,
                yield_token_pool=self.engine.yield_token_pool
            )
            
            # Set engine reference for proper integration
            agent.engine = self.engine
            
            agents.append(agent)
            
        return agents
    
    def run_longterm_simulation(self) -> Dict[str, Any]:
        """Run the complete long-term simulation"""
        
        print("🚀 LONG-TERM TIDAL PROTOCOL ANALYSIS")
        print("=" * 60)
        print(f"📅 Duration: {self.config.simulation_duration_months} months ({self.config.simulation_duration_hours:,} hours)")
        print(f"📊 Agents: {self.config.num_high_tide_agents} High Tide agents")
        print(f"🔄 Pool Arbitrage: {'Enabled' if self.config.enable_pool_arbing else 'Disabled'}")
        print(f"💰 BTC Initial Price: ${self.config.btc_initial_price:,.0f}")
        print()
        
        # Initialize simulation state
        current_minute = 0
        total_minutes = self.config.simulation_duration_hours * 60
        next_price_update = self.config.price_update_interval_minutes
        
        # Main simulation loop
        while current_minute < total_minutes:
            
            # Update BTC price if it's time
            if current_minute >= next_price_update:
                new_btc_price = self.price_generator.generate_next_price(current_minute)
                self._update_btc_price(new_btc_price)
                next_price_update += self.config.price_update_interval_minutes
                
                # Progress update
                progress_pct = (current_minute / total_minutes) * 100
                hours_elapsed = current_minute / 60
                print(f"⏱️  Hour {hours_elapsed:.0f}/{self.config.simulation_duration_hours:.0f} ({progress_pct:.1f}%) - BTC: ${new_btc_price:,.0f}")
            
            # Process pool rebalancing
            if self.config.enable_pool_arbing:
                protocol_state = {"current_minute": current_minute}
                asset_prices = {Asset.BTC: self.engine.state.current_prices.get(Asset.BTC, self.config.btc_initial_price)}
                
                rebalancing_events = self.pool_rebalancer.process_rebalancing(protocol_state, asset_prices)
                if rebalancing_events:
                    self.rebalancing_events.extend(rebalancing_events)
            
            # Process High Tide agents (only on price updates for efficiency)
            if current_minute >= next_price_update - self.config.price_update_interval_minutes:
                self._process_agents(current_minute)
            
            # Collect hourly snapshots
            if self.config.save_hourly_snapshots and current_minute % 60 == 0:
                self._collect_hourly_snapshot(current_minute)
            
            # Advance simulation
            current_minute += 1
        
        # Generate final results
        results = self._generate_final_results()
        
        print("\n✅ Long-term simulation completed!")
        self._print_summary(results)
        
        return results
    
    def _update_btc_price(self, new_price: float):
        """Update BTC price in the engine"""
        self.engine.state.current_prices[Asset.BTC] = new_price
        
        # Update all agents' health factors
        for agent in self.agents:
            if hasattr(agent.state, 'update_health_factor'):
                agent.state.update_health_factor(
                    self.engine.state.current_prices,
                    self.engine.state.collateral_factors
                )
    
    def _process_agents(self, current_minute: int):
        """Process all High Tide agents for rebalancing"""
        
        for agent in self.agents:
            if not agent.state.survived:
                continue
                
            # Check if agent needs rebalancing
            if agent.state.health_factor < agent.target_health_factor:
                # Attempt rebalancing
                try:
                    rebalance_result = agent.attempt_rebalancing(current_minute)
                    if rebalance_result.get("success", False):
                        # Record rebalancing event
                        self.rebalancing_events.append({
                            "type": "agent_rebalancing",
                            "agent_id": agent.agent_id,
                            "minute": current_minute,
                            "hour": current_minute / 60,
                            "rebalance_data": rebalance_result
                        })
                except Exception as e:
                    print(f"Warning: Agent {agent.agent_id} rebalancing failed: {e}")
    
    def _collect_hourly_snapshot(self, current_minute: int):
        """Collect comprehensive hourly snapshot"""
        
        hour = current_minute / 60
        current_btc_price = self.engine.state.current_prices.get(Asset.BTC, self.config.btc_initial_price)
        
        # Agent states
        agent_states = []
        for agent in self.agents:
            if hasattr(agent.state, 'yield_token_manager'):
                yt_value = agent.state.yield_token_manager.calculate_total_value(current_minute)
                yt_yield = agent.state.yield_token_manager.calculate_total_yield_accrued(current_minute)
            else:
                yt_value = 0.0
                yt_yield = 0.0
                
            agent_states.append({
                "agent_id": agent.agent_id,
                "health_factor": agent.state.health_factor,
                "btc_amount": agent.state.btc_amount,
                "moet_debt": agent.state.moet_debt,
                "collateral_value": agent.state.btc_amount * current_btc_price,
                "yield_token_value": yt_value,
                "yield_accrued": yt_yield,
                "net_position": (agent.state.btc_amount * current_btc_price) + yt_value - agent.state.moet_debt,
                "survived": agent.state.survived
            })
        
        # Pool states
        pool_states = {}
        if self.engine.yield_token_pool:
            pool_states["moet_yt_pool"] = self.engine.yield_token_pool.get_pool_state()
        
        # Rebalancer states
        rebalancer_summary = self.pool_rebalancer.get_rebalancer_summary()
        
        snapshot = {
            "hour": hour,
            "minute": current_minute,
            "btc_price": current_btc_price,
            "agent_states": agent_states,
            "pool_states": pool_states,
            "rebalancer_summary": rebalancer_summary,
            "total_agents_survived": sum(1 for state in agent_states if state["survived"])
        }
        
        self.hourly_snapshots.append(snapshot)
    
    def _generate_final_results(self) -> Dict[str, Any]:
        """Generate comprehensive final results"""
        
        # Price path statistics
        price_stats = self.price_generator.get_price_statistics()
        
        # Agent performance analysis
        agent_performance = self._analyze_agent_performance()
        
        # Pool arbitrage analysis
        arbitrage_analysis = self._analyze_arbitrage_performance()
        
        # Yield token performance
        yield_token_analysis = self._analyze_yield_token_performance()
        
        return {
            "simulation_metadata": self.config.get_simulation_metadata(),
            "price_path_statistics": price_stats,
            "agent_performance_analysis": agent_performance,
            "arbitrage_analysis": arbitrage_analysis,
            "yield_token_analysis": yield_token_analysis,
            "hourly_snapshots": self.hourly_snapshots,
            "rebalancing_events": self.rebalancing_events,
            "final_snapshot": self.hourly_snapshots[-1] if self.hourly_snapshots else {}
        }
    
    def _analyze_agent_performance(self) -> Dict[str, Any]:
        """Analyze agent performance over the simulation"""
        
        if not self.hourly_snapshots:
            return {}
        
        final_snapshot = self.hourly_snapshots[-1]
        initial_snapshot = self.hourly_snapshots[0]
        
        # Calculate performance metrics
        total_agents = len(self.agents)
        survived_agents = final_snapshot["total_agents_survived"]
        survival_rate = survived_agents / total_agents if total_agents > 0 else 0
        
        # Performance vs BTC holding
        initial_btc_price = self.price_generator.price_history[0]
        final_btc_price = self.price_generator.price_history[-1]
        btc_return = (final_btc_price / initial_btc_price) - 1
        
        agent_returns = []
        for agent_state in final_snapshot["agent_states"]:
            if agent_state["survived"]:
                # Calculate return based on net position vs initial BTC value
                initial_btc_value = 1.0 * initial_btc_price  # Each agent starts with 1 BTC equivalent
                final_net_position = agent_state["net_position"]
                agent_return = (final_net_position / initial_btc_value) - 1
                agent_returns.append(agent_return)
        
        avg_agent_return = np.mean(agent_returns) if agent_returns else 0
        outperformance = avg_agent_return - btc_return
        
        return {
            "total_agents": total_agents,
            "survived_agents": survived_agents,
            "survival_rate": survival_rate,
            "btc_return": btc_return,
            "average_agent_return": avg_agent_return,
            "outperformance_vs_btc": outperformance,
            "agent_returns": agent_returns
        }
    
    def _analyze_arbitrage_performance(self) -> Dict[str, Any]:
        """Analyze pool arbitrage performance"""
        
        if not self.config.enable_pool_arbing:
            return {"enabled": False}
        
        rebalancer_summary = self.pool_rebalancer.get_rebalancer_summary()
        
        # Count arbitrage events by type
        alm_events = [e for e in self.rebalancing_events if e.get("rebalancer") == "ALM"]
        algo_events = [e for e in self.rebalancing_events if e.get("rebalancer") == "Algo"]
        
        return {
            "enabled": True,
            "alm_rebalances": len(alm_events),
            "algo_rebalances": len(algo_events),
            "total_rebalances": len(alm_events) + len(algo_events),
            "rebalancer_summary": rebalancer_summary,
            "rebalancing_events": self.rebalancing_events
        }
    
    def _analyze_yield_token_performance(self) -> Dict[str, Any]:
        """Analyze yield token performance over time"""
        
        if not self.hourly_snapshots:
            return {}
        
        final_snapshot = self.hourly_snapshots[-1]
        
        total_yield_earned = sum(state["yield_accrued"] for state in final_snapshot["agent_states"])
        total_yt_value = sum(state["yield_token_value"] for state in final_snapshot["agent_states"])
        
        # Calculate effective yield rate
        simulation_duration_years = self.config.simulation_duration_months / 12
        expected_yield = total_yt_value * self.config.yield_apr * simulation_duration_years
        yield_efficiency = total_yield_earned / expected_yield if expected_yield > 0 else 0
        
        return {
            "total_yield_earned": total_yield_earned,
            "total_yt_value": total_yt_value,
            "expected_yield": expected_yield,
            "yield_efficiency": yield_efficiency,
            "effective_apr": (total_yield_earned / total_yt_value / simulation_duration_years) if total_yt_value > 0 else 0
        }
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print simulation summary"""
        
        price_stats = results.get("price_path_statistics", {})
        agent_perf = results.get("agent_performance_analysis", {})
        arbitrage = results.get("arbitrage_analysis", {})
        
        print("\n📊 SIMULATION SUMMARY")
        print("=" * 40)
        
        # Price path
        if price_stats:
            print(f"💰 BTC Performance:")
            print(f"   Initial: ${price_stats['initial_price']:,.0f}")
            print(f"   Final: ${price_stats['final_price']:,.0f}")
            print(f"   Return: {price_stats['total_return']:+.1%}")
            print(f"   Max Drawdown: {price_stats['max_drawdown']:.1%}")
        
        # Agent performance
        if agent_perf:
            print(f"\n👥 Agent Performance:")
            print(f"   Survival Rate: {agent_perf['survival_rate']:.1%}")
            print(f"   Avg Return: {agent_perf['average_agent_return']:+.1%}")
            print(f"   Outperformance: {agent_perf['outperformance_vs_btc']:+.1%}")
        
        # Arbitrage
        if arbitrage.get("enabled"):
            print(f"\n🔄 Pool Arbitrage:")
            print(f"   ALM Rebalances: {arbitrage['alm_rebalances']}")
            print(f"   Algo Rebalances: {arbitrage['algo_rebalances']}")
            print(f"   Total Profit: ${arbitrage['rebalancer_summary']['alm_rebalancer']['total_profit'] + arbitrage['rebalancer_summary']['algo_rebalancer']['total_profit']:,.0f}")


def save_results(results: Dict[str, Any], config: LongTermSimulationConfig):
    """Save results to files"""
    
    # Create results directory [[memory:7204822]]
    output_dir = Path("tidal_protocol_sim/results") / config.scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main results JSON
    results_path = output_dir / f"longterm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert for JSON serialization
    json_results = convert_for_json(results)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"📁 Results saved to: {results_path}")
    
    # Save price path CSV
    price_history = results.get("price_path_statistics", {})
    if "price_history" in results:
        price_df = pd.DataFrame({
            "hour": [i for i in range(len(results["price_history"]))],
            "btc_price": results["price_history"]
        })
        price_csv_path = output_dir / "btc_price_path.csv"
        price_df.to_csv(price_csv_path, index=False)
        print(f"📊 Price path CSV saved to: {price_csv_path}")


def convert_for_json(obj):
    """Convert objects to JSON-serializable format"""
    if isinstance(obj, dict):
        return {str(key): convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif hasattr(obj, 'dtype') and 'float' in str(obj.dtype):
        return float(obj)
    elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
        return int(obj)
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj


def generate_charts(results: Dict[str, Any], config: LongTermSimulationConfig):
    """Generate comprehensive charts for long-term analysis"""
    
    output_dir = Path("tidal_protocol_sim/results") / config.scenario_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Chart 1: BTC Price Evolution
    create_price_evolution_chart(results, output_dir)
    
    # Chart 2: Agent Performance Over Time
    create_agent_performance_chart(results, output_dir)
    
    # Chart 3: Pool Arbitrage Activity
    if results.get("arbitrage_analysis", {}).get("enabled"):
        create_arbitrage_activity_chart(results, output_dir)
    
    # Chart 4: Yield Token Performance
    create_yield_token_chart(results, output_dir)
    
    print(f"📊 Charts saved to: {output_dir}")


def create_price_evolution_chart(results: Dict[str, Any], output_dir: Path):
    """Create BTC price evolution chart"""
    
    snapshots = results.get("hourly_snapshots", [])
    if not snapshots:
        return
    
    hours = [s["hour"] for s in snapshots]
    prices = [s["btc_price"] for s in snapshots]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(hours, prices, linewidth=2, color='orange', label='BTC Price')
    ax.set_xlabel('Hours')
    ax.set_ylabel('BTC Price ($)')
    ax.set_title('BTC Price Evolution - Long Term Analysis')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(output_dir / "btc_price_evolution.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_agent_performance_chart(results: Dict[str, Any], output_dir: Path):
    """Create agent performance over time chart"""
    
    snapshots = results.get("hourly_snapshots", [])
    if not snapshots:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Agent Performance Analysis - Long Term', fontsize=16)
    
    hours = [s["hour"] for s in snapshots]
    
    # Chart 1: Survival rate over time
    survival_rates = [s["total_agents_survived"] / len(s["agent_states"]) for s in snapshots]
    ax1.plot(hours, survival_rates, linewidth=2, color='green')
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Survival Rate')
    ax1.set_title('Agent Survival Rate Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Chart 2: Average health factor
    avg_hfs = []
    for snapshot in snapshots:
        surviving_agents = [a for a in snapshot["agent_states"] if a["survived"]]
        if surviving_agents:
            avg_hf = np.mean([a["health_factor"] for a in surviving_agents])
            avg_hfs.append(avg_hf)
        else:
            avg_hfs.append(0)
    
    ax2.plot(hours, avg_hfs, linewidth=2, color='blue')
    ax2.set_xlabel('Hours')
    ax2.set_ylabel('Average Health Factor')
    ax2.set_title('Average Health Factor Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Chart 3: Total yield earned
    total_yields = []
    for snapshot in snapshots:
        total_yield = sum(a["yield_accrued"] for a in snapshot["agent_states"])
        total_yields.append(total_yield)
    
    ax3.plot(hours, total_yields, linewidth=2, color='purple')
    ax3.set_xlabel('Hours')
    ax3.set_ylabel('Total Yield Earned ($)')
    ax3.set_title('Cumulative Yield Token Earnings')
    ax3.grid(True, alpha=0.3)
    
    # Chart 4: Net position vs BTC
    net_positions = []
    btc_values = []
    for snapshot in snapshots:
        total_net = sum(a["net_position"] for a in snapshot["agent_states"] if a["survived"])
        btc_equivalent = len([a for a in snapshot["agent_states"] if a["survived"]]) * snapshot["btc_price"]
        net_positions.append(total_net)
        btc_values.append(btc_equivalent)
    
    ax4.plot(hours, net_positions, linewidth=2, color='green', label='High Tide Net Position')
    ax4.plot(hours, btc_values, linewidth=2, color='orange', label='BTC Holding Equivalent')
    ax4.set_xlabel('Hours')
    ax4.set_ylabel('Total Value ($)')
    ax4.set_title('High Tide vs BTC Holding Performance')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "agent_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_arbitrage_activity_chart(results: Dict[str, Any], output_dir: Path):
    """Create pool arbitrage activity chart"""
    
    events = results.get("arbitrage_analysis", {}).get("rebalancing_events", [])
    if not events:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Pool Arbitrage Activity Analysis', fontsize=16)
    
    # Separate ALM and Algo events
    alm_events = [e for e in events if e.get("rebalancer") == "ALM"]
    algo_events = [e for e in events if e.get("rebalancer") == "Algo"]
    
    # Chart 1: Arbitrage events over time
    if alm_events:
        alm_hours = [e["hour"] for e in alm_events]
        ax1.scatter(alm_hours, [1] * len(alm_hours), alpha=0.7, s=50, color='blue', label='ALM Rebalances')
    
    if algo_events:
        algo_hours = [e["hour"] for e in algo_events]
        ax1.scatter(algo_hours, [0.5] * len(algo_hours), alpha=0.7, s=50, color='red', label='Algo Rebalances')
    
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Rebalancer Type')
    ax1.set_title('Arbitrage Events Timeline')
    ax1.set_yticks([0.5, 1])
    ax1.set_yticklabels(['Algo', 'ALM'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Cumulative arbitrage profit
    all_events_sorted = sorted(events, key=lambda x: x.get("hour", 0))
    cumulative_profit = 0
    hours_profit = []
    profits = []
    
    for event in all_events_sorted:
        event_profit = event.get("params", {}).get("profit", 0)  # This would need to be added to the event data
        cumulative_profit += event_profit
        hours_profit.append(event.get("hour", 0))
        profits.append(cumulative_profit)
    
    if profits:
        ax2.plot(hours_profit, profits, linewidth=2, color='green')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('Cumulative Profit ($)')
        ax2.set_title('Cumulative Arbitrage Profit')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "arbitrage_activity_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


def create_yield_token_chart(results: Dict[str, Any], output_dir: Path):
    """Create yield token performance chart"""
    
    snapshots = results.get("hourly_snapshots", [])
    if not snapshots:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Yield Token Performance Analysis', fontsize=16)
    
    hours = [s["hour"] for s in snapshots]
    
    # Chart 1: Total yield token value over time
    yt_values = []
    for snapshot in snapshots:
        total_yt = sum(a["yield_token_value"] for a in snapshot["agent_states"])
        yt_values.append(total_yt)
    
    ax1.plot(hours, yt_values, linewidth=2, color='purple', label='Total YT Value')
    ax1.set_xlabel('Hours')
    ax1.set_ylabel('Total Yield Token Value ($)')
    ax1.set_title('Yield Token Portfolio Value Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Chart 2: Yield accrual over time
    yield_accrueds = []
    for snapshot in snapshots:
        total_yield = sum(a["yield_accrued"] for a in snapshot["agent_states"])
        yield_accrueds.append(total_yield)
    
    ax2.plot(hours, yield_accrueds, linewidth=2, color='green', label='Accrued Yield')
    
    # Add expected yield line
    config_yield_apr = 0.10  # From config
    expected_yields = []
    for hour in hours:
        years_elapsed = hour / (365 * 24)
        expected_yield = yt_values[0] * config_yield_apr * years_elapsed if yt_values else 0
        expected_yields.append(expected_yield)
    
    ax2.plot(hours, expected_yields, linewidth=2, color='orange', linestyle='--', label='Expected Yield (10% APR)')
    
    ax2.set_xlabel('Hours')
    ax2.set_ylabel('Yield Earned ($)')
    ax2.set_title('Yield Token Earnings vs Expected')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "yield_token_performance.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main execution function"""
    
    print("Long-Term Tidal Protocol Analysis")
    print("=" * 50)
    print()
    print("This analysis will:")
    print("• Run 12-month simulation with hourly BTC price updates")
    print("• Test High Tide agent performance over extended periods")
    print("• Analyze pool arbitrage effectiveness (ALM + Algo rebalancers)")
    print("• Track yield token performance and earnings")
    print("• Generate comprehensive long-term performance charts")
    print()
    
    # Create configuration
    config = LongTermSimulationConfig()
    
    # Optional: Customize configuration
    # config.simulation_duration_months = 6  # Shorter test
    # config.enable_pool_arbing = True  # Enable arbitrage
    # config.enable_flash_crashes = True  # Add some excitement
    
    # Create and run analysis
    analysis_engine = LongTermAnalysisEngine(config)
    results = analysis_engine.run_longterm_simulation()
    
    # Save results [[memory:7204822]]
    save_results(results, config)
    
    # Generate charts
    if config.generate_charts:
        generate_charts(results, config)
    
    print("\n✅ Long-term analysis completed!")
    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
