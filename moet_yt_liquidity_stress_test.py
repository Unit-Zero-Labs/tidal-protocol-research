#!/usr/bin/env python3
"""
MOET:YT Liquidity Pool Stress Test
Single Agent Liquidity Breaking Point Analysis

This test focuses on identifying the exact breaking point of the MOET:YT liquidity pool
by using a single agent with aggressive rebalancing parameters that will consistently
trigger rebalancing events during the BTC decline scenario.

Test Parameters:
- Single agent with 1.30 initial HF, 1.25 target HF
- MOET:YT pool: $250k each side
- Multiple runs to demonstrate consistent pool breaking behavior
- Detailed pool state tracking and visualization

This implementation follows the established production architecture patterns from
balanced_scenario_monte_carlo.py and comprehensive_ht_vs_aave_analysis.py
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
from tidal_protocol_sim.core.protocol import TidalProtocol, Asset

# Import the established analysis patterns
from balanced_scenario_monte_carlo import AnalysisHighTideEngine, create_custom_ht_agents_with_scenario_ranges


class LiquidityStressTestConfig:
    """Configuration for MOET:YT liquidity stress test following production patterns"""
    
    def __init__(self):
        # Single agent parameters for stress testing
        self.initial_health_factor = 1.30
        self.target_health_factor = 1.25
        self.num_test_runs = 5
        
        # BTC scenario parameters
        self.btc_initial_price = 100_000.0
        self.btc_final_price = 76_300.0  # 23.7% decline
        self.btc_decline_duration = 60  # minutes
        
        # Critical pool sizing - SMALL MOET:YT pool to test breaking point
        self.moet_yt_pool_size = 250_000.0  # $250k each side (SMALL!)
        self.yield_token_concentration = 0.95  # 95% concentrated at 1:1 peg
        
        # Other pools (normal size for context)
        self.moet_btc_pool_size = 5_000_000.0  # $5M each side
        self.moet_btc_concentration = 0.80  # 80% concentrated
        
        # Configuration options
        self.use_direct_minting_for_initial = True
        self.random_seed_base = 42
        
        # Analysis configuration following production patterns
        self.scenario_name = "MOET_YT_Liquidity_Stress_Test"
        self.collect_pool_state_history = True
        self.collect_trading_activity = True
        self.collect_slippage_metrics = True
        self.collect_agent_portfolio_snapshots = True


class LiquidityStressTestEngine(AnalysisHighTideEngine):
    """Extends the production AnalysisHighTideEngine with liquidity stress test specific tracking"""
    
    def __init__(self, config: HighTideConfig):
        super().__init__(config)
        # Add liquidity stress test specific tracking
        self.pool_breaking_detected = False
        self.pool_breaking_minute = None
        self.liquidity_events = []
        
    def _process_high_tide_agents(self, minute: int):
        """Extends parent processing with liquidity stress test specific checks"""
        
        # Call parent processing (includes built-in time series tracking)
        result = super()._process_high_tide_agents(minute)
        
        # Add liquidity stress test specific pool breaking detection
        self._check_pool_breaking_conditions(minute)
        
        return result
    
    
    def _check_pool_breaking_conditions(self, minute: int):
        """Check if the pool has broken (liquidity exhausted or price out of range)"""
        if self.pool_breaking_detected:
            return  # Already detected
            
        try:
            if hasattr(self, 'yield_token_pool') and self.yield_token_pool:
                uniswap_pool = self.yield_token_pool.get_uniswap_pool()
                if uniswap_pool:
                    # Get current pool metrics
                    current_price = uniswap_pool.get_price()
                    active_liquidity = uniswap_pool._calculate_active_liquidity_from_ticks(uniswap_pool.tick_current) / 1e6
                    total_liquidity = uniswap_pool.total_liquidity
                    utilization_rate = active_liquidity / total_liquidity if total_liquidity > 0 else 0
                    
                    conditions_met = []
                    
                    # Condition 1: Low active liquidity (< 5%)
                    if utilization_rate < 0.05:
                        conditions_met.append(f"Low active liquidity: {utilization_rate:.1%}")
                    
                    # Condition 2: Price deviation from 1:1 peg (> 10%)
                    price_deviation = abs(current_price - 1.0)
                    if price_deviation > 0.10:
                        conditions_met.append(f"Price deviation from peg: {price_deviation:.1%}")
                    
                    # Condition 3: Reserve imbalance (> 3:1 ratio)
                    moet_reserves = uniswap_pool.token0_reserve
                    yt_reserves = uniswap_pool.token1_reserve
                    imbalance_ratio = 0
                    if moet_reserves > 0 and yt_reserves > 0:
                        imbalance_ratio = max(moet_reserves, yt_reserves) / min(moet_reserves, yt_reserves)
                        if imbalance_ratio > 3.0:  # 3:1 imbalance
                            conditions_met.append(f"Reserve imbalance: {imbalance_ratio:.1f}:1")
                    
                    if conditions_met:
                        self.pool_breaking_detected = True
                        self.pool_breaking_minute = minute
                        
                        breaking_event = {
                            "minute": minute,
                            "conditions_met": conditions_met,
                            "pool_state": {
                                "price": current_price,
                                "price_deviation": price_deviation,
                                "active_liquidity": active_liquidity,
                                "utilization_rate": utilization_rate,
                                "moet_reserves": moet_reserves,
                                "yt_reserves": yt_reserves,
                                "imbalance_ratio": imbalance_ratio if moet_reserves > 0 and yt_reserves > 0 else 0
                            }
                        }
                        
                        self.liquidity_events.append(breaking_event)
                        print(f"🚨 Pool breaking detected at minute {minute}: {', '.join(conditions_met)}")
            
        except Exception as e:
            print(f"Warning: Failed to check pool breaking conditions at minute {minute}: {e}")
    
    def get_liquidity_analysis_data(self):
        """Get comprehensive liquidity analysis data following production patterns"""
        # Use parent's time series data as base
        time_series_data = self.get_time_series_data()
        
        # Add liquidity stress test specific data
        return {
            "time_series_data": time_series_data,
            "liquidity_events": self.liquidity_events,
            "pool_breaking_detected": self.pool_breaking_detected,
            "pool_breaking_minute": self.pool_breaking_minute
        }


class LiquidityStressTestRunner:
    """Runner for MOET:YT liquidity pool stress tests following production patterns"""
    
    def __init__(self, config: LiquidityStressTestConfig = None):
        self.config = config or LiquidityStressTestConfig()
        
        # Follow production results structure pattern
        self.results = {
            "analysis_metadata": {
                "analysis_type": "MOET_YT_Liquidity_Stress_Test",
                "timestamp": datetime.now().isoformat(),
                "num_test_runs": config.num_test_runs,
                "btc_decline_percent": ((config.btc_initial_price - config.btc_final_price) / config.btc_initial_price) * 100,
                "pool_configurations": {
                    "moet_btc_pool_size": config.moet_btc_pool_size,
                    "moet_yt_pool_size": config.moet_yt_pool_size,
                    "yield_token_concentration": config.yield_token_concentration
                }
            },
            "run_results": [],
            "aggregate_analysis": {}
        }
        
        # Use production results directory pattern
        self.results_dir = Path("tidal_protocol_sim/results") / config.scenario_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def run_stress_tests(self) -> Dict[str, Any]:
        """Run multiple stress test iterations following production patterns"""
        print("🚨 MOET:YT LIQUIDITY POOL STRESS TEST")
        print("=" * 60)
        print(f"📊 Configuration:")
        print(f"   Agent: {self.config.initial_health_factor} → {self.config.target_health_factor} HF")
        print(f"   MOET:YT Pool: ${self.config.moet_yt_pool_size:,.0f} each side ({self.config.yield_token_concentration:.1%} concentrated)")
        print(f"   BTC Scenario: ${self.config.btc_initial_price:,.0f} → ${self.config.btc_final_price:,.0f} over {self.config.btc_decline_duration} minutes")
        print(f"   Test Runs: {self.config.num_test_runs}")
        print()
        
        # Run multiple test iterations
        for run_id in range(self.config.num_test_runs):
            print(f"🔄 Running stress test iteration {run_id + 1}/{self.config.num_test_runs}")
            
            # Use different seed for each run
            seed = self.config.random_seed_base + run_id
            random.seed(seed)
            np.random.seed(seed)
            
            run_result = self._run_single_test(run_id, seed)
            self.results["run_results"].append(run_result)
            
            # Print key results using production data structure
            breaking_detected = run_result.get("pool_breaking_detected", False)
            breaking_minute = run_result.get("pool_breaking_minute")
            agent_outcomes = run_result.get("agent_outcomes", [])
            rebalancing_events = sum(agent.get("rebalancing_events", 0) for agent in agent_outcomes)
            
            if breaking_detected:
                print(f"   🚨 Pool broke at minute {breaking_minute} after {rebalancing_events} rebalancing events")
            else:
                print(f"   ✅ Pool survived with {rebalancing_events} rebalancing events")
            print()
        
        # Generate aggregate analysis following production patterns
        print("📊 Generating aggregate analysis...")
        self._generate_aggregate_analysis()
        
        # Save results using production patterns
        self._save_results()
        
        # Generate charts
        print("📊 Generating charts...")
        self._generate_charts()
        
        print("✅ Liquidity stress test completed!")
        return self.results
    
    def _run_single_test(self, run_id: int, seed: int) -> Dict[str, Any]:
        """Run a single test iteration following production patterns"""
        
        # Create High Tide configuration following production patterns
        ht_config = HighTideConfig()
        ht_config.num_high_tide_agents = 0  # We'll create custom agent
        ht_config.btc_decline_duration = self.config.btc_decline_duration
        ht_config.btc_initial_price = self.config.btc_initial_price
        ht_config.btc_final_price_range = (self.config.btc_final_price, self.config.btc_final_price)
        
        # Configure pools - CRITICAL: Small MOET:YT pool for stress testing
        ht_config.moet_btc_pool_size = self.config.moet_btc_pool_size
        ht_config.moet_btc_concentration = self.config.moet_btc_concentration
        ht_config.moet_yield_pool_size = self.config.moet_yt_pool_size  # SMALL POOL
        ht_config.yield_token_concentration = self.config.yield_token_concentration
        ht_config.use_direct_minting_for_initial = self.config.use_direct_minting_for_initial
        
        # Create stress test engine (extends AnalysisHighTideEngine)
        engine = LiquidityStressTestEngine(ht_config)
        
        # Create single agent using production agent creation patterns
        hf_scenario = {
            "initial_hf_range": (self.config.initial_health_factor, self.config.initial_health_factor),
            "target_hf": self.config.target_health_factor
        }
        
        custom_agents = create_custom_ht_agents_with_scenario_ranges(
            target_hf=hf_scenario["target_hf"],
            initial_hf_range=hf_scenario["initial_hf_range"],
            num_agents=1,  # Single agent for stress test
            run_num=run_id,
            agent_type=f"stress_test_agent",
            yield_token_pool=engine.yield_token_pool
        )
        
        engine.high_tide_agents = custom_agents
        
        # Add agents to engine following production patterns
        for agent in custom_agents:
            engine.agents[agent.agent_id] = agent
            agent.engine = engine  # Set engine reference for real swap recording
        
        # Run simulation using production tracking patterns
        print(f"      🚀 Running simulation with single agent...")
        simulation_results = engine.run_simulation()
        
        # Extract comprehensive data using production patterns
        enhanced_results = self._extract_comprehensive_data(simulation_results, engine)
        
        # Add liquidity stress test specific data
        liquidity_analysis_data = engine.get_liquidity_analysis_data()
        enhanced_results.update({
            "pool_breaking_detected": liquidity_analysis_data["pool_breaking_detected"],
            "pool_breaking_minute": liquidity_analysis_data["pool_breaking_minute"],
            "liquidity_events": liquidity_analysis_data["liquidity_events"],
            "time_series_data": liquidity_analysis_data["time_series_data"]
        })
        
        # Add run metadata
        enhanced_results["run_metadata"] = {
            "run_id": run_id,
            "seed": seed,
            "num_agents": len(custom_agents),
            "agent_parameters": {
                "initial_health_factor": self.config.initial_health_factor,
                "target_health_factor": self.config.target_health_factor
            }
        }
        
        return enhanced_results
    
    def _extract_comprehensive_data(self, results: Dict, engine) -> Dict[str, Any]:
        """Extract comprehensive data following production patterns"""
        enhanced_results = results.copy()
        
        # Extract pool state data following production patterns
        if self.config.collect_pool_state_history:
            pool_state_data = self._extract_pool_state_data(engine)
            enhanced_results["pool_state_data"] = pool_state_data
        
        # Extract trading activity data
        if self.config.collect_trading_activity:
            trading_data = self._extract_trading_activity_data(results)
            enhanced_results["trading_activity_data"] = trading_data
        
        # Extract slippage metrics
        if self.config.collect_slippage_metrics:
            slippage_data = self._extract_slippage_metrics_data(enhanced_results)
            enhanced_results["slippage_metrics_data"] = slippage_data
        
        # Extract agent portfolio snapshots
        if self.config.collect_agent_portfolio_snapshots:
            portfolio_data = self._extract_agent_portfolio_data(results)
            enhanced_results["agent_portfolio_data"] = portfolio_data
        
        return enhanced_results
    
    def _extract_pool_state_data(self, engine) -> Dict[str, Any]:
        """Extract pool state data from engine following production patterns"""
        pool_state_data = {}
        
        try:
            # Extract yield token pool state
            if hasattr(engine, 'yield_token_pool') and engine.yield_token_pool:
                yt_pool_state = engine.yield_token_pool.get_pool_state()
                pool_state_data["moet_yt_pool"] = yt_pool_state
                
                # Get underlying Uniswap V3 pool data
                uniswap_pool = engine.yield_token_pool.get_uniswap_pool()
                if uniswap_pool:
                    pool_state_data["moet_yt_pool"]["uniswap_v3_data"] = {
                        "tick_current": uniswap_pool.tick_current,
                        "concentration": uniswap_pool.concentration,
                        "fee_tier": uniswap_pool.fee_tier,
                        "active_liquidity": uniswap_pool._calculate_active_liquidity_from_ticks(uniswap_pool.tick_current) / 1e6
                    }
        except Exception as e:
            pool_state_data = {"error": str(e)}
        
        return pool_state_data
    
    def _extract_trading_activity_data(self, results: Dict) -> Dict[str, Any]:
        """Extract trading activity data following production patterns"""
        trading_data = {}
        
        # Extract yield token trading data
        if "yield_token_trades" in results:
            trading_data["yield_token_trades"] = results["yield_token_trades"]
        
        # Extract rebalancing events
        if "rebalancing_events" in results:
            trading_data["rebalancing_events"] = results["rebalancing_events"]
        
        return trading_data
    
    def _extract_slippage_metrics_data(self, results: Dict) -> Dict[str, Any]:
        """Extract slippage metrics following production patterns"""
        slippage_data = {}
        
        agent_outcomes = results.get("agent_outcomes", [])
        total_slippage_costs = sum(agent.get("total_slippage_costs", 0) for agent in agent_outcomes)
        
        slippage_data = {
            "total_slippage_costs": total_slippage_costs,
            "average_slippage_per_agent": total_slippage_costs / len(agent_outcomes) if agent_outcomes else 0.0
        }
        
        return slippage_data
    
    def _extract_agent_portfolio_data(self, results: Dict) -> Dict[str, Any]:
        """Extract agent portfolio data following production patterns"""
        portfolio_data = {}
        
        agent_outcomes = results.get("agent_outcomes", [])
        portfolio_data["agent_count"] = len(agent_outcomes)
        portfolio_data["agent_outcomes"] = agent_outcomes
        
        return portfolio_data
    
    def _generate_aggregate_analysis(self):
        """Generate aggregate analysis across all runs following production patterns"""
        
        # Count pool breaking occurrences following production patterns
        breaking_count = sum(1 for run in self.results["run_results"] if run.get("pool_breaking_detected", False))
        breaking_rate = breaking_count / len(self.results["run_results"])
        
        # Extract breaking minutes
        breaking_minutes = [run.get("pool_breaking_minute", 0)
                          for run in self.results["run_results"] 
                          if run.get("pool_breaking_detected", False)]
        
        # Extract agent outcomes
        all_agent_outcomes = []
        for run in self.results["run_results"]:
            all_agent_outcomes.extend(run.get("agent_outcomes", []))
        
        # Calculate aggregate metrics
        self.results["aggregate_analysis"] = {
            "pool_breaking_statistics": {
                "total_runs": len(self.results["run_results"]),
                "breaking_occurrences": breaking_count,
                "breaking_rate": breaking_rate,
                "average_breaking_minute": np.mean(breaking_minutes) if breaking_minutes else None
            },
            "agent_performance": {
                "total_agents": len(all_agent_outcomes),
                "average_rebalancing_events": np.mean([agent.get("rebalancing_events", 0) for agent in all_agent_outcomes]),
                "average_slippage_costs": np.mean([agent.get("total_slippage_costs", 0) for agent in all_agent_outcomes])
            }
        }
    
    def _save_results(self):
        """Save results following production patterns"""
        results_file = self.results_dir / "liquidity_stress_test_results.json"
        
        # Convert for JSON serialization following production patterns
        json_safe_results = self._convert_for_json(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(json_safe_results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_file}")
    
    def _convert_for_json(self, obj):
        """Convert objects to JSON-serializable format following production patterns"""
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
    
    def _generate_charts(self):
        """Generate visualization charts following production patterns"""
        charts_dir = self.results_dir / "charts"
        charts_dir.mkdir(exist_ok=True)
        
        # Set style following production patterns
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Generate key charts
        self._create_pool_breaking_timeline_chart(charts_dir)
        self._create_liquidity_depletion_chart(charts_dir)
        
        print(f"📊 Charts generated in: {charts_dir}")
    
    def _create_pool_breaking_timeline_chart(self, charts_dir: Path):
        """Create pool breaking timeline chart following production patterns"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Extract data
        run_ids = []
        breaking_minutes = []
        
        for i, run in enumerate(self.results["run_results"]):
            if run.get("pool_breaking_detected", False):
                run_ids.append(i)
                breaking_minutes.append(run.get("pool_breaking_minute", 0))
        
        if breaking_minutes:
            ax.scatter(breaking_minutes, run_ids, s=100, alpha=0.7, color='red')
            ax.axvline(np.mean(breaking_minutes), color='red', linestyle='--', alpha=0.5, 
                       label=f'Average: {np.mean(breaking_minutes):.1f} min')
            
        ax.set_xlabel('Breaking Time (minutes)')
        ax.set_ylabel('Run ID')
        ax.set_title('MOET:YT Pool Breaking Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(charts_dir / "pool_breaking_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_liquidity_depletion_chart(self, charts_dir: Path):
        """Create liquidity depletion analysis chart following production patterns"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # This would create detailed liquidity analysis
        # For now, create a placeholder
        ax.text(0.5, 0.5, 'Liquidity Depletion Analysis\n(Detailed implementation pending)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('MOET:YT Pool Liquidity Depletion Analysis')
        
        plt.tight_layout()
        plt.savefig(charts_dir / "liquidity_depletion_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main execution function following production patterns"""
    print("🚀 Starting MOET:YT Liquidity Pool Stress Test")
    
    # Create configuration
    config = LiquidityStressTestConfig()
    
    # Run stress tests
    runner = LiquidityStressTestRunner(config)
    results = runner.run_stress_tests()
    
    # Print summary following production patterns
    print("\n" + "="*60)
    print("📊 STRESS TEST SUMMARY")
    print("="*60)
    
    aggregate = results["aggregate_analysis"]
    
    print("🚨 Pool Breaking Analysis:")
    breaking_stats = aggregate["pool_breaking_statistics"]
    print(f"   Breaking Rate: {breaking_stats['breaking_rate']:.1%} ({breaking_stats['breaking_occurrences']}/{breaking_stats['total_runs']} runs)")
    if breaking_stats['average_breaking_minute']:
        print(f"   Average Breaking Time: {breaking_stats['average_breaking_minute']:.1f} minutes")
    
    print("\n💰 Agent Performance:")
    agent_perf = aggregate["agent_performance"]
    print(f"   Average Rebalancing Events: {agent_perf['average_rebalancing_events']:.1f}")
    print(f"   Average Slippage Cost: ${agent_perf['average_slippage_costs']:,.0f}")
    
    print("\n✅ Test completed successfully!")
    print(f"📁 Results saved in: tidal_protocol_sim/results/{config.scenario_name}/")
    
    return results


if __name__ == "__main__":
    main()
