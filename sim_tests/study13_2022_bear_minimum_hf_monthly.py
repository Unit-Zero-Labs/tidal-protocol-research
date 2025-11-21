#!/usr/bin/env python3
"""
Study 13: 2022 Bear Market - Minimum Viable Health Factor (Monthly Rebalancing)

Optimization study to find the absolute minimum health factor that could survive
the 2022 bear market on Aave with monthly rebalancing.

Strategy:
- Uses binary search to efficiently find minimum viable HF
- Runs fail-fast simulations using the standard FullYearSimulation architecture
- Compares optimal Aave HF vs High Tide at 1.2 HF
- Memory-efficient: cleans up after each test run
"""

import sys
import gc
from pathlib import Path
from typing import Dict, Any, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_tests.full_year_sim import FullYearSimConfig, FullYearSimulation


class MinimumHFOptimizer:
    """Binary search optimizer to find minimum viable health factor"""
    
    def __init__(self, precision: float = 0.01):
        """
        Args:
            precision: Convergence precision (default 0.01 means within 1% HF)
        """
        self.precision = precision
        self.test_history = []
        self.min_hf_lower_bound = 1.05  # Must be above 1.0 (liquidation threshold)
        self.max_hf_upper_bound = 2.0   # Start conservative
        
    def find_minimum_hf(self) -> Tuple[float, Dict[str, Any]]:
        """
        Binary search to find minimum viable health factor for Aave during 2022.
        
        Returns:
            Tuple of (minimum_viable_hf, search_summary)
        """
        print("\n" + "=" * 80)
        print("BINARY SEARCH: Finding Minimum Viable Health Factor")
        print("=" * 80)
        print(f"Search range: [{self.min_hf_lower_bound:.2f}, {self.max_hf_upper_bound:.2f}]")
        print(f"Precision target: ±{self.precision:.3f}")
        print(f"Market: 2022 Bear Market (BTC -64.2%)")
        print(f"Strategy: Monthly rebalancing with historical Aave rates")
        print("=" * 80)
        print()
        
        lower = self.min_hf_lower_bound
        upper = self.max_hf_upper_bound
        iteration = 0
        last_successful_hf = None
        
        while (upper - lower) > self.precision:
            iteration += 1
            test_hf = (lower + upper) / 2.0
            
            print(f"\n{'─' * 80}")
            print(f"Iteration {iteration}: Testing HF = {test_hf:.3f}")
            print(f"  Search range: [{lower:.3f}, {upper:.3f}] (width: {upper - lower:.3f})")
            print(f"{'─' * 80}")
            
            # Run fail-fast simulation using standard architecture
            survived, liquidation_minute = self._test_health_factor(test_hf)
            
            # Record test
            test_record = {
                "iteration": iteration,
                "health_factor": test_hf,
                "survived": survived,
                "liquidation_minute": liquidation_minute,
                "search_range": (lower, upper)
            }
            self.test_history.append(test_record)
            
            if survived:
                # Success - can try lower
                print(f"  ✅ SURVIVED - Agent maintained position for full year")
                print(f"  → Trying lower HF (new upper bound: {test_hf:.3f})")
                upper = test_hf
                last_successful_hf = test_hf
            else:
                # Liquidated - need higher HF
                days_survived = liquidation_minute / 1440 if liquidation_minute else 0
                hours_survived = liquidation_minute / 60 if liquidation_minute else 0
                
                if liquidation_minute == 0:
                    print(f"  ❌ LIQUIDATED IMMEDIATELY (minute 0) - HF too low for initial position")
                elif liquidation_minute < 1440:
                    print(f"  ❌ LIQUIDATED at minute {liquidation_minute:,} ({hours_survived:.1f} hours / day {days_survived:.2f})")
                else:
                    print(f"  ❌ LIQUIDATED at minute {liquidation_minute:,} (day {days_survived:.1f})")
                
                print(f"  → Trying higher HF (new lower bound: {test_hf:.3f})")
                lower = test_hf
            
            # Show new range
            print(f"  → New range: [{lower:.4f}, {upper:.4f}] (width: {upper - lower:.4f})")
            
            # Force garbage collection after each test
            gc.collect()
        
        # Convergence reached
        minimum_viable_hf = upper  # Upper bound is the minimum that worked
        
        print("\n" + "=" * 80)
        print("BINARY SEARCH COMPLETE")
        print("=" * 80)
        print(f"Minimum Viable HF: {minimum_viable_hf:.3f}")
        print(f"Iterations: {iteration}")
        print(f"Precision achieved: ±{upper - lower:.4f}")
        print("=" * 80)
        print()
        
        summary = {
            "minimum_viable_hf": minimum_viable_hf,
            "iterations": iteration,
            "final_precision": upper - lower,
            "test_history": self.test_history,
            "search_bounds": {
                "initial_lower": self.min_hf_lower_bound,
                "initial_upper": self.max_hf_upper_bound,
                "final_lower": lower,
                "final_upper": upper
            }
        }
        
        return minimum_viable_hf, summary
    
    def _test_health_factor(self, test_hf: float) -> Tuple[bool, int]:
        """
        Run fail-fast simulation for a given health factor using standard architecture.
        
        Args:
            test_hf: Health factor to test
            
        Returns:
            Tuple of (survived: bool, liquidation_minute: int)
        """
        # Create standard config with fail-fast mode enabled
        config = FullYearSimConfig()
        config.test_name = f"_temp_hf_test_{test_hf:.3f}"  # Underscore prefix for cleanup
        config.simulation_duration_days = 365
        config.num_agents = 1
        config.initial_btc_per_agent = 1.0
        
        # Market data
        config.market_year = 2022
        config.use_historical_btc_data = True
        config.use_historical_aave_rates = True
        
        # Test HF (Aave only)
        config.aave_initial_hf = test_hf
        
        # MONTHLY REBALANCING
        config.leverage_frequency_minutes = 43200  # 30 days = 43,200 minutes
        
        # Disable unnecessary features for speed
        config.use_advanced_moet = False
        config.enable_weekly_yield_harvest = False  # Monthly rebalancing, no weekly harvest
        config.ecosystem_growth_enabled = False
        
        # CRITICAL: Enable fail-fast mode and suppress verbose output
        config.fail_fast_on_liquidation = True
        config.suppress_progress_output = True
        config.generate_charts = False  # Don't generate charts for test runs
        config.save_detailed_csv = False  # Don't save CSVs for test runs
        
        # Run simulation using standard architecture
        try:
            sim = FullYearSimulation(config)
            
            # Load market data
            config.load_market_data()
            
            # Create Aave engine directly (skip run_test which has no Aave-only path)
            aave_engine = sim._create_aave_engine()
            
            # CRITICAL FIX: Replace agents with single agent at exact test HF
            from tidal_protocol_sim.agents.aave_agent import AaveAgent
            from tidal_protocol_sim.core.protocol import Asset
            
            test_agent = AaveAgent(
                agent_id="test_agent_hf_optimization",
                initial_hf=test_hf,
                rebalancing_hf=test_hf,  # Not used by Aave but required
                target_hf=test_hf,        # Not used by Aave but required
                initial_balance=config.btc_initial_price  # Use BTC price as balance
            )
            
            # CRITICAL: Set yield_token_pool reference so agent can buy YT
            test_agent.state.yield_token_manager.yield_token_pool = aave_engine.yield_token_pool
            
            # Clear old agents' positions from protocol (avoid double-counting)
            btc_pool = aave_engine.protocol.asset_pools[Asset.BTC]
            btc_pool.total_supplied = 0.0  # Reset BTC pool
            aave_engine.protocol.moet_system.total_supply = 0.0  # Reset MOET
            
            # CRITICAL: Set BTC price to match what agent was created with
            aave_engine.state.current_prices[Asset.BTC] = config.btc_initial_price
            
            # Replace agents in engine
            aave_engine.aave_agents = [test_agent]
            aave_engine.agents = {test_agent.agent_id: test_agent}
            
            # Re-initialize positions with the correct agent
            aave_engine._setup_aave_positions()
            
            # Run Aave simulation with fail-fast
            results = sim._run_custom_simulation_aave(aave_engine)
            
            # Check if agent survived
            survived = results.get("summary", {}).get("survived", 0) > 0
            liquidation_minute = None
            
            # DEBUG: Print results structure
            print(f"  [DEBUG] Results keys: {list(results.keys())}")
            print(f"  [DEBUG] Survived: {survived}")
            print(f"  [DEBUG] Summary: {results.get('summary', {})}")
            if results.get("simulation_metadata"):
                print(f"  [DEBUG] Metadata: {results['simulation_metadata']}")
            if results.get("agent_outcomes"):
                print(f"  [DEBUG] Agent outcomes (first): {results['agent_outcomes'][0] if results['agent_outcomes'] else 'None'}")
            if results.get("liquidation_events"):
                print(f"  [DEBUG] Liquidation events (first): {results['liquidation_events'][0] if results['liquidation_events'] else 'None'}")
            
            # Get liquidation minute if failed
            if not survived:
                # Try multiple locations for liquidation minute
                if results.get("simulation_metadata", {}).get("terminated_early"):
                    # Fail-fast mode - get from metadata or agent_outcomes
                    liquidation_minute = results.get("simulation_metadata", {}).get("duration_minutes")
                    if liquidation_minute is None and results.get("agent_outcomes"):
                        liquidation_minute = results["agent_outcomes"][0].get("liquidation_minute", 0)
                elif results.get("liquidation_events"):
                    liquidation_minute = results["liquidation_events"][0].get("minute", 0)
                elif results.get("agent_outcomes") and results["agent_outcomes"][0].get("liquidation_minute") is not None:
                    liquidation_minute = results["agent_outcomes"][0].get("liquidation_minute", 0)
                else:
                    liquidation_minute = 0
                
                print(f"  [DEBUG] Extracted liquidation_minute: {liquidation_minute}")
            
            # SAVE DETAILED AGENT DATA before cleanup
            self._save_test_iteration_data(test_hf, survived, liquidation_minute, results)
            
            # Cleanup temporary results
            self._cleanup_temp_results(config.test_name)
            
            return survived, liquidation_minute
            
        except Exception as e:
            print(f"  ⚠️  Simulation error: {e}")
            # On error, assume failure (conservative)
            return False, 0
        finally:
            # Force cleanup
            del config
            del sim
            gc.collect()
    
    def _save_test_iteration_data(self, test_hf: float, survived: bool, liquidation_minute: int, results: Dict[str, Any]):
        """Save detailed agent data for this test iteration"""
        try:
            import json
            
            # Create directory for test iteration data
            output_dir = Path(__file__).parent.parent / "tidal_protocol_sim" / "results" / "Study_13_2022_Bear_Minimum_HF_Monthly_Test_Data"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract agent snapshots (HF over time)
            agent_snapshots = []
            if results.get("agent_outcomes"):
                for outcome in results["agent_outcomes"]:
                    if "snapshots" in outcome:
                        agent_snapshots = outcome["snapshots"]
                        break
            
            # Create iteration data
            iteration_data = {
                "health_factor": test_hf,
                "survived": survived,
                "liquidation_minute": liquidation_minute,
                "agent_snapshots": agent_snapshots,
                "summary": results.get("summary", {}),
                "simulation_metadata": results.get("simulation_metadata", {})
            }
            
            # Save to file
            filename = f"test_hf_{test_hf:.4f}_{'survived' if survived else 'liquidated'}.json"
            filepath = output_dir / filename
            with open(filepath, 'w') as f:
                json.dump(iteration_data, f, indent=2)
            
            print(f"  💾 Saved test data: {filepath}")
            
        except Exception as e:
            print(f"  ⚠️  Failed to save test data: {e}")
    
    def _cleanup_temp_results(self, test_name: str):
        """Remove temporary result files"""
        try:
            import shutil
            results_dir = Path(__file__).parent.parent / "tidal_protocol_sim" / "results" / test_name
            if results_dir.exists():
                shutil.rmtree(results_dir)
        except Exception:
            pass  # Ignore cleanup errors


def run_comparison_with_optimal_hf(optimal_aave_hf: float, optimization_summary: Dict[str, Any]):
    """
    Run full detailed comparison between:
    - Aave at optimal (minimum viable) HF
    - High Tide at 1.2 HF
    
    Both with monthly rebalancing during 2022 bear market.
    """
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON: Optimal Aave vs High Tide")
    print("=" * 80)
    print(f"Aave Configuration:")
    print(f"  - Health Factor: {optimal_aave_hf:.3f} (minimum viable)")
    print(f"  - Strategy: Monthly rebalancing")
    print(f"  - Rates: Historical Aave 2022")
    print()
    print(f"High Tide Configuration:")
    print(f"  - Health Factor: 1.20")
    print(f"  - Strategy: Monthly rebalancing")
    print(f"  - Rates: Historical Aave 2022")
    print()
    print(f"Market: 2022 Bear Market (BTC -64.2%)")
    print("=" * 80)
    print()
    
    # Create configuration
    config = FullYearSimConfig()
    
    # Study 13 parameters
    config.test_name = f"Study_13_2022_Bear_Minimum_HF_Monthly_Aave_{optimal_aave_hf:.2f}_vs_HT_1.20"
    config.simulation_duration_days = 365
    config.num_agents = 1  # Single agent per protocol
    config.initial_btc_per_agent = 1.0
    
    # Market data
    config.market_year = 2022
    config.use_historical_btc_data = True
    config.use_historical_aave_rates = True
    
    # Health factors
    config.agent_initial_hf = 1.2
    config.agent_rebalancing_hf = 1.0
    config.agent_target_hf = 1.1
    config.aave_initial_hf = optimal_aave_hf  # Use optimized minimum HF
    
    # MONTHLY REBALANCING
    config.leverage_frequency_minutes = 43200  # 30 days = 43,200 minutes
    
    # Advanced MOET: OFF for symmetric study
    config.use_advanced_moet = False
    
    # No weekly yield harvesting for monthly study
    config.enable_weekly_yield_harvest = False
    
    # Comparison mode
    config.comparison_mode = True
    config.run_high_tide_only = False
    
    # Ecosystem growth: OFF for clean comparison
    config.ecosystem_growth_enabled = False
    
    # Normal output mode (not fail-fast)
    config.fail_fast_on_liquidation = False
    config.suppress_progress_output = False
    config.generate_charts = True
    config.save_detailed_csv = True
    
    print("Starting detailed comparison simulation...")
    print("Expected runtime: ~5-10 minutes")
    print()
    
    # Run simulation
    sim = FullYearSimulation(config)
    results = sim.run_test()
    
    # Add optimization summary to results
    if "high_tide_results" in results:
        results["optimization_summary"] = optimization_summary
    
    # SAVE OPTIMIZATION SUMMARY TO DEDICATED FILE
    import json
    from pathlib import Path
    results_dir = Path(__file__).parent.parent / "tidal_protocol_sim" / "results" / config.test_name
    results_dir.mkdir(parents=True, exist_ok=True)
    optimization_file = results_dir / "optimization_summary.json"
    
    with open(optimization_file, 'w') as f:
        json.dump(optimization_summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("STUDY 13 COMPLETE")
    print("=" * 80)
    print(f"Minimum viable Aave HF: {optimal_aave_hf:.3f}")
    print(f"Optimization summary saved to: {optimization_file}")
    print(f"Results saved to: tidal_protocol_sim/results/{config.test_name}/")
    print()
    
    return results


def main():
    print("\n" + "🎯" * 40)
    print("STUDY 13: 2022 Bear Market - Minimum Viable Health Factor (Monthly)")
    print("🎯" * 40)
    print()
    print("Objective: Find the absolute minimum health factor that could have")
    print("           survived the 2022 bear market on Aave with monthly rebalancing.")
    print()
    print("Method: Binary search optimization with fail-fast liquidation detection")
    print("        Then compare optimal Aave HF vs High Tide at 1.2 HF")
    print()
    print("🎯" * 40)
    print()
    
    # Phase 1: Find minimum viable HF
    optimizer = MinimumHFOptimizer(precision=0.01)
    optimal_hf, optimization_summary = optimizer.find_minimum_hf()
    
    # Phase 2: Run detailed comparison
    results = run_comparison_with_optimal_hf(optimal_hf, optimization_summary)
    
    print("\n" + "=" * 80)
    print("STUDY 13 COMPLETE - KEY FINDINGS")
    print("=" * 80)
    print()
    print(f"📊 Minimum Viable Aave HF: {optimal_hf:.3f}")
    print(f"   → Required {(optimal_hf - 1.0) * 100:.1f}% safety margin above liquidation")
    print()
    print(f"🔍 Binary Search Statistics:")
    print(f"   → Iterations: {optimization_summary['iterations']}")
    print(f"   → Precision: ±{optimization_summary['final_precision']:.4f}")
    print()
    print(f"📁 Full results saved to:")
    print(f"   tidal_protocol_sim/results/Study_13_2022_Bear_Minimum_HF_Monthly_Aave_{optimal_hf:.2f}_vs_HT_1.20/")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

