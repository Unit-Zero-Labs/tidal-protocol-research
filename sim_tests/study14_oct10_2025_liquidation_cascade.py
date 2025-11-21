#!/usr/bin/env python3
"""
Study 14: October 10th 2025 Liquidation Cascade Stress Test

This study uses the largest liquidation event in crypto history (Oct 10, 2025)
to determine the minimum viable Health Factor needed to survive extreme volatility.

Market Context:
- BTC fell from $121,713 to $108,931 (-11.01% max drawdown)
- Single day stress test with minute-by-minute price data
- Tests Aave agents with various HFs to find minimum survival threshold
- Compares against High Tide at 1.2 HF

Methodology:
- Binary search optimization to find minimum viable Aave HF
- Fail-fast on liquidation for efficiency
- Single agent per protocol for clean comparison
- No rebalancing (buy-and-hold stress test)
"""

import sys
import csv
from pathlib import Path
from typing import Dict, Tuple, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sim_tests.full_year_sim import FullYearSimConfig, FullYearSimulation
from tidal_protocol_sim.core.protocol import Asset

class MinimumHFOptimizer:
    """Binary search optimizer to find minimum viable HF for October 10th crash"""
    
    def __init__(self):
        self.test_history = []
        
    def _load_oct10_data(self) -> List[float]:
        """Load minute-by-minute BTC price data from October 10th 2025"""
        csv_path = Path(__file__).parent.parent / 'dune_query_6227486.csv'
        
        prices = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prices.append(float(row['btc_price_usd']))
        
        print(f"✅ Loaded {len(prices)} minutes of October 10th 2025 data")
        print(f"   Start: ${prices[0]:,.2f}")
        print(f"   Low: ${min(prices):,.2f}")
        print(f"   End: ${prices[-1]:,.2f}")
        print(f"   Max Drawdown: {((min(prices) / max(prices)) - 1) * 100:.2f}%")
        
        return prices
        
    def _test_health_factor(self, test_hf: float) -> Tuple[bool, int]:
        """
        Test if an agent with given HF survives October 10th crash
        
        Returns:
            Tuple of (survived: bool, liquidation_minute: int)
        """
        # Create standard config with fail-fast mode enabled
        config = FullYearSimConfig()
        config.test_name = f"_temp_hf_test_{test_hf:.3f}"  # Underscore prefix for cleanup
        config.simulation_duration_minutes = 1440  # Single day = 1,440 minutes
        config.num_agents = 1
        config.initial_btc_per_agent = 1.0
        
        # Load October 10th 2025 data
        oct10_prices = self._load_oct10_data()
        config.btc_data = oct10_prices
        config.btc_initial_price = oct10_prices[0]
        config.btc_final_price = oct10_prices[-1]
        
        # Override get_btc_price_at_minute to use minute-level data directly
        def get_minute_price(minute: int) -> float:
            if minute >= len(oct10_prices):
                return oct10_prices[-1]
            return oct10_prices[minute]
        
        config.get_btc_price_at_minute = get_minute_price
        
        # Test HF (Aave only)
        config.aave_initial_hf = test_hf
        
        # NO REBALANCING - pure stress test
        config.leverage_frequency_minutes = 999999  # Effectively disabled
        
        # Disable unnecessary features for speed
        config.use_advanced_moet = False
        config.enable_weekly_yield_harvest = False
        config.ecosystem_growth_enabled = False
        config.use_historical_aave_rates = False  # Use default rates for single day
        config.use_historical_btc_data = False  # CRITICAL: Don't reload market data (we set it manually)
        
        # CRITICAL: Enable fail-fast mode and suppress verbose output
        config.fail_fast_on_liquidation = True
        config.suppress_progress_output = True
        config.generate_charts = False  # Don't generate charts for test runs
        config.save_detailed_csv = False  # Don't save CSVs for test runs
        
        # CRITICAL: Capture minute-by-minute health factor snapshots for chart
        config.agent_snapshot_frequency_minutes = 1  # Every minute for detailed tracking
        
        # Run simulation using standard architecture
        try:
            sim = FullYearSimulation(config)
            
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
            import gc
            gc.collect()
    
    def _save_test_iteration_data(self, test_hf: float, survived: bool, liquidation_minute: int, results: Dict[str, Any]):
        """Save detailed agent data for this test iteration"""
        try:
            import json
            
            # Create directory for test iteration data
            output_dir = Path(__file__).parent.parent / "tidal_protocol_sim" / "results" / "Study_14_Oct10_2025_Liquidation_Cascade_Test_Data"
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
                "agent_snapshots": agent_snapshots
            }
            
            # Save to file
            status = "survived" if survived else "liquidated"
            filename = f"test_hf_{test_hf:.4f}_{status}.json"
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(iteration_data, f, indent=2)
            
            print(f"  💾 Saved test data: {filepath}")
            
        except Exception as e:
            print(f"  ⚠️  Failed to save test iteration data: {e}")
    
    def _cleanup_temp_results(self, test_name: str):
        """Clean up temporary result files"""
        try:
            import shutil
            results_dir = Path(__file__).parent.parent / "tidal_protocol_sim" / "results"
            
            # Remove any directories starting with underscore (temp tests)
            for item in results_dir.glob("_temp_*"):
                if item.is_dir():
                    shutil.rmtree(item)
        except Exception as e:
            # Silently fail - cleanup is not critical
            pass
    
    def optimize(self, lower_bound: float = 1.05, upper_bound: float = 3.0, precision: float = 0.015) -> float:
        """
        Binary search to find minimum viable HF
        
        Args:
            lower_bound: Minimum HF to test (default 1.05)
            upper_bound: Maximum HF to test (default 3.0 for extreme volatility)
            precision: Stop when range is smaller than this (default 0.015)
            
        Returns:
            Minimum viable HF that survived
        """
        print("="*80)
        print("STARTING BINARY SEARCH OPTIMIZATION")
        print("="*80)
        print(f"Search range: [{lower_bound}, {upper_bound}]")
        print(f"Target precision: ±{precision}")
        print()
        
        iteration = 0
        
        while (upper_bound - lower_bound) > precision:
            iteration += 1
            test_hf = (lower_bound + upper_bound) / 2
            
            print(f"\nIteration {iteration}:")
            print(f"  Testing HF: {test_hf:.4f}")
            print(f"  Search range: [{lower_bound:.4f}, {upper_bound:.4f}]")
            
            survived, liquidation_minute = self._test_health_factor(test_hf)
            
            # Record result
            self.test_history.append({
                'iteration': iteration,
                'health_factor': test_hf,
                'survived': survived,
                'liquidation_minute': liquidation_minute,
                'search_range': [lower_bound, upper_bound]
            })
            
            if survived:
                print(f"  ✅ SURVIVED - Agent made it through the crash")
                upper_bound = test_hf  # Can try lower
            else:
                liq_min_str = f"minute {liquidation_minute}" if liquidation_minute else "unknown"
                print(f"  ❌ LIQUIDATED at {liq_min_str}")
                lower_bound = test_hf  # Need higher HF
        
        # Final minimum viable HF is the upper bound
        optimal_hf = upper_bound
        
        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        print(f"Minimum Viable HF: {optimal_hf:.4f}")
        print(f"Iterations: {iteration}")
        print(f"Final precision: ±{(upper_bound - lower_bound):.4f}")
        print("="*80)
        
        return optimal_hf

def run_comparison_with_optimal_hf(optimal_aave_hf: float, optimization_summary: Dict) -> Dict:
    """Run final comparison between Aave (optimal HF) and High Tide (1.2 HF)"""
    
    print("\n" + "="*80)
    print("RUNNING FINAL COMPARISON")
    print("="*80)
    print(f"Aave HF: {optimal_aave_hf:.2f} (Optimized Minimum)")
    print(f"High Tide HF: 1.10 (Initial) → 1.015 (Rebalancing) → 1.05 (Target)")
    print(f"Market: October 10th 2025 Liquidation Cascade")
    print("="*80)
    print()
    
    # Create configuration
    config = FullYearSimConfig()
    
    # Study 14 parameters
    config.test_name = f"Study_14_Oct10_2025_Liquidation_Cascade_Aave_{optimal_aave_hf:.2f}_vs_HT_1.10"
    config.simulation_duration_minutes = 1440  # Single day
    config.num_agents = 1  # Single agent per protocol
    config.initial_btc_per_agent = 1.0
    
    # Load October 10th 2025 data
    csv_path = Path(__file__).parent.parent / 'dune_query_6227486.csv'
    prices = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prices.append(float(row['btc_price_usd']))
    
    config.btc_data = prices
    config.btc_initial_price = prices[0]
    config.btc_final_price = prices[-1]
    
    # CRITICAL: Disable automatic market data loading to prevent config reset
    config.use_historical_btc_data = False
    config.use_historical_aave_rates = False  # Use default rates for single day test
    
    # Override get_btc_price_at_minute to use minute-level data directly
    def get_minute_price(minute: int) -> float:
        if minute >= len(prices):
            return prices[-1]
        return prices[minute]
    
    config.get_btc_price_at_minute = get_minute_price
    
    # Health factors (High Tide: aggressive 1.1 HF with tight rebalancing)
    config.agent_initial_hf = 1.1
    config.agent_rebalancing_hf = 1.015
    config.agent_target_hf = 1.05
    config.aave_initial_hf = optimal_aave_hf  # Use optimized minimum HF
    
    # DEBUG: Verify BTC price is correct
    print(f"🔧 DEBUG: config.btc_initial_price = ${config.btc_initial_price:,.2f}")
    print(f"🔧 DEBUG: Expected MOET debt at HF {config.agent_initial_hf}: ${(config.btc_initial_price * 0.85 / config.agent_initial_hf):,.2f}")
    
    # NO REBALANCING - stress test
    config.leverage_frequency_minutes = 999999  # Effectively disabled
    
    # Advanced MOET: OFF for symmetric study
    config.use_advanced_moet = False
    
    # No weekly yield harvesting for single day
    config.enable_weekly_yield_harvest = False
    
    # Comparison mode
    config.comparison_mode = True
    config.run_high_tide_only = False
    
    # Ecosystem growth: OFF for clean comparison
    config.ecosystem_growth_enabled = False
    
    # Use default rates for single day test
    config.use_historical_aave_rates = False
    
    # Normal output mode (not fail-fast)
    config.fail_fast_on_liquidation = False
    config.suppress_progress_output = False
    config.generate_charts = True
    config.save_detailed_csv = True
    
    # CRITICAL: Capture minute-by-minute health factor snapshots for chart
    config.agent_snapshot_frequency_minutes = 1  # Every minute for detailed LTV progression chart
    
    print("Starting detailed comparison simulation...")
    print("Expected runtime: ~2-3 minutes")
    print()
    
    # Run simulation
    sim = FullYearSimulation(config)
    results = sim.run_test()
    
    # Add optimization summary to results
    if "high_tide_results" in results:
        results["optimization_summary"] = optimization_summary
    
    return results

def main():
    print("="*80)
    print("STUDY 14: OCTOBER 10TH 2025 LIQUIDATION CASCADE")
    print("="*80)
    print()
    print("Market Event: Largest liquidation cascade in crypto history")
    print("Date: October 10th, 2025")
    print("BTC: $121,713 → $108,931 (-11.01% max drawdown)")
    print()
    print("Objective: Find minimum viable HF to survive extreme volatility")
    print("Method: Binary search optimization with fail-fast liquidation detection")
    print("Comparison: Optimized Aave HF vs High Tide 1.2 HF")
    print("="*80)
    print()
    
    # Run optimization
    optimizer = MinimumHFOptimizer()
    optimal_hf = optimizer.optimize(
        lower_bound=1.05,
        upper_bound=3.0,  # Higher upper bound for extreme event
        precision=0.015
    )
    
    # Create optimization summary
    optimization_summary = {
        "minimum_viable_hf": optimal_hf,
        "iterations": len(optimizer.test_history),
        "final_precision": 0.015,
        "test_history": optimizer.test_history,
        "search_bounds": {
            "initial_lower": 1.05,
            "initial_upper": 3.0,
            "final_lower": optimizer.test_history[-1]['search_range'][0],
            "final_upper": optimal_hf
        }
    }
    
    # Save optimization summary
    import json
    output_dir = Path(__file__).parent.parent / "tidal_protocol_sim" / "results" / f"Study_14_Oct10_2025_Liquidation_Cascade_Aave_{optimal_hf:.2f}_vs_HT_1.10"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_dir / "optimization_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(optimization_summary, f, indent=2)
    
    print(f"\n✅ Optimization summary saved to: {summary_path}")
    
    # Run final comparison
    results = run_comparison_with_optimal_hf(optimal_hf, optimization_summary)
    
    print("\n" + "="*80)
    print("STUDY 14 COMPLETE")
    print("="*80)
    print(f"Minimum viable Aave HF: {optimal_hf:.3f}")
    print(f"Optimization summary saved to: {summary_path}")
    print(f"Results saved to: {output_dir}/")
    print()
    
    print("\n" + "="*80)
    print("STUDY 14 COMPLETE - KEY FINDINGS")
    print("="*80)
    print()
    print(f"📊 Minimum Viable Aave HF: {optimal_hf:.3f}")
    print(f"   → Required {(optimal_hf - 1.0) * 100:.1f}% safety margin above liquidation")
    print()
    print(f"🔍 Binary Search Statistics:")
    print(f"   → Iterations: {len(optimizer.test_history)}")
    print(f"   → Precision: ±{0.015:.4f}")
    print()
    print(f"📁 Full results saved to:")
    print(f"   {output_dir}/")
    print()
    print("="*80)

if __name__ == '__main__':
    main()

