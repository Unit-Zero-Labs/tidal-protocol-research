#!/usr/bin/env python3
"""
Yield Token Pool Capacity Analysis
Testing Swap Capacity vs Pool Size Without Agents

This simulation directly tests the MOET:YT Uniswap V3 pool to determine:
1. Maximum trade size that keeps price within 1% of peg for different pool sizes
2. Pool capacity scaling relationships 
3. Optimal pool sizing for protocol operations

Based on the architecture from moet_yt_liquidity_stress_test.py but focused
purely on pool mechanics without agent complexity.
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
import math

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tidal_protocol_sim.core.uniswap_v3_math import create_yield_token_pool, UniswapV3SlippageCalculator
from tidal_protocol_sim.core.yield_tokens import YieldTokenPool


class YieldTokenPoolCapacityTester:
    """
    Direct testing of Yield Token Pool capacity using Uniswap V3 math
    
    Tests pool behavior under different swap sizes and pool configurations
    without the complexity of agent interactions.
    """
    
    def __init__(self):
        self.results = {
            "analysis_metadata": {
                "analysis_type": "Yield_Token_Pool_Capacity_Analysis", 
                "timestamp": datetime.now().isoformat(),
                "description": "Direct testing of MOET:YT pool swap capacity vs pool size"
            },
            "pool_capacity_tests": [],
            "price_impact_analysis": {},
            "scaling_analysis": {}
        }
        
        # Test configurations (MOET reserve amounts - each side of pool)
        # Using 500k gives us the desired $250k:$250k pool with correct 47M liquidity
        self.pool_sizes_usd = [
            250_000,    # Target config: $250k:$250k = $500k total
            500_000,    # 2x target: $500k:$500k = $1M total (current actual size)
            1_000_000,  # 4x target: $1M:$1M = $2M total
            2_500_000,  # 10x target: $2.5M:$2.5M = $5M total
            5_000_000,  # 20x target: $5M:$5M = $10M total
            10_000_000, # 40x target: $10M:$10M = $20M total
            25_000_000  # 100x target: $25M:$25M = $50M total
        ]
        
        self.test_swap_sizes = [
            70_000,  # Incremental test 2
            80_000,  # Incremental test 3
            90_000,  # Incremental test 4
            100_000, # Large capacity test
            125_000, # Extended capacity test 1
            150_000, # Extended capacity test 2
            175_000, # Extended capacity test 3
            200_000, # Maximum capacity test
            225_000, # Stress test 1
            250_000, # Stress test 2 (pool size match)
            275_000, # Stress test 3
            300_000, # Ultimate stress test
            350_000, # Stress test 4
            400_000, # Stress test 5
            500_000, # Ultimate stress test
            600_000, # Ultimate stress test 2
            700_000, # Ultimate stress test 3
            800_000, # Ultimate stress test 4
            900_000, # Ultimate stress test 5
            1_000_000, # Ultimate stress test 6
            1_500_000, # Ultimate stress test 7
            2_000_000, # Ultimate stress test 8
        ]
        
        self.concentration = 0.95  # 95% concentration at peg
        self.price_impact_threshold = 0.01  # 1% price impact threshold
        
        # Pool rebalancing/arbitrage configuration
        self.enable_pool_arbing = False  # Default to False for backward compatibility
        self.alm_rebalance_interval_minutes = 720  # 12 hours for ALM rebalancer
        self.algo_deviation_threshold_bps = 50.0  # 50 basis points for Algo rebalancer
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive pool capacity analysis"""
        print("🔍 YIELD TOKEN POOL CAPACITY ANALYSIS")
        print("=" * 60)
        print(f"📊 Testing {len(self.pool_sizes_usd)} pool sizes")
        print(f"📊 Testing {len(self.test_swap_sizes)} swap sizes (including 10k increments from $50k-$100k)")
        print(f"📊 Price impact threshold: {self.price_impact_threshold:.1%}")
        print()
        
        # Test 1: Target pool analysis ($250k:$250k = $500k total)
        print("🔍 Test 1: Target Pool Analysis ($250k:$250k)")
        current_pool_results = self._test_single_pool_capacity(250_000)
        self.results["current_pool_analysis"] = current_pool_results
        
        # Test 2: Pool scaling analysis
        print("\n🔍 Test 2: Pool Scaling Analysis")
        scaling_results = self._test_pool_scaling()
        self.results["scaling_analysis"] = scaling_results
        
        # Test 3: Price impact threshold analysis
        print("\n🔍 Test 3: Price Impact Threshold Analysis")
        threshold_results = self._analyze_price_impact_thresholds()
        self.results["price_impact_analysis"] = threshold_results
        
        # Test 4: Repeated small swap analysis
        print("\n🔍 Test 4: Repeated Small Swap Analysis ($2,000 swaps)")
        repeated_swap_results = self._analyze_repeated_small_swaps()
        self.results["repeated_swap_analysis"] = repeated_swap_results
        
        # Generate summary and recommendations
        print("\n📊 Generating Analysis Summary...")
        self._generate_analysis_summary()
        
        # Save results
        self._save_results()
        
        # Generate charts
        print("📊 Generating Charts...")
        self._generate_charts()
        
        print("\n✅ Pool capacity analysis completed!")
        return self.results
    
    def _test_single_pool_capacity(self, pool_size_usd: float) -> Dict[str, Any]:
        """Test capacity of a single pool size across different swap amounts"""
        print(f"   Testing pool size: ${pool_size_usd:,.0f}:${pool_size_usd:,.0f} (${pool_size_usd*2:,.0f} total)")
        
        # Create fresh pool for testing
        # pool_size_usd represents the MOET reserve size for backwards compatibility
        # Convert to new format: total pool size and token ratio
        total_pool_size = pool_size_usd * 2  # Convert from single-side to total pool size
        token0_ratio = 0.75  # Use 75% MOET, 25% YT ratio
        
        pool = YieldTokenPool(
            total_pool_size=total_pool_size,
            token0_ratio=token0_ratio,
            concentration=self.concentration
        )
        
        test_results = {
            "pool_size_usd": pool_size_usd,
            "concentration": self.concentration,
            "swap_tests": [],
            "max_safe_trade": 0.0,
            "pool_breaking_point": None
        }
        
        max_safe_trade = 0.0
        pool_breaking_point = None
        
        for swap_size in self.test_swap_sizes:
            # Test swap impact without executing (simulation only)
            swap_result = self._simulate_swap_impact(pool, swap_size, "Yield_Token")
            
            # Record results
            swap_test = {
                "swap_size_usd": swap_size,
                "price_before": swap_result["price_before"],
                "price_after": swap_result["price_after"], 
                "price_impact_percent": swap_result["price_impact_percent"],
                "moet_received": swap_result["moet_received"],
                "slippage_percent": swap_result["slippage_percent"],
                "active_liquidity_before": swap_result["active_liquidity_before"],
                "active_liquidity_after": swap_result["active_liquidity_after"],
                "liquidity_utilization": swap_result["liquidity_utilization"],
                "within_threshold": swap_result["price_impact_percent"] <= self.price_impact_threshold * 100,
                "swap_successful": swap_result["swap_successful"]
            }
            
            test_results["swap_tests"].append(swap_test)
            
            # Track maximum safe trade size
            if swap_test["within_threshold"] and swap_test["swap_successful"]:
                max_safe_trade = max(max_safe_trade, swap_size)
            
            # Track pool breaking point
            if not swap_test["swap_successful"] and pool_breaking_point is None:
                pool_breaking_point = swap_size
            
            print(f"      ${swap_size:>6,.0f} → {swap_result['price_impact_percent']:>5.1f}% impact, "
                  f"{swap_result['slippage_percent']:>5.1f}% slippage, "
                  f"{'✅' if swap_test['within_threshold'] else '❌'}")
        
        test_results["max_safe_trade"] = max_safe_trade
        test_results["pool_breaking_point"] = pool_breaking_point
        
        print(f"   📊 Max safe trade: ${max_safe_trade:,.0f}")
        if pool_breaking_point:
            print(f"   🚨 Pool breaks at: ${pool_breaking_point:,.0f}")
        
        return test_results
    
    def _test_pool_scaling(self) -> Dict[str, Any]:
        """Test how pool capacity scales with pool size"""
        scaling_results = {
            "pool_tests": [],
            "scaling_metrics": {}
        }
        
        for pool_size in self.pool_sizes_usd:
            pool_result = self._test_single_pool_capacity(pool_size)
            scaling_results["pool_tests"].append(pool_result)
        
        # Calculate scaling metrics
        pool_sizes = [test["pool_size_usd"] for test in scaling_results["pool_tests"]]
        max_trades = [test["max_safe_trade"] for test in scaling_results["pool_tests"]]
        
        # Calculate scaling efficiency (max_trade / pool_size)
        scaling_efficiencies = [
            (max_trade / pool_size) * 100 if pool_size > 0 else 0
            for max_trade, pool_size in zip(max_trades, pool_sizes)
        ]
        
        scaling_results["scaling_metrics"] = {
            "pool_sizes": pool_sizes,
            "max_safe_trades": max_trades,
            "scaling_efficiencies": scaling_efficiencies,
            "average_efficiency": np.mean(scaling_efficiencies),
            "efficiency_std": np.std(scaling_efficiencies)
        }
        
        return scaling_results
    
    def _analyze_price_impact_thresholds(self) -> Dict[str, Any]:
        """Analyze price impact at different thresholds"""
        thresholds = [0.005, 0.01, 0.02, 0.05]  # 0.5%, 1%, 2%, 5%
        
        threshold_analysis = {
            "thresholds_tested": thresholds,
            "threshold_results": []
        }
        
        # Test each threshold against the target $250k pool
        pool_size = 250_000
        total_pool_size = pool_size * 2  # Convert to total pool size
        token0_ratio = 0.75  # Use 75% MOET, 25% YT ratio
        
        pool = YieldTokenPool(
            total_pool_size=total_pool_size,
            token0_ratio=token0_ratio,
            concentration=self.concentration
        )
        
        for threshold in thresholds:
            print(f"   Testing {threshold:.1%} price impact threshold...")
            
            max_trade_for_threshold = 0.0
            
            # Binary search for maximum trade size at this threshold
            low, high = 0, 100_000
            while high - low > 100:  # $100 precision
                mid = (low + high) // 2
                
                swap_result = self._simulate_swap_impact(pool, mid, "Yield_Token")
                price_impact = swap_result["price_impact_percent"] / 100
                
                if price_impact <= threshold and swap_result["swap_successful"]:
                    max_trade_for_threshold = mid
                    low = mid
                else:
                    high = mid
            
            threshold_results = {
                "threshold_percent": threshold * 100,
                "max_trade_size": max_trade_for_threshold,
                "pool_utilization": (max_trade_for_threshold / pool_size) * 100
            }
            
            threshold_analysis["threshold_results"].append(threshold_results)
            
            print(f"      Max trade: ${max_trade_for_threshold:,.0f} ({max_trade_for_threshold/pool_size:.1%} of pool)")
        
        return threshold_analysis
    
    def _analyze_repeated_small_swaps(self) -> Dict[str, Any]:
        """Analyze how many repeated $2,000 swaps it takes to reach the 1% price impact threshold"""
        swap_size = 2_000  # $2k per swap
        pool_size = 250_000  # Target pool size
        
        print(f"   Testing repeated ${swap_size:,} swaps on ${pool_size:,} pool...")
        
        # Create fresh pool for testing
        total_pool_size = pool_size * 2  # Convert to total pool size
        token0_ratio = 0.75  # Use 75% MOET, 25% YT ratio
        
        pool = YieldTokenPool(
            total_pool_size=total_pool_size,
            token0_ratio=token0_ratio,
            concentration=self.concentration
        )
        
        repeated_swap_results = {
            "swap_size": swap_size,
            "pool_size": pool_size,
            "concentration": self.concentration,
            "swap_history": [],
            "total_swaps_executed": 0,
            "cumulative_volume": 0.0,
            "final_price_impact": 0.0,
            "threshold_reached": False,
            "threshold_swap_number": None
        }
        
        swap_count = 0
        cumulative_volume = 0.0
        initial_price = pool.uniswap_pool.get_price()
        
        # Keep swapping until we hit the 1% threshold or max iterations
        max_swaps = 100  # Safety limit
        
        while swap_count < max_swaps:
            swap_count += 1
            
            try:
                # Execute the swap (this permanently modifies pool state)
                moet_received = pool.execute_yield_token_sale(swap_size)
                swap_successful = moet_received > 0
                
                if not swap_successful:
                    print(f"      Swap #{swap_count}: FAILED - Pool exhausted")
                    break
                
                # Get post-swap metrics
                current_price = pool.uniswap_pool.get_price()
                price_impact_percent = abs((current_price - initial_price) / initial_price) * 100 if initial_price > 0 else 0
                active_liquidity = pool.uniswap_pool._calculate_active_liquidity_from_ticks(
                    pool.uniswap_pool.tick_current
                ) / 1e6
                
                cumulative_volume += swap_size
                
                # Calculate slippage for this individual swap
                expected_moet = swap_size  # 1:1 expected
                slippage_percent = abs((expected_moet - moet_received) / expected_moet) * 100 if expected_moet > 0 else 0
                
                # Record swap details
                swap_record = {
                    "swap_number": swap_count,
                    "cumulative_volume": cumulative_volume,
                    "price_before_swap": pool.uniswap_pool.get_price() if swap_count == 1 else repeated_swap_results["swap_history"][-1]["price_after_swap"],
                    "price_after_swap": current_price,
                    "price_impact_cumulative": price_impact_percent,
                    "moet_received": moet_received,
                    "slippage_percent": slippage_percent,
                    "active_liquidity": active_liquidity,
                    "swap_successful": swap_successful
                }
                
                repeated_swap_results["swap_history"].append(swap_record)
                
                # Check if we've reached the threshold
                if price_impact_percent >= self.price_impact_threshold * 100 and not repeated_swap_results["threshold_reached"]:
                    repeated_swap_results["threshold_reached"] = True
                    repeated_swap_results["threshold_swap_number"] = swap_count
                    print(f"      🎯 Threshold reached at swap #{swap_count}: {price_impact_percent:.2f}% cumulative impact")
                
                # Print progress every 5 swaps or at threshold
                if swap_count % 5 == 0 or repeated_swap_results["threshold_reached"]:
                    print(f"      Swap #{swap_count:2d}: ${cumulative_volume:6,.0f} total → {price_impact_percent:5.2f}% impact, {slippage_percent:5.2f}% slippage")
                
                # Stop if we've reached threshold
                if repeated_swap_results["threshold_reached"]:
                    break
                    
            except Exception as e:
                print(f"      Swap #{swap_count}: ERROR - {str(e)}")
                break
        
        # Final results
        repeated_swap_results["total_swaps_executed"] = swap_count
        repeated_swap_results["cumulative_volume"] = cumulative_volume
        repeated_swap_results["final_price_impact"] = price_impact_percent if 'price_impact_percent' in locals() else 0.0
        
        # Summary
        if repeated_swap_results["threshold_reached"]:
            threshold_volume = repeated_swap_results["threshold_swap_number"] * swap_size
            print(f"   📊 Summary: {repeated_swap_results['threshold_swap_number']} swaps (${threshold_volume:,}) to reach 1% threshold")
        else:
            print(f"   📊 Summary: {swap_count} swaps executed (${cumulative_volume:,} total), final impact: {repeated_swap_results['final_price_impact']:.2f}%")
        
        return repeated_swap_results
    
    def _simulate_swap_impact(self, pool: YieldTokenPool, swap_amount_usd: float, token_in: str) -> Dict[str, Any]:
        """
        Simulate swap impact without permanently modifying pool state
        
        This creates a temporary copy of the pool to test swap impact,
        then restores the original state.
        """
        
        # Store original pool state
        original_pool_state = self._capture_pool_state(pool)
        
        try:
            # Get initial state
            initial_price = pool.uniswap_pool.get_price()
            initial_active_liquidity = pool.uniswap_pool._calculate_active_liquidity_from_ticks(
                pool.uniswap_pool.tick_current
            ) / 1e6
            
            # Execute the swap to measure impact
            if token_in == "Yield_Token":
                moet_received = pool.execute_yield_token_sale(swap_amount_usd)
                swap_successful = moet_received > 0
            else:  # "MOET"
                yt_received = pool.execute_yield_token_purchase(swap_amount_usd) 
                swap_successful = yt_received > 0
                moet_received = yt_received  # For consistency in return format
            
            # Get post-swap state
            final_price = pool.uniswap_pool.get_price()
            final_active_liquidity = pool.uniswap_pool._calculate_active_liquidity_from_ticks(
                pool.uniswap_pool.tick_current
            ) / 1e6
            
            # Calculate metrics
            price_impact_percent = abs((final_price - initial_price) / initial_price) * 100 if initial_price > 0 else 0
            expected_moet = swap_amount_usd  # 1:1 expected without slippage
            slippage_percent = abs((expected_moet - moet_received) / expected_moet) * 100 if expected_moet > 0 else 0
            liquidity_utilization = ((initial_active_liquidity - final_active_liquidity) / initial_active_liquidity) * 100 if initial_active_liquidity > 0 else 0
            
            result = {
                "price_before": initial_price,
                "price_after": final_price,
                "price_impact_percent": price_impact_percent,
                "moet_received": moet_received,
                "slippage_percent": slippage_percent,
                "active_liquidity_before": initial_active_liquidity,
                "active_liquidity_after": final_active_liquidity,
                "liquidity_utilization": liquidity_utilization,
                "swap_successful": swap_successful
            }
            
        except Exception as e:
            # Handle swap failures
            result = {
                "price_before": pool.uniswap_pool.get_price(),
                "price_after": pool.uniswap_pool.get_price(),
                "price_impact_percent": 0.0,
                "moet_received": 0.0,
                "slippage_percent": 100.0,
                "active_liquidity_before": 0.0,
                "active_liquidity_after": 0.0,
                "liquidity_utilization": 0.0,
                "swap_successful": False,
                "error": str(e)
            }
        
        finally:
            # Restore original pool state
            self._restore_pool_state(pool, original_pool_state)
        
        return result
    
    def _capture_pool_state(self, pool: YieldTokenPool) -> Dict[str, Any]:
        """Capture current pool state for restoration"""
        uniswap_pool = pool.uniswap_pool
        
        return {
            "sqrt_price_x96": uniswap_pool.sqrt_price_x96,
            "tick_current": uniswap_pool.tick_current,
            "liquidity": uniswap_pool.liquidity,
            "token0_reserve": uniswap_pool.token0_reserve,
            "token1_reserve": uniswap_pool.token1_reserve,
            "moet_reserve": pool.moet_reserve,
            "yield_token_reserve": pool.yield_token_reserve
        }
    
    def _restore_pool_state(self, pool: YieldTokenPool, state: Dict[str, Any]):
        """Restore pool to captured state"""
        uniswap_pool = pool.uniswap_pool
        
        uniswap_pool.sqrt_price_x96 = state["sqrt_price_x96"]
        uniswap_pool.tick_current = state["tick_current"] 
        uniswap_pool.liquidity = state["liquidity"]
        uniswap_pool.token0_reserve = state["token0_reserve"]
        uniswap_pool.token1_reserve = state["token1_reserve"]
        
        pool.moet_reserve = state["moet_reserve"]
        pool.yield_token_reserve = state["yield_token_reserve"]
    
    def _generate_analysis_summary(self):
        """Generate comprehensive analysis summary"""
        current_analysis = self.results["current_pool_analysis"]
        scaling_analysis = self.results["scaling_analysis"]
        
        summary = {
            "current_pool_capacity": {
                "pool_size": current_analysis["pool_size_usd"],
                "max_safe_trade": current_analysis["max_safe_trade"],
                "pool_utilization": (current_analysis["max_safe_trade"] / current_analysis["pool_size_usd"]) * 100,
                "breaking_point": current_analysis["pool_breaking_point"]
            },
            "scaling_insights": {
                "efficiency_range": f"{min(scaling_analysis['scaling_metrics']['scaling_efficiencies']):.2f}% - {max(scaling_analysis['scaling_metrics']['scaling_efficiencies']):.2f}%",
                "optimal_pool_sizes": [],
                "recommendations": []
            }
        }
        
        # Find optimal pool sizes (highest efficiency)
        scaling_metrics = scaling_analysis["scaling_metrics"]
        max_efficiency_idx = np.argmax(scaling_metrics["scaling_efficiencies"])
        optimal_pool_size = scaling_metrics["pool_sizes"][max_efficiency_idx]
        optimal_efficiency = scaling_metrics["scaling_efficiencies"][max_efficiency_idx]
        
        summary["scaling_insights"]["optimal_pool_sizes"].append({
            "pool_size": optimal_pool_size,
            "efficiency": optimal_efficiency,
            "max_trade": scaling_metrics["max_safe_trades"][max_efficiency_idx]
        })
        
        # Generate recommendations
        current_max_trade = current_analysis["max_safe_trade"]
        
        if current_max_trade < 2500:  # Less than typical rebalancing
            summary["scaling_insights"]["recommendations"].append(
                f"🚨 CRITICAL: Current pool can only handle ${current_max_trade:,.0f} trades. Increase to at least ${optimal_pool_size:,.0f} for safe operations."
            )
        
        if current_analysis["pool_breaking_point"] and current_analysis["pool_breaking_point"] < 10000:
            summary["scaling_insights"]["recommendations"].append(
                f"⚠️ WARNING: Pool breaks at ${current_analysis['pool_breaking_point']:,.0f}. Consider 10x increase for resilience."
            )
        
        # Add repeated swap analysis to summary
        if "repeated_swap_analysis" in self.results:
            repeated_analysis = self.results["repeated_swap_analysis"]
            summary["repeated_swap_insights"] = {
                "swap_size": repeated_analysis["swap_size"],
                "threshold_reached": repeated_analysis["threshold_reached"],
                "swaps_to_threshold": repeated_analysis.get("threshold_swap_number"),
                "volume_to_threshold": repeated_analysis.get("threshold_swap_number", 0) * repeated_analysis["swap_size"] if repeated_analysis.get("threshold_swap_number") else None,
                "total_swaps_executed": repeated_analysis["total_swaps_executed"],
                "final_cumulative_volume": repeated_analysis["cumulative_volume"],
                "final_price_impact": repeated_analysis["final_price_impact"]
            }
        
        self.results["analysis_summary"] = summary
    
    def _save_results(self):
        """Save results to JSON file"""
        results_dir = Path("tidal_protocol_sim/results") / "Yield_Token_Pool_Capacity_Analysis"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"pool_capacity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types to Python types for JSON serialization
        json_safe_results = self._convert_for_json(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(json_safe_results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_file}")
    
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
    
    def _generate_charts(self):
        """Generate visualization charts"""
        charts_dir = Path("tidal_protocol_sim/results") / "Yield_Token_Pool_Capacity_Analysis" / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Chart 1: Pool Scaling Analysis
        self._create_pool_scaling_chart(charts_dir)
        
        # Chart 2: Price Impact Analysis
        self._create_price_impact_chart(charts_dir)
        
        # Chart 3: Current Pool Capacity Detail
        self._create_current_pool_detail_chart(charts_dir)
        
        # Chart 4: All Pools Detailed Analysis
        self._create_all_pools_detail_charts(charts_dir)
        
        # Chart 4: Repeated Swap Analysis
        if "repeated_swap_analysis" in self.results:
            self._create_repeated_swap_chart(charts_dir)
        
        print(f"📊 Charts saved to: {charts_dir}")
    
    def _create_pool_scaling_chart(self, charts_dir: Path):
        """Create pool scaling analysis chart"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        scaling_data = self.results["scaling_analysis"]["scaling_metrics"]
        pool_sizes = np.array(scaling_data["pool_sizes"]) / 1e6  # Convert to millions
        max_trades = np.array(scaling_data["max_safe_trades"]) / 1e3  # Convert to thousands
        
        # Top chart: Max Trade Size vs Pool Size
        ax1.plot(pool_sizes, max_trades, 'o-', linewidth=3, markersize=8, color='#2E86AB')
        ax1.set_xlabel('Pool Size ($ Millions)', fontsize=12)
        ax1.set_ylabel('Max Safe Trade Size ($ Thousands)', fontsize=12)
        ax1.set_title('Pool Capacity Scaling: Max Safe Trade Size vs Pool Size\n(1% Price Impact Threshold)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add annotations for key points
        current_idx = 0  # $250k pool
        ax1.annotate(f'Current Pool\n${pool_sizes[current_idx]:.1f}M → ${max_trades[current_idx]:.1f}k', 
                    xy=(pool_sizes[current_idx], max_trades[current_idx]),
                    xytext=(pool_sizes[current_idx] + 2, max_trades[current_idx] + 5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, color='red', fontweight='bold')
        
        # Bottom chart: Scaling Efficiency
        efficiencies = scaling_data["scaling_efficiencies"]
        ax2.bar(pool_sizes, efficiencies, color='#A23B72', alpha=0.7, width=0.8)
        ax2.set_xlabel('Pool Size ($ Millions)', fontsize=12)
        ax2.set_ylabel('Pool Utilization Efficiency (%)', fontsize=12)
        ax2.set_title('Pool Utilization Efficiency: Max Trade / Pool Size', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add efficiency line
        avg_efficiency = scaling_data["average_efficiency"]
        ax2.axhline(y=avg_efficiency, color='orange', linestyle='--', linewidth=2, 
                   label=f'Average: {avg_efficiency:.2f}%')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(charts_dir / "pool_scaling_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_price_impact_chart(self, charts_dir: Path):
        """Create price impact analysis chart"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get current pool data
        current_pool = self.results["current_pool_analysis"]
        swap_tests = current_pool["swap_tests"]
        
        swap_sizes = [test["swap_size_usd"] / 1e3 for test in swap_tests]  # Convert to thousands
        price_impacts = [test["price_impact_percent"] for test in swap_tests]
        slippage_percents = [test["slippage_percent"] for test in swap_tests]
        
        # Create dual-axis plot
        ax1 = ax
        ax2 = ax1.twinx()
        
        # Price impact line
        line1 = ax1.plot(swap_sizes, price_impacts, 'o-', linewidth=3, markersize=8, 
                        color='#E63946', label='Price Impact (%)')
        ax1.set_xlabel('Swap Size ($ Thousands)', fontsize=12)
        ax1.set_ylabel('Price Impact (%)', color='#E63946', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='#E63946')
        
        # Slippage line
        line2 = ax2.plot(swap_sizes, slippage_percents, 's-', linewidth=3, markersize=8, 
                        color='#457B9D', label='Slippage (%)')
        ax2.set_ylabel('Slippage (%)', color='#457B9D', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='#457B9D')
        
        # Add threshold line
        ax1.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, alpha=0.8, 
                   label='1% Price Impact Threshold')
        
        # Title and grid
        ax1.set_title(f'Price Impact & Slippage Analysis\nCurrent Pool: ${current_pool["pool_size_usd"]:,.0f} (95% Concentration)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(charts_dir / "price_impact_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_current_pool_detail_chart(self, charts_dir: Path):
        """Create detailed analysis of current pool"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        current_pool = self.results["current_pool_analysis"]
        swap_tests = current_pool["swap_tests"]
        
        swap_sizes = [test["swap_size_usd"] for test in swap_tests]
        price_impacts = [test["price_impact_percent"] for test in swap_tests]
        slippage_percents = [test["slippage_percent"] for test in swap_tests]
        liquidity_utils = [test["liquidity_utilization"] for test in swap_tests]
        moet_received = [test["moet_received"] for test in swap_tests]
        
        # Chart 1: Price Impact
        colors1 = ['green' if impact <= 1.0 else 'red' for impact in price_impacts]
        ax1.bar(range(len(swap_sizes)), price_impacts, color=colors1, alpha=0.7)
        ax1.set_xlabel('Swap Size')
        ax1.set_ylabel('Price Impact (%)')
        ax1.set_title('Price Impact by Swap Size')
        ax1.axhline(y=1.0, color='orange', linestyle='--', label='1% Threshold')
        ax1.set_xticks(range(len(swap_sizes)))
        ax1.set_xticklabels([f'${s/1e3:.0f}k' for s in swap_sizes], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Slippage
        ax2.plot(range(len(swap_sizes)), slippage_percents, 'o-', color='#457B9D', linewidth=2, markersize=6)
        ax2.set_xlabel('Swap Size') 
        ax2.set_ylabel('Slippage (%)')
        ax2.set_title('Slippage by Swap Size')
        ax2.set_xticks(range(len(swap_sizes)))
        ax2.set_xticklabels([f'${s/1e3:.0f}k' for s in swap_sizes], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Liquidity Utilization
        ax3.bar(range(len(swap_sizes)), liquidity_utils, color='#F1C40F', alpha=0.7)
        ax3.set_xlabel('Swap Size')
        ax3.set_ylabel('Liquidity Utilization (%)')
        ax3.set_title('Liquidity Pool Utilization')
        ax3.set_xticks(range(len(swap_sizes)))
        ax3.set_xticklabels([f'${s/1e3:.0f}k' for s in swap_sizes], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: MOET Received vs Expected
        expected_moet = swap_sizes  # 1:1 expected
        ax4.plot(range(len(swap_sizes)), expected_moet, '--', color='gray', linewidth=2, label='Expected (1:1)')
        ax4.plot(range(len(swap_sizes)), moet_received, 'o-', color='#2E86AB', linewidth=2, markersize=6, label='Actual Received')
        ax4.set_xlabel('Swap Size')
        ax4.set_ylabel('MOET Amount ($)')
        ax4.set_title('MOET Received: Expected vs Actual')
        ax4.set_xticks(range(len(swap_sizes)))
        ax4.set_xticklabels([f'${s/1e3:.0f}k' for s in swap_sizes], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Current Pool Detailed Analysis: ${current_pool["pool_size_usd"]:,.0f}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(charts_dir / "current_pool_detailed_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_all_pools_detail_charts(self, charts_dir: Path):
        """Create detailed analysis charts for all pools being tested"""
        scaling_data = self.results["scaling_analysis"]["pool_tests"]
        
        # Create a figure with subplots for each pool
        num_pools = len(scaling_data)
        cols = 3  # 3 columns
        rows = (num_pools + cols - 1) // cols  # Calculate rows needed
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 6 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, pool_data in enumerate(scaling_data):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            # Get data for this pool
            swap_tests = pool_data["swap_tests"]
            pool_size = pool_data["pool_size_usd"]
            
            swap_sizes = [test["swap_size_usd"] for test in swap_tests]
            price_impacts = [test["price_impact_percent"] for test in swap_tests]
            slippage_percents = [test["slippage_percent"] for test in swap_tests]
            liquidity_utils = [test["liquidity_utilization"] for test in swap_tests]
            moet_received = [test["moet_received"] for test in swap_tests]
            
            # Create 2x2 subplot within each pool
            gs = ax.get_gridspec()
            ax.remove()
            sub_ax = fig.add_subplot(gs[row, col])
            
            # Create 2x2 subplots for this pool
            sub_fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            
            # Chart 1: Price Impact
            colors1 = ['green' if impact <= 1.0 else 'red' for impact in price_impacts]
            ax1.bar(range(len(swap_sizes)), price_impacts, color=colors1, alpha=0.7)
            ax1.set_xlabel('Swap Size')
            ax1.set_ylabel('Price Impact (%)')
            ax1.set_title(f'Price Impact by Swap Size\nPool: ${pool_size:,.0f}')
            ax1.axhline(y=1.0, color='orange', linestyle='--', label='1% Threshold')
            ax1.set_xticks(range(0, len(swap_sizes), max(1, len(swap_sizes)//8)))
            ax1.set_xticklabels([f'${s/1e3:.0f}k' for s in swap_sizes[::max(1, len(swap_sizes)//8)]], rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Chart 2: Slippage
            ax2.plot(range(len(swap_sizes)), slippage_percents, 'o-', color='#457B9D', linewidth=2, markersize=4)
            ax2.set_xlabel('Swap Size') 
            ax2.set_ylabel('Slippage (%)')
            ax2.set_title('Slippage by Swap Size')
            ax2.set_xticks(range(0, len(swap_sizes), max(1, len(swap_sizes)//8)))
            ax2.set_xticklabels([f'${s/1e3:.0f}k' for s in swap_sizes[::max(1, len(swap_sizes)//8)]], rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Chart 3: Liquidity Utilization
            ax3.bar(range(len(swap_sizes)), liquidity_utils, color='#F1C40F', alpha=0.7)
            ax3.set_xlabel('Swap Size')
            ax3.set_ylabel('Liquidity Utilization (%)')
            ax3.set_title('Liquidity Pool Utilization')
            ax3.set_xticks(range(0, len(swap_sizes), max(1, len(swap_sizes)//8)))
            ax3.set_xticklabels([f'${s/1e3:.0f}k' for s in swap_sizes[::max(1, len(swap_sizes)//8)]], rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Chart 4: MOET Received vs Expected
            expected_moet = swap_sizes  # 1:1 expected
            ax4.plot(range(len(swap_sizes)), expected_moet, '--', color='gray', linewidth=2, label='Expected (1:1)')
            ax4.plot(range(len(swap_sizes)), moet_received, 'o-', color='#2E86AB', linewidth=2, markersize=4, label='Actual Received')
            ax4.set_xlabel('Swap Size')
            ax4.set_ylabel('MOET Amount ($)')
            ax4.set_title('MOET Received: Expected vs Actual')
            ax4.set_xticks(range(0, len(swap_sizes), max(1, len(swap_sizes)//8)))
            ax4.set_xticklabels([f'${s/1e3:.0f}k' for s in swap_sizes[::max(1, len(swap_sizes)//8)]], rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Add pool summary info
            max_safe = pool_data["max_safe_trade"]
            breaking_point = pool_data.get("pool_breaking_point")
            efficiency = (max_safe / pool_size) * 100 if pool_size > 0 else 0
            
            # Format breaking point safely
            breaking_str = f"${breaking_point:,.0f}" if breaking_point is not None else "N/A"
            
            sub_fig.suptitle(f'Pool Analysis: ${pool_size:,.0f}\n'
                           f'Max Safe: ${max_safe:,.0f} | Breaking: {breaking_str} | Efficiency: {efficiency:.1f}%', 
                           fontsize=14, fontweight='bold')
            sub_fig.tight_layout()
            
            # Save individual pool chart
            pool_filename = f"pool_{pool_size//1000}k_detailed_analysis.png"
            sub_fig.savefig(charts_dir / pool_filename, dpi=300, bbox_inches='tight')
            plt.close(sub_fig)
        
        # Close the main figure
        plt.close(fig)
        
        print(f"📊 Individual pool charts saved to: {charts_dir}")
        print(f"   Generated {num_pools} detailed pool analysis charts")
    
    def _create_repeated_swap_chart(self, charts_dir: Path):
        """Create repeated swap analysis chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        repeated_data = self.results["repeated_swap_analysis"]
        swap_history = repeated_data["swap_history"]
        
        if not swap_history:
            # Create placeholder if no data
            ax1.text(0.5, 0.5, 'No swap data available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Repeated Swap Analysis - No Data')
            plt.tight_layout()
            plt.savefig(charts_dir / "repeated_swap_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        swap_numbers = [swap["swap_number"] for swap in swap_history]
        cumulative_volumes = [swap["cumulative_volume"] for swap in swap_history]
        price_impacts = [swap["price_impact_cumulative"] for swap in swap_history]
        slippage_percents = [swap["slippage_percent"] for swap in swap_history]
        active_liquidities = [swap["active_liquidity"] for swap in swap_history]
        
        # Chart 1: Cumulative Price Impact
        ax1.plot(swap_numbers, price_impacts, 'o-', color='#E63946', linewidth=2, markersize=4)
        ax1.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, alpha=0.8, label='1% Threshold')
        ax1.set_xlabel('Swap Number')
        ax1.set_ylabel('Cumulative Price Impact (%)')
        ax1.set_title(f'Cumulative Price Impact: ${repeated_data["swap_size"]:,} Swaps')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Mark threshold point if reached
        if repeated_data["threshold_reached"]:
            threshold_swap = repeated_data["threshold_swap_number"]
            threshold_impact = next(swap["price_impact_cumulative"] for swap in swap_history if swap["swap_number"] == threshold_swap)
            ax1.scatter([threshold_swap], [threshold_impact], color='red', s=100, zorder=5)
            ax1.annotate(f'Threshold\nSwap #{threshold_swap}', 
                        xy=(threshold_swap, threshold_impact),
                        xytext=(threshold_swap + 2, threshold_impact + 0.2),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=9, color='red', fontweight='bold')
        
        # Chart 2: Individual Swap Slippage
        ax2.plot(swap_numbers, slippage_percents, 's-', color='#457B9D', linewidth=2, markersize=4)
        ax2.set_xlabel('Swap Number')
        ax2.set_ylabel('Individual Swap Slippage (%)')
        ax2.set_title('Slippage per Individual Swap')
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Cumulative Volume
        ax3.plot(swap_numbers, cumulative_volumes, '^-', color='#F1C40F', linewidth=2, markersize=4)
        ax3.set_xlabel('Swap Number')
        ax3.set_ylabel('Cumulative Volume ($)')
        ax3.set_title('Cumulative Trading Volume')
        ax3.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Chart 4: Active Liquidity Depletion
        ax4.plot(swap_numbers, active_liquidities, 'd-', color='#2E86AB', linewidth=2, markersize=4)
        ax4.set_xlabel('Swap Number')
        ax4.set_ylabel('Active Liquidity ($)')
        ax4.set_title('Pool Liquidity Depletion')
        ax4.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.suptitle(f'Repeated ${repeated_data["swap_size"]:,} Swap Analysis\n'
                    f'Pool Size: ${repeated_data["pool_size"]:,} ({repeated_data["concentration"]:.0%} Concentrated)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(charts_dir / "repeated_swap_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main execution function"""
    print("🚀 Starting Yield Token Pool Capacity Analysis")
    
    # Create and run analysis
    tester = YieldTokenPoolCapacityTester()
    results = tester.run_comprehensive_analysis()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 ANALYSIS SUMMARY")
    print("="*60)
    
    current_analysis = results["analysis_summary"]["current_pool_capacity"]
    print(f"🔍 Current Pool ($250k):")
    print(f"   Max Safe Trade: ${current_analysis['max_safe_trade']:,.0f}")
    print(f"   Pool Utilization: {current_analysis['pool_utilization']:.1f}%")
    if current_analysis['breaking_point']:
        print(f"   Breaking Point: ${current_analysis['breaking_point']:,.0f}")
    
    scaling_insights = results["analysis_summary"]["scaling_insights"]
    print(f"\n📈 Scaling Insights:")
    print(f"   Efficiency Range: {scaling_insights['efficiency_range']}")
    
    for rec in scaling_insights["recommendations"]:
        print(f"   {rec}")
    
    # Print repeated swap results if available
    if "repeated_swap_insights" in results["analysis_summary"]:
        repeated_insights = results["analysis_summary"]["repeated_swap_insights"]
        print(f"\n🔄 Repeated Swap Analysis:")
        print(f"   Swap Size: ${repeated_insights['swap_size']:,}")
        if repeated_insights["threshold_reached"]:
            print(f"   Swaps to 1% Threshold: {repeated_insights['swaps_to_threshold']}")
            print(f"   Volume to Threshold: ${repeated_insights['volume_to_threshold']:,}")
        else:
            print(f"   Total Swaps Executed: {repeated_insights['total_swaps_executed']}")
            print(f"   Final Volume: ${repeated_insights['final_cumulative_volume']:,}")
            print(f"   Final Price Impact: {repeated_insights['final_price_impact']:.2f}%")
    
    print("\n✅ Analysis completed successfully!")
    print(f"📁 Results saved in: tidal_protocol_sim/results/Yield_Token_Pool_Capacity_Analysis/")
    
    return results


if __name__ == "__main__":
    main()
