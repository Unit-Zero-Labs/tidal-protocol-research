#!/usr/bin/env python3
"""
Rebalance Liquidity Test
Testing MOET:YT Pool Capacity Under Rebalancing Scenarios

This simulation tests the $250k:$250k MOET:YT Uniswap V3 pool to determine:
1. Maximum single swap size that breaks the concentrated liquidity range (fresh pool each test)
2. How many consecutive $2,000 rebalances can occur before breaking the range (persistent state)

Based on the architecture from balanced_scenario_monte_carlo.py and yield_token_pool_capacity_analysis.py
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

from tidal_protocol_sim.engine.high_tide_vault_engine import HighTideVaultEngine, HighTideConfig
from tidal_protocol_sim.core.protocol import TidalProtocol, Asset
from tidal_protocol_sim.core.yield_tokens import YieldTokenPool


class RebalanceLiquidityTester:
    """
    Test MOET:YT pool capacity under rebalancing scenarios
    
    Tests both single large swaps (fresh pool) and consecutive small rebalances (persistent state)
    """
    
    def __init__(self):
        self.results = {
            "analysis_metadata": {
                "analysis_type": "Rebalance_Liquidity_Test", 
                "timestamp": datetime.now().isoformat(),
                "description": "Testing MOET:YT pool capacity under rebalancing scenarios"
            },
            "test_1_single_swaps": {},
            "test_2_consecutive_rebalances": {}
        }
        
        # Pool configuration - $250k:$250k MOET:YT pool
        self.pool_size_usd = 250_000  # Each side of the pool
        self.concentration = 0.95  # 95% concentration at 1:1 peg
        
        # Test 1: Progressive single swap sizes (fresh pool each time)
        self.single_swap_sizes = [
            1_000,   # Small rebalance
            2_500,   # Typical rebalance  
            5_000,   # Large rebalance
            10_000,  # Very large rebalance
            15_000,  # Progressive test 1
            20_000,  # Progressive test 2
            25_000,  # Progressive test 3
            30_000,  # Progressive test 4
            35_000,  # Progressive test 5
            40_000,  # Progressive test 6
            45_000,  # Progressive test 7
            50_000,  # Progressive test 8
            60_000,  # Progressive test 9
            70_000,  # Progressive test 10
            80_000,  # Progressive test 11
            90_000,  # Progressive test 12
            100_000, # Progressive test 13
            125_000, # Progressive test 14
            150_000, # Progressive test 15
            175_000, # Progressive test 16
            200_000, # Progressive test 17
            225_000, # Progressive test 18
            250_000, # Pool size match
            275_000, # Beyond pool size 1
            300_000, # Beyond pool size 2
            350_000, # Stress test 1
            400_000, # Stress test 2
            500_000  # Ultimate stress test
        ]
        
        # Test 2: Consecutive $2,000 rebalances
        self.rebalance_size = 2_000
        
        # Pool breaking criteria (when concentrated liquidity range is exceeded)
        self.price_deviation_threshold = 0.05  # 5% deviation from 1:1 peg indicates range break
        
        # Pool rebalancing/arbitrage configuration
        self.enable_pool_arbing = False  # Default to False for backward compatibility
        self.alm_rebalance_interval_minutes = 720  # 12 hours for ALM rebalancer
        self.algo_deviation_threshold_bps = 50.0  # 50 basis points for Algo rebalancer
        
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive rebalancing liquidity tests"""
        print("🔍 REBALANCE LIQUIDITY TEST")
        print("=" * 60)
        print(f"📊 Pool Configuration: ${self.pool_size_usd:,}:${self.pool_size_usd:,} MOET:YT")
        print(f"📊 Concentration: {self.concentration:.0%} at 1:1 peg")
        print(f"📊 Breaking threshold: {self.price_deviation_threshold:.1%} price deviation")
        print()
        
        # Test 1: Single large swaps (fresh pool each time)
        print("🔍 Test 1: Single Large Swaps (Fresh Pool Each Test)")
        test1_results = self._test_single_large_swaps()
        self.results["test_1_single_swaps"] = test1_results
        
        # Test 2: Consecutive $2,000 rebalances (persistent state)
        print("\n🔍 Test 2: Consecutive $2,000 Rebalances (Persistent State)")
        test2_results = self._test_consecutive_rebalances()
        self.results["test_2_consecutive_rebalances"] = test2_results
        
        # Generate analysis summary
        print("\n📊 Generating Analysis Summary...")
        self._generate_analysis_summary()
        
        # Save results
        self._save_results()
        
        # Generate charts
        print("📊 Generating Charts...")
        self._generate_charts()
        
        print("\n✅ Rebalance liquidity test completed!")
        return self.results
    
    def _test_single_large_swaps(self) -> Dict[str, Any]:
        """Test single large swaps with fresh pool each time"""
        print(f"   Testing {len(self.single_swap_sizes)} progressive swap sizes...")
        
        test_results = {
            "pool_config": {
                "pool_size_usd": self.pool_size_usd,
                "concentration": self.concentration,
                "price_deviation_threshold": self.price_deviation_threshold
            },
            "swap_tests": [],
            "breaking_point": None,
            "max_safe_single_swap": 0.0
        }
        
        breaking_point = None
        max_safe_swap = 0.0
        
        for swap_size in self.single_swap_sizes:
            print(f"      Testing ${swap_size:,} single swap on fresh pool...", end=" ")
            
            # Create fresh pool for each test
            fresh_pool = self._create_fresh_pool()
            
            # Test the swap
            swap_result = self._test_single_swap(fresh_pool, swap_size)
            
            # Record results
            swap_test = {
                "swap_size_usd": swap_size,
                "price_before": swap_result["price_before"],
                "price_after": swap_result["price_after"],
                "price_deviation_percent": swap_result["price_deviation_percent"],
                "moet_received": swap_result["moet_received"],
                "slippage_percent": swap_result["slippage_percent"],
                "active_liquidity_before": swap_result["active_liquidity_before"],
                "active_liquidity_after": swap_result["active_liquidity_after"],
                "liquidity_utilization": swap_result["liquidity_utilization"],
                "tick_before": swap_result["tick_before"],
                "tick_after": swap_result["tick_after"],
                "concentrated_range_ticks": swap_result["concentrated_range_ticks"],
                "concentrated_range_prices": swap_result["concentrated_range_prices"],
                "within_concentrated_range": swap_result["within_concentrated_range"],
                "swap_successful": swap_result["swap_successful"]
            }
            
            test_results["swap_tests"].append(swap_test)
            
            # Track maximum safe swap and breaking point (use proper concentrated range check)
            if swap_test["within_concentrated_range"] and swap_test["swap_successful"]:
                max_safe_swap = swap_size
                print("✅")
            else:
                if breaking_point is None:
                    breaking_point = swap_size
                print(f"❌ CONCENTRATED RANGE BROKEN (tick: {swap_test['tick_after']}, range: {swap_test['concentrated_range_ticks']})")
                break  # Stop testing once we break the concentrated range
        
        test_results["breaking_point"] = breaking_point
        test_results["max_safe_single_swap"] = max_safe_swap
        
        print(f"   📊 Max safe single swap: ${max_safe_swap:,}")
        if breaking_point:
            print(f"   🚨 Range breaks at single swap: ${breaking_point:,}")
        
        return test_results
    
    def _test_consecutive_rebalances(self) -> Dict[str, Any]:
        """Test consecutive $2,000 rebalances with persistent pool state"""
        print(f"   Testing consecutive ${self.rebalance_size:,} rebalances...")
        
        test_results = {
            "rebalance_size": self.rebalance_size,
            "pool_config": {
                "pool_size_usd": self.pool_size_usd,
                "concentration": self.concentration,
                "price_deviation_threshold": self.price_deviation_threshold
            },
            "rebalance_history": [],
            "total_rebalances_executed": 0,
            "cumulative_volume": 0.0,
            "final_price_deviation": 0.0,
            "range_broken": False,
            "breaking_rebalance_number": None
        }
        
        # Create persistent pool for consecutive testing
        persistent_pool = self._create_fresh_pool()
        
        rebalance_count = 0
        cumulative_volume = 0.0
        initial_price = persistent_pool.uniswap_pool.get_price()
        
        print(f"      Starting consecutive rebalances (no max limit - stop at range break)...")
        
        while True:  # No max limit - continue until range breaks
            rebalance_count += 1
            
            try:
                # Execute the rebalance (permanently modifies pool state)
                moet_received = persistent_pool.execute_yield_token_sale(self.rebalance_size)
                swap_successful = moet_received > 0
                
                if not swap_successful:
                    print(f"      Rebalance #{rebalance_count}: FAILED - Pool exhausted")
                    break
                
                # Get post-rebalance metrics
                current_price = persistent_pool.uniswap_pool.get_price()
                price_deviation_percent = abs((current_price - 1.0) / 1.0) * 100  # Deviation from 1:1 peg
                active_liquidity = persistent_pool.uniswap_pool._calculate_active_liquidity_from_ticks(
                    persistent_pool.uniswap_pool.tick_current
                ) / 1e6
                
                cumulative_volume += self.rebalance_size
                
                # Calculate slippage for this rebalance
                expected_moet = self.rebalance_size  # 1:1 expected
                slippage_percent = abs((expected_moet - moet_received) / expected_moet) * 100 if expected_moet > 0 else 0
                
                # Record rebalance details
                rebalance_record = {
                    "rebalance_number": rebalance_count,
                    "cumulative_volume": cumulative_volume,
                    "price_before_rebalance": persistent_pool.uniswap_pool.get_price() if rebalance_count == 1 else test_results["rebalance_history"][-1]["price_after_rebalance"],
                    "price_after_rebalance": current_price,
                    "price_deviation_percent": price_deviation_percent,
                    "moet_received": moet_received,
                    "slippage_percent": slippage_percent,
                    "active_liquidity": active_liquidity,
                    "within_range": price_deviation_percent <= self.price_deviation_threshold * 100,
                    "swap_successful": swap_successful
                }
                
                test_results["rebalance_history"].append(rebalance_record)
                
                # Get concentrated position bounds for proper range checking
                concentrated_pos = persistent_pool.uniswap_pool.positions[0]
                tick_lower = concentrated_pos.tick_lower
                tick_upper = concentrated_pos.tick_upper
                current_tick = persistent_pool.uniswap_pool.tick_current
                
                # Check if we've broken the concentrated range (proper Uniswap V3 check)
                within_concentrated_range = (tick_lower <= current_tick <= tick_upper)
                
                if not within_concentrated_range:
                    test_results["range_broken"] = True
                    test_results["breaking_rebalance_number"] = rebalance_count
                    print(f"      🚨 Concentrated range broken at rebalance #{rebalance_count}: tick {current_tick} outside [{tick_lower}, {tick_upper}]")
                    break
                
                # Print progress every 10 rebalances or at range break
                if rebalance_count % 10 == 0:
                    print(f"      Rebalance #{rebalance_count:3d}: ${cumulative_volume:7,.0f} total → {price_deviation_percent:5.2f}% deviation, {slippage_percent:5.2f}% slippage")
                    
            except Exception as e:
                print(f"      Rebalance #{rebalance_count}: ERROR - {str(e)}")
                break
        
        # Final results
        test_results["total_rebalances_executed"] = rebalance_count
        test_results["cumulative_volume"] = cumulative_volume
        test_results["final_price_deviation"] = price_deviation_percent if 'price_deviation_percent' in locals() else 0.0
        
        # Summary
        if test_results["range_broken"]:
            breaking_volume = test_results["breaking_rebalance_number"] * self.rebalance_size
            print(f"   📊 Summary: {test_results['breaking_rebalance_number']} rebalances (${breaking_volume:,}) to break range")
        else:
            print(f"   📊 Summary: {rebalance_count} rebalances executed (${cumulative_volume:,} total), final deviation: {test_results['final_price_deviation']:.2f}%")
        
        return test_results
    
    def _create_fresh_pool(self) -> YieldTokenPool:
        """Create a fresh MOET:YT pool with proper configuration"""
        # Create pool directly with new interface
        total_pool_size = self.pool_size_usd * 2  # Convert to total pool size
        token0_ratio = 0.75  # Use 75% MOET, 25% YT ratio
        
        yield_token_pool = YieldTokenPool(
            total_pool_size=total_pool_size,
            token0_ratio=token0_ratio,
            concentration=self.concentration
        )
        
        return yield_token_pool
    
    def _test_single_swap(self, pool: YieldTokenPool, swap_amount_usd: float) -> Dict[str, Any]:
        """Test a single swap on the given pool"""
        try:
            # Get initial state
            initial_price = pool.uniswap_pool.get_price()
            initial_tick = pool.uniswap_pool.tick_current
            initial_active_liquidity = pool.uniswap_pool._calculate_active_liquidity_from_ticks(
                pool.uniswap_pool.tick_current
            ) / 1e6
            
            # Get concentrated position bounds for proper range checking
            concentrated_pos = pool.uniswap_pool.positions[0]  # Main concentrated position
            tick_lower = concentrated_pos.tick_lower
            tick_upper = concentrated_pos.tick_upper
            price_lower = (1.0001 ** tick_lower)
            price_upper = (1.0001 ** tick_upper)
            
            # Execute the swap
            moet_received = pool.execute_yield_token_sale(swap_amount_usd)
            swap_successful = moet_received > 0
            
            # Get post-swap state
            final_price = pool.uniswap_pool.get_price()
            final_tick = pool.uniswap_pool.tick_current
            final_active_liquidity = pool.uniswap_pool._calculate_active_liquidity_from_ticks(
                pool.uniswap_pool.tick_current
            ) / 1e6
            
            # Calculate metrics
            price_deviation_percent = abs((final_price - 1.0) / 1.0) * 100  # Deviation from 1:1 peg
            expected_moet = swap_amount_usd  # 1:1 expected without slippage
            slippage_percent = abs((expected_moet - moet_received) / expected_moet) * 100 if expected_moet > 0 else 0
            liquidity_utilization = ((initial_active_liquidity - final_active_liquidity) / initial_active_liquidity) * 100 if initial_active_liquidity > 0 else 0
            
            # Check if we're still within the concentrated range (proper Uniswap V3 check)
            within_concentrated_range = (tick_lower <= final_tick <= tick_upper) and (price_lower <= final_price <= price_upper)
            
            result = {
                "price_before": initial_price,
                "price_after": final_price,
                "price_deviation_percent": price_deviation_percent,
                "moet_received": moet_received,
                "slippage_percent": slippage_percent,
                "active_liquidity_before": initial_active_liquidity,
                "active_liquidity_after": final_active_liquidity,
                "liquidity_utilization": liquidity_utilization,
                "tick_before": initial_tick,
                "tick_after": final_tick,
                "concentrated_range_ticks": f"[{tick_lower}, {tick_upper}]",
                "concentrated_range_prices": f"[{price_lower:.6f}, {price_upper:.6f}]",
                "within_concentrated_range": within_concentrated_range,
                "swap_successful": swap_successful
            }
            
        except Exception as e:
            # Handle swap failures
            result = {
                "price_before": pool.uniswap_pool.get_price(),
                "price_after": pool.uniswap_pool.get_price(),
                "price_deviation_percent": 0.0,
                "moet_received": 0.0,
                "slippage_percent": 100.0,
                "active_liquidity_before": 0.0,
                "active_liquidity_after": 0.0,
                "liquidity_utilization": 0.0,
                "swap_successful": False,
                "error": str(e)
            }
        
        return result
    
    def _generate_analysis_summary(self):
        """Generate comprehensive analysis summary"""
        test1_results = self.results["test_1_single_swaps"]
        test2_results = self.results["test_2_consecutive_rebalances"]
        
        summary = {
            "pool_configuration": {
                "pool_size_usd": self.pool_size_usd,
                "total_pool_value": self.pool_size_usd * 2,
                "concentration": self.concentration,
                "price_deviation_threshold": self.price_deviation_threshold
            },
            "test_1_single_swaps_summary": {
                "max_safe_single_swap": test1_results["max_safe_single_swap"],
                "breaking_point": test1_results["breaking_point"],
                "pool_utilization_at_max": (test1_results["max_safe_single_swap"] / self.pool_size_usd) * 100 if test1_results["max_safe_single_swap"] > 0 else 0,
                "total_swaps_tested": len(test1_results["swap_tests"])
            },
            "test_2_consecutive_rebalances_summary": {
                "rebalance_size": test2_results["rebalance_size"],
                "total_rebalances_executed": test2_results["total_rebalances_executed"],
                "cumulative_volume": test2_results["cumulative_volume"],
                "range_broken": test2_results["range_broken"],
                "breaking_rebalance_number": test2_results.get("breaking_rebalance_number"),
                "final_price_deviation": test2_results["final_price_deviation"],
                "pool_utilization_at_break": (test2_results["cumulative_volume"] / self.pool_size_usd) * 100 if test2_results["cumulative_volume"] > 0 else 0
            },
            "comparative_analysis": {
                "single_swap_capacity": test1_results["max_safe_single_swap"],
                "consecutive_rebalance_capacity": test2_results["cumulative_volume"],
                "efficiency_ratio": test2_results["cumulative_volume"] / test1_results["max_safe_single_swap"] if test1_results["max_safe_single_swap"] > 0 else 0
            }
        }
        
        self.results["analysis_summary"] = summary
    
    def _save_results(self):
        """Save results to JSON file"""
        results_dir = Path("tidal_protocol_sim/results") / "Rebalance_Liquidity_Test"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"rebalance_liquidity_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
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
        charts_dir = Path("tidal_protocol_sim/results") / "Rebalance_Liquidity_Test" / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Chart 1: Single Swap Capacity Analysis
        self._create_single_swap_chart(charts_dir)
        
        # Chart 2: Consecutive Rebalance Analysis
        self._create_consecutive_rebalance_chart(charts_dir)
        
        # Chart 3: Comparative Analysis
        self._create_comparative_analysis_chart(charts_dir)
        
        print(f"📊 Charts saved to: {charts_dir}")
    
    def _create_single_swap_chart(self, charts_dir: Path):
        """Create single swap capacity analysis chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        test1_data = self.results["test_1_single_swaps"]
        swap_tests = test1_data["swap_tests"]
        
        swap_sizes = [test["swap_size_usd"] / 1e3 for test in swap_tests]  # Convert to thousands
        price_deviations = [test["price_deviation_percent"] for test in swap_tests]
        slippage_percents = [test["slippage_percent"] for test in swap_tests]
        liquidity_utils = [test["liquidity_utilization"] for test in swap_tests]
        within_range = [test["within_concentrated_range"] for test in swap_tests]
        
        # Chart 1: Price Deviation
        colors1 = ['green' if wr else 'red' for wr in within_range]
        bars1 = ax1.bar(range(len(swap_sizes)), price_deviations, color=colors1, alpha=0.7)
        ax1.set_xlabel('Swap Size')
        ax1.set_ylabel('Price Deviation from 1:1 Peg (%)')
        ax1.set_title('Price Deviation by Single Swap Size')
        ax1.axhline(y=self.price_deviation_threshold * 100, color='orange', linestyle='--', 
                   label=f'{self.price_deviation_threshold:.1%} Range Break Threshold')
        ax1.set_xticks(range(len(swap_sizes)))
        ax1.set_xticklabels([f'${s:.0f}k' for s in swap_sizes], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mark breaking point
        if test1_data["breaking_point"]:
            breaking_idx = next(i for i, test in enumerate(swap_tests) if test["swap_size_usd"] == test1_data["breaking_point"])
            ax1.annotate(f'Range Breaks\n${test1_data["breaking_point"]/1e3:.0f}k', 
                        xy=(breaking_idx, price_deviations[breaking_idx]),
                        xytext=(breaking_idx, price_deviations[breaking_idx] + 1),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2),
                        fontsize=10, color='red', fontweight='bold')
        
        # Chart 2: Slippage
        ax2.plot(range(len(swap_sizes)), slippage_percents, 'o-', color='#457B9D', linewidth=2, markersize=6)
        ax2.set_xlabel('Swap Size')
        ax2.set_ylabel('Slippage (%)')
        ax2.set_title('Slippage by Single Swap Size')
        ax2.set_xticks(range(len(swap_sizes)))
        ax2.set_xticklabels([f'${s:.0f}k' for s in swap_sizes], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Liquidity Utilization
        ax3.bar(range(len(swap_sizes)), liquidity_utils, color='#F1C40F', alpha=0.7)
        ax3.set_xlabel('Swap Size')
        ax3.set_ylabel('Liquidity Utilization (%)')
        ax3.set_title('Pool Liquidity Utilization by Swap Size')
        ax3.set_xticks(range(len(swap_sizes)))
        ax3.set_xticklabels([f'${s:.0f}k' for s in swap_sizes], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Success/Failure Status
        success_status = ['Success' if wr else 'Range Broken' for wr in within_range]
        success_counts = {'Success': success_status.count('Success'), 'Range Broken': success_status.count('Range Broken')}
        colors4 = ['green', 'red']
        ax4.pie(success_counts.values(), labels=success_counts.keys(), colors=colors4, autopct='%1.1f%%')
        ax4.set_title('Single Swap Test Results')
        
        plt.suptitle(f'Single Swap Capacity Analysis\nPool: ${self.pool_size_usd:,}:${self.pool_size_usd:,} MOET:YT ({self.concentration:.0%} Concentrated)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(charts_dir / "single_swap_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_consecutive_rebalance_chart(self, charts_dir: Path):
        """Create consecutive rebalance analysis chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        test2_data = self.results["test_2_consecutive_rebalances"]
        rebalance_history = test2_data["rebalance_history"]
        
        if not rebalance_history:
            # Create placeholder if no data
            ax1.text(0.5, 0.5, 'No rebalance data available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Consecutive Rebalance Analysis - No Data')
            plt.tight_layout()
            plt.savefig(charts_dir / "consecutive_rebalance_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        rebalance_numbers = [rb["rebalance_number"] for rb in rebalance_history]
        cumulative_volumes = [rb["cumulative_volume"] for rb in rebalance_history]
        price_deviations = [rb["price_deviation_percent"] for rb in rebalance_history]
        slippage_percents = [rb["slippage_percent"] for rb in rebalance_history]
        active_liquidities = [rb["active_liquidity"] for rb in rebalance_history]
        
        # Chart 1: Cumulative Price Deviation
        ax1.plot(rebalance_numbers, price_deviations, 'o-', color='#E63946', linewidth=2, markersize=4)
        ax1.axhline(y=self.price_deviation_threshold * 100, color='orange', linestyle='--', linewidth=2, alpha=0.8, 
                   label=f'{self.price_deviation_threshold:.1%} Range Break Threshold')
        ax1.set_xlabel('Rebalance Number')
        ax1.set_ylabel('Price Deviation from 1:1 Peg (%)')
        ax1.set_title(f'Price Deviation: Consecutive ${test2_data["rebalance_size"]:,} Rebalances')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Mark breaking point if reached
        if test2_data["range_broken"]:
            breaking_rebalance = test2_data["breaking_rebalance_number"]
            breaking_deviation = next(rb["price_deviation_percent"] for rb in rebalance_history if rb["rebalance_number"] == breaking_rebalance)
            ax1.scatter([breaking_rebalance], [breaking_deviation], color='red', s=100, zorder=5)
            ax1.annotate(f'Range Breaks\nRebalance #{breaking_rebalance}', 
                        xy=(breaking_rebalance, breaking_deviation),
                        xytext=(breaking_rebalance + 5, breaking_deviation + 0.5),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=9, color='red', fontweight='bold')
        
        # Chart 2: Individual Rebalance Slippage
        ax2.plot(rebalance_numbers, slippage_percents, 's-', color='#457B9D', linewidth=2, markersize=4)
        ax2.set_xlabel('Rebalance Number')
        ax2.set_ylabel('Individual Rebalance Slippage (%)')
        ax2.set_title('Slippage per Individual Rebalance')
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Cumulative Volume
        ax3.plot(rebalance_numbers, cumulative_volumes, '^-', color='#F1C40F', linewidth=2, markersize=4)
        ax3.set_xlabel('Rebalance Number')
        ax3.set_ylabel('Cumulative Volume ($)')
        ax3.set_title('Cumulative Rebalancing Volume')
        ax3.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Chart 4: Active Liquidity Depletion
        ax4.plot(rebalance_numbers, active_liquidities, 'd-', color='#2E86AB', linewidth=2, markersize=4)
        ax4.set_xlabel('Rebalance Number')
        ax4.set_ylabel('Active Liquidity ($)')
        ax4.set_title('Pool Liquidity Depletion')
        ax4.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.suptitle(f'Consecutive ${test2_data["rebalance_size"]:,} Rebalance Analysis\n'
                    f'Pool: ${self.pool_size_usd:,}:${self.pool_size_usd:,} MOET:YT ({self.concentration:.0%} Concentrated)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(charts_dir / "consecutive_rebalance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_comparative_analysis_chart(self, charts_dir: Path):
        """Create comparative analysis chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        summary = self.results["analysis_summary"]
        
        # Chart 1: Capacity Comparison
        categories = ['Single Large\nSwap', 'Consecutive Small\nRebalances']
        capacities = [
            summary["test_1_single_swaps_summary"]["max_safe_single_swap"],
            summary["test_2_consecutive_rebalances_summary"]["cumulative_volume"]
        ]
        colors = ['#2E86AB', '#A23B72']
        
        bars1 = ax1.bar(categories, capacities, color=colors, alpha=0.8)
        ax1.set_ylabel('Total Capacity ($)')
        ax1.set_title('Pool Capacity Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, capacity in zip(bars1, capacities):
            height = bar.get_height()
            ax1.annotate(f'${capacity:,.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Chart 2: Pool Utilization Comparison
        utilizations = [
            summary["test_1_single_swaps_summary"]["pool_utilization_at_max"],
            summary["test_2_consecutive_rebalances_summary"]["pool_utilization_at_break"]
        ]
        
        bars2 = ax2.bar(categories, utilizations, color=colors, alpha=0.8)
        ax2.set_ylabel('Pool Utilization (%)')
        ax2.set_title('Pool Utilization at Capacity')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, util in zip(bars2, utilizations):
            height = bar.get_height()
            ax2.annotate(f'{util:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.suptitle(f'Rebalance Liquidity Test: Comparative Analysis\n'
                    f'Pool: ${self.pool_size_usd:,}:${self.pool_size_usd:,} MOET:YT ({self.concentration:.0%} Concentrated)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(charts_dir / "comparative_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main execution function"""
    print("🚀 Starting Rebalance Liquidity Test")
    
    # Create and run test
    tester = RebalanceLiquidityTester()
    results = tester.run_comprehensive_test()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    summary = results["analysis_summary"]
    
    print(f"🏊 Pool Configuration:")
    print(f"   Pool Size: ${summary['pool_configuration']['pool_size_usd']:,}:${summary['pool_configuration']['pool_size_usd']:,} MOET:YT")
    print(f"   Total Pool Value: ${summary['pool_configuration']['total_pool_value']:,}")
    print(f"   Concentration: {summary['pool_configuration']['concentration']:.0%}")
    print(f"   Range Break Threshold: {summary['pool_configuration']['price_deviation_threshold']:.1%}")
    
    print(f"\n🔍 Test 1: Single Large Swaps (Fresh Pool Each Time)")
    test1_summary = summary["test_1_single_swaps_summary"]
    print(f"   Max Safe Single Swap: ${test1_summary['max_safe_single_swap']:,}")
    print(f"   Range Breaking Point: ${test1_summary['breaking_point']:,}" if test1_summary['breaking_point'] else "   Range Breaking Point: Not reached")
    print(f"   Pool Utilization at Max: {test1_summary['pool_utilization_at_max']:.1f}%")
    print(f"   Total Swap Sizes Tested: {test1_summary['total_swaps_tested']}")
    
    print(f"\n🔄 Test 2: Consecutive $2,000 Rebalances (Persistent State)")
    test2_summary = summary["test_2_consecutive_rebalances_summary"]
    print(f"   Rebalance Size: ${test2_summary['rebalance_size']:,}")
    print(f"   Total Rebalances Executed: {test2_summary['total_rebalances_executed']}")
    print(f"   Cumulative Volume: ${test2_summary['cumulative_volume']:,}")
    print(f"   Range Broken: {'Yes' if test2_summary['range_broken'] else 'No'}")
    if test2_summary['breaking_rebalance_number']:
        print(f"   Breaking Rebalance Number: {test2_summary['breaking_rebalance_number']}")
    print(f"   Final Price Deviation: {test2_summary['final_price_deviation']:.2f}%")
    print(f"   Pool Utilization at Break: {test2_summary['pool_utilization_at_break']:.1f}%")
    
    print(f"\n📊 Comparative Analysis:")
    comp_analysis = summary["comparative_analysis"]
    print(f"   Single Swap Capacity: ${comp_analysis['single_swap_capacity']:,}")
    print(f"   Consecutive Rebalance Capacity: ${comp_analysis['consecutive_rebalance_capacity']:,}")
    print(f"   Efficiency Ratio (Consecutive/Single): {comp_analysis['efficiency_ratio']:.2f}x")
    
    print("\n✅ Rebalance liquidity test completed successfully!")
    print(f"📁 Results saved in: tidal_protocol_sim/results/Rebalance_Liquidity_Test/")
    
    return results


if __name__ == "__main__":
    main()
