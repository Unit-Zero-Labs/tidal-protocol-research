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
        
        # Test configurations
        self.pool_sizes_usd = [
            250_000,    # Current size (problematic)
            500_000,    # 2x current
            1_000_000,  # 4x current  
            2_000_000,  # 8x current
            5_000_000,  # 20x current
            10_000_000, # 40x current
            20_000_000  # 80x current
        ]
        
        self.test_swap_sizes = [
            500,     # Small trade
            1_000,   # Typical rebalancing
            2_500,   # Medium rebalancing  
            5_000,   # Large rebalancing
            10_000,  # Very large rebalancing
            25_000,  # Extreme rebalancing
            50_000,  # Massive trade
            100_000  # Pool-breaking trade
        ]
        
        self.concentration = 0.95  # 95% concentration at peg
        self.price_impact_threshold = 0.01  # 1% price impact threshold
        
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive pool capacity analysis"""
        print("🔍 YIELD TOKEN POOL CAPACITY ANALYSIS")
        print("=" * 60)
        print(f"📊 Testing {len(self.pool_sizes_usd)} pool sizes")
        print(f"📊 Testing {len(self.test_swap_sizes)} swap sizes")
        print(f"📊 Price impact threshold: {self.price_impact_threshold:.1%}")
        print()
        
        # Test 1: Current pool analysis ($250k)
        print("🔍 Test 1: Current Pool Analysis ($250k)")
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
        print(f"   Testing pool size: ${pool_size_usd:,.0f}")
        
        # Create fresh pool for testing
        pool = YieldTokenPool(
            initial_moet_reserve=pool_size_usd / 2,  # Half of total pool size
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
        
        # Test each threshold against the current $250k pool
        pool_size = 250_000
        pool = YieldTokenPool(
            initial_moet_reserve=pool_size / 2,
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
    
    print("\n✅ Analysis completed successfully!")
    print(f"📁 Results saved in: tidal_protocol_sim/results/Yield_Token_Pool_Capacity_Analysis/")
    
    return results


if __name__ == "__main__":
    main()
