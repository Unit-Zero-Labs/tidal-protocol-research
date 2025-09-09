#!/usr/bin/env python3
"""
Liquidity Provider Curve Analysis

Tracks and visualizes how Uniswap v3 concentrated liquidity curves change over time
during High Tide rebalancing events, showing:
- LP curve shape evolution with discrete bins (bar charts)
- Concentration level changes
- Price impact visualization
- Pool utilization analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import seaborn as sns
from dataclasses import dataclass
import math

from ..core.uniswap_v3_math import UniswapV3Pool


@dataclass
class PoolSnapshot:
    """Snapshot of pool state at a specific time"""
    minute: int
    moet_reserve: float
    btc_reserve: float
    price: float
    liquidity: float
    concentration_range: float
    trade_amount: float = 0.0
    trade_type: str = ""  # "rebalance" or "liquidation"
    concentrated_pool: Optional[UniswapV3Pool] = None  # New field for bin data


class LPCurveTracker:
    """Tracks LP curve changes during simulation with discrete bins"""
    
    def __init__(self, initial_pool_size: float, concentration_range: float, pool_name: str = "Unknown", btc_price: float = 100_000.0):
        self.initial_pool_size = initial_pool_size
        self.concentration_range = concentration_range
        self.pool_name = pool_name  # "MOET:BTC" or "MOET:Yield_Token"
        self.btc_price = btc_price  # Store BTC price for correct price calculations
        self.snapshots: List[PoolSnapshot] = []
        self.cumulative_trade_volume = 0.0  # Track total trade volume for utilization calculation
        
        # Initialize the concentrated liquidity pool
        if "MOET:BTC" in pool_name:
            from ..core.uniswap_v3_math import create_moet_btc_pool
            self.concentrated_pool = create_moet_btc_pool(pool_size_usd=initial_pool_size, btc_price=btc_price, concentration=concentration_range)
        else:
            from ..core.uniswap_v3_math import create_yield_token_pool
            self.concentrated_pool = create_yield_token_pool(pool_size_usd=initial_pool_size, concentration=concentration_range)
        
        # Calculate correct initial price based on pool type
        if "MOET:BTC" in pool_name:
            # For MOET:BTC pool, price should be BTC per MOET
            # 1 BTC = 100,000 MOET, so 1 MOET = 0.00001 BTC
            initial_price = 1.0 / 100000.0  # BTC per MOET = 0.00001
        else:
            # For yield token pool, maintain 1:1 with MOET
            initial_price = 1.0
        
        # Initialize with starting state
        self.snapshots.append(PoolSnapshot(
            minute=0,
            moet_reserve=initial_pool_size / 2,  # $250k MOET (250,000 tokens)
            btc_reserve=initial_pool_size / 2,  # $250k BTC (2.5 tokens at $100k/BTC)
            price=initial_price,  # Correct initial price
            liquidity=initial_pool_size / 2,
            concentration_range=concentration_range,
            concentrated_pool=self.concentrated_pool
        ))
    
    def record_snapshot(self, pool_state: Dict[str, float], minute: int, 
                       trade_amount: float = 0.0, trade_type: str = ""):
        """Record a pool state snapshot with updated concentrated liquidity"""
        
        # Use the pool's built-in price calculation for accuracy
        price = self.concentrated_pool.get_price()
        
        # Update the concentrated liquidity pool if there was a trade
        if trade_amount > 0 and self.concentrated_pool:
            # Track cumulative trade volume for utilization calculation
            self.cumulative_trade_volume += trade_amount
            
            # Determine token being traded in based on trade type and pool
            if trade_type == "rebalance":
                # Rebalancing typically involves selling yield tokens for MOET
                if "Yield" in self.pool_name:
                    token_in = "Yield_Token"
                else:
                    token_in = "MOET"  # For MOET:BTC pool, selling MOET for BTC
            else:
                # Buy operations - opposite direction
                token_in = "MOET"
            
            # Execute actual swap to update pool state properly
            # Convert USD to scaled amount for Uniswap V3 math
            amount_in_scaled = int(trade_amount * 1e6)
            
            # Determine swap direction
            zero_for_one = token_in in ["MOET", "token0"]
            
            # Execute the swap to update pool state
            try:
                from ..core.uniswap_v3_math import MIN_SQRT_RATIO, MAX_SQRT_RATIO
                
                sqrt_price_limit = MIN_SQRT_RATIO + 1 if zero_for_one else MAX_SQRT_RATIO - 1
                
                self.concentrated_pool.swap(
                    zero_for_one=zero_for_one,
                    amount_specified=amount_in_scaled,
                    sqrt_price_limit_x96=sqrt_price_limit
                )
                
                # Update price after swap
                price = self.concentrated_pool.get_price()
                
            except (ValueError, ZeroDivisionError) as e:
                # If swap fails, use current price and continue
                print(f"Warning: Swap failed in LPCurveTracker: {e}")
                price = self.concentrated_pool.get_price()
        
        # Create snapshot using pool's current state for accuracy
        snapshot = PoolSnapshot(
            minute=minute,
            moet_reserve=self.concentrated_pool.token0_reserve or pool_state.get("token0_reserve", pool_state.get("moet_reserve", 0)),
            btc_reserve=self.concentrated_pool.token1_reserve or pool_state.get("token1_reserve", pool_state.get("btc_reserve", 0)),
            price=price,
            liquidity=self.concentrated_pool.get_total_active_liquidity() or pool_state.get("liquidity", 0),
            concentration_range=self.concentration_range,
            trade_amount=trade_amount,
            trade_type=trade_type,
            concentrated_pool=self.concentrated_pool
        )
        
        self.snapshots.append(snapshot)
    
    def get_snapshots(self) -> List[PoolSnapshot]:
        """Get all recorded snapshots"""
        return self.snapshots


class LPCurveAnalyzer:
    """Analyzes and visualizes LP curve evolution with discrete bins"""
    
    def __init__(self):
        self._setup_styling()
    
    def _setup_styling(self):
        """Setup chart styling"""
        plt.rcParams.update({
            'figure.figsize': (16, 12),
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 9,
            'figure.titlesize': 16
        })
    
    def create_lp_curve_evolution_chart(self, tracker: LPCurveTracker, charts_dir: Path) -> Path:
        """Create chart showing LP curve evolution over time with discrete bins (bar charts)"""
        
        snapshots = tracker.get_snapshots()
        if len(snapshots) < 2:
            # Not enough data for evolution chart
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        pool_display_name = tracker.pool_name.replace("_", " ")
        fig.suptitle(f"{pool_display_name} LP Curve Evolution During Rebalancing Events", 
                     fontsize=16, fontweight='bold')
        
        # 1. LP Curve Shape Evolution with Discrete Ticks (Bar Chart)
        self._create_tick_evolution_chart(ax1, snapshots, tracker.pool_name)
        
        # 2. Pool Reserve Changes
        self._create_reserve_changes_chart(ax2, snapshots)
        
        # 3. Price Impact Over Time
        self._create_price_impact_chart(ax3, snapshots, tracker.pool_name)
        
        # 4. Concentration Efficiency Analysis
        self._create_concentration_efficiency_chart(ax4, snapshots, tracker.pool_name)
        
        plt.tight_layout()
        
        # Save chart with pool-specific name
        chart_filename = f"{tracker.pool_name.lower()}_lp_curve_evolution.png"
        chart_path = charts_dir / chart_filename
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def create_concentration_comparison_chart(self, 
                                            tight_tracker: LPCurveTracker,
                                            conservative_tracker: LPCurveTracker,
                                            charts_dir: Path) -> Path:
        """Create chart comparing different concentration levels"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle("LP Concentration Level Comparison: Tight vs Conservative", 
                     fontsize=16, fontweight='bold')
        
        # Get final snapshots
        tight_final = tight_tracker.get_snapshots()[-1]
        conservative_final = conservative_tracker.get_snapshots()[-1]
        
        # Determine correct peg value based on pool type
        if "MOET:BTC" in tight_tracker.pool_name:
            correct_peg = 0.00001  # BTC per MOET
            price_label = "Price (BTC per MOET)"
        else:
            correct_peg = 1.0  # 1:1 for yield tokens
            price_label = "Price (MOET per Yield Token)"
        
        # Price range for comparison
        all_prices = [tight_final.price, conservative_final.price, correct_peg]
        min_price = min(all_prices) * 0.7
        max_price = max(all_prices) * 1.3
        price_range = (min_price, max_price)
        
        # 1. LP Curve Shape Comparison with Discrete Ticks
        self._create_tick_comparison_chart(ax1, tight_final, conservative_final, price_label)
        
        # 2. Trade Impact Comparison
        tight_snapshots = tight_tracker.get_snapshots()
        conservative_snapshots = conservative_tracker.get_snapshots()
        
        tight_minutes = [s.minute for s in tight_snapshots]
        conservative_minutes = [s.minute for s in conservative_snapshots]
        
        tight_impacts = [(s.price - correct_peg) / correct_peg * 100 for s in tight_snapshots]
        conservative_impacts = [(s.price - correct_peg) / correct_peg * 100 for s in conservative_snapshots]
        
        ax2.plot(tight_minutes, tight_impacts, color='#2E8B57', linewidth=2, 
                label="Tight Concentration", marker='o', markersize=3)
        ax2.plot(conservative_minutes, conservative_impacts, color='#FF6B35', linewidth=2,
                label="Conservative Concentration", marker='s', markersize=3)
        
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel("Time (minutes)")
        ax2.set_ylabel("Price Deviation from Peg (%)")
        ax2.set_title("Price Impact Timeline Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Liquidity Utilization
        tight_utils = []
        conservative_utils = []
        
        for snapshot in tight_snapshots:
            price_change = abs(snapshot.price - correct_peg)
            util = min(100, (price_change / (snapshot.concentration_range / 2)) * 100)
            tight_utils.append(util)
        
        for snapshot in conservative_snapshots:
            price_change = abs(snapshot.price - correct_peg)
            util = min(100, (price_change / (snapshot.concentration_range / 2)) * 100)
            conservative_utils.append(util)
        
        ax3.fill_between(tight_minutes, tight_utils, alpha=0.6, color='#2E8B57', 
                        label="Tight Concentration Usage")
        ax3.fill_between(conservative_minutes, conservative_utils, alpha=0.6, color='#FF6B35',
                        label="Conservative Concentration Usage")
        
        ax3.set_xlabel("Time (minutes)")
        ax3.set_ylabel("Concentration Range Utilization (%)")
        ax3.set_title("Liquidity Range Efficiency")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 120)
        
        # 4. Summary Statistics
        ax4.axis('off')
        
        # Calculate summary stats
        tight_final_impact = abs(tight_final.price - correct_peg) / correct_peg * 100
        conservative_final_impact = abs(conservative_final.price - correct_peg) / correct_peg * 100
        
        tight_max_util = max(tight_utils) if tight_utils else 0
        conservative_max_util = max(conservative_utils) if conservative_utils else 0
        
        summary_text = f"""
CONCENTRATION ANALYSIS SUMMARY

Tight Concentration ({tight_final.concentration_range*100:.0f}%):
• Final Price Deviation: {tight_final_impact:.2f}%
• Max Range Utilization: {tight_max_util:.1f}%
• Final MOET Reserve: ${tight_final.moet_reserve:,.0f}
• Final BTC Reserve: ${tight_final.btc_reserve:,.0f}

Conservative Concentration ({conservative_final.concentration_range*100:.0f}%):
• Final Price Deviation: {conservative_final_impact:.2f}%
• Max Range Utilization: {conservative_max_util:.1f}%
• Final MOET Reserve: ${conservative_final.moet_reserve:,.0f}
• Final BTC Reserve: ${conservative_final.btc_reserve:,.0f}

Key Insights:
• Tight concentration → Higher slippage, better peg maintenance
• Conservative concentration → Lower slippage, more price flexibility
• Range utilization shows efficiency of concentration choice
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Save chart
        chart_path = charts_dir / "lp_concentration_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
    
    def _create_tick_evolution_chart(self, ax, snapshots: List[PoolSnapshot], pool_name: str):
        """Create bar chart showing liquidity tick evolution over time with proper concentration visualization"""
        
        # Sample snapshots for clear visualization
        sample_snapshots = snapshots[::max(1, len(snapshots)//6)]  # Show 6 snapshots max
        
        first_snapshot = sample_snapshots[0]
        if not first_snapshot.concentrated_pool:
            ax.text(0.5, 0.5, "No concentrated liquidity data", 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Get all tick data (not just active ones) to show the full distribution
        tick_data = first_snapshot.concentrated_pool.get_tick_data_for_charts()
        
        if not tick_data:
            ax.text(0.5, 0.5, "No liquidity tick data", 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Sort ticks by price to show concentration properly
        tick_data.sort(key=lambda x: x['price'])
        
        # For proper visualization, show all ticks but limit to reasonable number
        total_ticks = len(tick_data)
        if total_ticks > 100:
            # For very large tick counts, show key ticks: peg tick + surrounding ticks
            peg_price = 0.00001 if "MOET:BTC" in pool_name else 1.0
            
            # Find peg tick
            peg_tick = min(tick_data, key=lambda x: abs(x['price'] - peg_price))
            peg_index = tick_data.index(peg_tick)
            
            # Show peg tick + 20 ticks on each side
            start_idx = max(0, peg_index - 20)
            end_idx = min(len(tick_data), peg_index + 21)
            sampled_ticks = tick_data[start_idx:end_idx]
        else:
            sampled_ticks = tick_data
        
        tick_indices = range(len(sampled_ticks))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sample_snapshots)))
        
        # Create grouped bar chart
        bar_width = 0.8 / len(sample_snapshots)
        
        for i, snapshot in enumerate(sample_snapshots):
            if snapshot.concentrated_pool:
                # Get updated tick data for this snapshot
                current_tick_data = snapshot.concentrated_pool.get_tick_data_for_charts()
                current_tick_data.sort(key=lambda x: x['price'])
                
                # Use the same tick selection as the reference
                if total_ticks > 100:
                    # Use the same range as the reference ticks
                    peg_price = 0.00001 if "MOET:BTC" in pool_name else 1.0
                    peg_tick = min(current_tick_data, key=lambda x: abs(x['price'] - peg_price))
                    peg_index = current_tick_data.index(peg_tick)
                    start_idx = max(0, peg_index - 20)
                    end_idx = min(len(current_tick_data), peg_index + 21)
                    sampled_current = current_tick_data[start_idx:end_idx]
                else:
                    sampled_current = current_tick_data
                
                liquidity_values = [tick_data["liquidity"] for tick_data in sampled_current]
                
                # Create bars for this snapshot
                x_positions = [idx + i * bar_width - (len(sample_snapshots) * bar_width / 2) + bar_width/2 
                             for idx in tick_indices]
                
                bars = ax.bar(x_positions, liquidity_values, 
                            width=bar_width, 
                            alpha=0.7,
                            color=colors[i],
                            label=f"Minute {snapshot.minute}")
        
        # Improve the visualization
        ax.set_xlabel("Active Liquidity Ticks (Ordered by Price)")
        ax.set_ylabel("Liquidity per Tick ($)")
        
        # Add peg reference line based on pool type
        if "MOET:BTC" in pool_name:
            # Find the tick closest to the peg price (0.00001 BTC per MOET)
            peg_price = 0.00001
            peg_tick_index = min(range(len(sampled_ticks)), 
                               key=lambda i: abs(sampled_ticks[i]['price'] - peg_price))
            ax.axvline(x=peg_tick_index, color='red', linestyle='--', alpha=0.7, 
                      label="Correct Peg (1 BTC = 100k MOET)")
            concentration_pct = int(sample_snapshots[0].concentrated_pool.concentration * 100)
            ax.set_title(f"MOET:BTC LP Ticks - {concentration_pct}% Concentrated Around Peg")
        else:
            # Find the tick closest to the peg price (1.0)
            peg_price = 1.0
            peg_tick_index = min(range(len(sampled_ticks)), 
                               key=lambda i: abs(sampled_ticks[i]['price'] - peg_price))
            ax.axvline(x=peg_tick_index, color='red', linestyle='--', alpha=0.7, 
                      label="Perfect Peg (1:1)")
            concentration_pct = int(sample_snapshots[0].concentrated_pool.concentration * 100)
            remaining_pct = 100 - concentration_pct
            ax.set_title(f"MOET:Yield Token LP Ticks - {concentration_pct}% at Peg, {remaining_pct}% Distributed")
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Improve x-axis labeling
        num_ticks_display = min(10, len(sampled_ticks))
        tick_positions = np.linspace(0, len(sampled_ticks)-1, num_ticks_display, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"Tick {i}" for i in tick_positions], rotation=45)
    
    def _create_reserve_changes_chart(self, ax, snapshots: List[PoolSnapshot]):
        """Create chart showing pool reserve changes over time"""
        
        minutes = [s.minute for s in snapshots]
        moet_reserves = [s.moet_reserve for s in snapshots]
        btc_reserves = [s.btc_reserve for s in snapshots]
        
        ax.plot(minutes, moet_reserves, label="MOET Reserve", color='#2E8B57', linewidth=2)
        ax.plot(minutes, btc_reserves, label="BTC Reserve", color='#FF6B35', linewidth=2)
        
        # Mark trade events
        for snapshot in snapshots:
            if snapshot.trade_amount > 0:
                ax.axvline(x=snapshot.minute, color='red', alpha=0.5, linestyle=':')
                ax.annotate(f"${snapshot.trade_amount:,.0f}", 
                           xy=(snapshot.minute, max(moet_reserves + btc_reserves) * 0.9),
                           rotation=90, fontsize=8, alpha=0.7)
        
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Reserve Amount ($)")
        ax.set_title("Pool Reserve Changes")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_price_impact_chart(self, ax, snapshots: List[PoolSnapshot], pool_name: str):
        """Create chart showing price impact over time"""
        
        minutes = [s.minute for s in snapshots]
        prices = [s.price for s in snapshots]
        
        # Calculate price deviation from correct peg
        if "MOET:BTC" in pool_name:
            correct_peg = 0.00001  # BTC per MOET
            price_impacts = [(p - correct_peg) / correct_peg * 100 for p in prices]  # % deviation from correct peg
            peg_label = "Correct Peg (1 BTC = 100k MOET)"
        else:
            correct_peg = 1.0  # 1:1 for yield tokens
            price_impacts = [(p - correct_peg) / correct_peg * 100 for p in prices]  # % deviation from 1:1 peg
            peg_label = "Perfect Peg (1:1)"
        
        ax.plot(minutes, price_impacts, color='#4169E1', linewidth=2, marker='o', markersize=4)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label=peg_label)
        
        # Mark significant trades
        for snapshot in snapshots:
            if snapshot.trade_amount > 5000:  # Significant trades
                if "MOET:BTC" in pool_name:
                    impact = (snapshot.price - 0.00001) / 0.00001 * 100
                else:
                    impact = (snapshot.price - 1.0) / 1.0 * 100
                ax.annotate(f"${snapshot.trade_amount/1000:.0f}k", 
                           xy=(snapshot.minute, impact),
                           xytext=(5, 10), textcoords='offset points',
                           fontsize=8, alpha=0.8,
                           arrowprops=dict(arrowstyle='->', alpha=0.5))
        
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Price Deviation from Peg (%)")
        ax.set_title("Price Impact Timeline")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_concentration_efficiency_chart(self, ax, snapshots: List[PoolSnapshot], pool_name: str):
        """Create chart showing how trade volume moves price through the concentrated range"""
        
        minutes = [s.minute for s in snapshots]
        price_utilizations = []
        cumulative_volumes = []
        
        # Determine correct peg value based on pool type
        if "MOET:BTC" in pool_name:
            correct_peg = 0.00001  # BTC per MOET
        else:
            correct_peg = 1.0  # 1:1 for yield tokens
        
        for i, snapshot in enumerate(snapshots):
            if i == 0:
                price_utilizations.append(0)
                cumulative_volumes.append(0)
            else:
                # Calculate price-based utilization (how much of concentrated range we've moved through)
                price_change = abs(snapshot.price - correct_peg)
                util = min(100, (price_change / (snapshot.concentration_range / 2)) * 100)
                price_utilizations.append(util)
                
                # Track cumulative trade volume
                cumulative_volume = sum(s.trade_amount for s in snapshots[:i+1])
                cumulative_volumes.append(cumulative_volume)
        
        # Create dual y-axis plot
        ax2 = ax.twinx()
        
        # Plot price utilization over time
        line1 = ax.plot(minutes, price_utilizations, color='#2E8B57', linewidth=2, 
                       label="Price Range Utilization (%)")
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, 
                   label="Concentrated Range Exhausted")
        
        # Plot cumulative trade volume over time
        line2 = ax2.plot(minutes, cumulative_volumes, color='#4169E1', linewidth=2, 
                        linestyle='--', alpha=0.8, label="Cumulative Trade Volume ($)")
        
        # Find when we hit 100% utilization (range exhausted)
        exhaustion_point = None
        for i, util in enumerate(price_utilizations):
            if util >= 100 and exhaustion_point is None:
                exhaustion_point = (minutes[i], cumulative_volumes[i])
                break
        
        if exhaustion_point:
            ax.axvline(x=exhaustion_point[0], color='red', alpha=0.5, linestyle=':')
            ax.annotate(f"Range exhausted at ${exhaustion_point[1]:,.0f}", 
                       xy=exhaustion_point[0], xytext=(10, 10),
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Style the plot
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Price Range Utilization (%)", color='#2E8B57')
        ax2.set_ylabel("Cumulative Trade Volume ($)", color='#4169E1')
        ax.set_title(f"Concentrated Range Exhaustion Analysis ({snapshots[0].concentration_range*100:.0f}% range)")
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 120)
    
    def _get_concentrated_liquidity(self, pool: UniswapV3Pool, pool_name: str) -> float:
        """Get total liquidity in the concentrated range for a pool"""
        if not pool or not pool.positions:
            return 0.0
        
        # Import required functions for tick-price conversion
        from ..core.uniswap_v3_math import tick_to_sqrt_price_x96, Q96
        import math
        
        if "MOET:BTC" in pool_name:
            # For MOET:BTC, concentrated liquidity is around the peg price
            peg_price = 0.00001  # BTC per MOET
            
            # Convert peg price to tick for comparison
            sqrt_peg_price = math.sqrt(peg_price)
            peg_sqrt_price_x96 = int(sqrt_peg_price * Q96)
            
            # Find the tick closest to peg price
            from ..core.uniswap_v3_math import sqrt_price_x96_to_tick
            peg_tick = sqrt_price_x96_to_tick(peg_sqrt_price_x96)
            
            # Define concentrated range around peg (±100 ticks ≈ ±1%)
            concentrated_tick_range = 100
            concentrated_min_tick = peg_tick - concentrated_tick_range
            concentrated_max_tick = peg_tick + concentrated_tick_range
            
            concentrated_liquidity = 0.0
            for position in pool.positions:
                # Check if position overlaps with concentrated range
                if (position.tick_lower <= concentrated_max_tick and 
                    position.tick_upper >= concentrated_min_tick):
                    # Add the liquidity from this position
                    concentrated_liquidity += position.liquidity
            
            return concentrated_liquidity / 1e6  # Convert back to USD
        else:
            # For yield tokens, concentrated range is very tight around 1:1 peg
            peg_price = 1.0
            
            # Convert peg price to tick (tick 0 = price 1.0)
            peg_tick = 0
            
            # Define very tight concentrated range around peg (±10 ticks ≈ ±0.1%)
            concentrated_tick_range = 10
            concentrated_min_tick = peg_tick - concentrated_tick_range
            concentrated_max_tick = peg_tick + concentrated_tick_range
            
            concentrated_liquidity = 0.0
            for position in pool.positions:
                # Check if position overlaps with concentrated range
                if (position.tick_lower <= concentrated_max_tick and 
                    position.tick_upper >= concentrated_min_tick):
                    # Add the liquidity from this position
                    concentrated_liquidity += position.liquidity
            
            return concentrated_liquidity / 1e6  # Convert back to USD

    def _create_tick_comparison_chart(self, ax, tight_final: PoolSnapshot, conservative_final: PoolSnapshot, price_label: str):
        """Create tick comparison chart for concentration comparison with proper scaling"""
        
        if not tight_final.concentrated_pool or not conservative_final.concentrated_pool:
            ax.text(0.5, 0.5, "No concentrated liquidity data", 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Get tick data for both pools and sort by price
        tight_ticks = tight_final.concentrated_pool.get_tick_data_for_charts()
        conservative_ticks = conservative_final.concentrated_pool.get_tick_data_for_charts()
        
        tight_ticks.sort(key=lambda x: x['price'])
        conservative_ticks.sort(key=lambda x: x['price'])
        
        # Sample ticks for visualization (avoid overcrowding)
        max_ticks = 30
        if len(tight_ticks) > max_ticks:
            step = len(tight_ticks) // max_ticks
            tight_ticks = tight_ticks[::step]
            conservative_ticks = conservative_ticks[::step]
        
        # Extract data for plotting
        tight_liquidity = [tick_data["liquidity"] for tick_data in tight_ticks]
        conservative_liquidity = [tick_data["liquidity"] for tick_data in conservative_ticks]
        
        x = np.arange(len(tight_ticks))
        width = 0.35
        
        # Create bars with better styling
        bars1 = ax.bar(x - width/2, tight_liquidity, width, 
                      label=f"Tight ({tight_final.concentration_range*100:.0f}%)", 
                      alpha=0.8, color='#2E8B57')
        bars2 = ax.bar(x + width/2, conservative_liquidity, width, 
                      label=f"Conservative ({conservative_final.concentration_range*100:.0f}%)", 
                      alpha=0.8, color='#FF6B35')
        
        # Add peg reference
        peg_position = len(tight_ticks) // 2
        ax.axvline(x=peg_position, color='red', linestyle='--', alpha=0.7, 
                  label="Peg Position")
        
        ax.set_xlabel("Liquidity Ticks (Ordered by Price)")
        ax.set_ylabel("Liquidity per Tick ($)")
        ax.set_title("Final LP Curve Shape Comparison: Concentrated Ticks")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Improve x-axis labels
        num_ticks_display = min(8, len(tight_ticks))
        if num_ticks_display > 0:
            tick_positions = np.linspace(0, len(tight_ticks)-1, num_ticks_display, dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"Tick {i}" for i in tick_positions], rotation=45)


def create_pool_dynamics_summary(trackers: Dict[str, LPCurveTracker], charts_dir: Path) -> Path:
    """Create summary chart of all pool dynamics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Pool Dynamics Summary: All Concentration Levels", 
                 fontsize=16, fontweight='bold')
    
    colors = ['#2E8B57', '#FF6B35', '#4169E1', '#9370DB']
    
    # Determine if we have MOET:BTC pools to set correct peg
    has_moet_btc = any("MOET:BTC" in name for name in trackers.keys())
    
    for i, (pool_name, tracker) in enumerate(trackers.items()):
        color = colors[i % len(colors)]
        snapshots = tracker.get_snapshots()
        
        if not snapshots:
            continue
        
        minutes = [s.minute for s in snapshots]
        prices = [s.price for s in snapshots]
        moet_reserves = [s.moet_reserve for s in snapshots]
        btc_reserves = [s.btc_reserve for s in snapshots]
        
        # Price evolution
        axes[0, 0].plot(minutes, prices, color=color, linewidth=2, 
                       label=f"{pool_name} ({tracker.concentration_range*100:.0f}%)")
        
        # Reserve changes
        axes[0, 1].plot(minutes, moet_reserves, color=color, linewidth=2, 
                       linestyle='-', alpha=0.8, label=f"{pool_name} MOET")
        axes[1, 0].plot(minutes, btc_reserves, color=color, linewidth=2,
                       linestyle='--', alpha=0.8, label=f"{pool_name} BTC")
        
        # Liquidity changes
        liquidity = [s.liquidity for s in snapshots]
        axes[1, 1].plot(minutes, liquidity, color=color, linewidth=2,
                       label=f"{pool_name} Liquidity")
    
    # Style all subplots
    axes[0, 0].set_title("Price Evolution")
    axes[0, 0].set_xlabel("Time (minutes)")
    
    # Set correct y-axis label and peg line based on pool type
    if has_moet_btc:
        axes[0, 0].set_ylabel("Price (BTC per MOET)")
        correct_peg = 0.00001  # BTC per MOET
        axes[0, 0].axhline(y=correct_peg, color='red', linestyle='--', alpha=0.5, 
                           label=f"Correct Peg ({correct_peg})")
    else:
        axes[0, 0].set_ylabel("Price (MOET per Yield Token)")
        correct_peg = 1.0  # 1:1 for yield tokens
        axes[0, 0].axhline(y=correct_peg, color='red', linestyle='--', alpha=0.5, 
                           label=f"Correct Peg ({correct_peg})")
    
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title("MOET Reserve Changes")
    axes[0, 1].set_xlabel("Time (minutes)")
    axes[0, 1].set_ylabel("MOET Reserve ($)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title("BTC Reserve Changes")
    axes[1, 0].set_xlabel("Time (minutes)")
    axes[1, 0].set_ylabel("BTC Reserve ($)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title("Total Liquidity")
    axes[1, 1].set_xlabel("Time (minutes)")
    axes[1, 1].set_ylabel("Liquidity")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save chart
    chart_path = charts_dir / "pool_dynamics_summary.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_path
