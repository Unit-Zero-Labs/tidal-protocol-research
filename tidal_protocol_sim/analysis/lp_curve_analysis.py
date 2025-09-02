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

# Import the concentrated liquidity system from uniswap_v3_math
from ..core.uniswap_v3_math import UniswapV3Pool, LiquidityBin


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
            self.concentrated_pool = create_moet_btc_pool(initial_pool_size, btc_price, concentration_range)
        else:
            from ..core.uniswap_v3_math import create_yield_token_pool
            self.concentrated_pool = create_yield_token_pool(initial_pool_size, btc_price, concentration_range)
        
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
        
        # Calculate correct price if not provided
        price = pool_state.get("price")
        if price is None:
            if "MOET:BTC" in self.pool_name:
                # For MOET:BTC pool, calculate price as BTC per MOET
                # token0_reserve = MOET (in USD), token1_reserve = BTC (in USD)
                moet_reserve_usd = pool_state.get("token0_reserve", pool_state.get("moet_reserve", 0))
                btc_reserve_usd = pool_state.get("token1_reserve", pool_state.get("btc_reserve", 0))
                
                if moet_reserve_usd > 0 and btc_reserve_usd > 0:
                    # Convert USD reserves to actual token amounts
                    moet_tokens = moet_reserve_usd  # MOET is 1:1 with USD
                    btc_tokens = btc_reserve_usd / self.btc_price  # Convert USD to BTC tokens
                    price = btc_tokens / moet_tokens  # BTC per MOET
                else:
                    price = 0.00001  # Default: 1 BTC = 100,000 MOET
            else:
                # For yield token pool, maintain 1:1 with MOET
                price = 1.0
        
        # Update the concentrated liquidity pool if there was a trade
        if trade_amount > 0 and self.concentrated_pool:
            # Track cumulative trade volume for utilization calculation
            self.cumulative_trade_volume += trade_amount
            
            # Simulate price impact on the concentrated liquidity
            trade_direction = "sell" if trade_type == "rebalance" else "buy"
            impact = self.concentrated_pool.simulate_price_impact(trade_amount, trade_direction)
            
            # Update the pool's liquidity distribution based on the trade
            if impact["price_impact"] > 0:
                self.concentrated_pool.update_liquidity_distribution(impact["price_impact"])
        
        snapshot = PoolSnapshot(
            minute=minute,
            moet_reserve=pool_state.get("token0_reserve", pool_state.get("moet_reserve", 0)),
            btc_reserve=pool_state.get("token1_reserve", pool_state.get("btc_reserve", 0)),
            price=price,
            liquidity=pool_state.get("liquidity", 0),
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
        
        # 1. LP Curve Shape Evolution with Discrete Bins (Bar Chart)
        self._create_bin_evolution_chart(ax1, snapshots, tracker.pool_name)
        
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
        
        # 1. LP Curve Shape Comparison with Discrete Bins
        self._create_bin_comparison_chart(ax1, tight_final, conservative_final, price_label)
        
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
    
    def _create_bin_evolution_chart(self, ax, snapshots: List[PoolSnapshot], pool_name: str):
        """Create bar chart showing liquidity bin evolution over time with proper concentration visualization"""
        
        # Sample snapshots for clear visualization
        sample_snapshots = snapshots[::max(1, len(snapshots)//6)]  # Show 6 snapshots max
        
        first_snapshot = sample_snapshots[0]
        if not first_snapshot.concentrated_pool:
            ax.text(0.5, 0.5, "No concentrated liquidity data", 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Get all bins (not just active ones) to show the full distribution
        bin_data = first_snapshot.concentrated_pool.get_bin_data_for_charts()
        
        if not bin_data:
            ax.text(0.5, 0.5, "No liquidity bin data", 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Sort bins by price to show concentration properly
        bin_data.sort(key=lambda x: x['price'])
        
        # For proper visualization, show all bins but limit to reasonable number
        total_bins = len(bin_data)
        if total_bins > 100:
            # For very large bin counts, show key bins: peg bin + surrounding bins
            peg_price = 0.00001 if "MOET:BTC" in pool_name else 1.0
            
            # Find peg bin
            peg_bin = min(bin_data, key=lambda x: abs(x['price'] - peg_price))
            peg_index = bin_data.index(peg_bin)
            
            # Show peg bin + 20 bins on each side
            start_idx = max(0, peg_index - 20)
            end_idx = min(len(bin_data), peg_index + 21)
            sampled_bins = bin_data[start_idx:end_idx]
        else:
            sampled_bins = bin_data
        
        bin_indices = range(len(sampled_bins))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sample_snapshots)))
        
        # Create grouped bar chart
        bar_width = 0.8 / len(sample_snapshots)
        
        for i, snapshot in enumerate(sample_snapshots):
            if snapshot.concentrated_pool:
                # Get updated bin data for this snapshot
                current_bin_data = snapshot.concentrated_pool.get_bin_data_for_charts()
                current_bin_data.sort(key=lambda x: x['price'])
                
                # Use the same bin selection as the reference
                if total_bins > 100:
                    # Use the same range as the reference bins
                    peg_price = 0.00001 if "MOET:BTC" in pool_name else 1.0
                    peg_bin = min(current_bin_data, key=lambda x: abs(x['price'] - peg_price))
                    peg_index = current_bin_data.index(peg_bin)
                    start_idx = max(0, peg_index - 20)
                    end_idx = min(len(current_bin_data), peg_index + 21)
                    sampled_current = current_bin_data[start_idx:end_idx]
                else:
                    sampled_current = current_bin_data
                
                liquidity_values = [bin_data["liquidity"] for bin_data in sampled_current]
                
                # Create bars for this snapshot
                x_positions = [idx + i * bar_width - (len(sample_snapshots) * bar_width / 2) + bar_width/2 
                             for idx in bin_indices]
                
                bars = ax.bar(x_positions, liquidity_values, 
                            width=bar_width, 
                            alpha=0.7,
                            color=colors[i],
                            label=f"Minute {snapshot.minute}")
        
        # Improve the visualization
        ax.set_xlabel("Active Liquidity Bins (Ordered by Price)")
        ax.set_ylabel("Liquidity per Bin ($)")
        
        # Add peg reference line based on pool type
        if "MOET:BTC" in pool_name:
            # Find the bin closest to the peg price (0.00001 BTC per MOET)
            peg_price = 0.00001
            peg_bin_index = min(range(len(sampled_bins)), 
                               key=lambda i: abs(sampled_bins[i]['price'] - peg_price))
            ax.axvline(x=peg_bin_index, color='red', linestyle='--', alpha=0.7, 
                      label="Correct Peg (1 BTC = 100k MOET)")
            concentration_pct = int(sample_snapshots[0].concentrated_pool.concentration * 100)
            ax.set_title(f"MOET:BTC LP Bins - {concentration_pct}% in Single Peg Bin, $100k in 100bp increments")
        else:
            # Find the bin closest to the peg price (1.0)
            peg_price = 1.0
            peg_bin_index = min(range(len(sampled_bins)), 
                               key=lambda i: abs(sampled_bins[i]['price'] - peg_price))
            ax.axvline(x=peg_bin_index, color='red', linestyle='--', alpha=0.7, 
                      label="Perfect Peg (1:1)")
            concentration_pct = int(sample_snapshots[0].concentrated_pool.concentration * 100)
            remaining_pct = 100 - concentration_pct
            ax.set_title(f"MOET:Yield Token LP Bins - {concentration_pct}% at Peg, {remaining_pct}% in 1bp increments")
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Improve x-axis labeling
        num_ticks = min(10, len(sampled_bins))
        tick_positions = np.linspace(0, len(sampled_bins)-1, num_ticks, dtype=int)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"Bin {i}" for i in tick_positions], rotation=45)
    
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
        """Create chart showing concentration efficiency over time"""
        
        minutes = [s.minute for s in snapshots]
        utilization_rates = []
        
        for i, snapshot in enumerate(snapshots):
            if i == 0:
                utilization_rates.append(0)
            else:
                # Calculate concentration efficiency based on cumulative trade volume
                # This represents how much of the concentrated liquidity has been utilized
                if hasattr(snapshots[0], 'concentrated_pool') and snapshots[0].concentrated_pool:
                    # Get initial concentrated liquidity
                    initial_concentrated_liquidity = self._get_concentrated_liquidity(snapshots[0].concentrated_pool, pool_name)
                    
                    # Calculate cumulative trade volume up to this point
                    cumulative_volume = 0.0
                    for j in range(i + 1):  # Include current snapshot
                        if j < len(snapshots) and hasattr(snapshots[j], 'trade_amount'):
                            cumulative_volume += snapshots[j].trade_amount
                    
                    # Calculate utilization as percentage of concentrated liquidity that has been traded
                    if initial_concentrated_liquidity > 0:
                        # Assume that trades consume liquidity from concentrated range first
                        # Scale factor to make utilization visible (concentrated liquidity is typically much larger than individual trades)
                        utilization_rate = min(100.0, (cumulative_volume / initial_concentrated_liquidity) * 100 * 2)
                    else:
                        utilization_rate = 0.0
                else:
                    # Fallback: calculate based on trade activity
                    if snapshot.trade_amount > 0:
                        # Estimate utilization based on trade size relative to pool size
                        pool_size = snapshot.moet_reserve + snapshot.btc_reserve
                        utilization_rate = min(100.0, (snapshot.trade_amount / pool_size) * 100 * 10)  # Scale factor
                    else:
                        utilization_rate = 0.0
                
                utilization_rates.append(utilization_rate)
        
        ax.fill_between(minutes, utilization_rates, alpha=0.6, color='#9370DB', 
                        label="Concentration Utilization")
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, 
                   label="Full Concentration Range")
        
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Concentration Utilization (%)")
        ax.set_title(f"LP Concentration Efficiency ({snapshots[0].concentration_range*100:.0f}% range)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 120)
    
    def _get_concentrated_liquidity(self, pool: UniswapV3Pool, pool_name: str) -> float:
        """Get total liquidity in the concentrated range for a pool"""
        if not pool or not pool.bins:
            return 0.0
        
        if "MOET:BTC" in pool_name:
            # For MOET:BTC, concentrated liquidity is the SINGLE peg bin
            peg_price = 0.00001  # BTC per MOET
            # Find the bin at the exact peg price
            for bin in pool.bins:
                if bin.is_active and abs(bin.price - peg_price) < 1e-8:  # Very close to exact peg
                    return bin.liquidity
            return 0.0
        else:
            # For yield tokens, concentrated range is tight peg
            peg_price = 1.0
            concentrated_min = peg_price * 0.999999
            concentrated_max = peg_price * 1.000001
            
            concentrated_liquidity = 0.0
            for bin in pool.bins:
                if bin.is_active and concentrated_min <= bin.price <= concentrated_max:
                    concentrated_liquidity += bin.liquidity
            
            return concentrated_liquidity

    def _create_bin_comparison_chart(self, ax, tight_final: PoolSnapshot, conservative_final: PoolSnapshot, price_label: str):
        """Create bin comparison chart for concentration comparison with proper scaling"""
        
        if not tight_final.concentrated_pool or not conservative_final.concentrated_pool:
            ax.text(0.5, 0.5, "No concentrated liquidity data", 
                   transform=ax.transAxes, ha='center', va='center')
            return
        
        # Get bin data for both pools and sort by price
        tight_bins = tight_final.concentrated_pool.get_bin_data_for_charts()
        conservative_bins = conservative_final.concentrated_pool.get_bin_data_for_charts()
        
        tight_bins.sort(key=lambda x: x['price'])
        conservative_bins.sort(key=lambda x: x['price'])
        
        # Sample bins for visualization (avoid overcrowding)
        max_bins = 30
        if len(tight_bins) > max_bins:
            step = len(tight_bins) // max_bins
            tight_bins = tight_bins[::step]
            conservative_bins = conservative_bins[::step]
        
        # Extract data for plotting
        tight_liquidity = [bin_data["liquidity"] for bin_data in tight_bins]
        conservative_liquidity = [bin_data["liquidity"] for bin_data in conservative_bins]
        
        x = np.arange(len(tight_bins))
        width = 0.35
        
        # Create bars with better styling
        bars1 = ax.bar(x - width/2, tight_liquidity, width, 
                      label=f"Tight ({tight_final.concentration_range*100:.0f}%)", 
                      alpha=0.8, color='#2E8B57')
        bars2 = ax.bar(x + width/2, conservative_liquidity, width, 
                      label=f"Conservative ({conservative_final.concentration_range*100:.0f}%)", 
                      alpha=0.8, color='#FF6B35')
        
        # Add peg reference
        peg_position = len(tight_bins) // 2
        ax.axvline(x=peg_position, color='red', linestyle='--', alpha=0.7, 
                  label="Peg Position")
        
        ax.set_xlabel("Liquidity Bins (Ordered by Price)")
        ax.set_ylabel("Liquidity per Bin ($)")
        ax.set_title("Final LP Curve Shape Comparison: Concentrated Bins")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Improve x-axis labels
        num_ticks = min(8, len(tight_bins))
        if num_ticks > 0:
            tick_positions = np.linspace(0, len(tight_bins)-1, num_ticks, dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([f"Bin {i}" for i in tick_positions], rotation=45)


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
