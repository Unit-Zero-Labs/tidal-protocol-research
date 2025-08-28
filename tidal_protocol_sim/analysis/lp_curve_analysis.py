#!/usr/bin/env python3
"""
Liquidity Provider Curve Analysis

Tracks and visualizes how Uniswap v3 concentrated liquidity curves change over time
during High Tide rebalancing events, showing:
- LP curve shape evolution
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


class LPCurveTracker:
    """Tracks LP curve changes during simulation"""
    
    def __init__(self, initial_pool_size: float, concentration_range: float, pool_name: str = "Unknown"):
        self.initial_pool_size = initial_pool_size
        self.concentration_range = concentration_range
        self.pool_name = pool_name  # "MOET:BTC" or "MOET:Yield_Token"
        self.snapshots: List[PoolSnapshot] = []
        
        # Initialize with starting state
        self.snapshots.append(PoolSnapshot(
            minute=0,
            moet_reserve=initial_pool_size / 2,
            btc_reserve=initial_pool_size / 2,
            price=1.0,  # Initial 1:1 price
            liquidity=initial_pool_size / 2,
            concentration_range=concentration_range
        ))
    
    def record_snapshot(self, pool_state: Dict[str, float], minute: int, 
                       trade_amount: float = 0.0, trade_type: str = ""):
        """Record a pool state snapshot"""
        
        snapshot = PoolSnapshot(
            minute=minute,
            moet_reserve=pool_state.get("token0_reserve", pool_state.get("moet_reserve", 0)),
            btc_reserve=pool_state.get("token1_reserve", pool_state.get("btc_reserve", 0)),
            price=pool_state.get("price", 1.0),
            liquidity=pool_state.get("liquidity", 0),
            concentration_range=self.concentration_range,
            trade_amount=trade_amount,
            trade_type=trade_type
        )
        
        self.snapshots.append(snapshot)
    
    def get_snapshots(self) -> List[PoolSnapshot]:
        """Get all recorded snapshots"""
        return self.snapshots


class LPCurveAnalyzer:
    """Analyzes and visualizes LP curve evolution"""
    
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
    
    def calculate_lp_curve(self, snapshot: PoolSnapshot, price_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the LP curve shape for a given pool snapshot
        
        Returns:
            Tuple of (prices, liquidity_density)
        """
        
        current_price = snapshot.price
        concentration = snapshot.concentration_range
        
        # Create price range around current price
        min_price, max_price = price_range
        prices = np.linspace(min_price, max_price, 200)
        
        # Calculate liquidity density based on concentration
        # Higher concentration = more liquidity near current price
        liquidity_density = np.zeros_like(prices)
        
        for i, price in enumerate(prices):
            # Distance from current price (normalized)
            price_distance = abs(price - current_price) / current_price
            
            # Liquidity falls off based on concentration
            if price_distance <= concentration / 2:
                # Within concentrated range - high liquidity
                liquidity_density[i] = snapshot.liquidity * (1 - (price_distance / (concentration / 2)) * 0.7)
            else:
                # Outside concentrated range - low liquidity
                falloff_factor = max(0, 1 - ((price_distance - concentration / 2) / concentration) * 2)
                liquidity_density[i] = snapshot.liquidity * 0.3 * falloff_factor
        
        return prices, liquidity_density
    
    def create_lp_curve_evolution_chart(self, tracker: LPCurveTracker, charts_dir: Path) -> Path:
        """Create chart showing LP curve evolution over time"""
        
        snapshots = tracker.get_snapshots()
        if len(snapshots) < 2:
            # Not enough data for evolution chart
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        pool_display_name = tracker.pool_name.replace("_", " ")
        fig.suptitle(f"{pool_display_name} LP Curve Evolution During Rebalancing Events", 
                     fontsize=16, fontweight='bold')
        
        # Determine price range for all curves
        all_prices = [s.price for s in snapshots]
        min_price = min(all_prices) * 0.8
        max_price = max(all_prices) * 1.2
        price_range = (min_price, max_price)
        
        # Colors for different time periods
        colors = plt.cm.viridis(np.linspace(0, 1, len(snapshots)))
        
        # 1. LP Curve Shape Evolution
        for i, snapshot in enumerate(snapshots[::max(1, len(snapshots)//8)]):  # Sample snapshots
            prices, liquidity = self.calculate_lp_curve(snapshot, price_range)
            
            alpha = 0.3 + 0.7 * (i / len(snapshots))  # More recent = more opaque
            label = f"Minute {snapshot.minute}"
            if snapshot.trade_amount > 0:
                label += f" (${snapshot.trade_amount:,.0f} {snapshot.trade_type})"
            
            ax1.plot(prices, liquidity, color=colors[i], alpha=alpha, 
                    linewidth=2, label=label if i % 2 == 0 else "")
        
        ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label="Initial Peg")
        ax1.set_xlabel("Price (BTC per MOET)")
        ax1.set_ylabel("Liquidity Density")
        ax1.set_title("LP Curve Shape Evolution")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Pool Reserve Changes
        minutes = [s.minute for s in snapshots]
        moet_reserves = [s.moet_reserve for s in snapshots]
        btc_reserves = [s.btc_reserve for s in snapshots]
        
        ax2.plot(minutes, moet_reserves, label="MOET Reserve", color='#2E8B57', linewidth=2)
        ax2.plot(minutes, btc_reserves, label="BTC Reserve", color='#FF6B35', linewidth=2)
        
        # Mark trade events
        for snapshot in snapshots:
            if snapshot.trade_amount > 0:
                ax2.axvline(x=snapshot.minute, color='red', alpha=0.5, linestyle=':')
                ax2.annotate(f"${snapshot.trade_amount:,.0f}", 
                           xy=(snapshot.minute, max(moet_reserves + btc_reserves) * 0.9),
                           rotation=90, fontsize=8, alpha=0.7)
        
        ax2.set_xlabel("Time (minutes)")
        ax2.set_ylabel("Reserve Amount ($)")
        ax2.set_title("Pool Reserve Changes")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Price Impact Over Time
        prices = [s.price for s in snapshots]
        price_impacts = [(p - 1.0) / 1.0 * 100 for p in prices]  # % deviation from peg
        
        ax3.plot(minutes, price_impacts, color='#4169E1', linewidth=2, marker='o', markersize=4)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7, label="Perfect Peg")
        
        # Mark significant trades
        for snapshot in snapshots:
            if snapshot.trade_amount > 5000:  # Significant trades
                impact = (snapshot.price - 1.0) / 1.0 * 100
                ax3.annotate(f"${snapshot.trade_amount/1000:.0f}k", 
                           xy=(snapshot.minute, impact),
                           xytext=(5, 10), textcoords='offset points',
                           fontsize=8, alpha=0.8,
                           arrowprops=dict(arrowstyle='->', alpha=0.5))
        
        ax3.set_xlabel("Time (minutes)")
        ax3.set_ylabel("Price Deviation from Peg (%)")
        ax3.set_title("Price Impact Timeline")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Concentration Efficiency Analysis
        total_liquidity = [s.liquidity for s in snapshots]
        utilization_rates = []
        
        for i, snapshot in enumerate(snapshots):
            if i == 0:
                utilization_rates.append(0)
            else:
                # Calculate how much of the concentrated liquidity was actually used
                price_change = abs(snapshot.price - snapshots[0].price) / snapshots[0].price
                concentration_usage = min(1.0, price_change / (snapshot.concentration_range / 2))
                utilization_rates.append(concentration_usage * 100)
        
        ax4.fill_between(minutes, utilization_rates, alpha=0.6, color='#9370DB', 
                        label="Concentration Utilization")
        ax4.axhline(y=100, color='red', linestyle='--', alpha=0.7, 
                   label="Full Concentration Range")
        
        ax4.set_xlabel("Time (minutes)")
        ax4.set_ylabel("Concentration Utilization (%)")
        ax4.set_title(f"LP Concentration Efficiency ({tracker.concentration_range*100:.0f}% range)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 120)
        
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
        
        # Price range for comparison
        all_prices = [tight_final.price, conservative_final.price, 1.0]
        min_price = min(all_prices) * 0.7
        max_price = max(all_prices) * 1.3
        price_range = (min_price, max_price)
        
        # 1. LP Curve Shape Comparison
        tight_prices, tight_liquidity = self.calculate_lp_curve(tight_final, price_range)
        conservative_prices, conservative_liquidity = self.calculate_lp_curve(conservative_final, price_range)
        
        ax1.fill_between(tight_prices, tight_liquidity, alpha=0.6, color='#2E8B57', 
                        label=f"Tight ({tight_final.concentration_range*100:.0f}% concentration)")
        ax1.fill_between(conservative_prices, conservative_liquidity, alpha=0.6, color='#FF6B35',
                        label=f"Conservative ({conservative_final.concentration_range*100:.0f}% concentration)")
        
        ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label="Initial Peg")
        ax1.axvline(x=tight_final.price, color='#2E8B57', linestyle=':', alpha=0.8, label="Tight Final Price")
        ax1.axvline(x=conservative_final.price, color='#FF6B35', linestyle=':', alpha=0.8, label="Conservative Final Price")
        
        ax1.set_xlabel("Price (BTC per MOET)")
        ax1.set_ylabel("Liquidity Density")
        ax1.set_title("Final LP Curve Shape Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Trade Impact Comparison
        tight_snapshots = tight_tracker.get_snapshots()
        conservative_snapshots = conservative_tracker.get_snapshots()
        
        tight_minutes = [s.minute for s in tight_snapshots]
        conservative_minutes = [s.minute for s in conservative_snapshots]
        
        tight_impacts = [(s.price - 1.0) / 1.0 * 100 for s in tight_snapshots]
        conservative_impacts = [(s.price - 1.0) / 1.0 * 100 for s in conservative_snapshots]
        
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
            price_change = abs(snapshot.price - 1.0)
            util = min(100, (price_change / (snapshot.concentration_range / 2)) * 100)
            tight_utils.append(util)
        
        for snapshot in conservative_snapshots:
            price_change = abs(snapshot.price - 1.0)
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
        tight_final_impact = abs(tight_final.price - 1.0) / 1.0 * 100
        conservative_final_impact = abs(conservative_final.price - 1.0) / 1.0 * 100
        
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


def create_pool_dynamics_summary(trackers: Dict[str, LPCurveTracker], charts_dir: Path) -> Path:
    """Create summary chart of all pool dynamics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Pool Dynamics Summary: All Concentration Levels", 
                 fontsize=16, fontweight='bold')
    
    colors = ['#2E8B57', '#FF6B35', '#4169E1', '#9370DB']
    
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
    axes[0, 0].set_ylabel("Price (BTC per MOET)")
    axes[0, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
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
