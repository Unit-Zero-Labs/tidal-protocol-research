#!/usr/bin/env python3
"""
Streamlined Chart Generator

Creates a single, meaningful time-series chart showing liquidation simulation dynamics.
Focus on showing actual data insights rather than multiple empty charts.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from ..core.protocol import Asset


class ScenarioChartGenerator:
    """Generates one meaningful time-series chart per scenario"""
    
    def __init__(self):
        self._setup_styling()
    
    def _setup_styling(self):
        """Setup clean, professional chart styling"""
        plt.style.use('default')
        
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
    
    def generate_scenario_charts(
        self, 
        scenario_name: str, 
        results: Dict[str, Any], 
        charts_dir: Path
    ) -> List[Path]:
        """Generate single meaningful time-series chart showing simulation dynamics"""
        
        print(f"Generating time-series chart for: {scenario_name}")
        
        # Ensure charts directory exists
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            return self._create_simulation_time_series(results, charts_dir, scenario_name)
        except Exception as e:
            print(f"Warning: Failed to generate chart: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_simulation_time_series(self, results: Dict, charts_dir: Path, scenario_name: str) -> List[Path]:
        """Create single time-series chart showing key simulation dynamics"""
        
        print(f"Creating time-series chart for {scenario_name}...")
        
        # Extract time-series data from results - try multiple sources
        scenario_results = results.get("scenario_results", results)
        
        # For Monte Carlo results, look for sample data
        if not scenario_results.get("metrics_history") and "sample_scenario_results" in results:
            scenario_results = results["sample_scenario_results"]
            print("Using sample scenario results from Monte Carlo run")
        
        metrics_history = scenario_results.get("metrics_history", [])
        
        print(f"Found {len(metrics_history)} time-series data points")
        
        if not metrics_history:
            print("No time-series data found - cannot create meaningful chart")
            return []
        
        # Create single comprehensive chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{scenario_name.replace("_", " ")} - Simulation Dynamics Over Time', 
                    fontsize=16, fontweight='bold')
        
        steps = [m.get("step", i) for i, m in enumerate(metrics_history)]
        
        # Chart 1: Asset Prices (focus on stressed asset)
        self._plot_asset_prices(ax1, metrics_history, steps, scenario_name)
        
        # Chart 2: Health Factors and Liquidations
        self._plot_health_factors(ax2, metrics_history, steps)
        
        # Chart 3: Protocol Metrics
        self._plot_protocol_metrics(ax3, metrics_history, steps)
        
        # Chart 4: Agent Activity
        self._plot_agent_activity(ax4, scenario_results, steps)
        
        plt.tight_layout()
        
        chart_path = charts_dir / f"{scenario_name.lower()}_simulation_dynamics.png"
        fig.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Chart saved: {chart_path}")
        return [chart_path]
    
    def _plot_asset_prices(self, ax, metrics_history, steps, scenario_name):
        """Plot asset price evolution during stress scenario"""
        
        # Determine which asset to focus on based on scenario
        focus_asset = "ETH"
        initial_price = 4400
        
        if "MOET" in scenario_name:
            focus_asset = "MOET"
            initial_price = 1.0
        elif "BTC" in scenario_name:
            focus_asset = "BTC"
            initial_price = 118000
        elif "FLOW" in scenario_name:
            focus_asset = "FLOW"
            initial_price = 0.4
        
        prices = []
        for m in metrics_history:
            asset_prices = m.get("asset_prices", {})
            # Handle both string keys and Asset enum keys
            price = asset_prices.get(focus_asset, asset_prices.get(f"Asset.{focus_asset}", initial_price))
            prices.append(price)
        
        if prices and any(p != initial_price for p in prices):
            ax.plot(steps, prices, linewidth=3, color='#E74C3C', label=f'{focus_asset} Price')
            ax.axhline(y=initial_price, color='gray', linestyle='--', alpha=0.7, label='Initial Price')
            
            # Add stress level indicator
            if focus_asset == "ETH" and initial_price == 4400:
                ax.axhline(y=initial_price*0.6, color='red', linestyle=':', alpha=0.7, label='40% Drop')
            elif focus_asset == "MOET":
                ax.axhline(y=0.95, color='red', linestyle=':', alpha=0.7, label='Depeg Level')
        else:
            # Show all major asset prices if no single asset dominates
            colors = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12']
            assets = ['ETH', 'BTC', 'FLOW', 'USDC']
            
            for i, asset in enumerate(assets):
                asset_prices = []
                for m in metrics_history:
                    prices_dict = m.get("asset_prices", {})
                    price = prices_dict.get(asset, prices_dict.get(f"Asset.{asset}", 0))
                    asset_prices.append(price)
                
                if asset_prices and max(asset_prices) > 0:
                    # Normalize prices for comparison
                    normalized = np.array(asset_prices) / asset_prices[0] * 100
                    ax.plot(steps, normalized, linewidth=2, color=colors[i], label=f'{asset} (normalized)')
        
        ax.set_title(f'{focus_asset} Price Evolution')
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_health_factors(self, ax, metrics_history, steps):
        """Plot health factors and liquidation risk"""
        
        avg_health_factors = []
        liquidatable_agents = []
        
        for m in metrics_history:
            hf = m.get("average_health_factor", 1.0)
            # Handle infinite health factors
            if hf == float('inf') or hf > 10:
                hf = 10  # Cap for visualization
            avg_health_factors.append(hf)
            liquidatable_agents.append(m.get("liquidatable_agents", 0))
        
        # Primary y-axis: Health factors
        line1 = ax.plot(steps, avg_health_factors, linewidth=3, color='#27AE60', label='Avg Health Factor')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Liquidation Threshold')
        ax.axhline(y=1.5, color='orange', linestyle=':', alpha=0.5, label='Safe Level')
        
        # Secondary y-axis: Liquidatable agents
        ax2 = ax.twinx()
        line2 = ax2.plot(steps, liquidatable_agents, linewidth=2, color='#F39C12', 
                        label='Liquidatable Agents', alpha=0.8)
        
        ax.set_title('Health Factors & Liquidation Risk')
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Average Health Factor', color='#27AE60')
        ax2.set_ylabel('Liquidatable Agents', color='#F39C12')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines] + ['Liquidation Threshold', 'Safe Level']
        ax.legend(lines + [ax.lines[0], ax.lines[1]], labels, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    def _plot_protocol_metrics(self, ax, metrics_history, steps):
        """Plot protocol treasury and utilization"""
        
        treasury_values = []
        total_borrowed = []
        total_supplied = []
        
        for m in metrics_history:
            treasury_values.append(m.get("protocol_treasury", 0))
            total_borrowed.append(m.get("total_borrowed", 0))
            total_supplied.append(m.get("total_supplied", 0))
        
        # Primary y-axis: Treasury
        line1 = ax.plot(steps, treasury_values, linewidth=3, color='#2ECC71', label='Protocol Treasury')
        
        # Secondary y-axis: Borrowed/Supplied ratio
        ax2 = ax.twinx()
        utilization = []
        for i in range(len(total_borrowed)):
            if total_supplied[i] > 0:
                util = (total_borrowed[i] / total_supplied[i]) * 100
                utilization.append(min(util, 100))  # Cap at 100%
            else:
                utilization.append(0)
        
        line2 = ax2.plot(steps, utilization, linewidth=2, color='#3498DB', 
                        label='Utilization %', alpha=0.8)
        
        ax.set_title('Protocol Health & Utilization')
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Treasury Balance ($)', color='#2ECC71')
        ax2.set_ylabel('Utilization (%)', color='#3498DB')
        
        # Format treasury axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True, alpha=0.3)
    
    def _plot_agent_activity(self, ax, scenario_results, steps):
        """Plot agent activity and liquidation events"""
        
        # Get liquidation events
        liquidation_events = scenario_results.get("liquidation_events", [])
        trade_events = scenario_results.get("trade_events", [])
        
        if liquidation_events:
            # Plot liquidation timeline
            liquidation_steps = [event.get("step", 0) for event in liquidation_events]
            liquidation_amounts = [event.get("repay_amount", 0) for event in liquidation_events]
            
            ax.scatter(liquidation_steps, liquidation_amounts, color='#E74C3C', 
                      s=50, alpha=0.7, label='Liquidations')
            ax.set_ylabel('Liquidation Amount ($)')
            ax.set_title('Liquidation Events Timeline')
        else:
            # Plot trade activity if no liquidations
            if trade_events:
                # Group trades by time windows
                max_step = max(steps) if steps else 100
                time_windows = np.arange(0, max_step + 10, max(1, max_step // 20))
                trade_counts = np.histogram([event.get("step", 0) for event in trade_events], 
                                          bins=time_windows)[0]
                
                ax.bar(time_windows[:-1], trade_counts, width=max_step//25, alpha=0.7, 
                      color='#9B59B6', label='Trade Activity')
                ax.set_ylabel('Number of Trades')
                ax.set_title('Trading Activity Over Time')
            else:
                # Show agent action counts from metrics
                ax.text(0.5, 0.5, 'No liquidation or trade events\nto display', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                ax.set_title('Agent Activity')
        
        ax.set_xlabel('Simulation Step')
        ax.legend()
        ax.grid(True, alpha=0.3)