#!/usr/bin/env python3
"""
Create Missing Comparison Charts
===============================

Creates the net_apy_analysis_comparison.png and yield_strategy_comparison.png charts
using available data from the simulation results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

class MissingComparisonCharts:
    def __init__(self):
        self.base_dir = Path("tidal_protocol_sim/results")
        self.output_dir = self.base_dir / "Leverage_Frequency_Comparison"
        self.charts_dir = self.output_dir / "charts"
        
        # Data storage
        self.data = {}
        self.frequencies = ["3min", "5min", "10min"]
        
        # Colors for each frequency
        self.colors = {
            "3min": "#FF6B6B",   # Red
            "5min": "#4ECDC4",   # Teal  
            "10min": "#45B7D1"   # Blue
        }
        
    def load_data(self):
        """Load simulation data from all three frequency runs"""
        print("📊 Loading simulation data...")
        
        directory_map = {
            "3min": "Full_Year_2024_BTC_Simulation_3min_leverage",
            "5min": "Full_Year_2024_BTC_Simulation_5min_leverage", 
            "10min": "Full_Year_2024_BTC_Simulation_10min_leverage"
        }
        
        for freq, dir_name in directory_map.items():
            dir_path = self.base_dir / dir_name
            
            if not dir_path.exists():
                continue
                
            # Find JSON file
            json_files = list(dir_path.glob("*.json"))
            if not json_files:
                continue
                
            json_file = json_files[0]
            print(f"   📄 Loading {freq}: {json_file.name}")
            
            with open(json_file, 'r') as f:
                self.data[freq] = json.load(f)
                
        print(f"✅ Loaded data for {len(self.data)} frequencies")
        
    def create_net_apy_analysis_comparison(self):
        """Create Net APY Analysis comparison chart"""
        print("📈 Creating Net APY Analysis comparison...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Net APY Analysis - Leverage Frequency Comparison', fontsize=16, fontweight='bold')
        
        # Agent Health Factor Evolution
        ax1.set_title('Average Agent Health Factor Over Time', fontweight='bold')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Health Factor')
        ax1.grid(True, alpha=0.3)
        
        for freq in self.frequencies:
            if freq not in self.data:
                continue
                
            sim_results = self.data[freq].get('simulation_results', {})
            health_history = sim_results.get('agent_health_history', [])
            
            if health_history:
                # Sample every 24 hours for readability
                sample_indices = list(range(0, len(health_history), 1440))
                sampled_hf = []
                hours = []
                
                for i in sample_indices:
                    if i < len(health_history):
                        health_entry = health_history[i]
                        if isinstance(health_entry, dict):
                            # Calculate average health factor across all agents
                            agent_hfs = []
                            for agent_id, hf in health_entry.items():
                                if isinstance(hf, (int, float)) and hf > 0:
                                    agent_hfs.append(hf)
                            
                            if agent_hfs:
                                avg_hf = np.mean(agent_hfs)
                                sampled_hf.append(avg_hf)
                                hours.append(i / 60)
                
                if sampled_hf:
                    ax1.plot(hours, sampled_hf, label=f'{freq} leverage checks', 
                            color=self.colors[freq], linewidth=2, alpha=0.8)
        
        ax1.legend()
        
        # BTC Price vs Agent Performance
        ax2.set_title('BTC Price Evolution', fontweight='bold')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('BTC Price ($)')
        ax2.grid(True, alpha=0.3)
        
        # Use BTC price from one simulation (should be identical across all)
        for freq in self.frequencies:
            if freq not in self.data:
                continue
                
            sim_results = self.data[freq].get('simulation_results', {})
            btc_history = sim_results.get('btc_price_history', [])
            
            if btc_history:
                # Sample every 24 hours
                sample_indices = list(range(0, len(btc_history), 1440))
                sampled_prices = []
                hours = []
                
                for i in sample_indices:
                    if i < len(btc_history):
                        price_entry = btc_history[i]
                        if isinstance(price_entry, dict) and 'price' in price_entry:
                            sampled_prices.append(price_entry['price'])
                            hours.append(i / 60)
                        elif isinstance(price_entry, (int, float)):
                            sampled_prices.append(price_entry)
                            hours.append(i / 60)
                
                if sampled_prices:
                    ax2.plot(hours, sampled_prices, label=f'BTC Price', 
                            color='orange', linewidth=2, alpha=0.8)
                    break  # Only plot once since BTC price is the same for all
        
        ax2.legend()
        
        # Yield Token Activity
        ax3.set_title('Yield Token Activity Comparison', fontweight='bold')
        frequencies = []
        yt_activity = []
        colors = []
        
        for freq in self.frequencies:
            if freq not in self.data:
                continue
                
            sim_results = self.data[freq].get('simulation_results', {})
            yt_trades = sim_results.get('yield_token_trades', [])
            
            frequencies.append(f'{freq}')
            yt_activity.append(len(yt_trades))
            colors.append(self.colors[freq])
        
        bars = ax3.bar(frequencies, yt_activity, color=colors, alpha=0.7)
        ax3.set_ylabel('YT Trade Events')
        
        # Add value labels on bars
        for bar, count in zip(bars, yt_activity):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(yt_activity)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Performance Summary
        ax4.set_title('APY Performance Summary', fontweight='bold')
        ax4.axis('off')
        
        summary_text = []
        for freq in self.frequencies:
            if freq not in self.data:
                continue
                
            sim_results = self.data[freq].get('simulation_results', {})
            
            # Calculate approximate APY metrics
            survival_stats = sim_results.get('survival_statistics', {})
            survival_rate = survival_stats.get('survival_rate', 0) * 100
            
            # Get cost analysis
            cost_analysis = sim_results.get('cost_analysis', {})
            total_slippage = cost_analysis.get('total_slippage_cost', 0)
            
            # Get final MOET rate as proxy for yield
            moet_state = sim_results.get('moet_system_state', {})
            tracking = moet_state.get('tracking_data', {})
            final_moet_rate = 0
            
            if 'moet_rate_history' in tracking:
                rate_data = tracking['moet_rate_history']
                if rate_data:
                    last_entry = rate_data[-1]
                    if isinstance(last_entry, dict) and 'rate' in last_entry:
                        final_moet_rate = last_entry['rate'] * 100
                    elif isinstance(last_entry, (int, float)):
                        final_moet_rate = last_entry * 100
            
            summary_text.append(f"{freq} Leverage Performance:")
            summary_text.append(f"  • Agent Survival: {survival_rate:.1f}%")
            summary_text.append(f"  • Final MOET Rate: {final_moet_rate:.4f}%")
            summary_text.append(f"  • Total Slippage: ${total_slippage:,.2f}")
            summary_text.append(f"  • YT Trades: {len(sim_results.get('yield_token_trades', [])):,}")
            summary_text.append("")
        
        ax4.text(0.05, 0.95, '\n'.join(summary_text), transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        output_path = self.charts_dir / "net_apy_analysis_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")
        
    def create_yield_strategy_comparison(self):
        """Create Yield Strategy comparison chart"""
        print("📈 Creating Yield Strategy comparison...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Yield Strategy Comparison - Leverage Frequency Analysis', fontsize=16, fontweight='bold')
        
        # Pool State Evolution
        ax1.set_title('Pool State Evolution', fontweight='bold')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Pool Value ($)')
        ax1.grid(True, alpha=0.3)
        
        for freq in self.frequencies:
            if freq not in self.data:
                continue
                
            sim_results = self.data[freq].get('simulation_results', {})
            pool_snapshots = sim_results.get('pool_state_snapshots', [])
            
            if pool_snapshots:
                # Sample every 24 hours
                sample_indices = list(range(0, len(pool_snapshots), 1440))
                sampled_values = []
                hours = []
                
                for i in sample_indices:
                    if i < len(pool_snapshots):
                        snapshot = pool_snapshots[i]
                        if isinstance(snapshot, dict):
                            # Calculate total pool value
                            moet_balance = snapshot.get('moet_balance', 0)
                            yt_balance = snapshot.get('yt_balance', 0)
                            yt_price = snapshot.get('yt_price', 1)
                            
                            total_value = moet_balance + (yt_balance * yt_price)
                            sampled_values.append(total_value)
                            hours.append(i / 60)
                
                if sampled_values:
                    ax1.plot(hours, sampled_values, label=f'{freq} leverage checks', 
                            color=self.colors[freq], linewidth=2, alpha=0.8)
        
        ax1.legend()
        
        # Strategy Efficiency Metrics
        ax2.set_title('Strategy Efficiency Comparison', fontweight='bold')
        frequencies = []
        efficiency_scores = []
        colors = []
        
        for freq in self.frequencies:
            if freq not in self.data:
                continue
                
            sim_results = self.data[freq].get('simulation_results', {})
            
            # Calculate efficiency as survival rate / total costs
            survival_stats = sim_results.get('survival_statistics', {})
            survival_rate = survival_stats.get('survival_rate', 0)
            
            cost_analysis = sim_results.get('cost_analysis', {})
            total_slippage = cost_analysis.get('total_slippage_cost', 1)  # Avoid division by zero
            
            # Simple efficiency metric
            efficiency = survival_rate / max(total_slippage / 1000, 0.001)  # Normalize slippage
            
            frequencies.append(f'{freq}')
            efficiency_scores.append(efficiency)
            colors.append(self.colors[freq])
        
        bars = ax2.bar(frequencies, efficiency_scores, color=colors, alpha=0.7)
        ax2.set_ylabel('Efficiency Score')
        
        # Add value labels on bars
        for bar, score in zip(bars, efficiency_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(efficiency_scores)*0.01,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Rebalancing Frequency Analysis
        ax3.set_title('Rebalancing Frequency Analysis', fontweight='bold')
        frequencies = []
        rebalance_rates = []
        colors = []
        
        for freq in self.frequencies:
            if freq not in self.data:
                continue
                
            sim_results = self.data[freq].get('simulation_results', {})
            rebalancing_events = sim_results.get('rebalancing_events', [])
            
            # Calculate rebalances per day (assuming 1 year = 8760 hours)
            total_hours = 8760
            rebalances_per_day = (len(rebalancing_events) / total_hours) * 24
            
            frequencies.append(f'{freq}')
            rebalance_rates.append(rebalances_per_day)
            colors.append(self.colors[freq])
        
        bars = ax3.bar(frequencies, rebalance_rates, color=colors, alpha=0.7)
        ax3.set_ylabel('Rebalances per Day')
        
        # Add value labels on bars
        for bar, rate in zip(bars, rebalance_rates):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(rebalance_rates)*0.01,
                    f'{rate:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Strategy Comparison Summary
        ax4.set_title('Strategy Performance Summary', fontweight='bold')
        ax4.axis('off')
        
        summary_text = []
        for freq in self.frequencies:
            if freq not in self.data:
                continue
                
            sim_results = self.data[freq].get('simulation_results', {})
            
            # Get key metrics
            survival_stats = sim_results.get('survival_statistics', {})
            survival_rate = survival_stats.get('survival_rate', 0) * 100
            
            rebalancing_events = sim_results.get('rebalancing_events', [])
            pool_activity = sim_results.get('pool_rebalancing_activity', [])
            
            cost_analysis = sim_results.get('cost_analysis', {})
            total_slippage = cost_analysis.get('total_slippage_cost', 0)
            
            summary_text.append(f"{freq} Strategy Results:")
            summary_text.append(f"  • Survival Rate: {survival_rate:.1f}%")
            summary_text.append(f"  • Agent Rebalances: {len(rebalancing_events):,}")
            summary_text.append(f"  • Pool Rebalances: {len(pool_activity):,}")
            summary_text.append(f"  • Total Slippage: ${total_slippage:,.2f}")
            summary_text.append("")
        
        ax4.text(0.05, 0.95, '\n'.join(summary_text), transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        output_path = self.charts_dir / "yield_strategy_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")
        
    def run_analysis(self):
        """Run the missing charts analysis"""
        print("🚀 Creating missing comparison charts...")
        
        # Load data
        self.load_data()
        
        if len(self.data) == 0:
            print("❌ No data loaded. Cannot proceed.")
            return
        
        # Create missing charts
        self.create_net_apy_analysis_comparison()
        self.create_yield_strategy_comparison()
        
        print(f"\n✅ Missing charts created successfully!")
        print(f"📁 Charts saved to: {self.charts_dir}")

if __name__ == "__main__":
    charts = MissingComparisonCharts()
    charts.run_analysis()

