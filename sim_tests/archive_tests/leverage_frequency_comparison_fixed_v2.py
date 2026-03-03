#!/usr/bin/env python3
"""
Fixed Leverage Frequency Comparison Analysis v2
===============================================

Creates working comparison charts using the actual data structures available in JSON files.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

class LeverageFrequencyComparisonFixed:
    def __init__(self):
        self.base_dir = Path("tidal_protocol_sim/results")
        self.output_dir = self.base_dir / "Leverage_Frequency_Comparison"
        self.output_dir.mkdir(exist_ok=True)
        
        # Chart output directory
        self.charts_dir = self.output_dir / "charts"
        self.charts_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.data = {}
        self.frequencies = ["3min", "5min", "10min"]
        
        # Colors for each frequency
        self.colors = {
            "3min": "#FF6B6B",  # Red
            "5min": "#4ECDC4",  # Teal  
            "10min": "#45B7D1"  # Blue
        }
        
    def load_data(self):
        """Load simulation data from all frequency directories"""
        print("🔄 Loading simulation data...")
        
        for freq in self.frequencies:
            dir_name = f"Full_Year_2024_BTC_Simulation_{freq}_leverage"
            data_dir = self.base_dir / dir_name
            
            if not data_dir.exists():
                print(f"❌ Directory not found: {data_dir}")
                continue
                
            # Find JSON file
            json_files = list(data_dir.glob("*.json"))
            if not json_files:
                print(f"❌ No JSON files found in {data_dir}")
                continue
                
            json_file = json_files[0]  # Take the first (should be only one)
            print(f"📄 Loading {freq}: {json_file.name}")
            
            with open(json_file, 'r') as f:
                self.data[freq] = json.load(f)
                
        print(f"✅ Loaded data for {len(self.data)} frequencies")
        
    def create_moet_system_comparison(self):
        """Create MOET system analysis comparison chart"""
        print("📊 Creating MOET system analysis comparison...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MOET System Analysis: Leverage Frequency Comparison', fontsize=16, fontweight='bold')
        
        for freq, data in self.data.items():
            sim_results = data.get('simulation_results', {})
            moet_state = sim_results.get('moet_system_state', {})
            tracking = moet_state.get('tracking_data', {})
            
            if not tracking:
                continue
                
            color = self.colors[freq]
            
            # MOET Interest Rate over time
            if 'moet_rate_history' in tracking:
                rate_data = tracking['moet_rate_history']
                if rate_data:
                    # Sample every 24 hours (1440 minutes) for readability
                    sample_indices = list(range(0, len(rate_data), 1440))
                    sampled_rates = [rate_data[i]['moet_interest_rate'] * 100 for i in sample_indices if i < len(rate_data)]
                    hours = [i / 60 for i in sample_indices if i < len(rate_data)]  # Convert to hours
                    
                    ax1.plot(hours, sampled_rates, label=f'{freq} checks', 
                            color=color, linewidth=2, alpha=0.8)
            
            # Bond APR History
            if 'bond_apr_history' in tracking:
                bond_data = tracking['bond_apr_history']
                if bond_data:
                    # Sample every 24 hours
                    sample_indices = list(range(0, len(bond_data), 1440))
                    sampled_aprs = [bond_data[i]['apr'] * 100 for i in sample_indices if i < len(bond_data)]
                    hours = [i / 60 for i in sample_indices if i < len(bond_data)]
                    
                    ax2.plot(hours, sampled_aprs, label=f'{freq} checks',
                            color=color, linewidth=2, alpha=0.8)
            
            # Reserve Ratio Evolution
            if 'reserve_history' in tracking:
                reserve_data = tracking['reserve_history']
                
                if reserve_data:
                    # Sample every 24 hours
                    sample_indices = list(range(0, len(reserve_data), 1440))
                    sampled_ratios = []
                    hours = []
                    
                    for i in sample_indices:
                        if i < len(reserve_data):
                            reserve_entry = reserve_data[i]
                            if isinstance(reserve_entry, dict):
                                ratio = reserve_entry.get('reserve_ratio', 0) * 100
                                sampled_ratios.append(ratio)
                                hours.append(i / 60)
                    
                    if sampled_ratios:
                        ax3.plot(hours, sampled_ratios, label=f'{freq} checks',
                                color=color, linewidth=2, alpha=0.8)
        
        # Format axes
        ax1.set_title('MOET Interest Rate Evolution', fontweight='bold')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Interest Rate (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Bond APR Evolution', fontweight='bold')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('Bond APR (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Reserve Ratio Evolution', fontweight='bold')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('Reserve Ratio (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Summary statistics in bottom right
        ax4.axis('off')
        summary_text = "📊 Final System State Comparison\\n\\n"
        
        for freq, data in self.data.items():
            sim_results = data.get('simulation_results', {})
            moet_summary = sim_results.get('moet_system_summary', {})
            
            final_rate = moet_summary.get('final_moet_rate', 0) * 100
            avg_rate = moet_summary.get('avg_moet_rate', 0) * 100
            max_bond_apr = moet_summary.get('max_bond_apr', 0) * 100
            
            summary_text += f"{freq} Leverage Checks:\\n"
            summary_text += f"  Final MOET Rate: {final_rate:.4f}%\\n"
            summary_text += f"  Avg MOET Rate: {avg_rate:.4f}%\\n"
            summary_text += f"  Max Bond APR: {max_bond_apr:.2f}%\\n\\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        output_file = self.charts_dir / "moet_system_analysis_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_file}")
        
    def create_net_apy_comparison(self):
        """Create Net APY analysis comparison chart"""
        print("📊 Creating Net APY analysis comparison...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Net APY Analysis: Leverage Frequency Comparison', fontsize=16, fontweight='bold')
        
        for freq, data in self.data.items():
            sim_results = data.get('simulation_results', {})
            color = self.colors[freq]
            
            # Agent health factor evolution
            if 'agent_health_history' in sim_results:
                health_history = sim_results['agent_health_history']
                if health_history:
                    hours = [entry['minute'] / 60 for entry in health_history]
                    avg_health_factors = []
                    
                    for entry in health_history:
                        agents = entry.get('agents', [])
                        if agents:
                            health_factors = [agent.get('health_factor', 0) for agent in agents]
                            avg_hf = np.mean(health_factors) if health_factors else 0
                            avg_health_factors.append(avg_hf)
                        else:
                            avg_health_factors.append(0)
                    
                    # Sample every 24 hours for readability
                    sample_indices = list(range(0, len(hours), 24))
                    sampled_hours = [hours[i] for i in sample_indices if i < len(hours)]
                    sampled_hf = [avg_health_factors[i] for i in sample_indices if i < len(avg_health_factors)]
                    
                    ax1.plot(sampled_hours, sampled_hf, label=f'{freq} checks',
                            color=color, linewidth=2, alpha=0.8)
            
            # Rebalancing activity over time
            if 'rebalancing_events' in sim_results:
                rebalancing_events = sim_results['rebalancing_events']
                if rebalancing_events:
                    # Group by day and count rebalances
                    daily_rebalances = {}
                    for event in rebalancing_events:
                        day = int(event['minute'] // 1440)  # Convert to day
                        daily_rebalances[day] = daily_rebalances.get(day, 0) + 1
                    
                    days = sorted(daily_rebalances.keys())
                    rebalance_counts = [daily_rebalances[day] for day in days]
                    
                    ax2.plot(days, rebalance_counts, label=f'{freq} checks',
                            color=color, linewidth=2, alpha=0.8)
            
            # Cost analysis
            if 'cost_analysis' in sim_results:
                cost_data = sim_results['cost_analysis']
                total_cost = cost_data.get('total_cost_of_rebalancing', 0)
                avg_cost = cost_data.get('average_cost_per_agent', 0)
                
                # Bar chart for costs
                x_pos = list(self.data.keys()).index(freq)
                ax3.bar(x_pos, total_cost, color=color, alpha=0.7, label=f'{freq} checks')
        
        # Format axes
        ax1.set_title('Average Health Factor Evolution', fontweight='bold')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Average Health Factor')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Daily Rebalancing Activity', fontweight='bold')
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Number of Rebalances')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Total Rebalancing Costs', fontweight='bold')
        ax3.set_xlabel('Leverage Check Frequency')
        ax3.set_ylabel('Total Cost ($)')
        ax3.set_xticks(range(len(self.frequencies)))
        ax3.set_xticklabels(self.frequencies)
        ax3.grid(True, alpha=0.3)
        
        # Performance summary
        ax4.axis('off')
        summary_text = "📈 Performance Summary\\n\\n"
        
        for freq, data in self.data.items():
            sim_results = data.get('simulation_results', {})
            survival_stats = sim_results.get('survival_statistics', {})
            cost_analysis = sim_results.get('cost_analysis', {})
            
            survival_rate = survival_stats.get('survival_rate', 0) * 100
            total_cost = cost_analysis.get('total_cost_of_rebalancing', 0)
            avg_cost = cost_analysis.get('average_cost_per_agent', 0)
            
            summary_text += f"{freq} Leverage Checks:\\n"
            summary_text += f"  Survival Rate: {survival_rate:.1f}%\\n"
            summary_text += f"  Total Cost: ${total_cost:,.0f}\\n"
            summary_text += f"  Avg Cost/Agent: ${avg_cost:.2f}\\n\\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        output_file = self.charts_dir / "net_apy_analysis_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_file}")
        
    def create_yield_strategy_comparison(self):
        """Create yield strategy comparison chart"""
        print("📊 Creating yield strategy comparison...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Yield Strategy Comparison: Leverage Frequency Analysis', fontsize=16, fontweight='bold')
        
        for freq, data in self.data.items():
            sim_results = data.get('simulation_results', {})
            color = self.colors[freq]
            
            # BTC price vs agent performance
            if 'btc_price_history' in sim_results and 'agent_health_history' in sim_results:
                btc_history = sim_results['btc_price_history']
                health_history = sim_results['agent_health_history']
                
                if btc_history and health_history:
                    # Sample every 7 days for readability
                    sample_indices = list(range(0, min(len(btc_history), len(health_history)), 7))
                    
                    btc_prices = [btc_history[i] for i in sample_indices if i < len(btc_history)]
                    days = [i for i in sample_indices]
                    
                    # Normalize BTC prices to percentage change from start
                    if btc_prices:
                        initial_price = btc_prices[0]
                        btc_pct_change = [(price / initial_price - 1) * 100 for price in btc_prices]
                        
                        ax1.plot(days, btc_pct_change, label=f'{freq} checks',
                                color=color, linewidth=2, alpha=0.8)
            
            # Yield token activity
            if 'yield_token_activity' in sim_results:
                yt_activity = sim_results['yield_token_activity']
                
                total_purchases = yt_activity.get('total_purchases', 0)
                total_sales = yt_activity.get('total_rebalancing_sales', 0)
                
                # Bar chart for YT activity
                x_pos = list(self.data.keys()).index(freq)
                width = 0.35
                ax2.bar(x_pos - width/2, total_purchases, width, color=color, alpha=0.7, label=f'{freq} purchases')
                ax2.bar(x_pos + width/2, total_sales, width, color=color, alpha=0.5, label=f'{freq} sales')
            
            # Pool rebalancing activity
            if 'pool_rebalancing_activity' in sim_results:
                pool_activity = sim_results['pool_rebalancing_activity']
                
                alm_rebalances = pool_activity.get('alm_rebalances', 0)
                algo_rebalances = pool_activity.get('algo_rebalances', 0)
                
                # Stacked bar chart
                x_pos = list(self.data.keys()).index(freq)
                ax3.bar(x_pos, alm_rebalances, color=color, alpha=0.7, label=f'{freq} ALM')
                ax3.bar(x_pos, algo_rebalances, bottom=alm_rebalances, color=color, alpha=0.5, label=f'{freq} Algo')
        
        # Format axes
        ax1.set_title('BTC Price Performance', fontweight='bold')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('BTC Price Change (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Yield Token Activity', fontweight='bold')
        ax2.set_xlabel('Leverage Check Frequency')
        ax2.set_ylabel('Number of Transactions')
        ax2.set_xticks(range(len(self.frequencies)))
        ax2.set_xticklabels(self.frequencies)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Pool Rebalancing Activity', fontweight='bold')
        ax3.set_xlabel('Leverage Check Frequency')
        ax3.set_ylabel('Number of Rebalances')
        ax3.set_xticks(range(len(self.frequencies)))
        ax3.set_xticklabels(self.frequencies)
        ax3.grid(True, alpha=0.3)
        
        # Strategy effectiveness summary
        ax4.axis('off')
        summary_text = "🎯 Strategy Effectiveness\\n\\n"
        
        for freq, data in self.data.items():
            sim_results = data.get('simulation_results', {})
            
            # Calculate some effectiveness metrics
            total_rebalances = len(sim_results.get('rebalancing_events', []))
            survival_rate = sim_results.get('survival_statistics', {}).get('survival_rate', 0) * 100
            
            yt_activity = sim_results.get('yield_token_activity', {})
            total_yt_trades = yt_activity.get('total_trades', 0)
            
            summary_text += f"{freq} Leverage Checks:\\n"
            summary_text += f"  Total Rebalances: {total_rebalances:,}\\n"
            summary_text += f"  Survival Rate: {survival_rate:.1f}%\\n"
            summary_text += f"  YT Trades: {total_yt_trades:,}\\n\\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        output_file = self.charts_dir / "yield_strategy_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_file}")
        
    def create_summary_report(self):
        """Create a summary report of the comparison"""
        print("📝 Creating summary report...")
        
        summary_file = self.output_dir / "leverage_frequency_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("LEVERAGE FREQUENCY COMPARISON ANALYSIS\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            f.write("SIMULATION CONFIGURATIONS:\\n")
            f.write("-" * 25 + "\\n")
            for freq, data in self.data.items():
                sim_config = data.get('simulation_results', {}).get('simulation_config', {})
                f.write(f"{freq} Leverage Checks:\\n")
                f.write(f"  Steps: {sim_config.get('steps', 'N/A'):,}\\n")
                f.write(f"  Agents: {sim_config.get('num_agents', 'N/A')}\\n")
                f.write(f"  Scenario: {sim_config.get('scenario_name', 'N/A')}\\n\\n")
            
            f.write("PERFORMANCE COMPARISON:\\n")
            f.write("-" * 22 + "\\n")
            for freq, data in self.data.items():
                sim_results = data.get('simulation_results', {})
                
                # Survival statistics
                survival_stats = sim_results.get('survival_statistics', {})
                survival_rate = survival_stats.get('survival_rate', 0) * 100
                
                # Cost analysis
                cost_analysis = sim_results.get('cost_analysis', {})
                total_cost = cost_analysis.get('total_cost_of_rebalancing', 0)
                
                # MOET system
                moet_summary = sim_results.get('moet_system_summary', {})
                final_moet_rate = moet_summary.get('final_moet_rate', 0) * 100
                
                # Activity metrics
                total_rebalances = len(sim_results.get('rebalancing_events', []))
                
                f.write(f"{freq} Leverage Checks:\\n")
                f.write(f"  Survival Rate: {survival_rate:.1f}%\\n")
                f.write(f"  Total Rebalances: {total_rebalances:,}\\n")
                f.write(f"  Total Cost: ${total_cost:,.2f}\\n")
                f.write(f"  Final MOET Rate: {final_moet_rate:.4f}%\\n\\n")
        
        print(f"✅ Saved: {summary_file}")
        
    def run_analysis(self):
        """Run the complete comparison analysis"""
        print("🚀 Starting Leverage Frequency Comparison Analysis...")
        
        self.load_data()
        
        if not self.data:
            print("❌ No data loaded. Exiting.")
            return
            
        self.create_moet_system_comparison()
        self.create_net_apy_comparison()
        self.create_yield_strategy_comparison()
        self.create_summary_report()
        
        print("\\n✅ Leverage Frequency Comparison Analysis Complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Charts saved to: {self.charts_dir}")

if __name__ == "__main__":
    analyzer = LeverageFrequencyComparisonFixed()
    analyzer.run_analysis()
