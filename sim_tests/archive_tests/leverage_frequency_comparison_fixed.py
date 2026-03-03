#!/usr/bin/env python3
"""
Fixed Leverage Frequency Comparison Analysis
===========================================

Compares simulation results across different agent leverage check frequencies:
- 3-minute leverage checks
- 5-minute leverage checks  
- 10-minute leverage checks

Generates comparison charts using available JSON data.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

class LeverageFrequencyComparison:
    def __init__(self):
        self.base_dir = Path("tidal_protocol_sim/results")
        self.output_dir = self.base_dir / "Leverage_Frequency_Comparison"
        self.output_dir.mkdir(exist_ok=True)
        
        # Chart output directory
        self.charts_dir = self.output_dir / "charts"
        self.charts_dir.mkdir(exist_ok=True)
        
        # Data storage
        self.data = {}
        
    def load_simulation_data(self):
        """Load data from all three simulation directories"""
        directories = {
            "3min": "Full_Year_2024_BTC_Simulation_3min_leverage",
            "5min": "Full_Year_2024_BTC_Simulation_5min_leverage", 
            "10min": "Full_Year_2024_BTC_Simulation_10min_leverage"
        }
        
        for freq, dir_name in directories.items():
            dir_path = self.base_dir / dir_name
            if not dir_path.exists():
                print(f"❌ Directory not found: {dir_path}")
                continue
                
            # Find JSON file
            json_files = list(dir_path.glob("*.json"))
            if not json_files:
                print(f"❌ No JSON files found in {dir_path}")
                continue
                
            json_file = json_files[0]  # Take the first (should be only one)
            print(f"📄 Loading {freq} data from: {json_file.name}")
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            self.data[freq] = data.get('simulation_results', {})
            
        print(f"✅ Loaded data for {len(self.data)} frequencies")
        
    def create_moet_system_comparison(self):
        """Create MOET system analysis comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MOET System Analysis: Leverage Frequency Comparison', fontsize=16, fontweight='bold')
        
        colors = {'3min': '#FF6B6B', '5min': '#4ECDC4', '10min': '#45B7D1'}
        
        for freq, data in self.data.items():
            moet_state = data.get('moet_system_state', {})
            tracking = moet_state.get('tracking_data', {})
            
            if not tracking:
                continue
                
            # MOET Interest Rate over time (sample every hour)
            if 'moet_rate_history' in tracking:
                rate_data = tracking['moet_rate_history']
                # Sample every 60 minutes (1 hour) to reduce data points
                sampled_data = rate_data[::60]  # Every 60th entry
                hours = list(range(len(sampled_data)))
                rates = [entry['moet_interest_rate']*100 for entry in sampled_data]
                ax1.plot(hours, rates, label=f'{freq} checks', 
                        color=colors[freq], linewidth=2, alpha=0.8)
        
        ax1.set_title('MOET Interest Rate Evolution', fontweight='bold')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Interest Rate (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bond APR History
        for freq, data in self.data.items():
            moet_state = data.get('moet_system_state', {})
            tracking = moet_state.get('tracking_data', {})
            
            if 'bond_apr_history' in tracking:
                bond_data = tracking['bond_apr_history']
                if bond_data:
                    # Sample every hour
                    sampled_data = bond_data[::60]
                    hours = list(range(len(sampled_data)))
                    aprs = [entry['bond_apr']*100 for entry in sampled_data]
                    ax2.plot(hours, aprs, label=f'{freq} checks',
                            color=colors[freq], linewidth=2, alpha=0.8)
        
        ax2.set_title('Bond APR History', fontweight='bold')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('Bond APR (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Reserve Ratio Evolution
        for freq, data in self.data.items():
            moet_state = data.get('moet_system_state', {})
            tracking = moet_state.get('tracking_data', {})
            
            if 'reserve_history' in tracking:
                reserve_data = tracking['reserve_history']
                if reserve_data:
                    # Sample every hour
                    sampled_data = reserve_data[::60]
                    hours = list(range(len(sampled_data)))
                    ratios = [entry['reserve_ratio']*100 for entry in sampled_data]
                    ax3.plot(hours, ratios, label=f'{freq} checks',
                            color=colors[freq], linewidth=2, alpha=0.8)
        
        ax3.set_title('Reserve Ratio Evolution', fontweight='bold')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('Reserve Ratio (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Summary Statistics
        summary_data = []
        for freq, data in self.data.items():
            moet_summary = data.get('moet_system_summary', {})
            summary_data.append({
                'Frequency': freq,
                'Final MOET Rate': f"{moet_summary.get('final_moet_rate', 0)*100:.4f}%",
                'Avg Bond APR': f"{moet_summary.get('avg_bond_apr', 0)*100:.4f}%",
                'Final Reserve Ratio': f"{moet_summary.get('final_reserve_ratio', 0)*100:.2f}%"
            })
        
        # Create summary table
        ax4.axis('tight')
        ax4.axis('off')
        if summary_data:
            df = pd.DataFrame(summary_data)
            table = ax4.table(cellText=df.values, colLabels=df.columns,
                             cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
        ax4.set_title('Summary Statistics', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'moet_system_analysis_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_agent_performance_comparison(self):
        """Create agent performance comparison using available data"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Agent Performance: Leverage Frequency Comparison', fontsize=16, fontweight='bold')
        
        colors = {'3min': '#FF6B6B', '5min': '#4ECDC4', '10min': '#45B7D1'}
        
        # Survival rates
        survival_data = []
        for freq, data in self.data.items():
            survival_stats = data.get('survival_statistics', {})
            survival_rate = survival_stats.get('survival_rate', 0) * 100
            survival_data.append((freq, survival_rate))
        
        if survival_data:
            freqs, rates = zip(*survival_data)
            bars = ax1.bar(freqs, rates, color=[colors[f] for f in freqs], alpha=0.7)
            ax1.set_title('Agent Survival Rates', fontweight='bold')
            ax1.set_ylabel('Survival Rate (%)')
            ax1.set_ylim(0, 105)
            
            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Total rebalancing activity
        rebalance_data = []
        for freq, data in self.data.items():
            rebalancing_events = data.get('rebalancing_events', [])
            total_rebalances = len(rebalancing_events)
            rebalance_data.append((freq, total_rebalances))
        
        if rebalance_data:
            freqs, counts = zip(*rebalance_data)
            bars = ax2.bar(freqs, counts, color=[colors[f] for f in freqs], alpha=0.7)
            ax2.set_title('Total Rebalancing Events', fontweight='bold')
            ax2.set_ylabel('Number of Rebalances')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Average slippage costs
        slippage_data = []
        for freq, data in self.data.items():
            cost_analysis = data.get('cost_analysis', {})
            avg_cost = cost_analysis.get('average_cost_per_agent', 0)
            slippage_data.append((freq, avg_cost))
        
        if slippage_data:
            freqs, costs = zip(*slippage_data)
            bars = ax3.bar(freqs, costs, color=[colors[f] for f in freqs], alpha=0.7)
            ax3.set_title('Average Slippage Cost per Agent', fontweight='bold')
            ax3.set_ylabel('Average Cost ($)')
            
            # Add value labels on bars
            for bar, cost in zip(bars, costs):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(costs)*0.01,
                        f'${cost:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Pool rebalancing comparison
        pool_data = []
        for freq, data in self.data.items():
            pool_activity = data.get('pool_rebalancing_activity', {})
            total_pool_rebalances = pool_activity.get('total_rebalances', 0)
            pool_data.append((freq, total_pool_rebalances))
        
        if pool_data:
            freqs, counts = zip(*pool_data)
            bars = ax4.bar(freqs, counts, color=[colors[f] for f in freqs], alpha=0.7)
            ax4.set_title('Pool Rebalancing Events', fontweight='bold')
            ax4.set_ylabel('Number of Pool Rebalances')
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                        f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.charts_dir / 'agent_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_summary_report(self):
        """Create a summary report of the comparison"""
        report_path = self.output_dir / "leverage_frequency_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("Leverage Frequency Comparison Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for freq, data in self.data.items():
                f.write(f"{freq.upper()} LEVERAGE CHECKS:\n")
                f.write("-" * 30 + "\n")
                
                # Basic stats
                survival_stats = data.get('survival_statistics', {})
                f.write(f"Survival Rate: {survival_stats.get('survival_rate', 0)*100:.1f}%\n")
                
                cost_analysis = data.get('cost_analysis', {})
                f.write(f"Avg Cost per Agent: ${cost_analysis.get('average_cost_per_agent', 0):.2f}\n")
                
                rebalancing_events = data.get('rebalancing_events', [])
                f.write(f"Total Rebalances: {len(rebalancing_events):,}\n")
                
                pool_activity = data.get('pool_rebalancing_activity', {})
                f.write(f"Pool Rebalances: {pool_activity.get('total_rebalances', 0):,}\n")
                
                # MOET system stats
                moet_summary = data.get('moet_system_summary', {})
                f.write(f"Final MOET Rate: {moet_summary.get('final_moet_rate', 0)*100:.4f}%\n")
                f.write(f"Avg Bond APR: {moet_summary.get('avg_bond_apr', 0)*100:.4f}%\n")
                f.write(f"Final Reserve Ratio: {moet_summary.get('final_reserve_ratio', 0)*100:.2f}%\n")
                
                f.write("\n")
        
        print(f"📄 Summary report saved to: {report_path}")
        
    def run_comparison(self):
        """Run the complete comparison analysis"""
        print("🔄 Starting Leverage Frequency Comparison Analysis...")
        
        # Load data
        self.load_simulation_data()
        
        if len(self.data) < 2:
            print("❌ Need at least 2 datasets to compare")
            return
            
        # Create charts
        print("📊 Creating MOET system comparison...")
        self.create_moet_system_comparison()
        
        print("📊 Creating agent performance comparison...")
        self.create_agent_performance_comparison()
        
        # Create summary
        print("📄 Creating summary report...")
        self.create_summary_report()
        
        print(f"✅ Comparison analysis complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Charts saved to: {self.charts_dir}")

if __name__ == "__main__":
    comparison = LeverageFrequencyComparison()
    comparison.run_comparison()
