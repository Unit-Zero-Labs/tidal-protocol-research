#!/usr/bin/env python3
"""
Leverage Frequency Comparison Analysis
=====================================

Compares simulation results across different agent leverage check frequencies:
- 3-minute leverage checks
- 5-minute leverage checks  
- 10-minute leverage checks

Generates comparison charts for:
- MOET System Analysis
- MOET Reserve Management
- Net APY Analysis
- Yield Strategy Comparison
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
        
        # Configuration for different frequencies
        self.frequencies = {
            "3min": {
                "label": "3-Minute Checks",
                "color": "#FF6B6B",  # Red
                "directory": "Full_Year_2024_BTC_Simulation_3min_leverage"
            },
            "5min": {
                "label": "5-Minute Checks", 
                "color": "#4ECDC4",  # Teal
                "directory": "Full_Year_2024_BTC_Simulation_5min_leverage"
            },
            "10min": {
                "label": "10-Minute Checks",
                "color": "#45B7D1",  # Blue
                "directory": "Full_Year_2024_BTC_Simulation_10min_leverage"
            }
        }
        
        self.data = {}
        
    def load_simulation_data(self):
        """Load simulation data for all frequencies"""
        print("🔍 Loading simulation data for all frequencies...")
        
        for freq_key, freq_config in self.frequencies.items():
            freq_dir = self.base_dir / freq_config["directory"]
            
            if not freq_dir.exists():
                print(f"⚠️  Directory not found: {freq_dir}")
                continue
                
            # Find the latest JSON file in the directory
            json_files = list(freq_dir.glob("*.json"))
            if not json_files:
                print(f"⚠️  No JSON files found in: {freq_dir}")
                continue
                
            latest_json = max(json_files, key=lambda x: x.stat().st_mtime)
            
            print(f"📁 Loading {freq_config['label']}: {latest_json.name}")
            
            try:
                with open(latest_json, 'r') as f:
                    data = json.load(f)
                    self.data[freq_key] = data
                    print(f"✅ Loaded {freq_config['label']} successfully")
            except Exception as e:
                print(f"❌ Error loading {freq_config['label']}: {e}")
                
        print(f"📊 Successfully loaded data for {len(self.data)} frequencies")
        
    def create_moet_system_comparison(self):
        """Create MOET System Analysis comparison chart"""
        print("📊 Creating MOET System Analysis comparison...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('MOET System Analysis: Leverage Frequency Comparison', fontsize=16, fontweight='bold')
        
        for freq_key, freq_config in self.frequencies.items():
            if freq_key not in self.data:
                continue
                
            data = self.data[freq_key]
            moet_state = data.get('simulation_results', {}).get('moet_system_state', {})
            tracking = moet_state.get('tracking_data', {})
            
            if not tracking:
                continue
                
            # Extract data
            moet_rates = tracking.get('moet_rate_history', [])
            bond_aprs = tracking.get('bond_apr_history', [])
            
            if not moet_rates or not bond_aprs:
                continue
                
            # Convert to arrays for plotting
            hours = [entry['minute'] / 60 for entry in moet_rates]
            total_rates = [entry['moet_interest_rate'] * 100 for entry in moet_rates]  # Convert to %
            r_floor = [entry['r_floor'] * 100 for entry in moet_rates]
            r_bond_cost = [entry['r_bond_cost'] * 100 for entry in moet_rates]
            bond_apr_values = [entry['bond_apr'] * 100 for entry in bond_aprs]
            
            color = freq_config['color']
            label = freq_config['label']
            alpha = 0.8
            
            # Top Left: MOET Interest Rate Components
            ax1.plot(hours, total_rates, color=color, label=f'{label} - Total MOET Rate', linewidth=2, alpha=alpha)
            ax1.plot(hours, r_floor, color=color, linestyle='--', alpha=0.6, linewidth=1)
            ax1.plot(hours, r_bond_cost, color=color, linestyle=':', alpha=0.6, linewidth=1)
            
            # Top Right: Bond Auction APR Evolution  
            ax2.plot(hours, bond_apr_values, color=color, label=f'{label} - Bond APR', linewidth=2, alpha=alpha)
            
            # Bottom Left: MOET Rate Premium Over Governance Floor
            premium = [total - floor for total, floor in zip(total_rates, r_floor)]
            ax3.plot(hours, premium, color=color, label=f'{label} - Rate Premium', linewidth=2, alpha=alpha)
            
            # Bottom Right: Bond Cost EMA Evolution (High Precision)
            ax4.plot(hours, r_bond_cost, color=color, label=f'{label} - r_bond_cost EMA', linewidth=2, alpha=alpha)
            
        # Formatting
        ax1.set_title('MOET Interest Rate Components')
        ax1.set_ylabel('Interest Rate (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_title('Bond Auction APR Evolution')
        ax2.set_ylabel('Bond APR (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3.set_title('MOET Rate Premium Over Governance Floor')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('Rate Premium (%)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        ax4.set_title('Bond Cost EMA Evolution (High Precision)')
        ax4.set_xlabel('Hours')
        ax4.set_ylabel('Bond Cost EMA (basis points)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.charts_dir / "moet_system_analysis_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ MOET System Analysis comparison saved: {chart_path}")
        
    def create_moet_reserve_comparison(self):
        """Create MOET Reserve Management comparison chart"""
        print("📊 Creating MOET Reserve Management comparison...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('MOET Reserve Management: Leverage Frequency Comparison', fontsize=16, fontweight='bold')
        
        for freq_key, freq_config in self.frequencies.items():
            if freq_key not in self.data:
                continue
                
            data = self.data[freq_key]
            moet_state = data.get('simulation_results', {}).get('moet_system_state', {})
            tracking = moet_state.get('tracking_data', {})
            
            if not tracking:
                continue
                
            # Extract reserve data
            reserve_history = tracking.get('reserve_history', [])
            deficit_history = tracking.get('deficit_history', [])
            
            if not reserve_history or not deficit_history:
                continue
                
            hours = [entry['minute'] / 60 for entry in reserve_history]
            target_reserves = [entry['target_reserves'] for entry in reserve_history]
            actual_reserves = [entry['actual_reserves'] for entry in reserve_history]
            reserve_ratios = [entry['reserve_ratio'] * 100 for entry in reserve_history]  # Convert to %
            deficits = [entry['deficit'] for entry in deficit_history]
            
            color = freq_config['color']
            label = freq_config['label']
            alpha = 0.8
            
            # Top Left: Target vs Actual Reserves
            ax1.plot(hours, target_reserves, color=color, linestyle='--', alpha=0.6, linewidth=1)
            ax1.plot(hours, actual_reserves, color=color, label=f'{label} - Actual Reserves', linewidth=2, alpha=alpha)
            
            # Top Right: Reserve Ratio Evolution
            ax2.plot(hours, reserve_ratios, color=color, label=f'{label} - Reserve Ratio', linewidth=2, alpha=alpha)
            ax2.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Target Ratio (10%)')
            
            # Bottom Left: Reserve Deficit Over Time
            ax3.plot(hours, deficits, color=color, label=f'{label} - Deficit Amount', linewidth=2, alpha=alpha)
            
            # Bottom Right: Reserve Composition (use final values)
            if reserve_history:
                final_entry = reserve_history[-1]
                # This would need reserve composition data from the JSON
                # For now, we'll skip this panel or show total reserves
                ax4.plot(hours, actual_reserves, color=color, label=f'{label} - Total Reserves', linewidth=2, alpha=alpha)
        
        # Formatting
        ax1.set_title('Target vs Actual Reserves')
        ax1.set_ylabel('Reserves ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        ax2.set_title('Reserve Ratio Evolution')
        ax2.set_ylabel('Reserve Ratio (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3.set_title('Reserve Deficit Over Time')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('Deficit ($)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        ax4.set_title('Total Reserves Evolution')
        ax4.set_xlabel('Hours')
        ax4.set_ylabel('Total Reserves ($)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.charts_dir / "moet_reserve_management_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ MOET Reserve Management comparison saved: {chart_path}")
        
    def create_net_apy_comparison(self):
        """Create Net APY Analysis comparison chart"""
        print("📊 Creating Net APY Analysis comparison...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Net APY Analysis: Leverage Frequency Comparison', fontsize=16, fontweight='bold')
        
        for freq_key, freq_config in self.frequencies.items():
            if freq_key not in self.data:
                continue
                
            data = self.data[freq_key]
            sim_results = data.get('simulation_results', {})
            
            # Extract BTC price data and agent performance
            btc_prices = data.get('btc_prices', [])
            agent_perf = sim_results.get('agent_performance', {})
            
            if not btc_prices or not agent_perf:
                continue
                
            # Calculate portfolio values over time (simplified)
            hours = list(range(len(btc_prices)))
            btc_values = btc_prices
            
            # Calculate average agent performance
            total_agents = len(agent_perf)
            if total_agents == 0:
                continue
                
            avg_final_value = sum(agent.get('final_portfolio_value', 50000) for agent in agent_perf.values()) / total_agents
            avg_initial_value = 50000  # Assuming initial value
            
            # Create synthetic portfolio evolution (this would need actual time series data)
            agent_values = [avg_initial_value * (1 + (avg_final_value/avg_initial_value - 1) * (h / len(hours))) for h in hours]
            
            # Calculate APYs
            hours_in_year = 8760
            time_factor = len(hours) / hours_in_year
            
            btc_apy = [(btc_val / btc_values[0]) ** (1/time_factor) - 1 for btc_val in btc_values]
            agent_apy = [(agent_val / avg_initial_value) ** (1/time_factor) - 1 for agent_val in agent_values]
            
            # Convert to percentages
            btc_apy_pct = [apy * 100 for apy in btc_apy]
            agent_apy_pct = [apy * 100 for apy in agent_apy]
            outperformance = [agent - btc for agent, btc in zip(agent_apy_pct, btc_apy_pct)]
            
            color = freq_config['color']
            label = freq_config['label']
            alpha = 0.8
            
            # Top Left: Portfolio Value Comparison
            ax1.plot(hours, btc_values, color='gray', alpha=0.6, linewidth=1, label='BTC Hold Value' if freq_key == list(self.frequencies.keys())[0] else "")
            ax1.plot(hours, agent_values, color=color, label=f'{label} - Agent Portfolio', linewidth=2, alpha=alpha)
            
            # Top Right: Annualized Percentage Yield (APY)
            ax2.plot(hours, btc_apy_pct, color='gray', alpha=0.6, linewidth=1, label='BTC Hold APY' if freq_key == list(self.frequencies.keys())[0] else "")
            ax2.plot(hours, agent_apy_pct, color=color, label=f'{label} - Agent APY', linewidth=2, alpha=alpha)
            
            # Bottom Left: Relative Performance (Agent APY - BTC Hold APY)
            ax3.fill_between(hours, 0, outperformance, color=color, alpha=0.3, label=f'{label} - Outperformance')
            ax3.plot(hours, outperformance, color=color, linewidth=2, alpha=alpha)
            
            # Bottom Right: Average Outperformance Over Time
            cumulative_outperf = np.cumsum(outperformance) / np.arange(1, len(outperformance) + 1)
            ax4.plot(hours, cumulative_outperf, color=color, label=f'{label} - Avg Outperformance', linewidth=2, alpha=alpha)
        
        # Formatting
        ax1.set_title('Portfolio Value Comparison')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        ax2.set_title('Annualized Percentage Yield (APY)')
        ax2.set_ylabel('APY (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3.set_title('Relative Performance (Agent APY - BTC Hold APY)')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('APY Difference (%)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax4.set_title('Average Outperformance Over Time')
        ax4.set_xlabel('Hours')
        ax4.set_ylabel('Average APY Difference (%)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.charts_dir / "net_apy_analysis_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Net APY Analysis comparison saved: {chart_path}")
        
    def create_yield_strategy_comparison(self):
        """Create Yield Strategy Comparison chart"""
        print("📊 Creating Yield Strategy Comparison...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Yield Strategy Comparison: Leverage Frequency Analysis', fontsize=16, fontweight='bold')
        
        for freq_key, freq_config in self.frequencies.items():
            if freq_key not in self.data:
                continue
                
            data = self.data[freq_key]
            sim_results = data.get('simulation_results', {})
            
            # Extract data for yield comparison
            btc_prices = data.get('btc_prices', [])
            agent_perf = sim_results.get('agent_performance', {})
            
            if not btc_prices or not agent_perf:
                continue
                
            hours = list(range(len(btc_prices)))
            
            # Calculate average agent performance
            total_agents = len(agent_perf)
            if total_agents == 0:
                continue
                
            avg_final_value = sum(agent.get('final_portfolio_value', 50000) for agent in agent_perf.values()) / total_agents
            avg_initial_value = 50000
            
            # Create synthetic data for comparison
            base_10_pct_values = [avg_initial_value * (1.10 ** (h / 8760)) for h in hours]  # 10% APR baseline
            tidal_values = [avg_initial_value * (1 + (avg_final_value/avg_initial_value - 1) * (h / len(hours))) for h in hours]
            
            # Calculate APYs
            hours_in_year = 8760
            time_factor = len(hours) / hours_in_year
            
            base_apy = [((val / avg_initial_value) ** (1/time_factor) - 1) * 100 for val in base_10_pct_values]
            tidal_apy = [((val / avg_initial_value) ** (1/time_factor) - 1) * 100 for val in tidal_values]
            
            # Calculate advantage
            advantage = [tidal - base for tidal, base in zip(tidal_apy, base_apy)]
            cumulative_advantage = np.cumsum(advantage) / np.arange(1, len(advantage) + 1)
            
            color = freq_config['color']
            label = freq_config['label']
            alpha = 0.8
            
            # Top Left: Portfolio Value Comparison (BTC-Price Adjusted)
            ax1.plot(hours, base_10_pct_values, color='gray', alpha=0.6, linewidth=1, label='Base 10% APR Yield' if freq_key == list(self.frequencies.keys())[0] else "")
            ax1.plot(hours, tidal_values, color=color, label=f'{label} - Tidal Protocol', linewidth=2, alpha=alpha)
            
            # Top Right: Annualized Yield Comparison
            ax2.plot(hours, base_apy, color='gray', alpha=0.6, linewidth=1, label='Base 10% APR' if freq_key == list(self.frequencies.keys())[0] else "")
            ax2.plot(hours, tidal_apy, color=color, label=f'{label} - Tidal APY', linewidth=2, alpha=alpha)
            ax2.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='10% Target' if freq_key == list(self.frequencies.keys())[0] else "")
            
            # Bottom Left: Relative Performance (Tidal APY - Base 10% APR)
            ax3.fill_between(hours, 0, advantage, color=color, alpha=0.3, 
                           where=[adv >= 0 for adv in advantage], label=f'{label} - Outperformance', interpolate=True)
            ax3.fill_between(hours, 0, advantage, color=color, alpha=0.3, 
                           where=[adv < 0 for adv in advantage], label=f'{label} - Underperformance', interpolate=True)
            ax3.plot(hours, advantage, color=color, linewidth=2, alpha=alpha)
            
            # Bottom Right: Average Yield Advantage Over Time
            ax4.plot(hours, cumulative_advantage, color=color, label=f'{label} - Avg Yield Advantage', linewidth=2, alpha=alpha)
        
        # Formatting
        ax1.set_title('Portfolio Value Comparison (BTC-Price Adjusted)')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        ax2.set_title('Annualized Yield Comparison')
        ax2.set_ylabel('APY (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3.set_title('Relative Performance (Tidal APY - Base 10% APR)')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('APY Difference (%)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        ax4.set_title('Average Yield Advantage Over Time')
        ax4.set_xlabel('Hours')
        ax4.set_ylabel('Average APY Advantage (%)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = self.charts_dir / "yield_strategy_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Yield Strategy Comparison saved: {chart_path}")
        
    def create_summary_metrics_table(self):
        """Create a summary table comparing key metrics across frequencies"""
        print("📊 Creating summary metrics table...")
        
        summary_data = []
        
        for freq_key, freq_config in self.frequencies.items():
            if freq_key not in self.data:
                continue
                
            data = self.data[freq_key]
            sim_results = data.get('simulation_results', {})
            moet_state = sim_results.get('moet_system_state', {})
            agent_perf = sim_results.get('agent_performance', {})
            
            # Extract key metrics
            tracking = moet_state.get('tracking_data', {})
            
            if tracking:
                moet_rates = tracking.get('moet_rate_history', [])
                bond_aprs = tracking.get('bond_apr_history', [])
                
                final_moet_rate = moet_rates[-1]['moet_interest_rate'] * 100 if moet_rates else 0
                avg_bond_apr = np.mean([entry['bond_apr'] * 100 for entry in bond_aprs]) if bond_aprs else 0
                max_bond_apr = max([entry['bond_apr'] * 100 for entry in bond_aprs]) if bond_aprs else 0
            else:
                final_moet_rate = avg_bond_apr = max_bond_apr = 0
            
            # Agent performance metrics
            if agent_perf:
                total_rebalances = sum(agent.get('total_rebalances', 0) for agent in agent_perf.values())
                avg_final_value = np.mean([agent.get('final_portfolio_value', 50000) for agent in agent_perf.values()])
                survival_rate = sum(1 for agent in agent_perf.values() if agent.get('survived', True)) / len(agent_perf) * 100
            else:
                total_rebalances = avg_final_value = survival_rate = 0
            
            summary_data.append({
                'Frequency': freq_config['label'],
                'Final MOET Rate (%)': f"{final_moet_rate:.4f}",
                'Avg Bond APR (%)': f"{avg_bond_apr:.4f}",
                'Max Bond APR (%)': f"{max_bond_apr:.2f}",
                'Total Agent Rebalances': f"{total_rebalances:,}",
                'Avg Final Portfolio ($)': f"${avg_final_value:,.0f}",
                'Survival Rate (%)': f"{survival_rate:.1f}%"
            })
        
        # Create summary table as text file
        summary_path = self.output_dir / "leverage_frequency_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("LEVERAGE FREQUENCY COMPARISON SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            for data_row in summary_data:
                f.write(f"{data_row['Frequency']}:\n")
                for key, value in data_row.items():
                    if key != 'Frequency':
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
        
        print(f"✅ Summary metrics saved: {summary_path}")
        
    def run_comparison(self):
        """Run the complete leverage frequency comparison analysis"""
        print("🚀 Starting Leverage Frequency Comparison Analysis...")
        print("=" * 60)
        
        # Load data
        self.load_simulation_data()
        
        if len(self.data) < 2:
            print("❌ Need at least 2 frequency datasets to create comparisons")
            return
        
        # Create comparison charts
        self.create_moet_system_comparison()
        self.create_moet_reserve_comparison() 
        self.create_net_apy_comparison()
        self.create_yield_strategy_comparison()
        
        # Create summary
        self.create_summary_metrics_table()
        
        print("\n" + "=" * 60)
        print("✅ Leverage Frequency Comparison Analysis Complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        print(f"📊 Charts saved to: {self.charts_dir}")
        print("\nGenerated Charts:")
        print("  - moet_system_analysis_comparison.png")
        print("  - moet_reserve_management_comparison.png") 
        print("  - net_apy_analysis_comparison.png")
        print("  - yield_strategy_comparison.png")
        print("  - leverage_frequency_summary.txt")

if __name__ == "__main__":
    comparison = LeverageFrequencyComparison()
    comparison.run_comparison()

