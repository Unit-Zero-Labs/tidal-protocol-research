#!/usr/bin/env python3
"""
Create Exact Comparison Charts
=============================

Creates side-by-side comparison of the exact same net_apy_analysis.png and 
yield_strategy_comparison.png charts from individual simulations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns

class ExactComparisonCharts:
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
        """Create Net APY Analysis comparison chart using exact same logic as individual sims"""
        print("📈 Creating Net APY Analysis comparison...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Net APY Analysis: Agent Performance vs BTC Hold Strategy - Leverage Frequency Comparison', 
                     fontsize=16, fontweight='bold')
        
        for i, freq in enumerate(self.frequencies):
            if freq not in self.data:
                continue
                
            # Get simulation results for this frequency
            results = {"simulation_results": self.data[freq].get("simulation_results", {})}
            
            # Create charts for this frequency using exact same logic
            self._create_individual_net_apy_chart(axes[:, i], results, freq)
        
        plt.tight_layout()
        output_path = self.charts_dir / "net_apy_analysis_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")
        
    def _create_individual_net_apy_chart(self, axes, results, freq_label):
        """Create individual Net APY chart using exact same logic as full_year_sim.py"""
        
        # Extract time series data from simulation results (EXACT COPY FROM ORIGINAL)
        simulation_results = results.get("simulation_results", {})
        
        # Get BTC price history
        btc_history = simulation_results.get("btc_price_history", [])
        btc_data = []
        for i, btc_entry in enumerate(btc_history):
            hour = i * 24.0  # Convert day index to hours
            btc_data.append({
                "hour": hour,
                "day": i,
                "btc_price": btc_entry
            })
        
        # Get agent health factor history and net position data
        agent_health_history = simulation_results.get("agent_health_history", [])
        agent_data = []
        
        # Extract data from agent health history snapshots
        for i, health_snapshot in enumerate(agent_health_history):
            hour = i * 24.0  # Convert day index to hours
            if health_snapshot and "agents" in health_snapshot:
                agents_list = health_snapshot["agents"]
                if agents_list:
                    # Use test_agent_03 as representative
                    target_agent = None
                    for agent in agents_list:
                        if agent.get("agent_id") == "test_agent_03":
                            target_agent = agent
                            break
                    
                    if not target_agent and agents_list:
                        target_agent = agents_list[0]
                    
                    if target_agent:
                        agent_data.append({
                            "hour": hour,
                            "day": i,
                            "net_position_value": target_agent.get("net_position_value", 100000),
                            "initial_position": 100000  # Assuming $100k initial position
                        })
        
        if not btc_data or not agent_data:
            print(f"⚠️  No data available for Net APY analysis ({freq_label})")
            return
        
        # Calculate performance metrics (EXACT COPY FROM ORIGINAL)
        initial_btc_price = btc_data[0]["btc_price"]
        initial_agent_value = agent_data[0]["net_position_value"]
        
        # Create performance time series
        hours = []
        btc_hold_values = []
        agent_net_values = []
        
        for btc_point, agent_point in zip(btc_data, agent_data):
            if btc_point["hour"] == agent_point["hour"]:
                hours.append(btc_point["hour"])
                
                # BTC Hold Strategy: Initial investment * (current_price / initial_price)
                btc_hold_value = initial_agent_value * (btc_point["btc_price"] / initial_btc_price)
                btc_hold_values.append(btc_hold_value)
                
                # Agent Net Position Value
                agent_net_values.append(agent_point["net_position_value"])
        
        if not hours:
            return
            
        # Calculate APY values using the EXACT same method as original
        btc_hold_apy = []
        agent_strategy_apy = []
        
        for i, hour in enumerate(hours):
            days_elapsed = hour / 24.0
            if days_elapsed > 0:
                # Calculate annualized returns (EXACT COPY FROM ORIGINAL)
                btc_return = (btc_hold_values[i] / initial_agent_value - 1) * (365 / days_elapsed) * 100
                agent_return = (agent_net_values[i] / initial_agent_value - 1) * (365 / days_elapsed) * 100
                
                btc_hold_apy.append(btc_return)
                agent_strategy_apy.append(agent_return)
            else:
                btc_hold_apy.append(0)
                agent_strategy_apy.append(0)
        
        # Chart 1: Portfolio Value Comparison (EXACT COPY)
        ax1 = axes[0]
        ax1.set_title(f'{freq_label} Leverage Checks\\nPortfolio Value Comparison', fontweight='bold')
        ax1.plot(hours, btc_hold_values, label='BTC Hold Value', color='orange', linewidth=2)
        ax1.plot(hours, agent_net_values, label='Agent Net Position', color='blue', linewidth=2)
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Chart 2: Annualized Percentage Yield (APY) (EXACT COPY)
        ax2 = axes[1]
        ax2.set_title(f'{freq_label} Leverage Checks\\nAnnualized Percentage Yield (APY)', fontweight='bold')
        ax2.plot(hours, btc_hold_apy, label='BTC Hold APY', color='orange', linewidth=2)
        ax2.plot(hours, agent_strategy_apy, label='Agent Strategy APY', color='blue', linewidth=2)
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('APY (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
    def create_yield_strategy_comparison(self):
        """Create Yield Strategy Comparison chart using exact same logic as individual sims"""
        print("📈 Creating Yield Strategy Comparison...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Yield Strategy Comparison: Tidal Protocol vs Base 10% APR Yield - Leverage Frequency Comparison', 
                     fontsize=16, fontweight='bold')
        
        for i, freq in enumerate(self.frequencies):
            if freq not in self.data:
                continue
                
            # Get simulation results for this frequency
            results = {"simulation_results": self.data[freq].get("simulation_results", {})}
            
            # Create charts for this frequency using exact same logic
            self._create_individual_yield_strategy_chart(axes[:, i], results, freq)
        
        plt.tight_layout()
        output_path = self.charts_dir / "yield_strategy_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")
        
    def _create_individual_yield_strategy_chart(self, axes, results, freq_label):
        """Create individual Yield Strategy chart using exact same logic as full_year_sim.py"""
        
        # Extract time series data from simulation results (EXACT COPY FROM ORIGINAL)
        simulation_results = results.get("simulation_results", {})
        
        # Get BTC price history
        btc_history = simulation_results.get("btc_price_history", [])
        btc_data = []
        for i, btc_entry in enumerate(btc_history):
            hour = i * 24.0  # Convert day index to hours
            btc_data.append({
                "hour": hour,
                "day": i,
                "btc_price": btc_entry
            })
        
        # Get agent health factor history and net position data
        agent_health_history = simulation_results.get("agent_health_history", [])
        agent_data = []
        
        # Extract data from agent health history snapshots
        for i, health_snapshot in enumerate(agent_health_history):
            hour = i * 24.0  # Convert day index to hours
            if health_snapshot and "agents" in health_snapshot:
                agents_list = health_snapshot["agents"]
                if agents_list:
                    # Use first agent as representative
                    target_agent = agents_list[0]
                    
                    if target_agent:
                        agent_data.append({
                            "hour": hour,
                            "day": i,
                            "net_position_value": target_agent.get("net_position_value", 100000),
                            "btc_price": btc_data[i]["btc_price"] if i < len(btc_data) else btc_data[-1]["btc_price"]
                        })
        
        if not btc_data or not agent_data:
            print(f"⚠️  No data available for Yield Strategy Comparison ({freq_label})")
            return
        
        # Calculate performance metrics (EXACT COPY FROM ORIGINAL)
        initial_btc_price = btc_data[0]["btc_price"]
        initial_agent_value = agent_data[0]["net_position_value"]
        base_apr = 0.10  # 10% APR
        
        # Create performance time series
        hours = []
        tidal_yield_adjusted = []  # Net Position Value / BTC Price
        base_yield_value = []      # Just 10% APR compounded
        
        for agent_point in agent_data:
            hours.append(agent_point["hour"])
            
            # Tidal Protocol yield (BTC-price adjusted)
            btc_adjusted_value = agent_point["net_position_value"] / agent_point["btc_price"] * initial_btc_price
            tidal_yield_adjusted.append(btc_adjusted_value)
            
            # Base 10% APR yield (compounded continuously)
            time_years = agent_point["hour"] / 8760.0
            base_value = initial_agent_value * np.exp(base_apr * time_years)
            base_yield_value.append(base_value)
        
        if not hours:
            return
            
        # Calculate APY values using the EXACT same method as original
        tidal_apy = []
        base_apy = []
        
        for i, hour in enumerate(hours):
            days_elapsed = hour / 24.0
            if days_elapsed > 0:
                # Calculate annualized returns (EXACT COPY FROM ORIGINAL)
                tidal_return = (tidal_yield_adjusted[i] / initial_agent_value - 1) * (365 / days_elapsed) * 100
                base_return = (base_yield_value[i] / initial_agent_value - 1) * (365 / days_elapsed) * 100
                
                tidal_apy.append(tidal_return)
                base_apy.append(base_return)
            else:
                tidal_apy.append(0)
                base_apy.append(0)
        
        # Chart 1: Portfolio Value Comparison (BTC-Price Adjusted) (EXACT COPY)
        ax1 = axes[0]
        ax1.set_title(f'{freq_label} Leverage Checks\\nPortfolio Value Comparison (BTC-Price Adjusted)', fontweight='bold')
        ax1.plot(hours, base_yield_value, label='Base 10% APR Yield', color='green', linewidth=2)
        ax1.plot(hours, tidal_yield_adjusted, label='Tidal Protocol (BTC-adjusted)', color='blue', linewidth=2)
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Chart 2: Annualized Yield Comparison (EXACT COPY)
        ax2 = axes[1]
        ax2.set_title(f'{freq_label} Leverage Checks\\nAnnualized Yield Comparison', fontweight='bold')
        ax2.plot(hours, base_apy, label='Base 10% APR', color='green', linewidth=2)
        ax2.plot(hours, tidal_apy, label='Tidal Protocol APY', color='blue', linewidth=2)
        ax2.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='10% Target')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('APY (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
    def run_analysis(self):
        """Run the exact comparison charts analysis"""
        print("🚀 Creating exact comparison charts...")
        
        # Load data
        self.load_data()
        
        if len(self.data) == 0:
            print("❌ No data loaded. Cannot proceed.")
            return
        
        # Set plotting style (same as original)
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create exact comparison charts
        self.create_net_apy_analysis_comparison()
        self.create_yield_strategy_comparison()
        
        print(f"\n✅ Exact comparison charts created successfully!")
        print(f"📁 Charts saved to: {self.charts_dir}")

if __name__ == "__main__":
    charts = ExactComparisonCharts()
    charts.run_analysis()
