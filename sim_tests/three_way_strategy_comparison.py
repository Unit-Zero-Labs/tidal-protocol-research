#!/usr/bin/env python3
"""
Three-Way Strategy Comparison Analysis

Compares three investment strategies using 2024 BTC data:
1. High Tide Strategy - Borrow MOET to buy Yield Tokens (from full year sim)
2. BTC Hold Strategy - Simply hold BTC (baseline)
3. Aave Leverage Strategy - Borrow MOET to buy more BTC at 2.0 HF (new sim)

This script pulls JSON data from existing simulation results and creates
comprehensive comparison charts, including the enhanced net APY analysis.
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
from typing import Dict, List, Any, Tuple, Optional
import glob

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ThreeWayStrategyComparison:
    """Main class for comparing High Tide vs BTC Hold vs Aave Leverage strategies"""
    
    def __init__(self):
        self.results_dir = Path("tidal_protocol_sim/results")
        self.output_dir = self.results_dir / "Three_Way_Strategy_Comparison"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Strategy data
        self.high_tide_data = None
        self.aave_leverage_data = None
        self.btc_2024_data = None
        
        # Comparison results
        self.comparison_results = {
            "metadata": {
                "analysis_name": "Three_Way_Strategy_Comparison",
                "timestamp": datetime.now().isoformat(),
                "strategies": ["High_Tide", "BTC_Hold", "Aave_Leverage"]
            },
            "strategy_performance": {},
            "time_series_comparison": {},
            "risk_analysis": {}
        }
    
    def load_simulation_data(self) -> bool:
        """Load data from existing simulation results"""
        
        print("📂 Loading simulation data...")
        
        # Load High Tide data (Full Year 2024 BTC Simulation)
        high_tide_success = self._load_high_tide_data()
        
        # Load Aave Leverage data
        aave_leverage_success = self._load_aave_leverage_data()
        
        # Load BTC 2024 data
        btc_data_success = self._load_btc_2024_data()
        
        if not high_tide_success:
            print("❌ Failed to load High Tide simulation data")
            return False
        
        if not aave_leverage_success:
            print("❌ Failed to load Aave Leverage simulation data")
            return False
        
        if not btc_data_success:
            print("❌ Failed to load BTC 2024 data")
            return False
        
        print("✅ All simulation data loaded successfully")
        return True
    
    def _load_high_tide_data(self) -> bool:
        """Load High Tide simulation data from Full Year 2024 BTC Simulation"""
        
        high_tide_dir = self.results_dir / "Full_Year_2024_BTC_Simulation"
        if not high_tide_dir.exists():
            print(f"⚠️  High Tide results directory not found: {high_tide_dir}")
            return False
        
        # Find the most recent JSON file
        json_files = list(high_tide_dir.glob("pool_rebalancer_test_*.json"))
        if not json_files:
            print(f"⚠️  No High Tide JSON results found in {high_tide_dir}")
            return False
        
        # Use the most recent file
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                self.high_tide_data = json.load(f)
            
            print(f"✅ Loaded High Tide data from: {latest_file.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading High Tide data: {e}")
            return False
    
    def _load_aave_leverage_data(self) -> bool:
        """Load Aave Leverage simulation data"""
        
        aave_dir = self.results_dir / "Aave_Leverage_Strategy_2024"
        if not aave_dir.exists():
            print(f"⚠️  Aave Leverage results directory not found: {aave_dir}")
            print("   Run the Aave leverage simulation first: python sim_tests/aave_leverage_strategy_sim.py")
            return False
        
        # Find the most recent JSON file
        json_files = list(aave_dir.glob("aave_leverage_sim_*.json"))
        if not json_files:
            print(f"⚠️  No Aave Leverage JSON results found in {aave_dir}")
            return False
        
        # Use the most recent file
        latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                self.aave_leverage_data = json.load(f)
            
            print(f"✅ Loaded Aave Leverage data from: {latest_file.name}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading Aave Leverage data: {e}")
            return False
    
    def _load_btc_2024_data(self) -> bool:
        """Load 2024 BTC pricing data"""
        
        btc_csv_path = project_root / "btc-usd-max.csv"
        if not btc_csv_path.exists():
            print(f"⚠️  BTC CSV file not found: {btc_csv_path}")
            return False
        
        try:
            import csv
            btc_prices = []
            
            with open(btc_csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if '2024-' in row['snapped_at']:
                        price = float(row['price'])
                        btc_prices.append(price)
            
            self.btc_2024_data = btc_prices
            print(f"✅ Loaded {len(btc_prices)} days of 2024 BTC data")
            return True
            
        except Exception as e:
            print(f"❌ Error loading BTC data: {e}")
            return False
    
    def run_comparison_analysis(self) -> Dict[str, Any]:
        """Run comprehensive three-way strategy comparison"""
        
        print("\n🔍 THREE-WAY STRATEGY COMPARISON ANALYSIS")
        print("=" * 70)
        print("Comparing:")
        print("1. 🌊 High Tide Strategy - Borrow MOET → Buy Yield Tokens")
        print("2. 🪙 BTC Hold Strategy - Simply hold BTC (baseline)")
        print("3. 🏦 Aave Leverage Strategy - Borrow MOET → Buy more BTC (2.0 HF)")
        print()
        
        # Extract and normalize data
        self._extract_strategy_performance()
        
        # Create time series comparison
        self._create_time_series_comparison()
        
        # Analyze risk metrics
        self._analyze_risk_metrics()
        
        # Generate comparison charts
        self._generate_comparison_charts()
        
        # Save results
        self._save_comparison_results()
        
        # Print summary
        self._print_comparison_summary()
        
        return self.comparison_results
    
    def _extract_strategy_performance(self):
        """Extract and normalize performance data from both simulations"""
        
        print("📊 Extracting strategy performance data...")
        
        # Extract High Tide performance
        ht_sim_results = self.high_tide_data.get("simulation_results", {})
        ht_agent_health = ht_sim_results.get("agent_health_history", [])
        ht_btc_history = ht_sim_results.get("btc_price_history", [])
        
        # Extract Aave Leverage performance
        # Handle both nested (simulation_results) and top-level data structures
        al_sim_results = self.aave_leverage_data.get("simulation_results", {})
        if al_sim_results:
            # Nested structure (like High Tide)
            al_agent_health = al_sim_results.get("agent_health_history", [])
            al_btc_history = al_sim_results.get("btc_price_history", [])
        else:
            # Top-level structure (like Aave v2)
            al_agent_health = self.aave_leverage_data.get("agent_health_history", [])
            al_btc_history = self.aave_leverage_data.get("btc_price_history", [])
        
        # Ensure we have the same BTC price timeline
        if len(ht_btc_history) != len(al_btc_history):
            print(f"⚠️  BTC history length mismatch: HT={len(ht_btc_history)}, AL={len(al_btc_history)}")
            # Use the shorter one
            min_length = min(len(ht_btc_history), len(al_btc_history))
            ht_btc_history = ht_btc_history[:min_length]
            al_btc_history = al_btc_history[:min_length]
            ht_agent_health = ht_agent_health[:min_length]
            al_agent_health = al_agent_health[:min_length]
        
        # Calculate strategy performance over time
        self.comparison_results["strategy_performance"] = {
            "high_tide": self._calculate_strategy_performance(ht_agent_health, ht_btc_history, "High Tide"),
            "aave_leverage": self._calculate_strategy_performance(al_agent_health, al_btc_history, "Aave Leverage"),
            "btc_hold": self._calculate_btc_hold_performance(ht_btc_history)  # Use HT BTC history as reference
        }
        
        print("✅ Strategy performance data extracted")
    
    def _calculate_strategy_performance(self, agent_health_history: List, btc_history: List, strategy_name: str) -> Dict:
        """Calculate performance metrics for a strategy"""
        
        if not agent_health_history or not btc_history:
            return {"error": "No data available"}
        
        # Extract time series data
        days = []
        avg_net_positions = []
        survival_rates = []
        avg_health_factors = []
        
        initial_position = 42208.20  # 1 BTC at 2024 starting price
        
        for i, (health_snapshot, btc_price) in enumerate(zip(agent_health_history, btc_history)):
            day = i
            agents = health_snapshot.get("agents", [])
            
            if agents:
                # Calculate averages for active agents
                active_agents = [a for a in agents if a.get("active", True)]
                total_agents = len(agents)
                
                if active_agents:
                    net_positions = [a.get("net_position_value", initial_position) for a in active_agents]
                    health_factors = [a.get("health_factor", 1.25) for a in active_agents]
                    
                    avg_net_position = np.mean(net_positions)
                    avg_health_factor = np.mean(health_factors)
                    survival_rate = len(active_agents) / total_agents
                else:
                    avg_net_position = 0
                    avg_health_factor = 0
                    survival_rate = 0
            else:
                avg_net_position = initial_position
                avg_health_factor = 1.25
                survival_rate = 1.0
            
            days.append(day)
            avg_net_positions.append(avg_net_position)
            survival_rates.append(survival_rate)
            avg_health_factors.append(avg_health_factor)
        
        # Calculate performance metrics
        if avg_net_positions:
            final_position = avg_net_positions[-1]
            total_return = (final_position / initial_position - 1) * 100
            annualized_return = total_return  # Already annualized for full year
            max_position = max(avg_net_positions)
            min_position = min(avg_net_positions)
            volatility = np.std(avg_net_positions) / np.mean(avg_net_positions) * 100
        else:
            final_position = initial_position
            total_return = 0
            annualized_return = 0
            max_position = initial_position
            min_position = initial_position
            volatility = 0
        
        return {
            "strategy_name": strategy_name,
            "time_series": {
                "days": days,
                "avg_net_positions": avg_net_positions,
                "survival_rates": survival_rates,
                "avg_health_factors": avg_health_factors
            },
            "performance_metrics": {
                "initial_position": initial_position,
                "final_position": final_position,
                "total_return_pct": total_return,
                "annualized_return_pct": annualized_return,
                "max_position": max_position,
                "min_position": min_position,
                "volatility_pct": volatility,
                "final_survival_rate": survival_rates[-1] if survival_rates else 1.0
            }
        }
    
    def _calculate_btc_hold_performance(self, btc_history: List) -> Dict:
        """Calculate performance for simple BTC hold strategy"""
        
        if not btc_history:
            return {"error": "No BTC data available"}
        
        initial_position = 42208.20  # 1 BTC at 2024 starting price
        initial_btc_price = btc_history[0]
        btc_amount = 1.0  # Exactly 1 BTC
        
        # Calculate position value over time
        days = []
        net_positions = []
        
        for i, btc_price in enumerate(btc_history):
            day = i
            position_value = btc_amount * btc_price
            
            days.append(day)
            net_positions.append(position_value)
        
        # Calculate performance metrics
        final_position = net_positions[-1]
        total_return = (final_position / initial_position - 1) * 100
        annualized_return = total_return  # Already annualized for full year
        max_position = max(net_positions)
        min_position = min(net_positions)
        volatility = np.std(net_positions) / np.mean(net_positions) * 100
        
        return {
            "strategy_name": "BTC Hold",
            "time_series": {
                "days": days,
                "avg_net_positions": net_positions,
                "survival_rates": [1.0] * len(days),  # Always 100% survival
                "avg_health_factors": [float('inf')] * len(days)  # No leverage
            },
            "performance_metrics": {
                "initial_position": initial_position,
                "final_position": final_position,
                "total_return_pct": total_return,
                "annualized_return_pct": annualized_return,
                "max_position": max_position,
                "min_position": min_position,
                "volatility_pct": volatility,
                "final_survival_rate": 1.0
            }
        }
    
    def _create_time_series_comparison(self):
        """Create time series comparison data"""
        
        print("📈 Creating time series comparison...")
        
        strategies = self.comparison_results["strategy_performance"]
        
        # Align all time series to the same length
        min_length = min(len(strategies[s]["time_series"]["days"]) for s in strategies if "error" not in strategies[s])
        
        aligned_data = {}
        for strategy_key, strategy_data in strategies.items():
            if "error" not in strategy_data:
                ts = strategy_data["time_series"]
                aligned_data[strategy_key] = {
                    "days": ts["days"][:min_length],
                    "net_positions": ts["avg_net_positions"][:min_length],
                    "survival_rates": ts["survival_rates"][:min_length]
                }
        
        self.comparison_results["time_series_comparison"] = aligned_data
        
        print("✅ Time series comparison data created")
    
    def _analyze_risk_metrics(self):
        """Analyze risk metrics for all strategies"""
        
        print("⚖️  Analyzing risk metrics...")
        
        strategies = self.comparison_results["strategy_performance"]
        risk_analysis = {}
        
        for strategy_key, strategy_data in strategies.items():
            if "error" not in strategy_data:
                metrics = strategy_data["performance_metrics"]
                ts = strategy_data["time_series"]
                
                # Calculate additional risk metrics
                net_positions = ts["avg_net_positions"]
                returns = [((net_positions[i] / net_positions[i-1]) - 1) * 100 for i in range(1, len(net_positions))]
                
                if returns:
                    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                    max_drawdown = self._calculate_max_drawdown(net_positions)
                    downside_deviation = np.std([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0
                else:
                    sharpe_ratio = 0
                    max_drawdown = 0
                    downside_deviation = 0
                
                risk_analysis[strategy_key] = {
                    "sharpe_ratio": sharpe_ratio,
                    "max_drawdown_pct": max_drawdown,
                    "downside_deviation": downside_deviation,
                    "volatility_pct": metrics["volatility_pct"],
                    "final_survival_rate": metrics["final_survival_rate"]
                }
        
        self.comparison_results["risk_analysis"] = risk_analysis
        
        print("✅ Risk metrics analyzed")
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown percentage"""
        if not values:
            return 0
        
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak * 100
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _generate_comparison_charts(self):
        """Generate comprehensive comparison charts"""
        
        print("📊 Generating three-way comparison charts...")
        
        charts_dir = self.output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Chart 1: Enhanced Net APY Analysis (3 strategies)
        self._create_enhanced_net_apy_chart(charts_dir)
        
        # Chart 2: Strategy Performance Overview
        self._create_strategy_overview_chart(charts_dir)
        
        # Chart 3: Risk-Return Analysis
        self._create_risk_return_chart(charts_dir)
        
        # Chart 4: Time Series Evolution (4-panel)
        self._create_time_series_evolution_chart(charts_dir)
        
        print(f"📊 Charts saved to: {charts_dir}")
    
    def _create_enhanced_net_apy_chart(self, output_dir: Path):
        """Create enhanced Net APY analysis with three strategies"""
        
        strategies = self.comparison_results["strategy_performance"]
        ts_data = self.comparison_results["time_series_comparison"]
        
        if not ts_data:
            print("⚠️  No time series data for Net APY chart")
            return
        
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Net APY Analysis: Agent Performance vs BTC Hold Strategy', 
                     fontsize=16, fontweight='bold')
        
        # Define colors for strategies (matching original)
        colors = {
            "high_tide": "blue",         # Blue (matching original Agent Strategy)
            "btc_hold": "orange",        # Orange (matching original BTC Hold)  
            "aave_leverage": "green"     # Green (new Aave strategy)
        }
        
        labels = {
            "high_tide": "High Tide Strategy",
            "btc_hold": "BTC Hold Value",
            "aave_leverage": "Aave Leverage Strategy"
        }
        
        # Get common time axis (days)
        days = ts_data[list(ts_data.keys())[0]]["days"]
        
        # Top Left: Absolute Portfolio Values
        for strategy_key, data in ts_data.items():
            if strategy_key in colors:
                ax1.plot(data["days"], data["net_positions"], 
                        linewidth=2, color=colors[strategy_key], 
                        label=labels[strategy_key], alpha=0.8)
        
        ax1.set_title('Portfolio Value Comparison')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Top Right: Annualized Returns (APY) - matching original calculation
        initial_value = 42208.20  # 1 BTC at 2024 starting price
        hours = [d * 24 for d in days]  # Convert days to hours to match original
        
        strategy_apy_data = {}
        for strategy_key, data in ts_data.items():
            if strategy_key in colors:
                net_positions = data["net_positions"]
                apy_values = []
                
                for i, (hour, position) in enumerate(zip(hours, net_positions)):
                    days_elapsed = hour / 24.0
                    if days_elapsed > 0:
                        # Calculate annualized returns (matching original formula)
                        apy = (position / initial_value - 1) * (365 / days_elapsed) * 100
                    else:
                        apy = 0
                    apy_values.append(apy)
                
                strategy_apy_data[strategy_key] = apy_values
                ax2.plot(hours, apy_values, 
                        linewidth=2, color=colors[strategy_key], 
                        label=f'{labels[strategy_key]} APY', alpha=0.8)
        
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Annualized Percentage Yield (APY)')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('APY (%)')
        ax2.set_xlim(0, 8760)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Bottom Left: Relative Performance (matching original format)
        btc_hold_apy = strategy_apy_data["btc_hold"]
        
        for strategy_key in ["high_tide", "aave_leverage"]:
            if strategy_key in strategy_apy_data:
                strategy_apy = strategy_apy_data[strategy_key]
                
                # Calculate relative performance (Strategy APY - BTC Hold APY)
                relative_performance = [s_apy - btc_apy for s_apy, btc_apy in zip(strategy_apy, btc_hold_apy)]
                
                # Plot positive and negative separately for color coding (matching original)
                positive_mask = [x >= 0 for x in relative_performance]
                negative_mask = [x < 0 for x in relative_performance]
                
                ax3.fill_between(hours, 0, relative_performance, 
                               where=positive_mask, color=colors[strategy_key], alpha=0.7, 
                               interpolate=True, label=f'{labels[strategy_key]} Outperformance')
                ax3.fill_between(hours, 0, relative_performance, 
                               where=negative_mask, color=colors[strategy_key], alpha=0.3, 
                               interpolate=True, label=f'{labels[strategy_key]} Underperformance')
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_title('Relative Performance (Agent APY - BTC Hold APY)')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('APY Difference (%)')
        ax3.set_xlim(0, 8760)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Bottom Right: Average Outperformance (matching original calculation)
        for strategy_key in ["high_tide", "aave_leverage"]:
            if strategy_key in strategy_apy_data:
                strategy_apy = strategy_apy_data[strategy_key]
                relative_performance = [s_apy - btc_apy for s_apy, btc_apy in zip(strategy_apy, btc_hold_apy)]
                
                # Calculate average outperformance (matching original formula)
                average_outperformance = []
                running_sum = 0
                for perf in relative_performance:
                    running_sum += perf / len(relative_performance)  # Average the performance
                    average_outperformance.append(running_sum)
                
                ax4.plot(hours, average_outperformance, linewidth=2, color=colors[strategy_key], 
                        label=f'{labels[strategy_key]} Avg Outperformance')
                ax4.fill_between(hours, 0, average_outperformance, 
                               where=[x >= 0 for x in average_outperformance], 
                               color=colors[strategy_key], alpha=0.3)
        
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('Average Outperformance')
        ax4.set_xlabel('Hours')
        ax4.set_ylabel('Outperformance (%)')
        ax4.set_xlim(0, 8760)
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "enhanced_net_apy_three_way_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Enhanced Net APY chart created")
    
    def _create_strategy_overview_chart(self, output_dir: Path):
        """Create strategy performance overview chart"""
        
        strategies = self.comparison_results["strategy_performance"]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Strategy Performance Overview: Key Metrics Comparison', 
                     fontsize=16, fontweight='bold')
        
        # Extract metrics
        strategy_names = []
        final_returns = []
        volatilities = []
        survival_rates = []
        max_drawdowns = []
        
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green
        
        for i, (strategy_key, strategy_data) in enumerate(strategies.items()):
            if "error" not in strategy_data:
                metrics = strategy_data["performance_metrics"]
                risk_metrics = self.comparison_results["risk_analysis"].get(strategy_key, {})
                
                strategy_names.append(strategy_data["strategy_name"])
                final_returns.append(metrics["total_return_pct"])
                volatilities.append(metrics["volatility_pct"])
                survival_rates.append(metrics["final_survival_rate"] * 100)
                max_drawdowns.append(risk_metrics.get("max_drawdown_pct", 0))
        
        # Chart 1: Final Returns
        bars1 = ax1.bar(strategy_names, final_returns, color=colors, alpha=0.7)
        ax1.set_title('Total Return (%)')
        ax1.set_ylabel('Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, final_returns):
            ax1.annotate(f'{value:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # Chart 2: Volatility
        bars2 = ax2.bar(strategy_names, volatilities, color=colors, alpha=0.7)
        ax2.set_title('Volatility (%)')
        ax2.set_ylabel('Volatility (%)')
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, volatilities):
            ax2.annotate(f'{value:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # Chart 3: Survival Rate
        bars3 = ax3.bar(strategy_names, survival_rates, color=colors, alpha=0.7)
        ax3.set_title('Survival Rate (%)')
        ax3.set_ylabel('Survival Rate (%)')
        ax3.set_ylim(0, 105)
        ax3.grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, survival_rates):
            ax3.annotate(f'{value:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # Chart 4: Max Drawdown
        bars4 = ax4.bar(strategy_names, max_drawdowns, color=colors, alpha=0.7)
        ax4.set_title('Maximum Drawdown (%)')
        ax4.set_ylabel('Max Drawdown (%)')
        ax4.grid(True, alpha=0.3)
        
        for bar, value in zip(bars4, max_drawdowns):
            ax4.annotate(f'{value:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "strategy_performance_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Strategy overview chart created")
    
    def _create_risk_return_chart(self, output_dir: Path):
        """Create risk-return scatter plot"""
        
        strategies = self.comparison_results["strategy_performance"]
        risk_analysis = self.comparison_results["risk_analysis"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Risk-Return Analysis: Strategy Comparison', fontsize=16, fontweight='bold')
        
        # Extract data
        returns = []
        volatilities = []
        sharpe_ratios = []
        max_drawdowns = []
        strategy_names = []
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        
        for strategy_key, strategy_data in strategies.items():
            if "error" not in strategy_data:
                metrics = strategy_data["performance_metrics"]
                risk_metrics = risk_analysis.get(strategy_key, {})
                
                returns.append(metrics["total_return_pct"])
                volatilities.append(metrics["volatility_pct"])
                sharpe_ratios.append(risk_metrics.get("sharpe_ratio", 0))
                max_drawdowns.append(risk_metrics.get("max_drawdown_pct", 0))
                strategy_names.append(strategy_data["strategy_name"])
        
        # Chart 1: Return vs Volatility
        scatter1 = ax1.scatter(volatilities, returns, c=colors, s=200, alpha=0.7)
        
        for i, name in enumerate(strategy_names):
            ax1.annotate(name, (volatilities[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        ax1.set_xlabel('Volatility (%)')
        ax1.set_ylabel('Total Return (%)')
        ax1.set_title('Return vs Volatility')
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Return vs Max Drawdown
        scatter2 = ax2.scatter(max_drawdowns, returns, c=colors, s=200, alpha=0.7)
        
        for i, name in enumerate(strategy_names):
            ax2.annotate(name, (max_drawdowns[i], returns[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold')
        
        ax2.set_xlabel('Maximum Drawdown (%)')
        ax2.set_ylabel('Total Return (%)')
        ax2.set_title('Return vs Maximum Drawdown')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "risk_return_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Risk-return chart created")
    
    def _create_time_series_evolution_chart(self, output_dir: Path):
        """Create comprehensive time series evolution chart"""
        
        ts_data = self.comparison_results["time_series_comparison"]
        
        if not ts_data:
            print("⚠️  No time series data for evolution chart")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Time Series Evolution: Strategy Performance Over 2024', 
                     fontsize=16, fontweight='bold')
        
        colors = {
            "high_tide": "#1f77b4",
            "btc_hold": "#ff7f0e",
            "aave_leverage": "#2ca02c"
        }
        
        labels = {
            "high_tide": "High Tide Strategy",
            "btc_hold": "BTC Hold Strategy", 
            "aave_leverage": "Aave Leverage Strategy"
        }
        
        # Chart 1: Portfolio Values Over Time
        for strategy_key, data in ts_data.items():
            if strategy_key in colors:
                ax1.plot(data["days"], data["net_positions"], 
                        linewidth=2, color=colors[strategy_key], 
                        label=labels[strategy_key], alpha=0.8)
        
        ax1.set_title('Portfolio Value Evolution')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Chart 2: Survival Rates Over Time
        for strategy_key, data in ts_data.items():
            if strategy_key in colors:
                survival_pct = [rate * 100 for rate in data["survival_rates"]]
                ax2.plot(data["days"], survival_pct, 
                        linewidth=2, color=colors[strategy_key], 
                        label=labels[strategy_key], alpha=0.8)
        
        ax2.set_title('Survival Rate Evolution')
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Survival Rate (%)')
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Chart 3: Rolling 30-Day Returns
        for strategy_key, data in ts_data.items():
            if strategy_key in colors:
                positions = data["net_positions"]
                rolling_returns = []
                
                for i in range(len(positions)):
                    if i >= 30:  # 30-day window
                        start_value = positions[i-30]
                        end_value = positions[i]
                        if start_value > 0:
                            rolling_return = ((end_value / start_value) - 1) * 100
                        else:
                            rolling_return = 0
                    else:
                        rolling_return = 0
                    rolling_returns.append(rolling_return)
                
                ax3.plot(data["days"], rolling_returns, 
                        linewidth=2, color=colors[strategy_key], 
                        label=labels[strategy_key], alpha=0.8)
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Rolling 30-Day Returns')
        ax3.set_xlabel('Days')
        ax3.set_ylabel('30-Day Return (%)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Chart 4: Cumulative Performance Ratio (vs initial)
        for strategy_key, data in ts_data.items():
            if strategy_key in colors:
                positions = data["net_positions"]
                initial_position = positions[0] if positions else 100_000
                
                performance_ratios = [pos / initial_position for pos in positions]
                
                ax4.plot(data["days"], performance_ratios, 
                        linewidth=2, color=colors[strategy_key], 
                        label=labels[strategy_key], alpha=0.8)
        
        ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Break-even')
        ax4.set_title('Cumulative Performance Ratio')
        ax4.set_xlabel('Days')
        ax4.set_ylabel('Performance Ratio (vs Initial)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "time_series_evolution_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Time series evolution chart created")
    
    def _save_comparison_results(self):
        """Save comparison results to JSON"""
        
        results_path = self.output_dir / f"three_way_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.comparison_results, f, indent=2)
        
        print(f"📁 Comparison results saved to: {results_path}")
    
    def _print_comparison_summary(self):
        """Print comprehensive comparison summary"""
        
        print("\n📊 THREE-WAY STRATEGY COMPARISON SUMMARY")
        print("=" * 70)
        
        strategies = self.comparison_results["strategy_performance"]
        risk_analysis = self.comparison_results["risk_analysis"]
        
        print(f"{'Strategy':<20} {'Return':<12} {'Volatility':<12} {'Survival':<12} {'Max DD':<12}")
        print("-" * 70)
        
        for strategy_key, strategy_data in strategies.items():
            if "error" not in strategy_data:
                metrics = strategy_data["performance_metrics"]
                risk_metrics = risk_analysis.get(strategy_key, {})
                
                name = strategy_data["strategy_name"]
                return_pct = metrics["total_return_pct"]
                volatility = metrics["volatility_pct"]
                survival = metrics["final_survival_rate"] * 100
                max_dd = risk_metrics.get("max_drawdown_pct", 0)
                
                print(f"{name:<20} {return_pct:>8.1f}%   {volatility:>8.1f}%   {survival:>8.1f}%   {max_dd:>8.1f}%")
        
        print("\n🏆 WINNER ANALYSIS:")
        
        # Find best performing strategy by different metrics
        best_return = max(strategies.items(), 
                         key=lambda x: x[1]["performance_metrics"]["total_return_pct"] if "error" not in x[1] else -float('inf'))
        
        best_survival = max(strategies.items(),
                           key=lambda x: x[1]["performance_metrics"]["final_survival_rate"] if "error" not in x[1] else 0)
        
        lowest_risk = min(strategies.items(),
                         key=lambda x: x[1]["performance_metrics"]["volatility_pct"] if "error" not in x[1] else float('inf'))
        
        print(f"🥇 Highest Return: {best_return[1]['strategy_name']} ({best_return[1]['performance_metrics']['total_return_pct']:.1f}%)")
        print(f"🛡️  Best Survival: {best_survival[1]['strategy_name']} ({best_survival[1]['performance_metrics']['final_survival_rate']*100:.1f}%)")
        print(f"⚖️  Lowest Risk: {lowest_risk[1]['strategy_name']} ({lowest_risk[1]['performance_metrics']['volatility_pct']:.1f}% volatility)")


def main():
    """Main execution function"""
    
    print("Three-Way Strategy Comparison Analysis")
    print("=" * 50)
    print()
    print("This analysis compares three investment strategies using 2024 BTC data:")
    print("1. 🌊 High Tide Strategy - Borrow MOET to buy Yield Tokens")
    print("2. 🪙 BTC Hold Strategy - Simply hold BTC (baseline)")
    print("3. 🏦 Aave Leverage Strategy - Borrow MOET to buy more BTC (2.0 HF)")
    print()
    
    # Create comparison analysis
    comparison = ThreeWayStrategyComparison()
    
    # Load simulation data
    if not comparison.load_simulation_data():
        print("❌ Failed to load required simulation data")
        print("\nTo run this analysis, you need:")
        print("1. Full Year 2024 BTC Simulation results (already exists)")
        print("2. Aave Leverage Strategy simulation results")
        print("\nRun: python sim_tests/aave_leverage_strategy_sim.py")
        return None
    
    # Run comparison analysis
    try:
        results = comparison.run_comparison_analysis()
        
        print(f"\n🎯 THREE-WAY COMPARISON COMPLETED!")
        print(f"📊 Results and charts saved to: {comparison.output_dir}")
        
        return results
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Analysis interrupted by user.")
        return None
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
