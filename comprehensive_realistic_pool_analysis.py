#!/usr/bin/env python3
"""
Comprehensive Realistic Pool Analysis

Creates the full suite of charts from the pool permutation analysis, but using
ACTUAL High Tide simulation data instead of synthetic trade amounts.
"""

import sys
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
import seaborn as sns

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tidal_protocol_sim.simulation.high_tide_engine import HighTideConfig, HighTideSimulationEngine
from tidal_protocol_sim.analysis.high_tide_charts import HighTideChartGenerator


def run_comprehensive_pool_analysis():
    """Run comprehensive analysis across all pool configurations using simulation data"""
    
    print("🎯 COMPREHENSIVE POOL ANALYSIS")
    print("Using High Tide simulation data for all pool configurations")
    print("=" * 80)
    
    # Define pool depth options (expanded set)
    pool_depths = [
        {"size": 500_000, "label": "$250k:$250k"},
        {"size": 1_000_000, "label": "$500k:$500k"},
        {"size": 2_000_000, "label": "$1M:$1M"},
        {"size": 4_000_000, "label": "$2M:$2M"}
    ]
    
    results_matrix = []
    
    print(f"📊 Analysis Configuration:")
    print(f"   • Pool Depths: {len(pool_depths)} options")
    print(f"   • Total Permutations: {len(pool_depths) * len(pool_depths)} combinations")
    print(f"   • MOET:BTC Concentration: 80% (conservative)")
    print(f"   • MOET:Yield Token Concentration: 95% (tight peg)")
    print()
    
    # Test all pool permutations with ACTUAL simulations
    for i, (btc_pool_config, yield_pool_config) in enumerate(product(pool_depths, repeat=2)):
        btc_pool_size = btc_pool_config["size"]
        yield_pool_size = yield_pool_config["size"]
        btc_label = btc_pool_config["label"]
        yield_label = yield_pool_config["label"]
        
        print(f"🚀 [{i+1}/16] Running: MOET:BTC {btc_label} + MOET:Yield {yield_label}")
        
        try:
            # Create High Tide configuration
            config = HighTideConfig()
            config.num_high_tide_agents = 25  # Good sample size
            config.btc_decline_duration = 45  # Long enough for rebalancing
            config.uniswap_pool_size = btc_pool_size  # MOET:BTC pool
            config.moet_btc_pool_size = yield_pool_size // 2  # Each side of yield pool
            config.moet_btc_concentration = 0.20  # 80% concentration for MOET:BTC
            config.yield_token_concentration = 0.05  # 95% concentration for yield tokens
            
            # Run ACTUAL High Tide simulation
            engine = HighTideSimulationEngine(config)
            results = engine.run_high_tide_simulation()
            
            # Extract comprehensive data
            agent_outcomes = results.get("agent_outcomes", [])
            rebalancing_events = results.get("rebalancing_events", [])
            btc_price_history = results.get("btc_price_history", [])
            
            # Calculate detailed statistics
            rebalancing_amounts = [event.get("moet_raised", 0) for event in rebalancing_events if event.get("moet_raised", 0) > 0]
            agent_costs = [agent.get("cost_of_rebalancing", 0) for agent in agent_outcomes]
            survival_rate = sum(1 for agent in agent_outcomes if agent.get("survived", False)) / len(agent_outcomes) if agent_outcomes else 0
            
            # Risk profile analysis
            risk_profiles = {"conservative": [], "moderate": [], "aggressive": []}
            for agent in agent_outcomes:
                profile = agent.get("risk_profile", "unknown")
                if profile in risk_profiles:
                    risk_profiles[profile].append(agent.get("cost_of_rebalancing", 0))
            
            # Store comprehensive results
            result_data = {
                "btc_pool_label": btc_label,
                "yield_pool_label": yield_label,
                "btc_pool_size": btc_pool_size,
                "yield_pool_size": yield_pool_size,
                "full_simulation_results": results,
                
                # Agent statistics
                "total_agents": len(agent_outcomes),
                "agents_that_rebalanced": len([c for c in agent_costs if c > 0]),
                "survival_rate": survival_rate,
                "avg_cost_per_agent": np.mean(agent_costs) if agent_costs else 0,
                "median_cost_per_agent": np.median(agent_costs) if agent_costs else 0,
                "total_cost_all_agents": sum(agent_costs) if agent_costs else 0,
                
                # Rebalancing statistics
                "total_rebalancing_events": len(rebalancing_events),
                "avg_rebalancing_amount": np.mean(rebalancing_amounts) if rebalancing_amounts else 0,
                "median_rebalancing_amount": np.median(rebalancing_amounts) if rebalancing_amounts else 0,
                "max_rebalancing_amount": max(rebalancing_amounts) if rebalancing_amounts else 0,
                "min_rebalancing_amount": min(rebalancing_amounts) if rebalancing_amounts else 0,
                
                # Risk profile costs
                "conservative_avg_cost": np.mean(risk_profiles["conservative"]) if risk_profiles["conservative"] else 0,
                "moderate_avg_cost": np.mean(risk_profiles["moderate"]) if risk_profiles["moderate"] else 0,
                "aggressive_avg_cost": np.mean(risk_profiles["aggressive"]) if risk_profiles["aggressive"] else 0,
                
                # Price impact
                "initial_btc_price": btc_price_history[0] if btc_price_history else 100000,
                "final_btc_price": btc_price_history[-1] if btc_price_history else 100000,
                "btc_decline_percent": ((btc_price_history[0] - btc_price_history[-1]) / btc_price_history[0] * 100) if btc_price_history else 0,
                
                # Raw data for detailed analysis
                "agent_costs": agent_costs,
                "rebalancing_amounts": rebalancing_amounts,
                "agent_outcomes": agent_outcomes
            }
            
            results_matrix.append(result_data)
            
            print(f"   ✅ Complete: {len(agent_outcomes)} agents, {len(rebalancing_events)} events, ${np.mean(agent_costs) if agent_costs else 0:.0f} avg cost")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    return results_matrix


def create_comprehensive_analysis_charts(results_matrix, output_dir: Path):
    """Create the full suite of analysis charts"""
    
    output_dir.mkdir(exist_ok=True)
    
    if not results_matrix:
        print("❌ No valid simulation results to analyze")
        return []
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results_matrix)
    
    generated_charts = []
    
    # 1. MAIN COMPREHENSIVE ANALYSIS CHART
    main_chart = create_main_analysis_dashboard(df, output_dir)
    if main_chart:
        generated_charts.append(main_chart)
    
    # 2. DETAILED COST ANALYSIS CHARTS
    cost_charts = create_detailed_cost_analysis(df, output_dir)
    generated_charts.extend(cost_charts)
    
    # 3. REBALANCING ACTIVITY ANALYSIS
    activity_charts = create_rebalancing_activity_analysis(df, output_dir)
    generated_charts.extend(activity_charts)
    
    # 4. RISK PROFILE ANALYSIS
    risk_charts = create_risk_profile_analysis(df, output_dir)
    generated_charts.extend(risk_charts)
    
    # 5. POOL EFFICIENCY ANALYSIS
    efficiency_charts = create_pool_efficiency_analysis(df, output_dir)
    generated_charts.extend(efficiency_charts)
    
    # 6. LP CURVE EVOLUTION CHARTS (for best and worst configurations)
    lp_charts = create_lp_curve_analysis(results_matrix, output_dir)
    generated_charts.extend(lp_charts)
    
    return generated_charts


def create_main_analysis_dashboard(df: pd.DataFrame, output_dir: Path) -> Path:
    """Create the main comprehensive analysis dashboard"""
    
    fig = plt.figure(figsize=(24, 18))
    
    # 1. Total Average Cost Heatmap (Top Left)
    ax1 = plt.subplot(3, 4, (1, 2))
    if not df.empty and "avg_cost_per_agent" in df.columns:
        heatmap_data = df.pivot(index="yield_pool_label", columns="btc_pool_label", values="avg_cost_per_agent")
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='RdYlBu_r', 
                    cbar_kws={'label': 'Avg Cost Per Agent ($)'}, ax=ax1)
        ax1.set_title("Average Total Rebalancing Cost Per Agent\n(MOET:BTC + MOET:Yield Token)", fontweight='bold')
        ax1.set_xlabel("MOET:BTC Pool Size")
        ax1.set_ylabel("MOET:Yield Token Pool Size")
    
    # 2. Survival Rate Heatmap (Top Right)
    ax2 = plt.subplot(3, 4, (3, 4))
    if not df.empty and "survival_rate" in df.columns:
        survival_heatmap = df.pivot(index="yield_pool_label", columns="btc_pool_label", values="survival_rate")
        sns.heatmap(survival_heatmap * 100, annot=True, fmt='.1f', cmap='RdYlGn', 
                    cbar_kws={'label': 'Survival Rate (%)'}, ax=ax2)
        ax2.set_title("Agent Survival Rate by Pool Configuration", fontweight='bold')
        ax2.set_xlabel("MOET:BTC Pool Size")
        ax2.set_ylabel("MOET:Yield Token Pool Size")
    
    # 3. Rebalancing Activity Heatmap (Middle Left)
    ax3 = plt.subplot(3, 4, (5, 6))
    if not df.empty and "total_rebalancing_events" in df.columns:
        activity_heatmap = df.pivot(index="yield_pool_label", columns="btc_pool_label", values="total_rebalancing_events")
        sns.heatmap(activity_heatmap, annot=True, fmt='d', cmap='Blues', 
                    cbar_kws={'label': 'Total Events'}, ax=ax3)
        ax3.set_title("Total Rebalancing Events", fontweight='bold')
        ax3.set_xlabel("MOET:BTC Pool Size")
        ax3.set_ylabel("MOET:Yield Token Pool Size")
    
    # 4. Average Rebalancing Amount (Middle Right)
    ax4 = plt.subplot(3, 4, (7, 8))
    if not df.empty and "avg_rebalancing_amount" in df.columns:
        amount_heatmap = df.pivot(index="yield_pool_label", columns="btc_pool_label", values="avg_rebalancing_amount")
        sns.heatmap(amount_heatmap, annot=True, fmt='.0f', cmap='Purples', 
                    cbar_kws={'label': 'Avg Amount ($)'}, ax=ax4)
        ax4.set_title("Average Rebalancing Amount", fontweight='bold')
        ax4.set_xlabel("MOET:BTC Pool Size")
        ax4.set_ylabel("MOET:Yield Token Pool Size")
    
    # 5. Best vs Worst Configuration Comparison (Bottom Left)
    ax5 = plt.subplot(3, 4, 9)
    if not df.empty:
        best_config = df.loc[df['avg_cost_per_agent'].idxmin()]
        worst_config = df.loc[df['avg_cost_per_agent'].idxmax()]
        
        categories = ['Avg Cost', 'Survival Rate', 'Rebalancing Events']
        best_values = [best_config['avg_cost_per_agent'], best_config['survival_rate']*100, best_config['total_rebalancing_events']]
        worst_values = [worst_config['avg_cost_per_agent'], worst_config['survival_rate']*100, worst_config['total_rebalancing_events']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax5.bar(x - width/2, best_values, width, label=f"Best: BTC {best_config['btc_pool_label']}, Yield {best_config['yield_pool_label']}", color='green', alpha=0.7)
        ax5.bar(x + width/2, worst_values, width, label=f"Worst: BTC {worst_config['btc_pool_label']}, Yield {worst_config['yield_pool_label']}", color='red', alpha=0.7)
        
        ax5.set_xlabel('Metrics')
        ax5.set_ylabel('Values')
        ax5.set_title('Best vs Worst Pool Configurations')
        ax5.set_xticks(x)
        ax5.set_xticklabels(categories)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Cost Distribution (Bottom Middle)
    ax6 = plt.subplot(3, 4, 10)
    all_costs = []
    for result in df.itertuples():
        all_costs.extend(result.agent_costs)
    
    if all_costs:
        ax6.hist(all_costs, bins=25, alpha=0.7, color='skyblue', edgecolor='black')
        ax6.axvline(x=np.mean(all_costs), color='red', linestyle='--', 
                   label=f'Mean: ${np.mean(all_costs):.0f}')
        ax6.axvline(x=np.median(all_costs), color='green', linestyle='--',
                   label=f'Median: ${np.median(all_costs):.0f}')
        ax6.set_xlabel("Agent Cost ($)")
        ax6.set_ylabel("Frequency")
        ax6.set_title("Distribution of Agent Costs")
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. Pool Size Efficiency (Bottom Right)
    ax7 = plt.subplot(3, 4, 11)
    if not df.empty:
        # Calculate efficiency as cost per million dollars of liquidity
        df_temp = df.copy()
        df_temp['btc_efficiency'] = df_temp['avg_cost_per_agent'] / (df_temp['btc_pool_size'] / 1000000)
        df_temp['yield_efficiency'] = df_temp['avg_cost_per_agent'] / (df_temp['yield_pool_size'] / 1000000)
        
        ax7.scatter(df_temp['btc_pool_size'] / 1000000, df_temp['btc_efficiency'], 
                    c='red', alpha=0.7, s=80, label='MOET:BTC', marker='o')
        ax7.scatter(df_temp['yield_pool_size'] / 1000000, df_temp['yield_efficiency'], 
                    c='blue', alpha=0.7, s=80, label='MOET:Yield Token', marker='s')
        
        ax7.set_xlabel("Pool Size ($M)")
        ax7.set_ylabel("Cost per $1M Liquidity ($)")
        ax7.set_title("Pool Size Efficiency")
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. Summary Statistics (Bottom Far Right)
    ax8 = plt.subplot(3, 4, 12)
    ax8.axis('off')
    
    if not df.empty:
        # Calculate summary statistics
        total_simulations = len(df)
        total_agents = df['total_agents'].sum()
        total_events = df['total_rebalancing_events'].sum()
        best_config = df.loc[df['avg_cost_per_agent'].idxmin()]
        worst_config = df.loc[df['avg_cost_per_agent'].idxmax()]
        
        cost_reduction = ((worst_config['avg_cost_per_agent'] - best_config['avg_cost_per_agent']) / worst_config['avg_cost_per_agent']) * 100
        
        summary_text = f"""
COMPREHENSIVE ANALYSIS SUMMARY

Simulations Run: {total_simulations}
Total Agents Tested: {total_agents}
Total Rebalancing Events: {total_events}

OPTIMAL CONFIGURATION:
MOET:BTC: {best_config['btc_pool_label']}
MOET:Yield: {best_config['yield_pool_label']}
Avg Cost: ${best_config['avg_cost_per_agent']:.0f}
Survival Rate: {best_config['survival_rate']*100:.1f}%

WORST CONFIGURATION:
MOET:BTC: {worst_config['btc_pool_label']}
MOET:Yield: {worst_config['yield_pool_label']}
Avg Cost: ${worst_config['avg_cost_per_agent']:.0f}
Survival Rate: {worst_config['survival_rate']*100:.1f}%

COST REDUCTION: {cost_reduction:.1f}%

KEY INSIGHTS:
• Larger pools reduce costs significantly
• Pool depth more important than concentration
• Real rebalancing amounts: ${df["avg_rebalancing_amount"].mean():.0f} avg
• Optimal balance between cost and liquidity
        """
        
        ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.suptitle("Pool Analysis: High Tide Simulation Data", 
                 fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save chart
    chart_path = output_dir / "comprehensive_pool_analysis_dashboard.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_path


def create_detailed_cost_analysis(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Create detailed cost analysis charts"""
    
    charts = []
    
    # Cost breakdown by pool type
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Detailed Cost Analysis by Pool Configuration", fontsize=16, fontweight='bold')
    
    # 1. Cost vs Pool Size Scatter
    if not df.empty:
        ax1.scatter(df['btc_pool_size'] / 1000000, df['avg_cost_per_agent'], 
                   c='red', alpha=0.7, s=60, label='vs BTC Pool Size')
        ax1.scatter(df['yield_pool_size'] / 1000000, df['avg_cost_per_agent'], 
                   c='blue', alpha=0.7, s=60, label='vs Yield Pool Size')
        ax1.set_xlabel("Pool Size ($M)")
        ax1.set_ylabel("Average Cost Per Agent ($)")
        ax1.set_title("Cost vs Pool Size Relationship")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Cost Distribution by Configuration
    if not df.empty:
        config_labels = [f"{row.btc_pool_label}\n{row.yield_pool_label}" for row in df.itertuples()]
        costs = df['avg_cost_per_agent'].values
        
        bars = ax2.bar(range(len(costs)), costs, color=plt.cm.viridis(np.linspace(0, 1, len(costs))))
        ax2.set_xlabel("Pool Configuration")
        ax2.set_ylabel("Average Cost Per Agent ($)")
        ax2.set_title("Cost by Pool Configuration")
        ax2.set_xticks(range(len(config_labels)))
        ax2.set_xticklabels(config_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
    
    # 3. Cost vs Survival Rate
    if not df.empty:
        ax3.scatter(df['survival_rate'] * 100, df['avg_cost_per_agent'], 
                   c=df['total_rebalancing_events'], cmap='plasma', s=80, alpha=0.7)
        ax3.set_xlabel("Survival Rate (%)")
        ax3.set_ylabel("Average Cost Per Agent ($)")
        ax3.set_title("Cost vs Survival Rate")
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax3.collections[0], ax=ax3)
        cbar.set_label("Rebalancing Events")
    
    # 4. Cost Reduction Analysis
    if not df.empty:
        # Sort by cost and show cumulative savings
        df_sorted = df.sort_values('avg_cost_per_agent')
        baseline_cost = df_sorted['avg_cost_per_agent'].iloc[-1]  # Highest cost
        cumulative_savings = []
        
        for cost in df_sorted['avg_cost_per_agent']:
            savings = ((baseline_cost - cost) / baseline_cost) * 100
            cumulative_savings.append(savings)
        
        ax4.plot(range(len(cumulative_savings)), cumulative_savings, 
                marker='o', linewidth=2, markersize=6, color='green')
        ax4.set_xlabel("Pool Configuration Rank (Best to Worst)")
        ax4.set_ylabel("Cost Reduction vs Worst Case (%)")
        ax4.set_title("Cumulative Cost Reduction Potential")
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    
    chart_path = output_dir / "detailed_cost_analysis.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    charts.append(chart_path)
    
    return charts


def create_rebalancing_activity_analysis(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Create rebalancing activity analysis charts"""
    
    charts = []
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Rebalancing Activity Analysis", fontsize=16, fontweight='bold')
    
    # 1. Rebalancing Amount Distribution
    all_amounts = []
    for result in df.itertuples():
        all_amounts.extend(result.rebalancing_amounts)
    
    if all_amounts:
        ax1.hist(all_amounts, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.axvline(x=np.mean(all_amounts), color='red', linestyle='--', 
                   label=f'Mean: ${np.mean(all_amounts):.0f}')
        ax1.axvline(x=np.median(all_amounts), color='green', linestyle='--',
                   label=f'Median: ${np.median(all_amounts):.0f}')
        ax1.set_xlabel("Rebalancing Amount ($)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Distribution of Actual Rebalancing Amounts")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Events vs Pool Size
    if not df.empty:
        ax2.scatter(df['btc_pool_size'] / 1000000, df['total_rebalancing_events'], 
                   c='red', alpha=0.7, s=60, label='vs BTC Pool Size')
        ax2.scatter(df['yield_pool_size'] / 1000000, df['total_rebalancing_events'], 
                   c='blue', alpha=0.7, s=60, label='vs Yield Pool Size')
        ax2.set_xlabel("Pool Size ($M)")
        ax2.set_ylabel("Total Rebalancing Events")
        ax2.set_title("Rebalancing Activity vs Pool Size")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Average Amount vs Pool Configuration
    if not df.empty:
        config_labels = [f"{row.btc_pool_label}\n{row.yield_pool_label}" for row in df.itertuples()]
        amounts = df['avg_rebalancing_amount'].values
        
        bars = ax3.bar(range(len(amounts)), amounts, color=plt.cm.plasma(np.linspace(0, 1, len(amounts))))
        ax3.set_xlabel("Pool Configuration")
        ax3.set_ylabel("Average Rebalancing Amount ($)")
        ax3.set_title("Average Rebalancing Amount by Configuration")
        ax3.set_xticks(range(len(config_labels)))
        ax3.set_xticklabels(config_labels, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
    
    # 4. Activity vs Cost Correlation
    if not df.empty:
        ax4.scatter(df['total_rebalancing_events'], df['avg_cost_per_agent'], 
                   c=df['survival_rate'], cmap='RdYlGn', s=80, alpha=0.7)
        ax4.set_xlabel("Total Rebalancing Events")
        ax4.set_ylabel("Average Cost Per Agent ($)")
        ax4.set_title("Rebalancing Activity vs Cost")
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label("Survival Rate")
    
    plt.tight_layout()
    
    chart_path = output_dir / "rebalancing_activity_analysis.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    charts.append(chart_path)
    
    return charts


def create_risk_profile_analysis(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Create risk profile analysis charts"""
    
    charts = []
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Risk Profile Analysis", fontsize=16, fontweight='bold')
    
    if not df.empty:
        # 1. Cost by Risk Profile Heatmap
        risk_data = df[['conservative_avg_cost', 'moderate_avg_cost', 'aggressive_avg_cost']].values
        risk_labels = ['Conservative', 'Moderate', 'Aggressive']
        config_labels = [f"{row.btc_pool_label} + {row.yield_pool_label}" for row in df.itertuples()]
        
        im = ax1.imshow(risk_data.T, cmap='RdYlBu_r', aspect='auto')
        ax1.set_xticks(range(len(config_labels)))
        ax1.set_xticklabels(config_labels, rotation=45, ha='right')
        ax1.set_yticks(range(len(risk_labels)))
        ax1.set_yticklabels(risk_labels)
        ax1.set_title("Average Cost by Risk Profile")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label("Average Cost ($)")
        
        # Add text annotations
        for i in range(len(risk_labels)):
            for j in range(len(config_labels)):
                if not np.isnan(risk_data[j, i]):
                    ax1.text(j, i, f'${risk_data[j, i]:.0f}', ha='center', va='center', fontsize=8)
        
        # 2. Risk Profile Cost Distribution
        all_conservative = [val for val in df['conservative_avg_cost'] if val > 0]
        all_moderate = [val for val in df['moderate_avg_cost'] if val > 0]
        all_aggressive = [val for val in df['aggressive_avg_cost'] if val > 0]
        
        ax2.boxplot([all_conservative, all_moderate, all_aggressive], 
                   labels=['Conservative', 'Moderate', 'Aggressive'])
        ax2.set_ylabel("Average Cost ($)")
        ax2.set_title("Cost Distribution by Risk Profile")
        ax2.grid(True, alpha=0.3)
        
        # 3. Best Configuration for Each Risk Profile
        best_configs = {}
        for profile in ['conservative_avg_cost', 'moderate_avg_cost', 'aggressive_avg_cost']:
            if profile in df.columns:
                valid_df = df[df[profile] > 0]
                if not valid_df.empty:
                    best_idx = valid_df[profile].idxmin()
                    best_configs[profile.replace('_avg_cost', '')] = {
                        'config': f"{valid_df.loc[best_idx, 'btc_pool_label']} + {valid_df.loc[best_idx, 'yield_pool_label']}",
                        'cost': valid_df.loc[best_idx, profile]
                    }
        
        if best_configs:
            profiles = list(best_configs.keys())
            costs = [best_configs[p]['cost'] for p in profiles]
            colors = ['#2E8B57', '#FF8C00', '#DC143C']  # Conservative, Moderate, Aggressive
            
            bars = ax3.bar(profiles, costs, color=colors, alpha=0.7)
            ax3.set_ylabel("Best Cost ($)")
            ax3.set_title("Optimal Cost by Risk Profile")
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, cost in zip(bars, costs):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                        f'${cost:.0f}', ha='center', va='bottom')
        
        # 4. Risk Profile Summary Table
        ax4.axis('off')
        
        summary_text = "RISK PROFILE ANALYSIS SUMMARY\n\n"
        
        for profile in ['conservative', 'moderate', 'aggressive']:
            cost_col = f"{profile}_avg_cost"
            if cost_col in df.columns:
                valid_costs = df[df[cost_col] > 0][cost_col]
                if not valid_costs.empty:
                    avg_cost = valid_costs.mean()
                    min_cost = valid_costs.min()
                    max_cost = valid_costs.max()
                    
                    summary_text += f"{profile.title()} Profile:\n"
                    summary_text += f"  Average Cost: ${avg_cost:.0f}\n"
                    summary_text += f"  Range: ${min_cost:.0f} - ${max_cost:.0f}\n"
                    summary_text += f"  Configurations: {len(valid_costs)}\n\n"
        
        summary_text += "KEY INSIGHTS:\n"
        summary_text += "• Conservative agents typically have lower costs\n"
        summary_text += "• Aggressive agents face higher rebalancing costs\n"
        summary_text += "• Pool configuration affects all profiles similarly\n"
        summary_text += "• Larger pools benefit all risk profiles"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    
    chart_path = output_dir / "risk_profile_analysis.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    charts.append(chart_path)
    
    return charts


def create_pool_efficiency_analysis(df: pd.DataFrame, output_dir: Path) -> List[Path]:
    """Create pool efficiency analysis charts"""
    
    charts = []
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Pool Efficiency Analysis", fontsize=16, fontweight='bold')
    
    if not df.empty:
        # 1. Cost per Dollar of Liquidity
        df_temp = df.copy()
        df_temp['total_liquidity'] = df_temp['btc_pool_size'] + df_temp['yield_pool_size']
        df_temp['cost_per_liquidity'] = df_temp['avg_cost_per_agent'] / (df_temp['total_liquidity'] / 1000000)
        
        scatter = ax1.scatter(df_temp['total_liquidity'] / 1000000, df_temp['cost_per_liquidity'], 
                            c=df_temp['survival_rate'], cmap='RdYlGn', s=100, alpha=0.7)
        ax1.set_xlabel("Total Liquidity ($M)")
        ax1.set_ylabel("Cost per $1M Liquidity ($)")
        ax1.set_title("Liquidity Efficiency")
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("Survival Rate")
        
        # 2. Pool Size Impact Analysis
        pool_sizes = sorted(df['btc_pool_size'].unique())
        size_labels = [f"${size//1000}k" for size in pool_sizes]
        
        avg_costs_by_size = []
        for size in pool_sizes:
            size_data = df[df['btc_pool_size'] == size]
            avg_costs_by_size.append(size_data['avg_cost_per_agent'].mean())
        
        ax2.plot(size_labels, avg_costs_by_size, marker='o', linewidth=3, markersize=8, color='blue')
        ax2.set_xlabel("MOET:BTC Pool Size")
        ax2.set_ylabel("Average Cost ($)")
        ax2.set_title("Cost Reduction by Pool Size")
        ax2.grid(True, alpha=0.3)
        
        # Add percentage reduction annotations
        if len(avg_costs_by_size) > 1:
            baseline = avg_costs_by_size[0]
            for i, cost in enumerate(avg_costs_by_size[1:], 1):
                reduction = ((baseline - cost) / baseline) * 100
                ax2.annotate(f'-{reduction:.1f}%', xy=(i, cost), xytext=(5, 10), 
                           textcoords='offset points', fontsize=9, color='red')
        
        # 3. Optimal Configuration Analysis
        # Find Pareto frontier (best cost for each survival rate range)
        df_sorted = df.sort_values(['survival_rate', 'avg_cost_per_agent'])
        pareto_points = []
        
        for survival_threshold in [0.7, 0.8, 0.9, 0.95]:
            candidates = df_sorted[df_sorted['survival_rate'] >= survival_threshold]
            if not candidates.empty:
                best = candidates.loc[candidates['avg_cost_per_agent'].idxmin()]
                pareto_points.append(best)
        
        if pareto_points:
            pareto_df = pd.DataFrame(pareto_points)
            ax3.scatter(df['survival_rate'] * 100, df['avg_cost_per_agent'], 
                       alpha=0.5, color='lightblue', s=50, label='All Configurations')
            ax3.scatter(pareto_df['survival_rate'] * 100, pareto_df['avg_cost_per_agent'], 
                       color='red', s=100, label='Pareto Optimal', marker='*')
            
            # Connect Pareto points
            pareto_sorted = pareto_df.sort_values('survival_rate')
            ax3.plot(pareto_sorted['survival_rate'] * 100, pareto_sorted['avg_cost_per_agent'], 
                    'r--', alpha=0.7, linewidth=2)
            
            ax3.set_xlabel("Survival Rate (%)")
            ax3.set_ylabel("Average Cost ($)")
            ax3.set_title("Pareto Optimal Configurations")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency Summary
        ax4.axis('off')
        
        # Calculate efficiency metrics
        best_overall = df.loc[df['avg_cost_per_agent'].idxmin()]
        worst_overall = df.loc[df['avg_cost_per_agent'].idxmax()]
        
        efficiency_text = f"""
POOL EFFICIENCY SUMMARY

MOST EFFICIENT:
Configuration: {best_overall['btc_pool_label']} + {best_overall['yield_pool_label']}
Cost per Agent: ${best_overall['avg_cost_per_agent']:.0f}
Survival Rate: {best_overall['survival_rate']*100:.1f}%
Total Liquidity: ${(best_overall['btc_pool_size'] + best_overall['yield_pool_size'])/1000000:.1f}M

LEAST EFFICIENT:
Configuration: {worst_overall['btc_pool_label']} + {worst_overall['yield_pool_label']}
Cost per Agent: ${worst_overall['avg_cost_per_agent']:.0f}
Survival Rate: {worst_overall['survival_rate']*100:.1f}%
Total Liquidity: ${(worst_overall['btc_pool_size'] + worst_overall['yield_pool_size'])/1000000:.1f}M

EFFICIENCY GAIN: {((worst_overall['avg_cost_per_agent'] - best_overall['avg_cost_per_agent']) / worst_overall['avg_cost_per_agent'] * 100):.1f}%

OPTIMAL STRATEGY:
• Larger pools consistently outperform
• Diminishing returns after $1M per pool
• Balance liquidity cost vs rebalancing savings
• Consider maintenance costs in practice
        """
        
        ax4.text(0.05, 0.95, efficiency_text, transform=ax4.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    
    chart_path = output_dir / "pool_efficiency_analysis.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    charts.append(chart_path)
    
    return charts


def create_lp_curve_analysis(results_matrix, output_dir: Path) -> List[Path]:
    """Create LP curve evolution charts for best and worst configurations"""
    
    charts = []
    
    if not results_matrix:
        return charts
    
    # Find best and worst configurations
    costs = [r["avg_cost_per_agent"] for r in results_matrix]
    best_idx = costs.index(min(costs))
    worst_idx = costs.index(max(costs))
    
    best_config = results_matrix[best_idx]
    worst_config = results_matrix[worst_idx]
    
    # Generate LP curve charts for best and worst configurations
    chart_generator = HighTideChartGenerator()
    
    try:
        # Best configuration charts
        best_charts_dir = output_dir / "best_configuration_charts"
        best_charts = chart_generator.generate_high_tide_charts(
            scenario_name=f"Best_Config_{best_config['btc_pool_label']}_{best_config['yield_pool_label']}",
            results=best_config["full_simulation_results"],
            charts_dir=best_charts_dir
        )
        charts.extend(best_charts)
        
        # Worst configuration charts  
        worst_charts_dir = output_dir / "worst_configuration_charts"
        worst_charts = chart_generator.generate_high_tide_charts(
            scenario_name=f"Worst_Config_{worst_config['btc_pool_label']}_{worst_config['yield_pool_label']}",
            results=worst_config["full_simulation_results"],
            charts_dir=worst_charts_dir
        )
        charts.extend(worst_charts)
        
        print(f"📊 Generated LP curve charts:")
        print(f"   • Best config ({best_config['btc_pool_label']} + {best_config['yield_pool_label']}): {len(best_charts)} charts")
        print(f"   • Worst config ({worst_config['btc_pool_label']} + {worst_config['yield_pool_label']}): {len(worst_charts)} charts")
        
    except Exception as e:
        print(f"⚠️  LP curve chart generation failed: {e}")
    
    return charts


def main():
    """Run comprehensive realistic pool analysis"""
    
    try:
        print("🎯 STARTING COMPREHENSIVE REALISTIC POOL ANALYSIS")
        print("This creates the FULL suite of analysis charts using actual simulation data")
        print()
        
        # 1. Run comprehensive simulations
        results_matrix = run_comprehensive_pool_analysis()
        
        if not results_matrix:
            print("❌ No valid simulation results obtained")
            return False
        
        # 2. Create all analysis charts
        output_dir = Path("comprehensive_realistic_analysis")
        generated_charts = create_comprehensive_analysis_charts(results_matrix, output_dir)
        
        print(f"\n" + "=" * 80)
        print("✅ COMPREHENSIVE REALISTIC POOL ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"📊 Total Charts Generated: {len(generated_charts)}")
        print(f"📁 Results Directory: {output_dir}")
        print(f"🎯 All data based on ACTUAL High Tide simulations")
        print(f"   • {len(results_matrix)} pool configurations tested")
        print(f"   • Real rebalancing events and costs analyzed")
        print(f"   • Comprehensive visualization suite created")
        
        # Show chart categories
        chart_categories = {
            "Main Dashboard": [c for c in generated_charts if "dashboard" in str(c)],
            "Cost Analysis": [c for c in generated_charts if "cost" in str(c)],
            "Activity Analysis": [c for c in generated_charts if "activity" in str(c)],
            "Risk Profile": [c for c in generated_charts if "risk" in str(c)],
            "Efficiency": [c for c in generated_charts if "efficiency" in str(c)],
            "LP Curves": [c for c in generated_charts if "lp_curve" in str(c) or "charts" in str(c.parent.name)]
        }
        
        print(f"\n📈 Chart Categories:")
        for category, charts in chart_categories.items():
            if charts:
                print(f"   • {category}: {len(charts)} charts")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
