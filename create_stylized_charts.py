#!/usr/bin/env python3
"""
Create Stylized Charts for Report
Enhanced visualization of liquidation capacity analysis
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

def load_data():
    """Load the JSON results data"""
    results_path = Path("tidal_protocol_sim/results/moet_yt_borrow_cap_analysis/run_001_20250904_130630/liquidation_capacity_results.json")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    return data, results_path.parent / "charts"

def create_slippage_cost_vs_liquidations_chart(data, output_dir):
    """Enhanced Slippage Cost vs Liquidations scatter plot"""
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color scheme
    colors = {1.0: '#2E8B57', 2.0: '#4682B4', 3.0: '#FF8C00', 4.0: '#DC143C', 5.0: '#8B0000'}
    shock_markers = {10: 'o', 15: 's', 25: '^'}
    shock_sizes = {10: 80, 15: 100, 25: 120}
    
    # Extract data for scatter plot
    for scenario in data["detailed_scenario_results"]:
        ratio = scenario["deposit_cap_ratio"]
        
        for shock_result in scenario["shock_scenario_results"]:
            shock_pct = shock_result["scenario_params"]["btc_shock_percent"]
            
            # Get liquidation events
            liquidation_events = shock_result["dex_capacity_analysis"]["liquidation_events"]
            
            for event in liquidation_events:
                cumulative_liquidations = event["cumulative_liquidations"]
                slippage_cost = event["slippage_cost"]
                
                ax.scatter(cumulative_liquidations, slippage_cost,
                          c=colors[ratio], marker=shock_markers[shock_pct], 
                          s=shock_sizes[shock_pct], alpha=0.8, 
                          edgecolors='black', linewidth=1)
    
    # Add trend lines for each ratio
    for ratio in [1.0, 2.0, 3.0, 4.0, 5.0]:
        x_vals = []
        y_vals = []
        
        for scenario in data["detailed_scenario_results"]:
            if scenario["deposit_cap_ratio"] == ratio:
                for shock_result in scenario["shock_scenario_results"]:
                    liquidation_events = shock_result["dex_capacity_analysis"]["liquidation_events"]
                    for event in liquidation_events:
                        x_vals.append(event["cumulative_liquidations"])
                        y_vals.append(event["slippage_cost"])
        
        if len(x_vals) > 1:
            # Fit polynomial trend line
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(x_vals), max(x_vals), 100)
            ax.plot(x_trend, p(x_trend), color=colors[ratio], alpha=0.6, linewidth=2, linestyle='--')
    
    # Formatting
    ax.set_xlabel('Cumulative Liquidations ($)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Slippage Cost per Liquidation ($)', fontsize=14, fontweight='bold')
    ax.set_title('Slippage Cost Efficiency vs Liquidation Volume\nHigher deposit ratios show increased slippage costs', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Create custom legends
    ratio_legend = [plt.scatter([], [], c=colors[r], s=100, edgecolors='black', 
                               label=f'{r:.0f}:1 Ratio') for r in colors.keys()]
    shock_legend = [plt.scatter([], [], c='gray', marker=shock_markers[s], s=shock_sizes[s],
                               edgecolors='black', label=f'{s}% Shock') for s in shock_markers.keys()]
    
    legend1 = ax.legend(handles=ratio_legend, loc='upper left', title='Deposit Cap Ratio', 
                       title_fontsize=12, fontsize=10)
    legend2 = ax.legend(handles=shock_legend, loc='upper right', title='BTC Shock Scenario',
                       title_fontsize=12, fontsize=10)
    ax.add_artist(legend1)
    
    # Grid and styling
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    output_path = output_dir / "enhanced_slippage_vs_liquidations.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path

def create_stacked_liquidation_dollars_chart(data, output_dir):
    """Enhanced stacked bar chart for liquidation dollars"""
    
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Extract data for stacked bars
    ratios = []
    shock_10_vals = []
    shock_15_vals = []
    shock_25_vals = []
    avg_liquidations = []
    
    for scenario in data["detailed_scenario_results"]:
        ratio = scenario["deposit_cap_ratio"]
        ratios.append(f"{ratio:.0f}:1")
        
        shock_data = {10: 0, 15: 0, 25: 0}
        total_liquidations = 0
        count_liquidations = 0
        
        for shock_result in scenario["shock_scenario_results"]:
            shock_pct = shock_result["scenario_params"]["btc_shock_percent"]
            total_liquidated = shock_result["liquidation_metrics"]["total_liquidated_value"]
            positions_liquidated = shock_result["liquidation_metrics"]["positions_successfully_liquidated"]
            
            shock_data[shock_pct] = total_liquidated
            
            if positions_liquidated > 0:
                total_liquidations += total_liquidated
                count_liquidations += positions_liquidated
        
        shock_10_vals.append(shock_data[10])
        shock_15_vals.append(shock_data[15])
        shock_25_vals.append(shock_data[25])
        avg_liquidations.append(total_liquidations / max(count_liquidations, 1))
    
    # Chart 1: Stacked Total Dollars
    width = 0.6
    x = np.arange(len(ratios))
    
    bars1 = ax1.bar(x, shock_10_vals, width, label='10% Shock', color='#90EE90', edgecolor='black')
    bars2 = ax1.bar(x, shock_15_vals, width, bottom=shock_10_vals, label='15% Shock', 
                    color='#FFB347', edgecolor='black')
    bars3 = ax1.bar(x, shock_25_vals, width, 
                    bottom=np.array(shock_10_vals) + np.array(shock_15_vals),
                    label='25% Shock', color='#FF6B6B', edgecolor='black')
    
    # Add value labels
    for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
        # Total at top
        total = shock_10_vals[i] + shock_15_vals[i] + shock_25_vals[i]
        ax1.text(bar3.get_x() + bar3.get_width()/2., total + 1000,
                f'${total:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax1.set_xlabel('Deposit Cap Ratio', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Dollars Liquidated ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Total Liquidation Volume by Shock Scenario\nStacked by shock severity', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ratios)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Chart 2: Average Dollars per Liquidation
    bars_avg = ax2.bar(ratios, avg_liquidations, width, color='#FFD700', 
                       edgecolor='black', alpha=0.8)
    
    # Add value labels
    for bar, avg_val in zip(bars_avg, avg_liquidations):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
                f'${avg_val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_xlabel('Deposit Cap Ratio', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Dollars per Liquidation ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Average Liquidation Size\nScale efficiency analysis', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "enhanced_liquidation_dollars_stacked.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path

def create_enhanced_liquidation_heatmap(data, output_dir):
    """Enhanced heatmap of liquidation dollars"""
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for heatmap
    heatmap_data = []
    ratios = []
    shocks = [10, 15, 25]
    
    for scenario in data["detailed_scenario_results"]:
        ratio = scenario["deposit_cap_ratio"]
        ratios.append(f"{ratio:.0f}:1")
        
        row_data = []
        for shock_pct in shocks:
            # Find matching shock scenario
            total_liquidated = 0
            for shock_result in scenario["shock_scenario_results"]:
                if shock_result["scenario_params"]["btc_shock_percent"] == shock_pct:
                    total_liquidated = shock_result["liquidation_metrics"]["total_liquidated_value"]
                    break
            row_data.append(total_liquidated)
        heatmap_data.append(row_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(heatmap_data, index=ratios, columns=[f'{s}%' for s in shocks])
    
    # Create enhanced heatmap
    sns.heatmap(df, annot=True, fmt=',.0f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Total Liquidated ($)'}, 
                annot_kws={'fontsize': 12, 'fontweight': 'bold'},
                linewidths=1, linecolor='black')
    
    # Formatting
    ax.set_xlabel('BTC Shock Severity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Deposit Cap Ratio', fontsize=14, fontweight='bold')
    ax.set_title('Liquidation Volume Heatmap\nTotal dollars liquidated by ratio and shock severity', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    output_path = output_dir / "enhanced_liquidation_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path

def create_concentration_risk_analysis(data, output_dir):
    """Create concentration risk vs liquidation efficiency chart"""
    
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Extract data for analysis
    ratios = []
    max_concentrations = []
    total_liquidations = []
    efficiency_ratios = []  # Liquidated value / Concentration used
    
    for scenario in data["detailed_scenario_results"]:
        ratio = scenario["deposit_cap_ratio"]
        ratios.append(ratio)
        
        max_conc = 0
        total_liq = 0
        
        for shock_result in scenario["shock_scenario_results"]:
            conc = shock_result["liquidation_metrics"]["concentration_utilization"]
            liquidated = shock_result["liquidation_metrics"]["total_liquidated_value"]
            
            max_conc = max(max_conc, conc)
            total_liq += liquidated
        
        max_concentrations.append(max_conc)
        total_liquidations.append(total_liq)
        efficiency_ratios.append(total_liq / max_conc if max_conc > 0 else 0)
    
    # Chart 1: Risk vs Capacity
    ax1_twin = ax1.twinx()
    
    # Bar chart for total liquidations
    bars = ax1.bar([f"{r:.0f}:1" for r in ratios], total_liquidations, 
                   color='#4682B4', alpha=0.7, edgecolor='black', 
                   label='Total Liquidation Capacity')
    
    # Line chart for max concentration
    line = ax1_twin.plot([f"{r:.0f}:1" for r in ratios], max_concentrations, 
                         color='#DC143C', marker='o', linewidth=3, markersize=8,
                         label='Max Concentration Risk')
    
    # Add value labels
    for i, (bar, conc, total) in enumerate(zip(bars, max_concentrations, total_liquidations)):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2000,
                f'${total:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax1_twin.text(i, conc + 1, f'{conc:.1f}%', ha='center', va='bottom', 
                     fontweight='bold', fontsize=10, color='#DC143C')
    
    ax1.set_xlabel('Deposit Cap Ratio', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Liquidation Capacity ($)', fontsize=12, fontweight='bold', color='#4682B4')
    ax1_twin.set_ylabel('Maximum Concentration Utilization (%)', fontsize=12, fontweight='bold', color='#DC143C')
    ax1.set_title('Liquidation Capacity vs Concentration Risk\nHigher ratios increase both capacity and risk', 
                  fontsize=14, fontweight='bold')
    
    # Color coordinate the y-axis labels
    ax1.tick_params(axis='y', labelcolor='#4682B4')
    ax1_twin.tick_params(axis='y', labelcolor='#DC143C')
    
    ax1.grid(True, alpha=0.3)
    
    # Chart 2: Efficiency Analysis
    bars2 = ax2.bar([f"{r:.0f}:1" for r in ratios], efficiency_ratios, 
                    color='#32CD32', alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, eff in zip(bars2, efficiency_ratios):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500,
                f'${eff:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_xlabel('Deposit Cap Ratio', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Liquidation Efficiency ($/% Concentration)', fontsize=12, fontweight='bold')
    ax2.set_title('Liquidation Efficiency Analysis\nDollars liquidated per % of concentration used', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "concentration_risk_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path

def main():
    """Generate all stylized charts"""
    print("🎨 Creating stylized charts for report...")
    
    data, charts_dir = load_data()
    charts_dir.mkdir(exist_ok=True)
    
    # Create enhanced charts
    chart1 = create_slippage_cost_vs_liquidations_chart(data, charts_dir)
    print(f"✅ Enhanced slippage chart: {chart1.name}")
    
    chart2 = create_stacked_liquidation_dollars_chart(data, charts_dir)
    print(f"✅ Stacked liquidation dollars: {chart2.name}")
    
    chart3 = create_enhanced_liquidation_heatmap(data, charts_dir)
    print(f"✅ Enhanced heatmap: {chart3.name}")
    
    chart4 = create_concentration_risk_analysis(data, charts_dir)
    print(f"✅ Concentration risk analysis: {chart4.name}")
    
    print(f"\n📊 All stylized charts saved to: {charts_dir}")

if __name__ == "__main__":
    main()