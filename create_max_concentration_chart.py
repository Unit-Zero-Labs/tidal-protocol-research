#!/usr/bin/env python3
"""
Create Maximum Concentration Utilization Chart
Shows the worst-case (maximum) concentration utilization per deposit cap ratio
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_max_concentration_chart():
    """Create chart showing maximum concentration utilization per deposit cap ratio"""
    
    # Load the JSON results
    results_path = Path("tidal_protocol_sim/results/moet_yt_borrow_cap_analysis/run_001_20250904_130630/liquidation_capacity_results.json")
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Extract max concentration utilization per deposit cap ratio
    deposit_ratios = []
    max_concentrations = []
    shock_scenarios = []
    
    for scenario in data["detailed_scenario_results"]:
        ratio = scenario["deposit_cap_ratio"]
        deposit_ratios.append(ratio)
        
        # Find maximum concentration across all shock scenarios for this ratio
        max_conc = 0
        worst_shock = 0
        
        for shock_result in scenario["shock_scenario_results"]:
            conc = shock_result["liquidation_metrics"]["concentration_utilization"]
            shock_pct = shock_result["scenario_params"]["btc_shock_percent"]
            
            if conc > max_conc:
                max_conc = conc
                worst_shock = shock_pct
        
        max_concentrations.append(max_conc)
        shock_scenarios.append(worst_shock)
    
    # Create the chart
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bar chart with color coding by shock scenario
    colors = ['#2E8B57', '#FF8C00', '#DC143C']  # Green, Orange, Red
    shock_colors = {10: colors[0], 15: colors[1], 25: colors[2]}
    bar_colors = [shock_colors[shock] for shock in shock_scenarios]
    
    bars = ax.bar([f"{r:.0f}:1" for r in deposit_ratios], 
                  max_concentrations, 
                  color=bar_colors, 
                  alpha=0.8, 
                  edgecolor='black', 
                  linewidth=1)
    
    # Add value labels on bars
    for i, (bar, conc, shock) in enumerate(zip(bars, max_concentrations, shock_scenarios)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{conc:.1f}%\n({shock}% shock)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add 100% reference line
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(0.02, 102, '100% Concentration Exhaustion', 
            transform=ax.get_yaxis_transform(), color='red', fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Deposit Cap Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Maximum Concentration Utilization (%)', fontsize=12, fontweight='bold')
    ax.set_title('Maximum Concentration Utilization by Deposit Cap Ratio\n(Worst-Case Shock Scenario)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=colors[0], label='10% BTC Shock'),
        plt.Rectangle((0,0),1,1, facecolor=colors[1], label='15% BTC Shock'),
        plt.Rectangle((0,0),1,1, facecolor=colors[2], label='25% BTC Shock')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    # Grid and styling
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max_concentrations) * 1.2)
    
    # Add subtitle with key insight
    plt.figtext(0.5, 0.02, 
                f'Key Finding: Maximum concentration utilization is {max(max_concentrations):.1f}% at 5:1 ratio (25% shock)',
                ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    
    # Save to results folder
    output_path = results_path.parent / "charts" / "max_concentration_utilization.png"
    output_path.parent.mkdir(exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"📊 Maximum concentration chart saved to: {output_path}")
    
    # Print summary data
    print("\nMaximum Concentration Utilization by Ratio:")
    for ratio, conc, shock in zip(deposit_ratios, max_concentrations, shock_scenarios):
        print(f"  {ratio:.0f}:1 ratio: {conc:.1f}% (worst case: {shock}% shock)")

if __name__ == "__main__":
    create_max_concentration_chart()