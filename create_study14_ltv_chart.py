#!/usr/bin/env python3
"""
Create LTV progression chart for Study 14 optimization iterations
Shows all test HFs over time PLUS High Tide (blue line) over October 10th 2025 liquidation cascade
Minute-by-minute data over single day (1440 minutes)
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
LIQUIDATION_THRESHOLD = 0.85
LIQUIDATION_LTV = 85.0  # 85% is the liquidation threshold

def hf_to_ltv(hf):
    """Convert Health Factor to LTV percentage
    
    LTV = Debt / Collateral
    At any HF: Debt = (Collateral × liquidation_threshold) / HF
    Therefore: LTV = liquidation_threshold / HF
    """
    return (LIQUIDATION_THRESHOLD / hf) * 100

def load_test_data():
    """Load all test iteration data from Study 14 (Aave optimization)"""
    test_dir = Path('tidal_protocol_sim/results/Study_14_Oct10_2025_Liquidation_Cascade_Test_Data')
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test data directory not found: {test_dir}")
    
    test_files = sorted(test_dir.glob('test_hf_*.json'))
    
    tests = []
    for filepath in test_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract data
        target_hf = data['health_factor']
        survived = data['survived']
        snapshots = data['agent_snapshots']
        
        # Convert timestamps to minutes
        timestamps_minutes = snapshots['timestamps']
        
        # Convert HFs to LTVs
        health_factors = snapshots['health_factors']
        ltvs = [hf_to_ltv(hf) for hf in health_factors]
        
        tests.append({
            'target_hf': target_hf,
            'survived': survived,
            'minutes': timestamps_minutes,
            'ltvs': ltvs,
            'filename': filepath.name
        })
    
    return tests

def load_high_tide_data():
    """Load High Tide comparison data from Study 14 final results"""
    # Find the Study 14 results directory
    results_base = Path('tidal_protocol_sim/results')
    study14_dirs = sorted(results_base.glob('Study_14_Oct10_2025_Liquidation_Cascade_Aave_*_vs_HT_1.10'))
    
    if not study14_dirs:
        print("⚠️  Warning: No High Tide data found for Study 14")
        return None
    
    # Use the most recent one
    study_dir = study14_dirs[-1]
    
    # Look for comparison JSON with agent data
    comparison_dir = study_dir.parent / (study_dir.name + "_HT_vs_AAVE_Comparison")
    if comparison_dir.exists():
        json_files = sorted(comparison_dir.glob('comparison_*.json'))
        if json_files:
            with open(json_files[-1], 'r') as f:
                data = json.load(f)
            
            # Extract High Tide agent health history (minute-by-minute snapshots)
            if 'high_tide_results' in data and 'agent_health_history' in data['high_tide_results']:
                health_history = data['high_tide_results']['agent_health_history']
                
                if len(health_history) > 0:
                    # Extract minutes and health factors from agent_health_history
                    minutes = []
                    health_factors = []
                    
                    for entry in health_history:
                        minutes.append(entry['minute'])
                        # Get the first agent's health factor (we only have 1 agent in Study 14)
                        if entry['agents']:
                            health_factors.append(entry['agents'][0]['health_factor'])
                    
                    # Convert HFs to LTVs
                    ltvs = [hf_to_ltv(hf) for hf in health_factors]
                    
                    # Get initial HF from agent_outcomes
                    initial_hf = 1.1  # Default
                    if 'agent_outcomes' in data['high_tide_results']:
                        agent = data['high_tide_results']['agent_outcomes'][0]
                        initial_hf = agent.get('initial_health_factor', 1.1)
                    
                    print(f"✅ Extracted {len(minutes)} High Tide minute-by-minute snapshots")
                    
                    return {
                        'minutes': minutes,
                        'ltvs': ltvs,
                        'initial_hf': initial_hf
                    }
    
    print("⚠️  Warning: Could not extract High Tide snapshot data")
    return None

def create_ltv_chart(aave_tests, high_tide_data):
    """Create chart showing LTV progression over time for all Aave tests + High Tide"""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Sort Aave tests by target HF
    tests_sorted = sorted(aave_tests, key=lambda x: x['target_hf'])
    
    # Plot each Aave test
    for test in tests_sorted:
        color = 'green' if test['survived'] else 'red'
        alpha = 0.7 if test['survived'] else 0.5
        linewidth = 2.5 if test['survived'] else 1.5
        linestyle = '-' if test['survived'] else '--'
        
        label = f"Aave HF {test['target_hf']:.3f} ({'Survived' if test['survived'] else 'Liquidated'})"
        
        ax.plot(test['minutes'], test['ltvs'], 
                color=color, 
                alpha=alpha, 
                linewidth=linewidth,
                linestyle=linestyle,
                label=label)
    
    # Plot High Tide data (BLUE LINE) - this should be the star of the show
    if high_tide_data:
        ax.plot(high_tide_data['minutes'], high_tide_data['ltvs'],
                color='blue',
                alpha=1.0,
                linewidth=4,
                linestyle='-',
                label=f"High Tide HF {high_tide_data['initial_hf']:.2f} (Active Rebalancing)",
                zorder=10)  # Draw on top
    
    # Add liquidation threshold line
    ax.axhline(y=LIQUIDATION_LTV, color='black', linestyle=':', linewidth=2.5, 
               label=f'Liquidation Threshold ({LIQUIDATION_LTV}% LTV)', zorder=1)
    
    # Formatting
    ax.set_xlabel('Minutes Since Midnight UTC (October 10, 2025)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loan-to-Value Ratio (%)', fontsize=14, fontweight='bold')
    ax.set_title('Study 14: LTV Progression During October 10th 2025 Liquidation Cascade\nAave Optimization Tests + High Tide Active Rebalancing',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend - outside plot area to avoid clutter
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=2)
    
    # Set y-axis limits with some padding
    all_ltvs = [ltv for test in aave_tests for ltv in test['ltvs']]
    if high_tide_data:
        all_ltvs.extend(high_tide_data['ltvs'])
    
    min_ltv = min(all_ltvs)
    max_ltv = max(all_ltvs)
    padding = (max_ltv - min_ltv) * 0.15  # 15% padding
    ax.set_ylim(min_ltv - padding, max(max_ltv + padding, LIQUIDATION_LTV + 5))
    
    # Set x-axis to show full day
    ax.set_xlim(0, 1440)
    
    # Add annotations
    annotation_text = 'Green = Aave Survived (passive)\nRed = Aave Liquidated\nBlue = High Tide (active rebalancing)'
    ax.text(0.02, 0.98, 
            annotation_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # Add market context annotation
    ax.text(0.98, 0.98,
            'Market Event:\nBTC: $121,713 → $108,931\nMax Drawdown: -10.5%',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Tight layout
    plt.tight_layout()
    
    # Save to the Study 14 results directory
    results_base = Path('tidal_protocol_sim/results')
    study14_dirs = sorted(results_base.glob('Study_14_Oct10_2025_Liquidation_Cascade_Aave_*_vs_HT_1.10'))
    
    if study14_dirs:
        output_dir = study14_dirs[-1]
    else:
        # Fallback to generic name
        output_dir = results_base / 'Study_14_Oct10_2025_Liquidation_Cascade_Aave_1.12_vs_HT_1.10'
    
    output_path = output_dir / 'ltv_progression_all_tests.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved to: {output_path}")
    
    plt.close()

def main():
    print("="*80)
    print("Creating LTV Progression Chart for Study 14 (October 10th Cascade)")
    print("="*80)
    print()
    
    # Load Aave test data
    print("Loading Aave optimization test data...")
    aave_tests = load_test_data()
    print(f"✅ Loaded {len(aave_tests)} Aave test iterations")
    print()
    
    # Summary
    survived_count = sum(1 for t in aave_tests if t['survived'])
    liquidated_count = len(aave_tests) - survived_count
    
    print(f"Aave Tests Summary:")
    print(f"  • Survived: {survived_count} tests (green)")
    print(f"  • Liquidated: {liquidated_count} tests (red)")
    print()
    
    # Load High Tide data
    print("Loading High Tide comparison data...")
    high_tide_data = load_high_tide_data()
    if high_tide_data:
        print(f"✅ Loaded High Tide data (HF {high_tide_data['initial_hf']:.2f}) - will be shown in BLUE")
    else:
        print("⚠️  No High Tide data found - chart will only show Aave tests")
    print()
    
    # Create chart
    print("Creating chart...")
    create_ltv_chart(aave_tests, high_tide_data)
    print()
    print("="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()

