#!/usr/bin/env python3
"""
Create LTV progression chart for Study 11 optimization iterations
Shows all test HFs over time, with liquidated tests in red and survived in green
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
    """Load all test iteration data from Study 11"""
    test_dir = Path('tidal_protocol_sim/results/Study_11_2022_Bear_Minimum_HF_Weekly_Test_Data')
    
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
        
        # Convert timestamps to days
        timestamps_minutes = snapshots['timestamps']
        days = [t / 1440 for t in timestamps_minutes]
        
        # Convert HFs to LTVs
        health_factors = snapshots['health_factors']
        ltvs = [hf_to_ltv(hf) for hf in health_factors]
        
        tests.append({
            'target_hf': target_hf,
            'survived': survived,
            'days': days,
            'ltvs': ltvs,
            'filename': filepath.name
        })
    
    return tests

def create_ltv_chart(tests):
    """Create chart showing LTV progression over time for all tests"""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Sort tests by target HF
    tests_sorted = sorted(tests, key=lambda x: x['target_hf'])
    
    # Plot each test
    for test in tests_sorted:
        color = 'green' if test['survived'] else 'red'
        alpha = 0.9 if test['survived'] else 0.6
        linewidth = 3 if test['survived'] else 2
        linestyle = '-' if test['survived'] else '--'
        
        label = f"HF {test['target_hf']:.3f} ({'Survived' if test['survived'] else 'Liquidated'})"
        
        ax.plot(test['days'], test['ltvs'], 
                color=color, 
                alpha=alpha, 
                linewidth=linewidth,
                linestyle=linestyle,
                label=label)
    
    # Add liquidation threshold line
    ax.axhline(y=LIQUIDATION_LTV, color='black', linestyle=':', linewidth=2, 
               label=f'Liquidation Threshold ({LIQUIDATION_LTV}% LTV)', zorder=1)
    
    # Formatting
    ax.set_xlabel('Day of Year (2022)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loan-to-Value Ratio (%)', fontsize=14, fontweight='bold')
    ax.set_title('Study 11: LTV Progression for All Test Iterations\nWeekly Rebalancing - 2022 Bear Market',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    
    # Set y-axis limits with some padding - find min/max across all tests
    all_ltvs = [ltv for test in tests for ltv in test['ltvs']]
    min_ltv = min(all_ltvs)
    max_ltv = max(all_ltvs)
    padding = (max_ltv - min_ltv) * 0.1  # 10% padding
    ax.set_ylim(min_ltv - padding, max_ltv + padding)
    
    # Add annotations
    ax.text(0.02, 0.98, 
            'Green = Survived entire year\nRed = Liquidated',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    output_dir = Path('tidal_protocol_sim/results/Study_11_2022_Bear_Minimum_HF_Weekly_Aave_1.35_vs_HT_1.20')
    output_path = output_dir / 'ltv_progression_all_tests.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Chart saved to: {output_path}")
    
    plt.close()

def main():
    print("="*80)
    print("Creating LTV Progression Chart for Study 11 (Weekly)")
    print("="*80)
    print()
    
    # Load data
    print("Loading test data...")
    tests = load_test_data()
    print(f"✅ Loaded {len(tests)} test iterations")
    print()
    
    # Summary
    survived_count = sum(1 for t in tests if t['survived'])
    liquidated_count = len(tests) - survived_count
    
    print(f"Summary:")
    print(f"  • Survived: {survived_count} tests (green)")
    print(f"  • Liquidated: {liquidated_count} tests (red)")
    print()
    
    # Create chart
    print("Creating chart...")
    create_ltv_chart(tests)
    print()
    print("="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()

