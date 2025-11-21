#!/usr/bin/env python3
"""
Create CSV exports for all three optimization studies
Each CSV contains: Day, BTC_Price, and LTV columns for each test iteration
"""

import json
import csv
from pathlib import Path

# Configuration
LIQUIDATION_THRESHOLD = 0.85

def hf_to_ltv(hf):
    """Convert Health Factor to LTV percentage"""
    return (LIQUIDATION_THRESHOLD / hf) * 100

def create_csv_for_study(study_number, study_name, test_data_dir, output_dir):
    """Create CSV export for a single study"""
    
    print(f"\n{'='*80}")
    print(f"Processing Study {study_number}: {study_name}")
    print(f"{'='*80}")
    
    # Load all test files
    test_dir = Path(test_data_dir)
    if not test_dir.exists():
        print(f"⚠️  Test data directory not found: {test_dir}")
        return
    
    test_files = sorted(test_dir.glob('test_hf_*.json'))
    
    # Load test data
    tests = []
    for filepath in test_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        target_hf = data['health_factor']
        survived = data['survived']
        snapshots = data['agent_snapshots']
        
        # Extract data
        timestamps_minutes = snapshots['timestamps']
        days = [t / 1440 for t in timestamps_minutes]
        health_factors = snapshots['health_factors']
        ltvs = [hf_to_ltv(hf) for hf in health_factors]
        collateral = snapshots['collateral']
        
        # Calculate BTC prices (assuming 1 BTC collateral initially)
        # We'll use the collateral values as proxy for BTC price
        btc_prices = collateral  # Already in dollar values
        
        tests.append({
            'target_hf': target_hf,
            'survived': survived,
            'days': days,
            'ltvs': ltvs,
            'btc_prices': btc_prices
        })
    
    print(f"✅ Loaded {len(tests)} test iterations")
    
    # Sort tests by target HF
    tests.sort(key=lambda x: x['target_hf'])
    
    # Build CSV data
    # Use the maximum length across all tests (survived tests have full year)
    num_days = max(len(test['days']) for test in tests)
    
    # Find a test with full data to use for Day and BTC_Price columns
    full_test = next(test for test in tests if len(test['days']) == num_days)
    
    # Create CSV
    output_path = Path(output_dir) / f'study{study_number}_ltv_data.csv'
    
    with open(output_path, 'w', newline='') as csvfile:
        # Build header
        header = ['Day', 'BTC_Price']
        for test in tests:
            status = 'Survived' if test['survived'] else 'Liquidated'
            header.append(f"HF_{test['target_hf']:.4f}_LTV_{status}")
        
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        # Write data rows
        for i in range(num_days):
            row = [
                full_test['days'][i],  # Day (from a complete test)
                full_test['btc_prices'][i]  # BTC price (from a complete test)
            ]
            
            # Add LTV for each test (or empty if test was liquidated before this day)
            for test in tests:
                if i < len(test['ltvs']):
                    row.append(test['ltvs'][i])
                else:
                    # Test was liquidated before this day
                    row.append('')  # Empty cell for liquidated tests
            
            writer.writerow(row)
    
    print(f"✅ CSV saved to: {output_path}")
    print(f"   Columns: Day, BTC_Price, + {len(tests)} LTV columns")
    
    return output_path

def main():
    print("="*80)
    print("CREATING CSV EXPORTS FOR ALL OPTIMIZATION STUDIES")
    print("="*80)
    
    studies = [
        {
            'number': 11,
            'name': 'Weekly Rebalancing',
            'test_data_dir': 'tidal_protocol_sim/results/Study_11_2022_Bear_Minimum_HF_Weekly_Test_Data',
            'output_dir': 'tidal_protocol_sim/results/Study_11_2022_Bear_Minimum_HF_Weekly_Aave_1.35_vs_HT_1.20'
        },
        {
            'number': 12,
            'name': 'Daily Rebalancing',
            'test_data_dir': 'tidal_protocol_sim/results/Study_12_2022_Bear_Minimum_HF_Daily_Test_Data',
            'output_dir': 'tidal_protocol_sim/results/Study_12_2022_Bear_Minimum_HF_Daily_Aave_1.19_vs_HT_1.20'
        },
        {
            'number': 13,
            'name': 'Monthly Rebalancing',
            'test_data_dir': 'tidal_protocol_sim/results/Study_13_2022_Bear_Minimum_HF_Monthly_Test_Data',
            'output_dir': 'tidal_protocol_sim/results/Study_13_2022_Bear_Minimum_HF_Monthly_Aave_1.57_vs_HT_1.20'
        }
    ]
    
    created_files = []
    
    for study in studies:
        csv_path = create_csv_for_study(
            study['number'],
            study['name'],
            study['test_data_dir'],
            study['output_dir']
        )
        if csv_path:
            created_files.append(csv_path)
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\n✅ Created {len(created_files)} CSV files:")
    for path in created_files:
        print(f"   • {path}")
    print()

if __name__ == '__main__':
    main()

