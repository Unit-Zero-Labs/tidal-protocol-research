#!/usr/bin/env python3
"""
Export Study 14 LTV progression data to CSV
Creates minute-by-minute LTV tracking for all Aave tests + High Tide
"""

import json
import csv
from pathlib import Path

# Configuration
LIQUIDATION_THRESHOLD = 0.85
LIQUIDATION_LTV = 85.0

def hf_to_ltv(hf):
    """Convert Health Factor to LTV percentage"""
    return (LIQUIDATION_THRESHOLD / hf) * 100

def load_test_data():
    """Load all Aave test iteration data"""
    test_dir = Path('tidal_protocol_sim/results/Study_14_Oct10_2025_Liquidation_Cascade_Test_Data')
    
    if not test_dir.exists():
        raise FileNotFoundError(f"Test data directory not found: {test_dir}")
    
    test_files = sorted(test_dir.glob('test_hf_*.json'))
    
    tests = []
    for filepath in test_files:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        target_hf = data['health_factor']
        survived = data['survived']
        snapshots = data['agent_snapshots']
        
        timestamps_minutes = snapshots['timestamps']
        health_factors = snapshots['health_factors']
        ltvs = [hf_to_ltv(hf) for hf in health_factors]
        
        tests.append({
            'target_hf': target_hf,
            'survived': survived,
            'minutes': timestamps_minutes,
            'ltvs': ltvs,
            'health_factors': health_factors,
            'filename': filepath.name
        })
    
    return tests

def load_high_tide_data():
    """Load High Tide comparison data"""
    results_base = Path('tidal_protocol_sim/results')
    study14_dirs = sorted(results_base.glob('Study_14_Oct10_2025_Liquidation_Cascade_Aave_*_vs_HT_1.10'))
    
    if not study14_dirs:
        print("⚠️  Warning: No High Tide data found")
        return None
    
    study_dir = study14_dirs[-1]
    comparison_dir = study_dir.parent / (study_dir.name + "_HT_vs_AAVE_Comparison")
    
    if comparison_dir.exists():
        json_files = sorted(comparison_dir.glob('comparison_*.json'))
        if json_files:
            with open(json_files[-1], 'r') as f:
                data = json.load(f)
            
            if 'high_tide_results' in data and 'agent_health_history' in data['high_tide_results']:
                health_history = data['high_tide_results']['agent_health_history']
                
                if len(health_history) > 0:
                    minutes = []
                    health_factors = []
                    
                    for entry in health_history:
                        minutes.append(entry['minute'])
                        if entry['agents']:
                            health_factors.append(entry['agents'][0]['health_factor'])
                    
                    ltvs = [hf_to_ltv(hf) for hf in health_factors]
                    
                    initial_hf = 1.1
                    if 'agent_outcomes' in data['high_tide_results']:
                        agent = data['high_tide_results']['agent_outcomes'][0]
                        initial_hf = agent.get('initial_health_factor', 1.1)
                    
                    return {
                        'minutes': minutes,
                        'ltvs': ltvs,
                        'health_factors': health_factors,
                        'initial_hf': initial_hf
                    }
    
    return None

def load_btc_prices():
    """Load BTC price data from October 10th 2025"""
    csv_path = Path('dune_query_6227486.csv')
    prices = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            prices.append(float(row['btc_price_usd']))
    
    return prices

def create_ltv_csv(aave_tests, high_tide_data, btc_prices):
    """Create CSV with minute-by-minute LTV data"""
    
    # Sort tests by target HF
    tests_sorted = sorted(aave_tests, key=lambda x: x['target_hf'])
    
    # Find the test with maximum minutes (should be 1440 for surviving tests)
    max_minutes = max(len(test['minutes']) for test in tests_sorted)
    
    # Build CSV data
    csv_data = []
    
    for minute in range(max_minutes):
        row = {
            'minute': minute,
            'btc_price': btc_prices[minute] if minute < len(btc_prices) else btc_prices[-1]
        }
        
        # Add each Aave test
        for test in tests_sorted:
            col_name = f"Aave_HF_{test['target_hf']:.4f}_{'Survived' if test['survived'] else 'Liquidated'}"
            
            if minute < len(test['ltvs']):
                row[col_name] = round(test['ltvs'][minute], 4)
            else:
                # Test was liquidated before this minute
                row[col_name] = None
        
        # Add High Tide
        if high_tide_data and minute < len(high_tide_data['ltvs']):
            row[f"HighTide_HF_{high_tide_data['initial_hf']:.2f}"] = round(high_tide_data['ltvs'][minute], 4)
        
        # Add liquidation threshold
        row['Liquidation_Threshold_LTV'] = LIQUIDATION_LTV
        
        csv_data.append(row)
    
    # Write CSV
    results_base = Path('tidal_protocol_sim/results')
    study14_dirs = sorted(results_base.glob('Study_14_Oct10_2025_Liquidation_Cascade_Aave_*_vs_HT_1.10'))
    
    if study14_dirs:
        output_dir = study14_dirs[-1]
    else:
        output_dir = results_base / 'Study_14_Oct10_2025_Liquidation_Cascade_Aave_1.12_vs_HT_1.10'
    
    output_path = output_dir / 'study14_ltv_data.csv'
    
    # Get fieldnames
    fieldnames = list(csv_data[0].keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"✅ CSV saved to: {output_path}")
    print(f"   Rows: {len(csv_data)} minutes")
    print(f"   Columns: {len(fieldnames)} (minute, BTC price, {len(tests_sorted)} Aave tests, High Tide, threshold)")
    
    return output_path

def main():
    print("="*80)
    print("Creating LTV Progression CSV for Study 14")
    print("="*80)
    print()
    
    # Load Aave test data
    print("Loading Aave optimization test data...")
    aave_tests = load_test_data()
    print(f"✅ Loaded {len(aave_tests)} Aave test iterations")
    
    survived_count = sum(1 for t in aave_tests if t['survived'])
    liquidated_count = len(aave_tests) - survived_count
    print(f"   • Survived: {survived_count}")
    print(f"   • Liquidated: {liquidated_count}")
    print()
    
    # Load High Tide data
    print("Loading High Tide comparison data...")
    high_tide_data = load_high_tide_data()
    if high_tide_data:
        print(f"✅ Loaded High Tide data (HF {high_tide_data['initial_hf']:.2f})")
        print(f"   • {len(high_tide_data['minutes'])} minute-by-minute snapshots")
    else:
        print("⚠️  No High Tide data found")
    print()
    
    # Load BTC prices
    print("Loading BTC price data...")
    btc_prices = load_btc_prices()
    print(f"✅ Loaded {len(btc_prices)} minutes of BTC prices")
    print(f"   • Start: ${btc_prices[0]:,.2f}")
    print(f"   • Low: ${min(btc_prices):,.2f}")
    print(f"   • End: ${btc_prices[-1]:,.2f}")
    print()
    
    # Create CSV
    print("Creating CSV...")
    output_path = create_ltv_csv(aave_tests, high_tide_data, btc_prices)
    print()
    print("="*80)
    print("COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()

