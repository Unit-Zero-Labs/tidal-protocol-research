"""
Generate Daily Performance CSV Files from Simulation Results

This script processes the JSON results from all 10 studies and generates:
1. Daily performance CSV for each study with detailed metrics
2. Summary table with aggregate statistics across all studies
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import glob
import csv
from datetime import datetime

# Study configurations
STUDIES = [
    {
        "id": 1,
        "name": "Full_Year_2021_BTC_Mixed_Market_Equal_HF_Weekly_Yield_Harvest_HT_vs_AAVE_Comparison",
        "short_name": "Study 1: 2021 Mixed Market (Symmetric)",
        "year": 2021,
        "market_type": "Mixed",
        "advanced_moet": False
    },
    {
        "id": 2,
        "name": "Full_Year_2024_BTC_Bull_Market_Equal_HF_Weekly_Yield_Harvest_HT_vs_AAVE_Comparison",
        "short_name": "Study 2: 2024 Bull Market (Symmetric)",
        "year": 2024,
        "market_type": "Bull",
        "advanced_moet": False
    },
    {
        "id": 3,
        "name": "Full_Year_2024_BTC_Capital_Efficiency_Realistic_HF_Weekly_Yield_Harvest_HT_vs_AAVE_Comparison",
        "short_name": "Study 3: 2024 Capital Efficiency (Symmetric)",
        "year": 2024,
        "market_type": "Capital Efficiency",
        "advanced_moet": False
    },
    {
        "id": 4,
        "name": "Full_Year_2022_BTC_Bear_Market_Equal_HF_Weekly_Yield_Harvest_HT_vs_AAVE_Comparison",
        "short_name": "Study 4: 2022 Bear Market (Symmetric)",
        "year": 2022,
        "market_type": "Bear",
        "advanced_moet": False
    },
    {
        "id": 5,
        "name": "Full_Year_2025_BTC_Low_Vol_Market_Equal_HF_Weekly_Yield_Harvest_HT_vs_AAVE_Comparison",
        "short_name": "Study 5: 2025 Low Vol Market (Symmetric)",
        "year": 2025,
        "market_type": "Low Vol",
        "advanced_moet": False
    },
    {
        "id": 6,
        "name": "Full_Year_2021_BTC_Mixed_Market_Advanced_MOET_vs_AAVE_Historical_HT_vs_AAVE_Comparison",
        "short_name": "Study 6: 2021 Mixed Market (Asymmetric)",
        "year": 2021,
        "market_type": "Mixed",
        "advanced_moet": True
    },
    {
        "id": 7,
        "name": "Full_Year_2024_BTC_Bull_Market_Advanced_MOET_vs_AAVE_Historical_HT_vs_AAVE_Comparison",
        "short_name": "Study 7: 2024 Bull Market (Asymmetric)",
        "year": 2024,
        "market_type": "Bull",
        "advanced_moet": True
    },
    {
        "id": 8,
        "name": "Full_Year_2024_BTC_Capital_Efficiency_Advanced_MOET_vs_AAVE_Historical_HT_vs_AAVE_Comparison",
        "short_name": "Study 8: 2024 Capital Efficiency (Asymmetric)",
        "year": 2024,
        "market_type": "Capital Efficiency",
        "advanced_moet": True
    },
    {
        "id": 9,
        "name": "Full_Year_2022_BTC_Bear_Market_Advanced_MOET_vs_AAVE_Historical_HT_vs_AAVE_Comparison",
        "short_name": "Study 9: 2022 Bear Market (Asymmetric)",
        "year": 2022,
        "market_type": "Bear",
        "advanced_moet": True
    },
    {
        "id": 10,
        "name": "Full_Year_2025_BTC_Low_Vol_Market_Advanced_MOET_vs_AAVE_Historical_HT_vs_AAVE_Comparison",
        "short_name": "Study 10: 2025 Low Vol Market (Asymmetric)",
        "year": 2025,
        "market_type": "Low Vol",
        "advanced_moet": True
    }
]

def load_historical_aave_rates(year: int, rates_csv_path: str = "rates_compute.csv") -> Dict[int, float]:
    """
    Load historical AAVE USDC borrow rates for a given year
    Returns: Dict mapping day_of_year (0-364) -> daily borrow rate (as decimal, e.g. 0.05 = 5%)
    """
    rates_by_day = {}
    
    try:
        with open(rates_csv_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                # Check if this is the correct year
                if f'{year}-' in row['date']:
                    # Parse date to get day of year
                    date_str = row['date'].split(' ')[0]  # Extract date part before time
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    day_of_year = date_obj.timetuple().tm_yday - 1  # 0-indexed
                    
                    # Extract borrow rate (avg_variableRate column)
                    borrow_rate = float(row['avg_variableRate'])
                    rates_by_day[day_of_year] = borrow_rate
        
        # Fill in missing days with interpolation
        if rates_by_day:
            # Determine if leap year
            is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            total_days = 366 if is_leap else 365
            
            # Interpolate missing days
            all_days = sorted(rates_by_day.keys())
            for day in range(total_days):
                if day not in rates_by_day:
                    # Find nearest neighbors
                    before = [d for d in all_days if d < day]
                    after = [d for d in all_days if d > day]
                    
                    if before and after:
                        # Linear interpolation
                        d1, d2 = before[-1], after[0]
                        r1, r2 = rates_by_day[d1], rates_by_day[d2]
                        weight = (day - d1) / (d2 - d1)
                        rates_by_day[day] = r1 + (r2 - r1) * weight
                    elif before:
                        # Use last known rate
                        rates_by_day[day] = rates_by_day[before[-1]]
                    elif after:
                        # Use first known rate
                        rates_by_day[day] = rates_by_day[after[0]]
        
        if not rates_by_day:
            print(f"⚠️  Warning: No {year} rates found in {rates_csv_path}. Using fallback 5% APR.")
            return {i: 0.05 for i in range(366)}
        
        print(f"📊 Loaded {len(rates_by_day)} days of {year} AAVE borrow rates")
        rates_list = [rates_by_day[i] for i in sorted(rates_by_day.keys())]
        print(f"   Rate Range: {min(rates_list)*100:.2f}% → {max(rates_list)*100:.2f}% APR")
        print(f"   Average Rate: {np.mean(rates_list)*100:.2f}% APR")
        
        return rates_by_day
        
    except FileNotFoundError:
        print(f"⚠️  Warning: {rates_csv_path} not found. Using fallback 5% APR.")
        return {i: 0.05 for i in range(366)}
    except Exception as e:
        print(f"❌ Error loading AAVE rates: {e}. Using fallback 5% APR.")
        return {i: 0.05 for i in range(366)}

def find_latest_json_file(study_folder: Path) -> Path:
    """Find the most recent comparison JSON file in the study folder"""
    # The comparison data is in the folder with suffix "_HT_vs_AAVE_Comparison"
    if not study_folder.exists():
        raise FileNotFoundError(f"Comparison folder not found: {study_folder}")
    
    json_files = list(study_folder.glob("comparison_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No comparison JSON files found in {study_folder}")
    
    # Return the most recent file
    return max(json_files, key=lambda p: p.stat().st_mtime)

def load_study_data(study: Dict[str, Any]) -> Dict[str, Any]:
    """Load JSON data for a study"""
    results_dir = Path("tidal_protocol_sim/tidal_protocol_sim/results")
    # The comparison folder has the suffix "_HT_vs_AAVE_Comparison"
    study_folder = results_dir / f"{study['name']}_HT_vs_AAVE_Comparison"
    
    if not study_folder.exists():
        raise FileNotFoundError(f"Study folder not found: {study_folder}")
    
    json_file = find_latest_json_file(study_folder)
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data

def extract_daily_metrics(data: Dict[str, Any], study: Dict[str, Any]) -> pd.DataFrame:
    """Extract daily performance metrics from simulation data"""
    
    # Get High Tide and AAVE data
    ht_data = data.get("high_tide_results", {})
    aave_data = data.get("aave_results", {})
    
    # Load historical AAVE rates for this study's year
    aave_rates = load_historical_aave_rates(study['year'])
    
    # Get MOET rate history for asymmetric studies
    moet_rate_history = []
    if study['advanced_moet'] and 'moet_system_state' in ht_data:
        moet_state = ht_data['moet_system_state']
        if 'tracking_data' in moet_state and 'moet_rate_history' in moet_state['tracking_data']:
            moet_rate_history = moet_state['tracking_data']['moet_rate_history']
            print(f"   📍 Loaded {len(moet_rate_history)} MOET rate entries for High Tide")
    
    # Extract time series data
    ht_health_history = ht_data.get("agent_health_history", [])
    aave_health_snapshots = aave_data.get("agent_health_snapshots", {})
    
    if not ht_health_history:
        print(f"⚠️  Warning: No High Tide health history found for {study['short_name']}")
        return pd.DataFrame()
    
    # Build daily data
    daily_records = []
    
    # Group by day (every 1440 minutes = 24 hours)
    minutes_per_day = 1440
    
    # Find the first AAVE agent's snapshots
    aave_agent_id = list(aave_health_snapshots.keys())[0] if aave_health_snapshots else None
    aave_snapshots = aave_health_snapshots.get(aave_agent_id, {}) if aave_agent_id else {}
    
    # AAVE snapshots are stored as parallel arrays
    aave_timestamps = aave_snapshots.get("timestamps", [])
    aave_health_factors = aave_snapshots.get("health_factors", [])
    aave_collateral = aave_snapshots.get("collateral", [])  # BTC value in USD
    aave_debt_values = aave_snapshots.get("debt", [])  # MOET debt in USD
    
    # Process each day
    day = 0
    for i, ht_snapshot in enumerate(ht_health_history):
        minute = ht_snapshot.get("minute", 0)
        
        # Only process entries at day boundaries (every 1440 minutes)
        if minute % minutes_per_day != 0 and minute != 0:
            continue
        
        day = minute // minutes_per_day
        
        # Get BTC price
        btc_price = ht_snapshot.get("btc_price", 0)
        
        # Calculate BTC daily return
        if day > 0 and i > 0:
            # Find previous day's snapshot
            prev_day_minute = (day - 1) * minutes_per_day
            prev_snapshot = None
            for prev_snap in ht_health_history:
                if prev_snap.get("minute", 0) == prev_day_minute:
                    prev_snapshot = prev_snap
                    break
            
            if prev_snapshot:
                prev_btc_price = prev_snapshot.get("btc_price", btc_price)
                btc_daily_return_pct = ((btc_price - prev_btc_price) / prev_btc_price) * 100 if prev_btc_price > 0 else 0
            else:
                btc_daily_return_pct = 0
        else:
            btc_daily_return_pct = 0
        
        # Get High Tide agent metrics (average across all agents)
        ht_agents = ht_snapshot.get("agents", [])
        if not ht_agents:
            continue
        
        # Calculate averages for High Tide
        num_agents = len(ht_agents)
        ht_net_position = sum(agent.get("net_position_value", 0) for agent in ht_agents) / num_agents
        ht_btc_amount = sum(agent.get("btc_amount", 0) for agent in ht_agents) / num_agents
        ht_health_factor = sum(agent.get("health_factor", 0) for agent in ht_agents) / num_agents
        
        # Calculate LTV for High Tide
        # LTV = debt / collateral_value
        # From net_position_value = collateral_value - debt_value
        # We need to derive this from the health factor and collateral
        # For now, approximate using typical values
        collateral_value = ht_btc_amount * btc_price
        debt_value = collateral_value - ht_net_position
        ht_ltv_pct = (debt_value / collateral_value * 100) if collateral_value > 0 else 0
        
        # Get AAVE agent metrics
        aave_net_position = 0
        aave_btc_amount = 0
        aave_health_factor = 0
        aave_ltv_pct = 0
        
        # Find closest AAVE snapshot by matching timestamp
        if aave_timestamps:
            # Find the index of the closest timestamp
            idx = None
            for i, ts in enumerate(aave_timestamps):
                if ts == minute:
                    idx = i
                    break
            
            if idx is None:
                # Find closest
                idx = min(range(len(aave_timestamps)), key=lambda i: abs(aave_timestamps[i] - minute))
            
            # Extract AAVE metrics from parallel arrays
            aave_collateral_value = aave_collateral[idx] if idx < len(aave_collateral) else 0
            aave_debt = aave_debt_values[idx] if idx < len(aave_debt_values) else 0
            aave_health_factor = aave_health_factors[idx] if idx < len(aave_health_factors) else 0
            
            # Calculate derived metrics
            aave_btc_amount = aave_collateral_value / btc_price if btc_price > 0 else 0
            aave_net_position = aave_collateral_value - aave_debt
            aave_ltv_pct = (aave_debt / aave_collateral_value * 100) if aave_collateral_value > 0 else 0
        
        # Calculate daily returns
        if day > 0:
            prev_day_minute = (day - 1) * minutes_per_day
            prev_ht_snapshot = None
            
            for prev_snap in ht_health_history:
                if prev_snap.get("minute", 0) == prev_day_minute:
                    prev_ht_snapshot = prev_snap
                    break
            
            if prev_ht_snapshot:
                prev_ht_agents = prev_ht_snapshot.get("agents", [])
                if prev_ht_agents:
                    prev_num_agents = len(prev_ht_agents)
                    prev_ht_net = sum(agent.get("net_position_value", 0) for agent in prev_ht_agents) / prev_num_agents
                    prev_ht_btc = sum(agent.get("btc_amount", 0) for agent in prev_ht_agents) / prev_num_agents
                    
                    ht_daily_return_usd_pct = ((ht_net_position - prev_ht_net) / prev_ht_net) * 100 if prev_ht_net > 0 else 0
                    ht_daily_yield_btc_pct = ((ht_btc_amount - prev_ht_btc) / prev_ht_btc) * 100 if prev_ht_btc > 0 else 0
                else:
                    ht_daily_return_usd_pct = 0
                    ht_daily_yield_btc_pct = 0
            else:
                ht_daily_return_usd_pct = 0
                ht_daily_yield_btc_pct = 0
            
            # AAVE daily returns
            if aave_timestamps and prev_ht_snapshot:
                # Find the index for previous day
                prev_idx = None
                for i, ts in enumerate(aave_timestamps):
                    if ts == prev_day_minute:
                        prev_idx = i
                        break
                
                if prev_idx is None and aave_timestamps:
                    # Find closest
                    prev_idx = min(range(len(aave_timestamps)), key=lambda i: abs(aave_timestamps[i] - prev_day_minute))
                
                if prev_idx is not None:
                    prev_aave_collateral_value = aave_collateral[prev_idx] if prev_idx < len(aave_collateral) else aave_collateral_value
                    prev_aave_debt = aave_debt_values[prev_idx] if prev_idx < len(aave_debt_values) else aave_debt
                    prev_btc_price = prev_ht_snapshot.get("btc_price", btc_price)
                    
                    prev_aave_btc = prev_aave_collateral_value / prev_btc_price if prev_btc_price > 0 else 0
                    prev_aave_net = prev_aave_collateral_value - prev_aave_debt
                    
                    aave_daily_return_usd_pct = ((aave_net_position - prev_aave_net) / prev_aave_net) * 100 if prev_aave_net > 0 else 0
                    aave_daily_yield_btc_pct = ((aave_btc_amount - prev_aave_btc) / prev_aave_btc) * 100 if prev_aave_btc > 0 else 0
                else:
                    aave_daily_return_usd_pct = 0
                    aave_daily_yield_btc_pct = 0
            else:
                aave_daily_return_usd_pct = 0
                aave_daily_yield_btc_pct = 0
        else:
            ht_daily_return_usd_pct = 0
            ht_daily_yield_btc_pct = 0
            aave_daily_return_usd_pct = 0
            aave_daily_yield_btc_pct = 0
        
        # Calculate borrow rates
        # AAVE always uses historical rates from rates_compute.csv
        aave_daily_borrow_rate_pct = aave_rates.get(day, 0.05) / 365  # Convert APR to daily
        
        # High Tide rate depends on study type
        if study['advanced_moet'] and moet_rate_history:
            # Asymmetric: Use MOET rates from simulation
            # Find the MOET rate at the end of this day (day * 1440 minutes)
            target_minute = day * 1440
            ht_apr = 0.05  # Default fallback
            
            # Find closest minute in MOET rate history
            for rate_entry in moet_rate_history:
                if rate_entry['minute'] >= target_minute:
                    ht_apr = rate_entry.get('moet_interest_rate', 0.05)
                    break
            
            ht_daily_borrow_rate_pct = ht_apr / 365  # Convert APR to daily
        else:
            # Symmetric: Both use AAVE rates
            ht_daily_borrow_rate_pct = aave_daily_borrow_rate_pct
        
        # Create record
        record = {
            "day": day,
            "btc_price": btc_price,
            "btc_daily_return_pct": btc_daily_return_pct,
            "ht_daily_borrow_rate_pct": ht_daily_borrow_rate_pct,
            "aave_daily_borrow_rate_pct": aave_daily_borrow_rate_pct,
            "ht_net_position": ht_net_position,
            "ht_daily_return_usd_pct": ht_daily_return_usd_pct,
            "ht_btc_amount": ht_btc_amount,
            "ht_daily_yield_btc_pct": ht_daily_yield_btc_pct,
            "ht_health_factor": ht_health_factor,
            "ht_ltv_pct": ht_ltv_pct,
            "aave_net_position": aave_net_position,
            "aave_daily_return_usd_pct": aave_daily_return_usd_pct,
            "aave_btc_amount": aave_btc_amount,
            "aave_daily_yield_btc_pct": aave_daily_yield_btc_pct,
            "aave_health": aave_health_factor,
            "aave_ltv_pct": aave_ltv_pct,
        }
        
        daily_records.append(record)
    
    return pd.DataFrame(daily_records)

def calculate_summary_stats(df: pd.DataFrame, protocol: str = "ht") -> Dict[str, float]:
    """Calculate summary statistics for a protocol"""
    if df.empty:
        return {
            "Daily Average USD Return %": 0,
            "Daily Average USD Return StdDev": 0,
            "Daily Average Borrow Cost": 0,
            "Daily Average BTC Return": 0,
            "Daily Average BTC StdDev": 0
        }
    
    prefix = protocol.lower()
    
    # Calculate metrics (excluding day 0 which has no returns)
    df_no_zero = df[df["day"] > 0].copy()
    
    if df_no_zero.empty:
        return {
            "Daily Average USD Return %": 0,
            "Daily Average USD Return StdDev": 0,
            "Daily Average Borrow Cost": 0,
            "Daily Average BTC Return": 0,
            "Daily Average BTC StdDev": 0
        }
    
    # Use protocol-specific borrow rate column
    borrow_rate_col = f"{prefix}_daily_borrow_rate_pct"
    
    return {
        "Daily Average USD Return %": df_no_zero[f"{prefix}_daily_return_usd_pct"].mean(),
        "Daily Average USD Return StdDev": df_no_zero[f"{prefix}_daily_return_usd_pct"].std(),
        "Daily Average Borrow Cost": df_no_zero[borrow_rate_col].mean(),
        "Daily Average BTC Return": df_no_zero[f"{prefix}_daily_yield_btc_pct"].mean(),
        "Daily Average BTC StdDev": df_no_zero[f"{prefix}_daily_yield_btc_pct"].std()
    }

def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("GENERATING DAILY PERFORMANCE CSV FILES")
    print("=" * 70 + "\n")
    
    # Create output directory  
    output_dir = Path("tidal_protocol_sim/tidal_protocol_sim/results/daily_performance_csvs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary table data
    summary_data = []
    
    # Process each study
    for study in STUDIES:
        print(f"\n📊 Processing {study['short_name']}...")
        
        try:
            # Load study data
            data = load_study_data(study)
            
            # Extract daily metrics
            df = extract_daily_metrics(data, study)
            
            if df.empty:
                print(f"   ⚠️  No daily data available - skipping")
                continue
            
            # Save to CSV
            csv_filename = f"study_{study['id']}_daily_performance.csv"
            csv_path = output_dir / csv_filename
            df.to_csv(csv_path, index=False)
            print(f"   ✅ Saved: {csv_filename}")
            print(f"      {len(df)} days of data")
            
            # Calculate summary statistics for both protocols
            ht_stats = calculate_summary_stats(df, "ht")
            aave_stats = calculate_summary_stats(df, "aave")
            
            # Add to summary table
            summary_data.append({
                "Study": study["short_name"],
                "Year": study["year"],
                "Market Type": study["market_type"],
                "Advanced MOET": "Yes" if study["advanced_moet"] else "No",
                
                # High Tide metrics
                "HT Daily Avg USD Return %": ht_stats["Daily Average USD Return %"],
                "HT Daily USD Return StdDev": ht_stats["Daily Average USD Return StdDev"],
                "HT Daily Avg Borrow Cost": ht_stats["Daily Average Borrow Cost"],
                "HT Daily Avg BTC Return %": ht_stats["Daily Average BTC Return"],
                "HT Daily BTC Return StdDev": ht_stats["Daily Average BTC StdDev"],
                
                # AAVE metrics
                "AAVE Daily Avg USD Return %": aave_stats["Daily Average USD Return %"],
                "AAVE Daily USD Return StdDev": aave_stats["Daily Average USD Return StdDev"],
                "AAVE Daily Avg Borrow Cost": aave_stats["Daily Average Borrow Cost"],
                "AAVE Daily Avg BTC Return %": aave_stats["Daily Average BTC Return"],
                "AAVE Daily BTC Return StdDev": aave_stats["Daily Average BTC StdDev"],
            })
            
        except FileNotFoundError as e:
            print(f"   ❌ Error: {e}")
            continue
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create and save summary table
    if summary_data:
        print("\n" + "=" * 70)
        print("GENERATING SUMMARY TABLE")
        print("=" * 70 + "\n")
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = output_dir / "all_studies_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"✅ Summary table saved: all_studies_summary.csv")
        print(f"\n📈 SUMMARY TABLE:\n")
        
        # Display formatted summary
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 40)
        
        print(summary_df.to_string(index=False))
        
    print("\n" + "=" * 70)
    print(f"✅ ALL CSV FILES SAVED TO: {output_dir}")
    print("=" * 70 + "\n")
    
    print(f"📁 Generated files:")
    print(f"   • {len(summary_data)} daily performance CSV files")
    print(f"   • 1 summary table CSV")
    print(f"\nLocation: {output_dir.absolute()}\n")

if __name__ == "__main__":
    main()

