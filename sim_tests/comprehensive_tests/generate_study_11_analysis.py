"""
Generate Study 11 Analysis: 2021 Mixed Market with AAVE Weekly Rebalancing

This study compares High Tide's automated rebalancing against AAVE with weekly manual rebalancing:
- AAVE now performs weekly checks and rebalances based on HF vs initial HF
- If HF < initial: Deleverages by selling YT → MOET → Pay down debt
- If HF > initial: Harvests profits by selling incremental weekly rebasing yield → BTC
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the latest comparison data
json_file = Path('tidal_protocol_sim/results/Full_Year_2021_BTC_Mixed_Market_Equal_HF_Weekly_Yield_Harvest_HT_vs_AAVE_Comparison_HT_vs_AAVE_Comparison/comparison_20251108_092836.json')

print("\n" + "=" * 80)
print("STUDY 11: 2021 Mixed Market - AAVE with Weekly Rebalancing")
print("=" * 80)

with open(json_file) as f:
    data = json.load(f)

# Extract High Tide and AAVE data
ht_data = data.get("high_tide_results", {})
aave_data = data.get("aave_results", {})

# Extract daily metrics
ht_health_history = ht_data.get("agent_health_history", [])
aave_health_snapshots = aave_data.get("agent_health_snapshots", {})

# Get AAVE agent data
aave_agent_id = list(aave_health_snapshots.keys())[0] if aave_health_snapshots else None
aave_snapshots = aave_health_snapshots.get(aave_agent_id, {}) if aave_agent_id else {}
aave_timestamps = aave_snapshots.get("timestamps", [])
aave_health_factors = aave_snapshots.get("health_factors", [])
aave_collateral = aave_snapshots.get("collateral", [])
aave_debt_values = aave_snapshots.get("debt", [])

# Build daily performance data
daily_records = []
minutes_per_day = 1440

for i, ht_snapshot in enumerate(ht_health_history):
    minute = ht_snapshot.get("minute", 0)
    
    # Only process entries at day boundaries
    if minute % minutes_per_day != 0 and minute != 0:
        continue
    
    day = minute // minutes_per_day
    btc_price = ht_snapshot.get("btc_price", 0)
    
    # Calculate BTC daily return
    if day > 0:
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
    
    # Get High Tide metrics
    ht_agents = ht_snapshot.get("agents", [])
    if not ht_agents:
        continue
    
    num_agents = len(ht_agents)
    ht_net_position = sum(agent.get("net_position_value", 0) for agent in ht_agents) / num_agents
    ht_btc_amount = sum(agent.get("btc_amount", 0) for agent in ht_agents) / num_agents
    ht_health_factor = sum(agent.get("health_factor", 0) for agent in ht_agents) / num_agents
    
    collateral_value = ht_btc_amount * btc_price
    debt_value = collateral_value - ht_net_position
    ht_ltv_pct = (debt_value / collateral_value * 100) if collateral_value > 0 else 0
    
    # Get AAVE metrics
    aave_net_position = 0
    aave_btc_amount = 0
    aave_health_factor = 0
    aave_ltv_pct = 0
    
    if aave_timestamps:
        idx = None
        for i_ts, ts in enumerate(aave_timestamps):
            if ts == minute:
                idx = i_ts
                break
        
        if idx is None and aave_timestamps:
            idx = min(range(len(aave_timestamps)), key=lambda i: abs(aave_timestamps[i] - minute))
        
        if idx is not None:
            aave_collateral_value = aave_collateral[idx] if idx < len(aave_collateral) else 0
            aave_debt = aave_debt_values[idx] if idx < len(aave_debt_values) else 0
            aave_health_factor = aave_health_factors[idx] if idx < len(aave_health_factors) else 0
            
            aave_btc_amount = aave_collateral_value / btc_price if btc_price > 0 else 0
            aave_net_position = aave_collateral_value - aave_debt
            aave_ltv_pct = (aave_debt / aave_collateral_value * 100) if aave_collateral_value > 0 else 0
    
    # Calculate daily returns
    if day > 0 and prev_snapshot:
        prev_ht_agents = prev_snapshot.get("agents", [])
        if prev_ht_agents:
            prev_num_agents = len(prev_ht_agents)
            prev_ht_net = sum(agent.get("net_position_value", 0) for agent in prev_ht_agents) / prev_num_agents
            prev_ht_btc = sum(agent.get("btc_amount", 0) for agent in prev_ht_agents) / prev_num_agents
            
            ht_daily_return_usd_pct = ((ht_net_position - prev_ht_net) / prev_ht_net) * 100 if prev_ht_net > 0 else 0
            ht_daily_yield_btc_pct = ((ht_btc_amount - prev_ht_btc) / prev_ht_btc) * 100 if prev_ht_btc > 0 else 0
        else:
            ht_daily_return_usd_pct = 0
            ht_daily_yield_btc_pct = 0
        
        # AAVE daily returns
        if aave_timestamps:
            prev_closest_minute = min(range(len(aave_timestamps)), key=lambda i: abs(aave_timestamps[i] - prev_day_minute))
            
            prev_aave_collateral_value = aave_collateral[prev_closest_minute] if prev_closest_minute < len(aave_collateral) else aave_collateral_value
            prev_aave_debt = aave_debt_values[prev_closest_minute] if prev_closest_minute < len(aave_debt_values) else aave_debt
            prev_btc_price = prev_snapshot.get("btc_price", btc_price)
            
            prev_aave_btc = prev_aave_collateral_value / prev_btc_price if prev_btc_price > 0 else 0
            prev_aave_net = prev_aave_collateral_value - prev_aave_debt
            
            aave_daily_return_usd_pct = ((aave_net_position - prev_aave_net) / prev_aave_net) * 100 if prev_aave_net > 0 else 0
            aave_daily_yield_btc_pct = ((aave_btc_amount - prev_aave_btc) / prev_aave_btc) * 100 if prev_aave_btc > 0 else 0
        else:
            aave_daily_return_usd_pct = 0
            aave_daily_yield_btc_pct = 0
    else:
        ht_daily_return_usd_pct = 0
        ht_daily_yield_btc_pct = 0
        aave_daily_return_usd_pct = 0
        aave_daily_yield_btc_pct = 0
    
    daily_borrow_rate_pct = 0.01  # Placeholder
    
    record = {
        "day": day,
        "btc_price": btc_price,
        "btc_daily_return_pct": btc_daily_return_pct,
        "daily_borrow_rate_pct": daily_borrow_rate_pct,
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

df = pd.DataFrame(daily_records)

# Save CSV
output_dir = Path("tidal_protocol_sim/results/daily_performance_csvs")
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = output_dir / "study_11_daily_performance.csv"
df.to_csv(csv_path, index=False)

print(f"\n✅ Saved daily performance CSV: {csv_path}")
print(f"   {len(df)} days of data")

# Calculate summary statistics
df_no_zero = df[df["day"] > 0]

ht_avg_usd_return = df_no_zero["ht_daily_return_usd_pct"].mean()
ht_std_usd_return = df_no_zero["ht_daily_return_usd_pct"].std()
ht_avg_btc_return = df_no_zero["ht_daily_yield_btc_pct"].mean()
ht_std_btc_return = df_no_zero["ht_daily_yield_btc_pct"].std()

aave_avg_usd_return = df_no_zero["aave_daily_return_usd_pct"].mean()
aave_std_usd_return = df_no_zero["aave_daily_return_usd_pct"].std()
aave_avg_btc_return = df_no_zero["aave_daily_yield_btc_pct"].mean()
aave_std_btc_return = df_no_zero["aave_daily_yield_btc_pct"].std()

# Get final outcomes
aave_outcomes = aave_data.get("agent_outcomes", [])
weekly_events = aave_data.get("weekly_rebalance_events", [])

if aave_outcomes:
    aave_agent = aave_outcomes[0]
    final_ht_btc = df.iloc[-1]["ht_btc_amount"]
    final_aave_btc = aave_agent.get("btc_amount", 0)
    
    final_ht_net = df.iloc[-1]["ht_net_position"]
    final_aave_net = aave_agent.get("net_position_value", 0)
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS - Study 11: AAVE with Weekly Rebalancing")
    print("=" * 80)
    
    print(f"\n📊 High Tide (Automated):")
    print(f"   Final BTC: {final_ht_btc:.4f} (+{(final_ht_btc - 1.0):.4f})")
    print(f"   Net Position: ${final_ht_net:,.2f}")
    print(f"   Daily Avg USD Return: {ht_avg_usd_return:.4f}%")
    print(f"   Daily Avg BTC Growth: {ht_avg_btc_return:.4f}%")
    
    print(f"\n📊 AAVE (Weekly Manual Rebalancing):")
    print(f"   Final BTC: {final_aave_btc:.4f} (+{(final_aave_btc - 1.0):.4f})")
    print(f"   Net Position: ${final_aave_net:,.2f}")
    print(f"   Weekly Rebalances: {len(weekly_events)}")
    deleverages = [e for e in weekly_events if e.get('action_taken') == 'deleverage']
    harvests = [e for e in weekly_events if e.get('action_taken') == 'harvest_profits']
    print(f"     Deleverages: {len(deleverages)}")
    print(f"     Harvests: {len(harvests)}")
    print(f"   Daily Avg USD Return: {aave_avg_usd_return:.4f}%")
    print(f"   Daily Avg BTC Growth: {aave_avg_btc_return:.4f}%")
    
    print(f"\n✅ HIGH TIDE ADVANTAGE:")
    btc_advantage = final_ht_btc - final_aave_btc
    net_advantage = final_ht_net - final_aave_net
    print(f"   BTC Accumulation: +{btc_advantage:.4f} BTC ({(btc_advantage / final_aave_btc * 100):.2f}% more)")
    print(f"   Net Position: +${net_advantage:,.2f} ({(net_advantage / final_aave_net * 100):.2f}% higher)")

# Create comparison charts
charts_dir = output_dir / "study_11_charts"
charts_dir.mkdir(exist_ok=True)

# Chart 1: BTC Accumulation Over Time
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df["day"], df["ht_btc_amount"], label="High Tide (Automated)", linewidth=2, color="#2E8B57")
ax.plot(df["day"], df["aave_btc_amount"], label="AAVE (Weekly Rebalancing)", linewidth=2, color="#FF8C00", linestyle="--")
ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label="Initial (1.0 BTC)")
ax.set_xlabel("Day", fontsize=12)
ax.set_ylabel("BTC Amount", fontsize=12)
ax.set_title("Study 11: BTC Accumulation - High Tide vs AAVE (Weekly Rebalancing)", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(charts_dir / "btc_accumulation_comparison.png", dpi=300, bbox_inches='tight')
print(f"\n📈 Chart saved: {charts_dir / 'btc_accumulation_comparison.png'}")

# Chart 2: Net Position Value Over Time
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df["day"], df["ht_net_position"], label="High Tide", linewidth=2, color="#2E8B57")
ax.plot(df["day"], df["aave_net_position"], label="AAVE", linewidth=2, color="#FF8C00", linestyle="--")
ax.set_xlabel("Day", fontsize=12)
ax.set_ylabel("Net Position Value ($)", fontsize=12)
ax.set_title("Study 11: Net Position Value - High Tide vs AAVE (Weekly Rebalancing)", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(charts_dir / "net_position_comparison.png", dpi=300, bbox_inches='tight')
print(f"📈 Chart saved: {charts_dir / 'net_position_comparison.png'}")

# Chart 3: Health Factor Over Time
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df["day"], df["ht_health_factor"], label="High Tide", linewidth=2, color="#2E8B57")
ax.plot(df["day"], df["aave_health"], label="AAVE", linewidth=2, color="#FF8C00", linestyle="--")
ax.axhline(y=1.3, color='blue', linestyle=':', alpha=0.5, label="Initial HF (1.3)")
ax.axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label="Liquidation (1.0)")
ax.set_xlabel("Day", fontsize=12)
ax.set_ylabel("Health Factor", fontsize=12)
ax.set_title("Study 11: Health Factor Evolution - High Tide vs AAVE (Weekly Rebalancing)", fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(charts_dir / "health_factor_comparison.png", dpi=300, bbox_inches='tight')
print(f"📈 Chart saved: {charts_dir / 'health_factor_comparison.png'}")

print("\n" + "=" * 80)
print("✅ STUDY 11 ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nFiles generated:")
print(f"  • Daily Performance CSV: {csv_path}")
print(f"  • Charts Directory: {charts_dir}")
print(f"\nReady to add to whitepaper!")

