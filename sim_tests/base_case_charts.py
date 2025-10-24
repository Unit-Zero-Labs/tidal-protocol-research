"""
Comprehensive charting functions for base case comparison
Extended from full_year_sim charting suite
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List


def create_all_charts(comparison_obj):
    """Generate all comparison charts - full suite"""
    
    output_dir = Path("tidal_protocol_sim/results") / comparison_obj.config.test_name / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📊 Generating comprehensive chart suite...")
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # COMPARISON CHARTS (both systems side-by-side)
    print("   Creating comparison charts...")
    create_btc_price_chart(comparison_obj, output_dir)
    create_survival_comparison_chart(comparison_obj, output_dir)
    create_net_apy_comparison_chart(comparison_obj, output_dir)
    create_health_factor_evolution_chart(comparison_obj, output_dir)
    create_yield_strategy_comparison_chart(comparison_obj, output_dir)
    create_agent_performance_distribution_comparison(comparison_obj, output_dir)
    
    # HIGH TIDE SYSTEM CHARTS
    print("   Creating High Tide charts...")
    create_system_charts(comparison_obj, output_dir, "high_tide", "#2E8B57")
    
    # AAVE SYSTEM CHARTS
    print("   Creating AAVE charts...")
    create_system_charts(comparison_obj, output_dir, "aave", "#DC143C")
    
    print(f"✅ All charts saved to: {output_dir}")


def create_system_charts(comparison_obj, output_dir: Path, system_type: str, color: str):
    """Create full chart suite for a specific system"""
    
    results = comparison_obj.results[f"{system_type}_results"]
    
    # Chart 1: Agent Health Factor Tracking
    create_agent_health_tracking_chart(comparison_obj, output_dir, system_type, results, color)
    
    # Chart 2: Pool State Evolution
    create_pool_evolution_chart(comparison_obj, output_dir, system_type, results, color)
    
    # Chart 3: Time Series Evolution (2x2 layout)
    create_time_series_evolution_chart(comparison_obj, output_dir, system_type, results, color)
    
    # Chart 4: Individual Agent Performance
    create_individual_agent_performance_chart(comparison_obj, output_dir, system_type, results, color)
    
    # Chart 5: Rebalancing Activity (High Tide only)
    if system_type == "high_tide" and results["time_series_data"]["rebalancing_events"]:
        create_rebalancing_activity_chart(comparison_obj, output_dir, results, color)
    
    # Chart 6: Liquidation Analysis (if any liquidations)
    if system_type == "aave" or results["final_metrics"]["liquidated_agents"] > 0:
        create_liquidation_analysis_chart(comparison_obj, output_dir, system_type, results, color)


def create_btc_price_chart(comparison_obj, output_dir: Path):
    """Create BTC price evolution chart"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    days = np.arange(len(comparison_obj.btc_price_path))
    ax.plot(days, comparison_obj.btc_price_path, linewidth=2.5, color='#FF9900', label='BTC Price', zorder=3)
    ax.axhline(comparison_obj.config.btc_initial_price, color='gray', linestyle='--', alpha=0.6, 
               label='Initial Price ($100k)', linewidth=1.5)
    ax.fill_between(days, comparison_obj.config.btc_min_price, comparison_obj.config.btc_max_price, 
                     alpha=0.15, color='gray', label='±15% Range')
    
    ax.set_xlabel('Days', fontsize=13, fontweight='bold')
    ax.set_ylabel('BTC Price (USD)', fontsize=13, fontweight='bold')
    ax.set_title('BTC Price Evolution - 3 Month Base Case Scenario', fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'btc_price_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_survival_comparison_chart(comparison_obj, output_dir: Path):
    """Create survival rate comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ht_metrics = comparison_obj.results["high_tide_results"]["final_metrics"]
    aave_metrics = comparison_obj.results["aave_results"]["final_metrics"]
    
    x = np.arange(2)
    width = 0.35
    
    systems = ['High Tide', 'AAVE']
    survival_rates = [ht_metrics['survival_rate'] * 100, aave_metrics['survival_rate'] * 100]
    liquidation_rates = [(1 - ht_metrics['survival_rate']) * 100, (1 - aave_metrics['survival_rate']) * 100]
    colors_survival = ['#2E8B57', '#DC143C']
    colors_liquidation = ['#FFB6C1', '#8B0000']
    
    bars1 = ax.bar(x, survival_rates, width, label='Survived', color=colors_survival, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, liquidation_rates, width, bottom=survival_rates, label='Liquidated', 
                   color=colors_liquidation, alpha=0.6, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (surv, liq) in enumerate(zip(survival_rates, liquidation_rates)):
        ax.text(i, surv/2, f'{surv:.1f}%', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        if liq > 2:  # Only show if significant
            ax.text(i, surv + liq/2, f'{liq:.1f}%', ha='center', va='center', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Percentage of Agents (%)', fontsize=13, fontweight='bold')
    ax.set_title('Agent Survival vs Liquidation: High Tide vs AAVE', fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'survival_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_net_apy_comparison_chart(comparison_obj, output_dir: Path):
    """Create net APY comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ht_metrics = comparison_obj.results["high_tide_results"]["final_metrics"]
    aave_metrics = comparison_obj.results["aave_results"]["final_metrics"]
    
    systems = ['High Tide', 'AAVE']
    net_apys = [ht_metrics['avg_net_apy'] * 100, aave_metrics['avg_net_apy'] * 100]
    colors = ['#2E8B57', '#DC143C']
    
    bars = ax.bar(systems, net_apys, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, net_apys):
        height = bar.get_height()
        y_pos = height if height > 0 else height
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{val:.2f}%',
               ha='center', va='bottom' if height > 0 else 'top', 
               fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Average Net APY (%)', fontsize=13, fontweight='bold')
    ax.set_title('Average Net APY: High Tide vs AAVE\n(Yield Earned - Interest Paid)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add delta annotation
    delta = net_apys[0] - net_apys[1]
    ax.text(0.5, max(net_apys) * 0.9, f'Δ = {delta:+.2f}%', 
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'net_apy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_health_factor_evolution_chart(comparison_obj, output_dir: Path):
    """Create health factor evolution comparison"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    
    # High Tide health factors
    ht_data = comparison_obj.results["high_tide_results"]["time_series_data"]
    plot_health_factors(ax1, ht_data, "High Tide", "#2E8B57", comparison_obj.config)
    
    # AAVE health factors
    aave_data = comparison_obj.results["aave_results"]["time_series_data"]
    plot_health_factors(ax2, aave_data, "AAVE", "#DC143C", comparison_obj.config)
    
    ax2.set_xlabel('Days', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'health_factor_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_health_factors(ax, data, system_name, color, config):
    """Plot health factors for a system"""
    
    timestamps = np.array(data["timestamps"]) / 1440  # Convert to days
    
    # Extract health factors for all agents at each timestamp
    all_hfs_avg = []
    all_hfs_min = []
    all_hfs_max = []
    
    for hf_snapshot in data["agent_health_factors"]:
        if hf_snapshot:
            hfs = [agent["health_factor"] for agent in hf_snapshot if agent["health_factor"] < 10]
            if hfs:
                all_hfs_avg.append(np.mean(hfs))
                all_hfs_min.append(np.min(hfs))
                all_hfs_max.append(np.max(hfs))
            else:
                all_hfs_avg.append(np.nan)
                all_hfs_min.append(np.nan)
                all_hfs_max.append(np.nan)
        else:
            all_hfs_avg.append(np.nan)
            all_hfs_min.append(np.nan)
            all_hfs_max.append(np.nan)
    
    # Plot with fill_between for range
    ax.plot(timestamps, all_hfs_avg, color=color, linewidth=2.5, label='Average HF', zorder=3)
    ax.fill_between(timestamps, all_hfs_min, all_hfs_max, alpha=0.2, color=color, label='Min-Max Range')
    ax.axhline(1.0, color='red', linestyle='--', label='Liquidation Threshold', linewidth=2, zorder=2)
    ax.axhline(config.agent_initial_hf, color='gray', linestyle='--', alpha=0.5, label=f'Initial HF ({config.agent_initial_hf})', linewidth=1.5)
    ax.axhline(config.agent_rebalancing_hf, color='orange', linestyle=':', alpha=0.7, 
               label=f'Rebalancing Trigger ({config.agent_rebalancing_hf})', linewidth=1.5)
    
    ax.set_ylabel('Health Factor', fontsize=12, fontweight='bold')
    ax.set_title(f'{system_name} - Agent Health Factor Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0.8, 2.2)


def create_yield_strategy_comparison_chart(comparison_obj, output_dir: Path):
    """Create yield strategy comparison"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    ht_perf = comparison_obj.results["high_tide_results"]["agent_performance"]
    aave_perf = comparison_obj.results["aave_results"]["agent_performance"]
    
    # Extract metrics for active agents only
    ht_yields = [p["yield_token_portfolio"]["total_yield_earned"] for p in ht_perf.values() 
                 if "yield_token_portfolio" in p and "total_yield_earned" in p["yield_token_portfolio"]]
    ht_interest = [p["total_interest_accrued"] for p in ht_perf.values() if "total_interest_accrued" in p]
    
    aave_yields = [p["yield_token_portfolio"]["total_yield_earned"] for p in aave_perf.values() 
                   if "yield_token_portfolio" in p and "total_yield_earned" in p["yield_token_portfolio"]]
    aave_interest = [p["total_interest_accrued"] for p in aave_perf.values() if "total_interest_accrued" in p]
    
    # Panel 1: Yield Earned Distribution
    ax1.hist(ht_yields, bins=20, alpha=0.6, color='#2E8B57', label='High Tide', edgecolor='black')
    ax1.hist(aave_yields, bins=20, alpha=0.6, color='#DC143C', label='AAVE', edgecolor='black')
    ax1.set_xlabel('Total Yield Earned ($)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Agents', fontsize=11, fontweight='bold')
    ax1.set_title('Yield Earned Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Interest Paid Distribution
    ax2.hist(ht_interest, bins=20, alpha=0.6, color='#2E8B57', label='High Tide', edgecolor='black')
    ax2.hist(aave_interest, bins=20, alpha=0.6, color='#DC143C', label='AAVE', edgecolor='black')
    ax2.set_xlabel('Total Interest Paid ($)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Agents', fontsize=11, fontweight='bold')
    ax2.set_title('Interest Paid Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Net Position (Yield - Interest)
    ht_net = [y - i for y, i in zip(ht_yields, ht_interest)]
    aave_net = [y - i for y, i in zip(aave_yields, aave_interest)]
    
    ax3.hist(ht_net, bins=20, alpha=0.6, color='#2E8B57', label='High Tide', edgecolor='black')
    ax3.hist(aave_net, bins=20, alpha=0.6, color='#DC143C', label='AAVE', edgecolor='black')
    ax3.axvline(0, color='black', linestyle='--', linewidth=1)
    ax3.set_xlabel('Net Position ($)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Agents', fontsize=11, fontweight='bold')
    ax3.set_title('Net Position Distribution (Yield - Interest)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Box Plot Comparison
    data_to_plot = [ht_net, aave_net]
    bp = ax4.boxplot(data_to_plot, labels=['High Tide', 'AAVE'], patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.6),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    
    colors = ['#2E8B57', '#DC143C']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax4.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_ylabel('Net Position ($)', fontsize=11, fontweight='bold')
    ax4.set_title('Net Position Box Plot', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'yield_strategy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_agent_performance_distribution_comparison(comparison_obj, output_dir: Path):
    """Create agent performance distribution comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    ht_perf = comparison_obj.results["high_tide_results"]["agent_performance"]
    aave_perf = comparison_obj.results["aave_results"]["agent_performance"]
    
    # Final portfolio values
    ht_final = [p.get("net_position_value", 0) for p in ht_perf.values()]
    aave_final = [p.get("net_position_value", 0) for p in aave_perf.values()]
    
    # Panel 1: Histogram
    ax1.hist(ht_final, bins=25, alpha=0.6, color='#2E8B57', label='High Tide', edgecolor='black')
    ax1.hist(aave_final, bins=25, alpha=0.6, color='#DC143C', label='AAVE', edgecolor='black')
    ax1.axvline(100_000, color='gray', linestyle='--', label='Initial Value ($100k)', linewidth=2)
    ax1.set_xlabel('Final Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Agents', fontsize=12, fontweight='bold')
    ax1.set_title('Final Portfolio Value Distribution', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Cumulative Distribution
    ht_sorted = np.sort(ht_final)
    aave_sorted = np.sort(aave_final)
    ht_cum = np.arange(1, len(ht_sorted)+1) / len(ht_sorted) * 100
    aave_cum = np.arange(1, len(aave_sorted)+1) / len(aave_sorted) * 100
    
    ax2.plot(ht_sorted, ht_cum, color='#2E8B57', linewidth=2.5, label='High Tide')
    ax2.plot(aave_sorted, aave_cum, color='#DC143C', linewidth=2.5, label='AAVE')
    ax2.axvline(100_000, color='gray', linestyle='--', label='Initial Value', linewidth=2)
    ax2.set_xlabel('Final Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Distribution Function', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'agent_performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


# Additional system-specific charts continue in next section...
def create_agent_health_tracking_chart(comparison_obj, output_dir: Path, system_type: str, results: Dict, color: str):
    """Create detailed agent health factor tracking"""
    fig, ax = plt.subplots(figsize=(15, 8))
    
    timestamps = np.array(results["time_series_data"]["timestamps"]) / 1440  # Convert to days
    
    # Plot individual agent health factors (sample if too many)
    agent_ids = list(results["agent_performance"].keys())
    sample_size = min(10, len(agent_ids))  # Sample 10 agents
    sampled_agents = agent_ids[::len(agent_ids)//sample_size][:sample_size]
    
    for agent_id in sampled_agents:
        agent_hfs = []
        for hf_snapshot in results["time_series_data"]["agent_health_factors"]:
            agent_data = next((a for a in hf_snapshot if a["agent_id"] == agent_id), None)
            if agent_data:
                agent_hfs.append(agent_data["health_factor"])
            else:
                agent_hfs.append(np.nan)
        
        ax.plot(timestamps, agent_hfs, alpha=0.4, linewidth=1)
    
    # Calculate and plot average
    all_hfs_avg = []
    for hf_snapshot in results["time_series_data"]["agent_health_factors"]:
        if hf_snapshot:
            hfs = [agent["health_factor"] for agent in hf_snapshot if agent["health_factor"] < 10]
            all_hfs_avg.append(np.mean(hfs) if hfs else np.nan)
        else:
            all_hfs_avg.append(np.nan)
    
    ax.plot(timestamps, all_hfs_avg, color=color, linewidth=3, label='Average HF', zorder=10)
    ax.axhline(1.0, color='red', linestyle='--', label='Liquidation Threshold', linewidth=2)
    ax.axhline(comparison_obj.config.agent_rebalancing_hf, color='orange', linestyle=':', 
               label=f'Rebalancing Trigger', linewidth=1.5)
    
    ax.set_xlabel('Days', fontsize=12, fontweight='bold')
    ax.set_ylabel('Health Factor', fontsize=12, fontweight='bold')
    ax.set_title(f'{system_type.upper()} - Agent Health Factor Tracking (Sample)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 3.0)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{system_type}_agent_health_tracking.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_pool_evolution_chart(comparison_obj, output_dir: Path, system_type: str, results: Dict, color: str):
    """Create pool state evolution chart"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))
    
    pool_states = results["time_series_data"]["pool_states"]
    timestamps = [ps["minute"] / 1440 for ps in pool_states]  # Convert to days
    
    # Panel 1: MOET Reserve
    moet_reserves = [ps["yt_pool_moet_reserve"] for ps in pool_states]
    ax1.plot(timestamps, moet_reserves, color=color, linewidth=2)
    ax1.set_xlabel('Days', fontsize=11, fontweight='bold')
    ax1.set_ylabel('MOET Reserve', fontsize=11, fontweight='bold')
    ax1.set_title('YT Pool - MOET Reserve Evolution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    
    # Panel 2: YT Reserve
    yt_reserves = [ps["yt_pool_yt_reserve"] for ps in pool_states]
    ax2.plot(timestamps, yt_reserves, color=color, linewidth=2)
    ax2.set_xlabel('Days', fontsize=11, fontweight='bold')
    ax2.set_ylabel('YT Reserve', fontsize=11, fontweight='bold')
    ax2.set_title('YT Pool - Yield Token Reserve Evolution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    
    # Panel 3: Pool Price (MOET/YT ratio)
    pool_prices = [ps["yt_pool_price"] for ps in pool_states]
    ax3.plot(timestamps, pool_prices, color=color, linewidth=2)
    ax3.axhline(1.0, color='gray', linestyle='--', label='1:1 Peg', linewidth=1.5)
    ax3.set_xlabel('Days', fontsize=11, fontweight='bold')
    ax3.set_ylabel('MOET/YT Price', fontsize=11, fontweight='bold')
    ax3.set_title('YT Pool - Marginal Price Evolution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Total Pool TVL
    pool_tvl = [ps["yt_pool_moet_reserve"] + ps["yt_pool_yt_reserve"] for ps in pool_states]
    ax4.plot(timestamps, pool_tvl, color=color, linewidth=2)
    ax4.set_xlabel('Days', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Total TVL ($)', fontsize=11, fontweight='bold')
    ax4.set_title('YT Pool - Total Value Locked', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{system_type}_pool_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_time_series_evolution_chart(comparison_obj, output_dir: Path, system_type: str, results: Dict, color: str):
    """Create time series evolution analysis (2x2)"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))
    
    timestamps_days = np.array(results["time_series_data"]["timestamps"]) / 1440
    btc_prices = results["time_series_data"]["btc_prices"]
    
    # Panel 1: BTC Price
    ax1.plot(timestamps_days, btc_prices, color='#FF9900', linewidth=2)
    ax1.set_xlabel('Days', fontsize=11, fontweight='bold')
    ax1.set_ylabel('BTC Price ($)', fontsize=11, fontweight='bold')
    ax1.set_title('BTC Price Evolution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    
    # Panel 2: Active Agents Count
    active_counts = []
    for hf_snapshot in results["time_series_data"]["agent_health_factors"]:
        active_counts.append(len(hf_snapshot))
    
    ax2.plot(timestamps_days, active_counts, color=color, linewidth=2)
    ax2.set_xlabel('Days', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Active Agents', fontsize=11, fontweight='bold')
    ax2.set_title('Active Agent Count Over Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Average Debt
    avg_debt = []
    for hf_snapshot in results["time_series_data"]["agent_health_factors"]:
        if hf_snapshot:
            debts = [agent["moet_debt"] for agent in hf_snapshot]
            avg_debt.append(np.mean(debts))
        else:
            avg_debt.append(0)
    
    ax3.plot(timestamps_days, avg_debt, color=color, linewidth=2)
    ax3.set_xlabel('Days', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Average MOET Debt ($)', fontsize=11, fontweight='bold')
    ax3.set_title('Average Agent Debt Evolution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    
    # Panel 4: Average BTC Collateral
    avg_btc = []
    for hf_snapshot in results["time_series_data"]["agent_health_factors"]:
        if hf_snapshot:
            btc_amounts = [agent["btc_amount"] for agent in hf_snapshot]
            avg_btc.append(np.mean(btc_amounts))
        else:
            avg_btc.append(0)
    
    ax4.plot(timestamps_days, avg_btc, color=color, linewidth=2)
    ax4.set_xlabel('Days', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Average BTC Collateral', fontsize=11, fontweight='bold')
    ax4.set_title('Average Agent BTC Holdings', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{system_type}_time_series_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_individual_agent_performance_chart(comparison_obj, output_dir: Path, system_type: str, results: Dict, color: str):
    """Create individual agent performance chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    agent_perfs = results["agent_performance"]
    
    # Extract metrics
    agent_ids = []
    final_values = []
    yields_earned = []
    interest_paid = []
    is_active = []
    
    for agent_id, perf in agent_perfs.items():
        agent_ids.append(agent_id)
        final_values.append(perf.get("net_position_value", 0))
        
        yt_portfolio = perf.get("yield_token_portfolio", {})
        yields_earned.append(yt_portfolio.get("total_yield_earned", 0))
        interest_paid.append(perf.get("total_interest_accrued", 0))
        
        # Check if agent survived
        is_active.append(perf.get("health_factor", 0) > 0)
    
    # Panel 1: Final Values (sorted)
    sorted_indices = np.argsort(final_values)
    sorted_values = [final_values[i] for i in sorted_indices]
    sorted_active = [is_active[i] for i in sorted_indices]
    
    colors_list = [color if active else '#8B0000' for active in sorted_active]
    
    ax1.bar(range(len(sorted_values)), sorted_values, color=colors_list, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.axhline(100_000, color='gray', linestyle='--', label='Initial Value', linewidth=2)
    ax1.set_xlabel('Agent (sorted by performance)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Final Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{system_type.upper()} - Agent Performance Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    
    # Panel 2: Yield vs Interest Scatter
    active_mask = np.array(is_active)
    ax2.scatter(np.array(interest_paid)[active_mask], np.array(yields_earned)[active_mask], 
                c=color, alpha=0.6, s=50, label='Active Agents', edgecolors='black')
    ax2.scatter(np.array(interest_paid)[~active_mask], np.array(yields_earned)[~active_mask], 
                c='#8B0000', alpha=0.6, s=50, label='Liquidated', edgecolors='black', marker='x')
    
    # Add diagonal line (break-even)
    max_val = max(max(interest_paid), max(yields_earned))
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='Break-even')
    
    ax2.set_xlabel('Interest Paid ($)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Yield Earned ($)', fontsize=12, fontweight='bold')
    ax2.set_title('Yield vs Interest: Agent-by-Agent', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{system_type}_individual_agent_performance.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_rebalancing_activity_chart(comparison_obj, output_dir: Path, results: Dict, color: str):
    """Create rebalancing activity analysis (High Tide only)"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))
    
    rebal_events = results["time_series_data"]["rebalancing_events"]
    
    if not rebal_events:
        # No rebalancing events
        fig.text(0.5, 0.5, 'No Rebalancing Events Recorded', 
                ha='center', va='center', fontsize=20)
        plt.savefig(output_dir / 'high_tide_rebalancing_activity.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Separate ALM and Algo events
    alm_events = [e for e in rebal_events if e.get("rebalancer", "").upper() == "ALM"]
    algo_events = [e for e in rebal_events if e.get("rebalancer", "").upper() == "ALGO"]
    
    # Panel 1: Event Timeline
    alm_times = [e["minute"] / 1440 for e in alm_events]
    algo_times = [e["minute"] / 1440 for e in algo_events]
    
    ax1.scatter(alm_times, [1]*len(alm_times), c='blue', s=100, alpha=0.6, label='ALM Rebalancing', marker='o')
    ax1.scatter(algo_times, [2]*len(algo_times), c='orange', s=100, alpha=0.6, label='Algo Rebalancing', marker='^')
    ax1.set_xlabel('Days', fontsize=11, fontweight='bold')
    ax1.set_yticks([1, 2])
    ax1.set_yticklabels(['ALM', 'Algo'])
    ax1.set_title('Rebalancing Event Timeline', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Panel 2: Event Counts
    event_types = ['ALM', 'Algo']
    event_counts = [len(alm_events), len(algo_events)]
    colors_bar = ['blue', 'orange']
    
    bars = ax2.bar(event_types, event_counts, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    for bar, count in zip(bars, event_counts):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{count}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax2.set_ylabel('Number of Events', fontsize=11, fontweight='bold')
    ax2.set_title('Rebalancing Event Counts', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Cumulative Events
    all_times = sorted(alm_times + algo_times)
    cumulative = range(1, len(all_times) + 1)
    
    ax3.plot(all_times, cumulative, color=color, linewidth=2)
    ax3.set_xlabel('Days', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Cumulative Rebalancing Events', fontsize=11, fontweight='bold')
    ax3.set_title('Cumulative Rebalancing Activity', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Summary Text
    ax4.axis('off')
    summary_text = f"""
    REBALANCING SUMMARY
    
    Total Events: {len(rebal_events)}
    ALM Events: {len(alm_events)}
    Algo Events: {len(algo_events)}
    
    Duration: {comparison_obj.config.simulation_duration_days} days
    Event Frequency: {len(rebal_events) / comparison_obj.config.simulation_duration_days:.1f} per day
    """
    ax4.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=13, 
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'high_tide_rebalancing_activity.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_liquidation_analysis_chart(comparison_obj, output_dir: Path, system_type: str, results: Dict, color: str):
    """Create liquidation analysis chart"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))
    
    # Extract liquidation data from agent performance
    liquidated_agents = []
    active_agents = []
    
    for agent_id, perf in results["agent_performance"].items():
        if perf.get("health_factor", 0) > 0:
            active_agents.append(perf)
        else:
            liquidated_agents.append(perf)
    
    # Panel 1: Liquidation Summary
    categories = ['Survived', 'Liquidated']
    counts = [len(active_agents), len(liquidated_agents)]
    colors_pie = ['#2E8B57', '#8B0000']
    
    ax1.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors_pie, startangle=90,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title(f'{system_type.upper()} - Liquidation Summary', fontsize=13, fontweight='bold')
    
    # Panel 2: Health Factor Distribution at End
    active_hfs = [perf.get("health_factor", 0) for perf in active_agents if perf.get("health_factor", 0) < 10]
    
    if active_hfs:
        ax2.hist(active_hfs, bins=20, color=color, alpha=0.7, edgecolor='black')
        ax2.axvline(comparison_obj.config.agent_rebalancing_hf, color='orange', linestyle='--', 
                   label='Rebalancing Trigger', linewidth=2)
        ax2.axvline(1.0, color='red', linestyle='--', label='Liquidation Threshold', linewidth=2)
    
    ax2.set_xlabel('Health Factor', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Agents', fontsize=11, fontweight='bold')
    ax2.set_title('Health Factor Distribution (Surviving Agents)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Final Portfolio Values
    active_values = [perf.get("net_position_value", 0) for perf in active_agents]
    liquidated_values = [perf.get("net_position_value", 0) for perf in liquidated_agents]
    
    if active_values:
        ax3.hist(active_values, bins=20, alpha=0.6, color='#2E8B57', label='Survived', edgecolor='black')
    if liquidated_values:
        ax3.hist(liquidated_values, bins=20, alpha=0.6, color='#8B0000', label='Liquidated', edgecolor='black')
    
    ax3.axvline(100_000, color='gray', linestyle='--', label='Initial Value', linewidth=2)
    ax3.set_xlabel('Final Portfolio Value ($)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Number of Agents', fontsize=11, fontweight='bold')
    ax3.set_title('Final Portfolio Value Distribution', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Summary Statistics
    ax4.axis('off')
    
    avg_active_value = np.mean(active_values) if active_values else 0
    avg_liquidated_value = np.mean(liquidated_values) if liquidated_values else 0
    avg_active_hf = np.mean(active_hfs) if active_hfs else 0
    
    summary_text = f"""
    LIQUIDATION ANALYSIS
    
    Total Agents: {len(active_agents) + len(liquidated_agents)}
    Survived: {len(active_agents)} ({len(active_agents)/(len(active_agents) + len(liquidated_agents))*100:.1f}%)
    Liquidated: {len(liquidated_agents)} ({len(liquidated_agents)/(len(active_agents) + len(liquidated_agents))*100:.1f}%)
    
    Avg Final Value (Survived): ${avg_active_value:,.0f}
    Avg Final Value (Liquidated): ${avg_liquidated_value:,.0f}
    
    Avg Health Factor (Survived): {avg_active_hf:.2f}
    """
    
    ax4.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=12,
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{system_type}_liquidation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()



##############################################################################
# MONTE CARLO CHARTING FUNCTIONS
##############################################################################

def create_monte_carlo_charts(comparison_obj):
    """Generate Monte Carlo-specific charts with aggregated results"""
    
    config = comparison_obj.config
    agg_results = comparison_obj.aggregated_results
    
    # Output directory with MC suffix
    output_dir = Path("tidal_protocol_sim/results") / f"{config.test_name}_MC{config.num_monte_carlo_runs}" / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📊 Generating Monte Carlo chart suite ({config.num_monte_carlo_runs} runs)...")
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # MC Chart 1: BTC Price Evolution - All runs overlaid
    print("   Creating MC BTC price evolution chart...")
    create_mc_btc_price_chart(comparison_obj, agg_results, output_dir)
    
    # MC Chart 2: Health Factor Evolution - Average with confidence bands
    print("   Creating MC health factor evolution chart...")
    create_mc_health_factor_chart(comparison_obj, agg_results, output_dir)
    
    # MC Chart 3: Time Series Evolution - Averages with confidence bands
    print("   Creating MC High Tide time series chart...")
    create_mc_time_series_chart(comparison_obj, agg_results, output_dir, "high_tide")
    
    print("   Creating MC AAVE time series chart...")
    create_mc_time_series_chart(comparison_obj, agg_results, output_dir, "aave")
    
    # MC Chart 4: Metrics Distribution - Box plots / violin plots
    print("   Creating MC metrics distribution chart...")
    create_mc_metrics_distribution_chart(comparison_obj, agg_results, output_dir)
    
    # MC Chart 5: APY Comparison with uncertainty
    print("   Creating MC APY comparison chart...")
    create_mc_apy_comparison_chart(comparison_obj, agg_results, output_dir)
    
    print(f"✅ All Monte Carlo charts saved to: {output_dir}")


def create_mc_btc_price_chart(comparison_obj, agg_results, output_dir):
    """BTC price evolution - each run in different color with average overlay"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    btc_paths = agg_results["btc_price_paths"]
    num_runs = len(btc_paths)
    days = np.arange(len(btc_paths[0]))
    
    # Use colormap for individual runs
    colors = plt.cm.tab20(np.linspace(0, 1, num_runs))
    
    # Plot individual runs with transparency
    for i, path in enumerate(btc_paths):
        ax.plot(days, path, alpha=0.5, linewidth=1.5, color=colors[i], 
                label=f'Run {i+1}' if num_runs <= 10 else None)
    
    # Calculate and plot average path in bold
    avg_path = np.mean(btc_paths, axis=0)
    ax.plot(days, avg_path, linewidth=3.5, color='black', label='Average Path', zorder=100)
    
    # Add percentile bands
    p10_path = np.percentile(btc_paths, 10, axis=0)
    p90_path = np.percentile(btc_paths, 90, axis=0)
    ax.fill_between(days, p10_path, p90_path, alpha=0.2, color='gray', 
                     label='10th-90th Percentile', zorder=50)
    
    # Reference lines
    ax.axhline(comparison_obj.config.btc_initial_price, color='gray', linestyle='--', 
               alpha=0.6, label='Initial Price ($100k)', linewidth=1.5, zorder=10)
    
    ax.set_xlabel('Days', fontsize=14, fontweight='bold')
    ax.set_ylabel('BTC Price (USD)', fontsize=14, fontweight='bold')
    ax.set_title(f'BTC Price Evolution - {num_runs} Monte Carlo Runs', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best', ncol=2 if num_runs <= 10 else 1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}k'))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mc_btc_price_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_mc_health_factor_chart(comparison_obj, agg_results, output_dir):
    """Health factor evolution - average with confidence bands for both systems"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    days = np.arange(comparison_obj.config.simulation_duration_days + 1)
    
    # High Tide
    ht_data = agg_results["high_tide"]["time_series"]
    ax1.plot(days, ht_data["avg_health_factors"], linewidth=3, color='#2E8B57', 
             label='Average HF', zorder=10)
    ax1.fill_between(days, 
                     ht_data["p10_health_factors"], 
                     ht_data["p90_health_factors"],
                     alpha=0.3, color='#2E8B57', label='10th-90th Percentile', zorder=5)
    
    # Add ±1 std band
    avg_hf = np.array(ht_data["avg_health_factors"])
    std_hf = np.array(ht_data["std_health_factors"])
    ax1.fill_between(days, avg_hf - std_hf, avg_hf + std_hf,
                     alpha=0.2, color='#2E8B57', label='±1 Std Dev', zorder=3)
    
    ax1.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Liquidation Threshold', zorder=15)
    ax1.axhline(comparison_obj.config.agent_rebalancing_hf, color='orange', linestyle=':', 
                linewidth=2, label=f'Rebalancing HF ({comparison_obj.config.agent_rebalancing_hf})', zorder=15)
    
    ax1.set_xlabel('Days', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Health Factor', fontsize=13, fontweight='bold')
    ax1.set_title(f'High Tide Health Factor ({agg_results["num_runs"]} MC Runs)', 
                  fontsize=15, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0.9)
    
    # AAVE
    aave_data = agg_results["aave"]["time_series"]
    ax2.plot(days, aave_data["avg_health_factors"], linewidth=3, color='#DC143C', 
             label='Average HF', zorder=10)
    ax2.fill_between(days, 
                     aave_data["p10_health_factors"], 
                     aave_data["p90_health_factors"],
                     alpha=0.3, color='#DC143C', label='10th-90th Percentile', zorder=5)
    
    avg_hf_aave = np.array(aave_data["avg_health_factors"])
    std_hf_aave = np.array(aave_data["std_health_factors"])
    ax2.fill_between(days, avg_hf_aave - std_hf_aave, avg_hf_aave + std_hf_aave,
                     alpha=0.2, color='#DC143C', label='±1 Std Dev', zorder=3)
    
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Liquidation Threshold', zorder=15)
    
    ax2.set_xlabel('Days', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Health Factor', fontsize=13, fontweight='bold')
    ax2.set_title(f'AAVE Health Factor ({agg_results["num_runs"]} MC Runs)', 
                  fontsize=15, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0.9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mc_health_factor_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_mc_time_series_chart(comparison_obj, agg_results, output_dir, system_type):
    """Time series evolution - averages with confidence bands (2x2 layout)"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()
    
    data = agg_results[system_type]["time_series"]
    days = np.arange(comparison_obj.config.simulation_duration_days + 1)
    color = '#2E8B57' if system_type == "high_tide" else '#DC143C'
    system_name = "High Tide" if system_type == "high_tide" else "AAVE"
    
    # Collateral
    avg_collateral = np.array(data["avg_collateral"])
    std_collateral = np.array(data["std_collateral"])
    axes[0].plot(days, avg_collateral / 1e6, linewidth=2.5, color=color, label='Average')
    axes[0].fill_between(days, (avg_collateral - std_collateral) / 1e6,
                        (avg_collateral + std_collateral) / 1e6,
                        alpha=0.3, color=color, label='±1 Std Dev')
    axes[0].set_title(f'{system_name}: Total Collateral Value', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Collateral ($M)', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Debt
    avg_debt = np.array(data["avg_debt"])
    std_debt = np.array(data["std_debt"])
    axes[1].plot(days, avg_debt / 1e6, linewidth=2.5, color=color, label='Average')
    axes[1].fill_between(days, (avg_debt - std_debt) / 1e6,
                        (avg_debt + std_debt) / 1e6,
                        alpha=0.3, color=color, label='±1 Std Dev')
    axes[1].set_title(f'{system_name}: Total Debt', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Debt ($M)', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # YT Value
    avg_yt = np.array(data["avg_yt_value"])
    std_yt = np.array(data["std_yt_value"])
    axes[2].plot(days, avg_yt / 1e6, linewidth=2.5, color=color, label='Average')
    axes[2].fill_between(days, (avg_yt - std_yt) / 1e6,
                        (avg_yt + std_yt) / 1e6,
                        alpha=0.3, color=color, label='±1 Std Dev')
    axes[2].set_title(f'{system_name}: Yield Token Value', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Days', fontsize=12)
    axes[2].set_ylabel('YT Value ($M)', fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    
    # Net Position
    avg_net = np.array(data["avg_net_position"])
    std_net = np.array(data["std_net_position"])
    axes[3].plot(days, avg_net / 1e6, linewidth=2.5, color=color, label='Average')
    axes[3].fill_between(days, (avg_net - std_net) / 1e6,
                        (avg_net + std_net) / 1e6,
                        alpha=0.3, color=color, label='±1 Std Dev')
    axes[3].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[3].set_title(f'{system_name}: Net Position Value', fontsize=13, fontweight='bold')
    axes[3].set_xlabel('Days', fontsize=12)
    axes[3].set_ylabel('Net Position ($M)', fontsize=12)
    axes[3].legend(fontsize=10)
    axes[3].grid(True, alpha=0.3)
    
    plt.suptitle(f'{system_name} Time Series Evolution ({agg_results["num_runs"]} MC Runs)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / f'mc_{system_type}_time_series_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_mc_metrics_distribution_chart(comparison_obj, agg_results, output_dir):
    """Distribution of final metrics across MC runs"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    ht_metrics = agg_results["high_tide"]["final_metrics"]
    aave_metrics = agg_results["aave"]["final_metrics"]
    
    # Net APY Distribution
    ax = axes[0, 0]
    ht_apys = np.array(ht_metrics["distribution_net_apy"]) * 100
    aave_apys = np.array(aave_metrics["distribution_net_apy"]) * 100
    
    positions = [1, 2]
    bp = ax.boxplot([ht_apys, aave_apys], positions=positions, widths=0.6,
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('#2E8B57')
    bp['boxes'][1].set_facecolor('#DC143C')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['High Tide', 'AAVE'], fontsize=12, fontweight='bold')
    ax.set_ylabel('Net APY (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Net APY Distribution ({agg_results["num_runs"]} Runs)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Survival Rate Distribution  
    ax = axes[0, 1]
    ht_survival = np.array(ht_metrics["distribution_survival_rate"]) * 100
    aave_survival = np.array(aave_metrics["distribution_survival_rate"]) * 100
    
    bp = ax.boxplot([ht_survival, aave_survival], positions=positions, widths=0.6,
                     patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('#2E8B57')
    bp['boxes'][1].set_facecolor('#DC143C')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['High Tide', 'AAVE'], fontsize=12, fontweight='bold')
    ax.set_ylabel('Survival Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Survival Rate Distribution ({agg_results["num_runs"]} Runs)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Histograms
    ax = axes[1, 0]
    ax.hist(ht_apys, bins=15, alpha=0.7, color='#2E8B57', label='High Tide', edgecolor='black')
    ax.hist(aave_apys, bins=15, alpha=0.7, color='#DC143C', label='AAVE', edgecolor='black')
    ax.axvline(np.mean(ht_apys), color='#2E8B57', linestyle='--', linewidth=2, label=f'HT Mean: {np.mean(ht_apys):.1f}%')
    ax.axvline(np.mean(aave_apys), color='#DC143C', linestyle='--', linewidth=2, label=f'AAVE Mean: {np.mean(aave_apys):.1f}%')
    ax.set_xlabel('Net APY (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Net APY Histogram', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Summary Statistics Table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'High Tide', 'AAVE', 'Delta'],
        ['Avg Net APY', f'{ht_metrics["avg_net_apy"]*100:.2f}%', f'{aave_metrics["avg_net_apy"]*100:.2f}%', 
         f'{(ht_metrics["avg_net_apy"]-aave_metrics["avg_net_apy"])*100:+.2f}%'],
        ['Std Net APY', f'{ht_metrics["std_net_apy"]*100:.2f}%', f'{aave_metrics["std_net_apy"]*100:.2f}%', ''],
        ['Min Net APY', f'{ht_metrics["min_net_apy"]*100:.2f}%', f'{aave_metrics["min_net_apy"]*100:.2f}%', ''],
        ['Max Net APY', f'{ht_metrics["max_net_apy"]*100:.2f}%', f'{aave_metrics["max_net_apy"]*100:.2f}%', ''],
        ['Avg Survival', f'{ht_metrics["avg_survival_rate"]*100:.1f}%', f'{aave_metrics["avg_survival_rate"]*100:.1f}%',
         f'{(ht_metrics["avg_survival_rate"]-aave_metrics["avg_survival_rate"])*100:+.1f}%'],
    ]
    
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#CCCCCC')
        table[(0, i)].set_text_props(weight='bold')
    
    ax.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle(f'Monte Carlo Metrics Distribution ({agg_results["num_runs"]} Runs)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'mc_metrics_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_mc_apy_comparison_chart(comparison_obj, agg_results, output_dir):
    """APY comparison with error bars showing uncertainty"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ht_metrics = agg_results["high_tide"]["final_metrics"]
    aave_metrics = agg_results["aave"]["final_metrics"]
    
    systems = ['High Tide', 'AAVE']
    avgs = [ht_metrics["avg_net_apy"] * 100, aave_metrics["avg_net_apy"] * 100]
    stds = [ht_metrics["std_net_apy"] * 100, aave_metrics["std_net_apy"] * 100]
    colors = ['#2E8B57', '#DC143C']
    
    bars = ax.bar(systems, avgs, yerr=stds, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2, width=0.5, capsize=10)
    
    # Add value labels on bars
    for i, (bar, avg, std) in enumerate(zip(bars, avgs, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 1,
                f'{avg:.2f}% ± {std:.2f}%',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Delta annotation
    delta = avgs[0] - avgs[1]
    ax.text(0.5, max(avgs) * 0.85, f'Δ = {delta:+.2f}%', 
            transform=ax.transData, fontsize=16, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            ha='center')
    
    ax.set_ylabel('Net APY (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Net APY Comparison with Uncertainty ({agg_results["num_runs"]} MC Runs)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mc_apy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

