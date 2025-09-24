#!/usr/bin/env python3
"""
Pool Utilization Verification
Verify that the fixed pool behavior is working correctly and pools aren't being exhausted
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def verify_pool_utilization():
    """Verify pool utilization with the fixed pool behavior"""
    
    # Load results
    with open('tidal_protocol_sim/results/Tri_Health_Factor_Analysis_RebHF_1025/tri_health_factor_analysis_results.json', 'r') as f:
        results = json.load(f)
    
    print("🔍 POOL UTILIZATION VERIFICATION")
    print("Checking if pools are being exhausted with fixed behavior")
    print("=" * 70)
    
    # Pool constants
    INITIAL_POOL_SIZE = 500_000  # $500K total
    INITIAL_MOET_RESERVE = 250_000  # $250K MOET
    INITIAL_YT_RESERVE = 250_000  # $250K Yield Tokens
    
    scenario_data = []
    total_moet_extracted = 0
    
    for scenario in results["scenario_results"]:
        scenario_name = scenario["scenario_name"]
        target_hf = scenario["scenario_params"]["target_hf"]
        
        # Extract MOET usage from rebalancing events
        scenario_moet_extracted = 0
        total_rebalancing_events = 0
        failed_trades = 0
        
        # Look in high_tide_summary for rebalancing data
        # Use a set to track unique rebalancing events and avoid double-counting
        unique_events = set()
        
        for agent in scenario.get("high_tide_summary", {}).get("all_agent_outcomes", []):
            for rebalancing_event in agent.get("rebalancing_events_list", []):
                if "moet_raised" in rebalancing_event:
                    # Create unique key to avoid counting duplicates
                    event_key = (
                        rebalancing_event.get("agent_id", "unknown"),
                        rebalancing_event.get("minute", 0),
                        rebalancing_event.get("moet_raised", 0),
                        rebalancing_event.get("rebalancing_type", "unknown")
                    )
                    
                    # Only count unique yield_token_sale events (the actual pool extraction)
                    if (event_key not in unique_events and 
                        rebalancing_event.get("rebalancing_type") == "yield_token_sale"):
                        unique_events.add(event_key)
                        
                        moet_raised = rebalancing_event["moet_raised"]
                        if moet_raised > 0:
                            scenario_moet_extracted += moet_raised
                            total_rebalancing_events += 1
                        else:
                            failed_trades += 1  # Trade failed due to pool exhaustion
        
        utilization_pct = (scenario_moet_extracted / INITIAL_MOET_RESERVE) * 100
        
        scenario_data.append({
            'name': scenario_name,
            'target_hf': target_hf,
            'moet_extracted': scenario_moet_extracted,
            'utilization_pct': utilization_pct,
            'rebalancing_events': total_rebalancing_events,
            'failed_trades': failed_trades
        })
        
        total_moet_extracted += scenario_moet_extracted
        
        print(f"\n📊 {scenario_name}")
        print(f"   Target HF: {target_hf}")
        print(f"   MOET Extracted: ${scenario_moet_extracted:,.0f}")
        print(f"   Pool Utilization: {utilization_pct:.1f}% of ${INITIAL_MOET_RESERVE:,} MOET reserve")
        print(f"   Successful Trades: {total_rebalancing_events}")
        print(f"   Failed Trades: {failed_trades}")
        
        if utilization_pct > 95:
            print(f"   🚨 WARNING: High utilization - approaching pool limits!")
        elif utilization_pct > 100:
            print(f"   ❌ ERROR: Pool exhausted - impossible utilization!")
        else:
            print(f"   ✅ Healthy utilization - within pool limits")
    
    # Summary
    print(f"\n🎯 SUMMARY:")
    print(f"   Total MOET Extracted Across All Scenarios: ${total_moet_extracted:,.0f}")
    print(f"   Available MOET Reserve Per Scenario: ${INITIAL_MOET_RESERVE:,}")
    print(f"   Average Utilization: {np.mean([s['utilization_pct'] for s in scenario_data]):.1f}%")
    print(f"   Maximum Utilization: {max([s['utilization_pct'] for s in scenario_data]):.1f}%")
    print(f"   Total Failed Trades: {sum([s['failed_trades'] for s in scenario_data])}")
    
    # Create visualization
    create_pool_utilization_chart(scenario_data)
    
    # Validate results
    max_util = max([s['utilization_pct'] for s in scenario_data])
    total_failed = sum([s['failed_trades'] for s in scenario_data])
    
    print(f"\n✅ POOL BEHAVIOR VALIDATION:")
    if max_util <= 100:
        print(f"   ✅ Pool utilization is realistic (max {max_util:.1f}%)")
    else:
        print(f"   ❌ Pool utilization exceeds limits (max {max_util:.1f}%)")
    
    if total_failed > 0:
        print(f"   ✅ Pool correctly failed {total_failed} trades when exhausted")
    else:
        print(f"   ✅ No trade failures - pool had sufficient liquidity")
    
    return scenario_data

def create_pool_utilization_chart(scenario_data):
    """Create pool utilization visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Pool Liquidity Utilization Verification\n$250K MOET Reserve Usage Analysis', fontsize=16, fontweight='bold')
    
    scenarios = [s['name'].replace('Target_HF_', '').replace('_', ' ') for s in scenario_data]
    target_hfs = [s['target_hf'] for s in scenario_data]
    utilizations = [s['utilization_pct'] for s in scenario_data]
    moet_extracted = [s['moet_extracted'] for s in scenario_data]
    failed_trades = [s['failed_trades'] for s in scenario_data]
    
    # 1. Pool Utilization Percentage
    bars1 = ax1.bar(scenarios, utilizations, color='skyblue', alpha=0.7)
    ax1.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Pool Limit (100%)')
    ax1.set_title('Pool Utilization by Scenario', fontweight='bold')
    ax1.set_ylabel('Utilization (%)')
    ax1.set_ylim(0, max(120, max(utilizations) * 1.1))
    ax1.legend()
    
    # Add value labels on bars
    for bar, util in zip(bars1, utilizations):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{util:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. MOET Extracted (Dollar Amounts)
    bars2 = ax2.bar(scenarios, moet_extracted, color='lightgreen', alpha=0.7)
    ax2.axhline(y=250000, color='red', linestyle='--', linewidth=2, label='MOET Reserve ($250K)')
    ax2.set_title('MOET Extracted by Scenario', fontweight='bold')
    ax2.set_ylabel('MOET Extracted ($)')
    ax2.legend()
    
    # Add value labels
    for bar, moet in zip(bars2, moet_extracted):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000, 
                f'${moet/1000:.0f}K', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # 3. Target HF vs Utilization (Scatter)
    ax3.scatter(target_hfs, utilizations, s=100, alpha=0.7, c='orange')
    ax3.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Pool Limit')
    ax3.set_title('Target Health Factor vs Pool Utilization', fontweight='bold')
    ax3.set_xlabel('Target Health Factor')
    ax3.set_ylabel('Pool Utilization (%)')
    ax3.legend()
    
    # Add scenario labels to points
    for i, scenario in enumerate(scenarios):
        ax3.annotate(scenario.split()[1], (target_hfs[i], utilizations[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Failed Trades Analysis
    bars4 = ax4.bar(scenarios, failed_trades, color='red', alpha=0.7)
    ax4.set_title('Failed Trades Due to Pool Exhaustion', fontweight='bold')
    ax4.set_ylabel('Number of Failed Trades')
    
    # Add value labels
    for bar, failed in zip(bars4, failed_trades):
        if failed > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{failed}', ha='center', va='bottom', fontweight='bold')
    
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save chart
    output_dir = Path("tidal_protocol_sim/results/Tri_Health_Factor_Analysis/charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "pool_utilization_verification.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Pool utilization chart saved to: {output_dir}/pool_utilization_verification.png")

if __name__ == "__main__":
    verify_pool_utilization()
