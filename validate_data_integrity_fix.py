#!/usr/bin/env python3
"""
Data Integrity Fix Validation Script

This script validates that the critical data integrity fixes are working correctly
by checking that High Tide costs now reflect real Uniswap V3 swap data instead of 
agent portfolio calculations.
"""

import sys
import json
from pathlib import Path

def validate_engine_data_structure(results_data):
    """Validate that engine data is properly included in results"""
    issues = []
    
    # Check for engine data in High Tide results
    scenario_results = results_data.get("scenario_results", [])
    for scenario in scenario_results:
        # Check High Tide runs
        ht_runs = scenario.get("high_tide_runs", [])
        for i, run in enumerate(ht_runs):
            if "engine_data" not in run:
                issues.append(f"❌ High Tide run {i} in scenario '{scenario['scenario_name']}' missing engine_data")
            else:
                engine_data = run["engine_data"]
                if "rebalancing_events" not in engine_data:
                    issues.append(f"❌ High Tide run {i} engine_data missing rebalancing_events")
                if "yield_token_trades" not in engine_data:
                    issues.append(f"❌ High Tide run {i} engine_data missing yield_token_trades")
    
    return issues

def validate_agent_outcome_data_source(results_data):
    """Validate that agent outcomes use real engine data"""
    issues = []
    
    scenario_results = results_data.get("scenario_results", [])
    for scenario in scenario_results:
        # Check High Tide agent outcomes
        ht_summary = scenario.get("high_tide_summary", {})
        agent_outcomes = ht_summary.get("all_agent_outcomes", [])
        
        for agent in agent_outcomes:
            if "data_source" not in agent:
                issues.append(f"❌ Agent {agent.get('agent_id', 'unknown')} missing data_source flag")
            elif agent["data_source"] != "engine_real_swaps":
                issues.append(f"❌ Agent {agent.get('agent_id', 'unknown')} using wrong data source: {agent['data_source']}")
            
            # Check if costs are realistic (not near-zero for High Tide)
            cost_of_rebalancing = agent.get("cost_of_rebalancing", 0)
            rebalancing_events = agent.get("rebalancing_events", 0)
            
            if rebalancing_events > 0 and cost_of_rebalancing < 1.0:
                issues.append(f"❌ Agent {agent.get('agent_id', 'unknown')} has {rebalancing_events} rebalancing events but cost of ${cost_of_rebalancing:.2f} (too low for real swaps)")
    
    return issues

def validate_slippage_metrics_data_source(results_data):
    """Validate that slippage metrics use engine data"""
    issues = []
    
    scenario_results = results_data.get("scenario_results", [])
    for scenario in scenario_results:
        ht_runs = scenario.get("high_tide_runs", [])
        for i, run in enumerate(ht_runs):
            slippage_data = run.get("slippage_metrics_data", {})
            if "data_source" not in slippage_data:
                issues.append(f"❌ High Tide run {i} slippage_metrics_data missing data_source flag")
            elif slippage_data["data_source"] != "engine_real_swaps":
                issues.append(f"❌ High Tide run {i} slippage_metrics_data using wrong source: {slippage_data['data_source']}")
    
    return issues

def validate_rebalancing_events_data_source(results_data):
    """Validate that rebalancing events use engine data"""
    issues = []
    
    scenario_results = results_data.get("scenario_results", [])
    for scenario in scenario_results:
        ht_runs = scenario.get("high_tide_runs", [])
        for i, run in enumerate(ht_runs):
            rebalancing_data = run.get("rebalancing_events_data", {})
            rebalancing_events = rebalancing_data.get("rebalancing_events", [])
            
            for event in rebalancing_events:
                if "data_source" not in event:
                    issues.append(f"❌ Rebalancing event missing data_source flag")
                elif event["data_source"] != "engine_real_swaps":
                    issues.append(f"❌ Rebalancing event using wrong source: {event['data_source']}")
    
    return issues

def validate_cost_realism(results_data):
    """Validate that High Tide costs are realistic for Uniswap V3 swaps"""
    issues = []
    warnings = []
    
    scenario_results = results_data.get("scenario_results", [])
    for scenario in scenario_results:
        ht_summary = scenario.get("high_tide_summary", {})
        mean_total_cost = ht_summary.get("mean_total_cost", 0)
        
        # High Tide costs should not be near-zero if there are rebalancing events
        if mean_total_cost < 10.0:  # Arbitrary threshold - real Uniswap swaps should cost more
            warnings.append(f"⚠️  Scenario '{scenario['scenario_name']}' High Tide mean cost ${mean_total_cost:.2f} seems low for real Uniswap V3 swaps")
        
        # Check individual agent costs
        agent_outcomes = ht_summary.get("all_agent_outcomes", [])
        for agent in agent_outcomes:
            cost = agent.get("cost_of_rebalancing", 0)
            events = agent.get("rebalancing_events", 0)
            
            if events > 0 and cost == 0:
                issues.append(f"❌ Agent {agent.get('agent_id', 'unknown')} has {events} rebalancing events but $0 cost")
    
    return issues, warnings

def main():
    """Main validation function"""
    print("🔍 VALIDATING DATA INTEGRITY FIXES")
    print("=" * 50)
    
    # Look for the most recent results file
    results_dir = Path("tidal_protocol_sim/results")
    if not results_dir.exists():
        results_dir = Path("results")
    
    if not results_dir.exists():
        print("❌ No results directory found. Please run the comprehensive analysis first.")
        return 1
    
    # Find the most recent comprehensive analysis results
    comprehensive_dirs = list(results_dir.glob("Comprehensive_HT_vs_Aave_Analysis"))
    if not comprehensive_dirs:
        print("❌ No Comprehensive_HT_vs_Aave_Analysis results found.")
        print("   Please run comprehensive_ht_vs_aave_analysis.py first to generate test data.")
        return 1
    
    # Find the most recent run
    latest_dir = max(comprehensive_dirs, key=lambda p: p.stat().st_mtime)
    json_files = list(latest_dir.glob("comprehensive_ht_vs_aave_results.json"))
    
    if not json_files:
        print(f"❌ No JSON results file found in {latest_dir}")
        return 1
    
    results_file = json_files[0]
    print(f"📂 Validating results from: {results_file}")
    
    try:
        with open(results_file, 'r') as f:
            results_data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading results file: {e}")
        return 1
    
    # Run validation checks
    all_issues = []
    all_warnings = []
    
    print("\n🔍 Checking engine data structure...")
    issues = validate_engine_data_structure(results_data)
    all_issues.extend(issues)
    if not issues:
        print("✅ Engine data structure: PASS")
    
    print("\n🔍 Checking agent outcome data source...")
    issues = validate_agent_outcome_data_source(results_data)
    all_issues.extend(issues)
    if not issues:
        print("✅ Agent outcome data source: PASS")
    
    print("\n🔍 Checking slippage metrics data source...")
    issues = validate_slippage_metrics_data_source(results_data)
    all_issues.extend(issues)
    if not issues:
        print("✅ Slippage metrics data source: PASS")
    
    print("\n🔍 Checking rebalancing events data source...")
    issues = validate_rebalancing_events_data_source(results_data)
    all_issues.extend(issues)
    if not issues:
        print("✅ Rebalancing events data source: PASS")
    
    print("\n🔍 Checking cost realism...")
    issues, warnings = validate_cost_realism(results_data)
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    if not issues:
        print("✅ Cost realism: PASS")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    if not all_issues and not all_warnings:
        print("🎉 ALL VALIDATION CHECKS PASSED!")
        print("   The data integrity fixes are working correctly.")
        print("   High Tide costs now reflect real Uniswap V3 swap data.")
        return 0
    
    if all_issues:
        print(f"❌ FOUND {len(all_issues)} CRITICAL ISSUES:")
        for issue in all_issues:
            print(f"   {issue}")
    
    if all_warnings:
        print(f"\n⚠️  FOUND {len(all_warnings)} WARNINGS:")
        for warning in all_warnings:
            print(f"   {warning}")
    
    if all_issues:
        print("\n❌ DATA INTEGRITY FIXES NEED MORE WORK")
        return 1
    else:
        print("\n✅ DATA INTEGRITY FIXES WORKING (with warnings)")
        return 0

if __name__ == "__main__":
    sys.exit(main())
