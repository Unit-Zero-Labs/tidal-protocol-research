#!/usr/bin/env python3
"""
MOET:YT Pool Borrow Cap Analysis

Tests whether borrow caps should be set as a percentage of liquidity in the MOET:YT pool.
Uses baseline $250K:$250K MOET:YT pool and tests against many agents with tight 
HF ranges who will initiate frequent rebalances.

Key Question: Is there a borrow cap that should be set as a % of Liquidity in the MOET:YT pool?
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tidal_protocol_sim.simulation.high_tide_engine import HighTideConfig, HighTideSimulationEngine
from tidal_protocol_sim.simulation.aave_engine import AaveConfig, AaveSimulationEngine
from tidal_protocol_sim.agents.high_tide_agent import HighTideAgent
from tidal_protocol_sim.agents.aave_agent import AaveAgent


def run_borrow_cap_analysis():
    """Run comprehensive MOET:YT pool borrow cap analysis"""
    
    print("=" * 80)
    print("MOET:YT POOL BORROW CAP ANALYSIS")
    print("=" * 80)
    print("Testing impact of agent borrowing on MOET:YT pool capacity")
    print("Baseline: $250K:$250K MOET:YT pool")
    print("Question: Should we set borrow caps as % of pool liquidity?")
    print()
    
    # Baseline pool configuration
    baseline_moet_yt_pool = 250_000  # $250K each side
    
    # Test different numbers of agents (representing different borrow loads)
    agent_counts = [20, 50, 100, 150, 200]  # Progressive stress testing
    
    # Test with agents that will rebalance frequently (tight HF ranges)
    tight_hf_scenarios = [
        {"initial_hf": 1.25, "target_hf": 1.20, "buffer": 0.05, "profile": "conservative_tight"},
        {"initial_hf": 1.20, "target_hf": 1.15, "buffer": 0.05, "profile": "moderate_tight"},
        {"initial_hf": 1.15, "target_hf": 1.10, "buffer": 0.05, "profile": "aggressive_tight"},
        {"initial_hf": 1.10, "target_hf": 1.07, "buffer": 0.03, "profile": "ultra_aggressive"}
    ]
    
    results_matrix = []
    
    for agent_count in agent_counts:
        print(f"\n👥 Testing with {agent_count} agents")
        
        # Calculate total borrowing capacity
        total_borrowing = agent_count * calculate_agent_borrowing_capacity()
        pool_utilization = total_borrowing / baseline_moet_yt_pool
        
        print(f"   Total borrowing: ${total_borrowing:,.0f}")
        print(f"   Pool utilization: {pool_utilization:.1%}")
        
        for scenario in tight_hf_scenarios:
            result = run_borrow_cap_scenario(
                agent_count, scenario, baseline_moet_yt_pool, pool_utilization
            )
            result["scenario_params"]["agent_count"] = agent_count
            result["scenario_params"]["pool_utilization"] = pool_utilization
            results_matrix.append(result)
    
    # Generate comprehensive analysis
    analysis = analyze_borrow_cap_results(results_matrix)
    
    # Save results
    save_borrow_cap_results(analysis, results_matrix)
    
    # Print summary
    print_borrow_cap_summary(analysis)
    
    return analysis


def calculate_agent_borrowing_capacity() -> float:
    """Calculate how much each agent can borrow based on 1 BTC collateral"""
    btc_price = 100_000.0  # $100K BTC
    btc_collateral_factor = 0.80  # 80% collateral factor
    average_initial_hf = 1.2  # Representative initial health factor
    
    effective_collateral = btc_price * btc_collateral_factor  # $80K
    borrowing_capacity = effective_collateral / average_initial_hf  # ~$67K
    
    return borrowing_capacity


def run_borrow_cap_scenario(agent_count: int, hf_scenario: Dict, 
                           pool_size: int, pool_utilization: float) -> Dict:
    """Run scenario testing pool capacity with specific agent load"""
    
    initial_hf = hf_scenario["initial_hf"]
    target_hf = hf_scenario["target_hf"]
    profile = hf_scenario["profile"]
    
    print(f"      {profile} (HF {initial_hf:.2f}→{target_hf:.2f}): ", end="")
    
    monte_carlo_runs = 5  # Lighter testing due to large agent counts
    
    ht_results = []
    aave_results = []
    
    for run_num in range(monte_carlo_runs):
        # High Tide simulation with many tight-range agents
        ht_config = HighTideConfig()
        ht_config.num_high_tide_agents = 0  # Custom agents
        ht_config.btc_decline_duration = 45  # Shorter for performance
        ht_config.moet_btc_pool_size = 250_000  # Standard liquidation pool
        ht_config.moet_yield_pool_size = pool_size  # Baseline YT pool
        
        # Create many agents with tight HF ranges
        tight_ht_agents = create_tight_range_agents(
            initial_hf, target_hf, agent_count, "high_tide", run_num, profile
        )
        
        ht_engine = HighTideSimulationEngine(ht_config)
        ht_engine.high_tide_agents = tight_ht_agents
        for agent in tight_ht_agents:
            ht_engine.agents[agent.agent_id] = agent
        
        ht_result = ht_engine.run_high_tide_simulation()
        ht_results.append(ht_result)
        
        # Matching Aave scenario
        aave_config = AaveConfig()
        aave_config.num_aave_agents = 0
        aave_config.btc_decline_duration = 45
        aave_config.moet_btc_pool_size = 250_000
        aave_config.moet_yield_pool_size = pool_size
        
        tight_aave_agents = create_tight_range_agents(
            initial_hf, target_hf, agent_count, "aave", run_num, profile
        )
        
        aave_engine = AaveSimulationEngine(aave_config)
        aave_engine.aave_agents = tight_aave_agents
        for agent in tight_aave_agents:
            aave_engine.agents[agent.agent_id] = agent
        
        aave_result = aave_engine.run_aave_simulation()
        aave_results.append(aave_result)
    
    # Aggregate results
    scenario_result = aggregate_borrow_cap_scenario(
        ht_results, aave_results, hf_scenario, agent_count, pool_utilization
    )
    
    # Quick feedback
    ht_rebalances = scenario_result["high_tide_metrics"]["total_rebalancing_events"]
    ht_liquidations = scenario_result["high_tide_metrics"]["liquidation_rate_percentage"]
    pool_stress = scenario_result["pool_stress_analysis"]["stress_level"]
    
    print(f"{ht_rebalances:.0f} rebalances, {ht_liquidations:.1f}% liquidations, {pool_stress} stress")
    
    return scenario_result


def create_tight_range_agents(initial_hf: float, target_hf: float, num_agents: int,
                             agent_type: str, run_num: int, profile: str) -> List:
    """Create agents with tight HF ranges for maximum rebalancing activity"""
    
    agents = []
    
    for i in range(num_agents):
        agent_id = f"tight_{agent_type}_{profile}_r{run_num}_a{i}"
        
        if agent_type == "high_tide":
            agent = HighTideAgent(agent_id, initial_hf, target_hf)
        else:  # aave
            agent = AaveAgent(agent_id, initial_hf, target_hf)
        
        # Set risk profile based on scenario
        if "conservative" in profile:
            agent.risk_profile = "conservative"
            agent.color = "#2E8B57"
        elif "moderate" in profile:
            agent.risk_profile = "moderate"
            agent.color = "#FF8C00"
        else:
            agent.risk_profile = "aggressive"
            agent.color = "#DC143C"
        
        agents.append(agent)
    
    return agents


def aggregate_borrow_cap_scenario(ht_results: List, aave_results: List,
                                 hf_scenario: Dict, agent_count: int, 
                                 pool_utilization: float) -> Dict:
    """Aggregate results for borrow cap scenario"""
    
    # High Tide metrics
    ht_survival_rates = []
    ht_rebalancing_events = []
    ht_liquidations = []
    ht_pool_stress_events = []
    
    for run in ht_results:
        survival_stats = run.get("survival_statistics", {})
        ht_survival_rates.append(survival_stats.get("survival_rate", 0.0))
        
        # Rebalancing activity
        rebalancing_activity = run.get("yield_token_activity", {})
        ht_rebalancing_events.append(rebalancing_activity.get("rebalancing_events", 0))
        
        # Emergency liquidations
        agent_outcomes = run.get("agent_outcomes", [])
        total_liquidations = sum(outcome.get("emergency_liquidations", 0) for outcome in agent_outcomes)
        ht_liquidations.append(total_liquidations)
        
        # Pool stress indicators (high rebalancing concentration)
        rebalancing_events = run.get("rebalancing_events", [])
        stress_events = count_pool_stress_events(rebalancing_events)
        ht_pool_stress_events.append(stress_events)
    
    # Aave metrics for comparison
    aave_survival_rates = []
    aave_liquidations = []
    
    for run in aave_results:
        survival_stats = run.get("survival_statistics", {})
        aave_survival_rates.append(survival_stats.get("survival_rate", 0.0))
        
        liquidation_activity = run.get("liquidation_activity", {})
        aave_liquidations.append(liquidation_activity.get("total_liquidation_events", 0))
    
    # Calculate pool stress analysis
    pool_stress_analysis = analyze_pool_capacity_stress(
        ht_rebalancing_events, ht_pool_stress_events, pool_utilization
    )
    
    return {
        "scenario_params": {
            **hf_scenario,
            "agent_count": agent_count,
            "pool_utilization": pool_utilization,
            "total_borrowing_capacity": agent_count * calculate_agent_borrowing_capacity()
        },
        "high_tide_metrics": {
            "survival_rate": np.mean(ht_survival_rates),
            "survival_rate_std": np.std(ht_survival_rates),
            "total_rebalancing_events": np.mean(ht_rebalancing_events),
            "liquidation_frequency": np.mean(ht_liquidations) / agent_count,
            "liquidation_rate_percentage": (np.mean(ht_liquidations) / agent_count) * 100,
            "pool_stress_events": np.mean(ht_pool_stress_events)
        },
        "aave_metrics": {
            "survival_rate": np.mean(aave_survival_rates),
            "liquidation_frequency": np.mean(aave_liquidations) / agent_count,
            "liquidation_rate_percentage": (np.mean(aave_liquidations) / agent_count) * 100
        },
        "pool_stress_analysis": pool_stress_analysis,
        "borrow_cap_implications": generate_borrow_cap_implications(
            pool_utilization, pool_stress_analysis, np.mean(ht_rebalancing_events)
        )
    }


def count_pool_stress_events(rebalancing_events: List) -> int:
    """Count events that indicate pool stress (multiple rebalances in short time)"""
    
    # Group rebalancing events by time windows (5-minute windows)
    time_windows = {}
    
    for event in rebalancing_events:
        minute = event.get("minute", 0)
        window = minute // 5  # 5-minute windows
        
        if window not in time_windows:
            time_windows[window] = 0
        time_windows[window] += 1
    
    # Count windows with high activity (>5 rebalances in 5 minutes)
    stress_events = sum(1 for count in time_windows.values() if count > 5)
    
    return stress_events


def analyze_pool_capacity_stress(rebalancing_events: List, stress_events: List, 
                                pool_utilization: float) -> Dict:
    """Analyze how pool capacity handles rebalancing stress"""
    
    avg_rebalancing = np.mean(rebalancing_events)
    avg_stress_events = np.mean(stress_events)
    
    # Determine stress level based on pool utilization and activity
    if pool_utilization > 0.8 and avg_stress_events > 2:
        stress_level = "Critical"
    elif pool_utilization > 0.6 and avg_stress_events > 1:
        stress_level = "High"
    elif pool_utilization > 0.4 or avg_stress_events > 0:
        stress_level = "Moderate"
    else:
        stress_level = "Low"
    
    return {
        "pool_utilization": pool_utilization,
        "avg_rebalancing_events": avg_rebalancing,
        "avg_stress_events": avg_stress_events,
        "stress_level": stress_level,
        "capacity_analysis": {
            "rebalancing_per_dollar_liquidity": avg_rebalancing / 500_000,  # $500K total pool
            "stress_events_per_dollar": avg_stress_events / 500_000,
            "utilization_efficiency": avg_rebalancing / max(pool_utilization, 0.01)
        }
    }


def generate_borrow_cap_implications(pool_utilization: float, stress_analysis: Dict,
                                   rebalancing_events: float) -> Dict:
    """Generate implications for borrow cap policy"""
    
    stress_level = stress_analysis["stress_level"]
    
    # Determine if borrow cap is needed
    if stress_level in ["Critical", "High"]:
        borrow_cap_needed = True
        recommended_cap = pool_utilization * 0.8  # 80% of current utilization
    elif stress_level == "Moderate":
        borrow_cap_needed = False  # Monitor only
        recommended_cap = pool_utilization * 1.2  # 20% buffer
    else:
        borrow_cap_needed = False
        recommended_cap = None
    
    # Calculate implied caps as percentage of pool liquidity
    if recommended_cap:
        cap_as_percentage = recommended_cap * 100
    else:
        cap_as_percentage = None
    
    return {
        "borrow_cap_needed": borrow_cap_needed,
        "recommended_cap_percentage": cap_as_percentage,
        "current_utilization_percentage": pool_utilization * 100,
        "stress_level": stress_level,
        "reasoning": generate_cap_reasoning(stress_level, pool_utilization, rebalancing_events),
        "monitoring_thresholds": {
            "utilization_warning": 60,  # 60% of pool capacity
            "utilization_critical": 80,  # 80% of pool capacity
            "rebalancing_activity_warning": 100,  # 100+ rebalancing events
            "stress_events_warning": 2  # 2+ stress periods
        }
    }


def generate_cap_reasoning(stress_level: str, pool_utilization: float, 
                          rebalancing_events: float) -> str:
    """Generate reasoning for borrow cap recommendation"""
    
    if stress_level == "Critical":
        return f"Pool utilization at {pool_utilization:.1%} with {rebalancing_events:.0f} rebalancing events indicates severe stress. Borrow cap needed to prevent system failure."
    elif stress_level == "High":
        return f"High rebalancing activity ({rebalancing_events:.0f} events) at {pool_utilization:.1%} utilization. Borrow cap recommended to maintain system stability."
    elif stress_level == "Moderate":
        return f"Moderate stress at {pool_utilization:.1%} utilization. Monitor closely but no immediate cap needed."
    else:
        return f"Low stress at {pool_utilization:.1%} utilization. System operating within normal parameters."


def analyze_borrow_cap_results(results_matrix: List[Dict]) -> Dict:
    """Analyze all borrow cap scenarios to determine optimal policies"""
    
    # Create DataFrame for analysis
    df_data = []
    for result in results_matrix:
        params = result["scenario_params"]
        ht_metrics = result["high_tide_metrics"]
        stress_analysis = result["pool_stress_analysis"]
        implications = result["borrow_cap_implications"]
        
        df_data.append({
            "agent_count": params["agent_count"],
            "initial_hf": params["initial_hf"],
            "target_hf": params["target_hf"],
            "hf_buffer": params["buffer"],
            "pool_utilization": params["pool_utilization"],
            "profile": params["profile"],
            "survival_rate": ht_metrics["survival_rate"],
            "rebalancing_events": ht_metrics["total_rebalancing_events"],
            "liquidation_rate": ht_metrics["liquidation_rate_percentage"],
            "stress_level": stress_analysis["stress_level"],
            "pool_stress_events": ht_metrics["pool_stress_events"],
            "borrow_cap_needed": implications["borrow_cap_needed"],
            "recommended_cap_percentage": implications.get("recommended_cap_percentage", 0)
        })
    
    df = pd.DataFrame(df_data)
    
    # Generate comprehensive analysis
    analysis = {
        "capacity_thresholds": find_capacity_thresholds(df),
        "borrow_cap_recommendations": generate_borrow_cap_recommendations(df),
        "stress_level_analysis": analyze_stress_levels_by_utilization(df),
        "rebalancing_capacity_limits": find_rebalancing_capacity_limits(df),
        "risk_profile_impact": analyze_risk_profile_pool_impact(df),
        "raw_results_matrix": results_matrix
    }
    
    return analysis


def find_capacity_thresholds(df: pd.DataFrame) -> Dict:
    """Find critical capacity thresholds for the MOET:YT pool"""
    
    # Group by stress level to find thresholds
    stress_thresholds = {}
    
    for stress_level in ["Low", "Moderate", "High", "Critical"]:
        stress_data = df[df["stress_level"] == stress_level]
        
        if len(stress_data) > 0:
            stress_thresholds[stress_level] = {
                "min_utilization": stress_data["pool_utilization"].min(),
                "max_utilization": stress_data["pool_utilization"].max(),
                "min_agent_count": stress_data["agent_count"].min(),
                "max_agent_count": stress_data["agent_count"].max(),
                "avg_rebalancing_events": stress_data["rebalancing_events"].mean(),
                "avg_liquidation_rate": stress_data["liquidation_rate"].mean()
            }
    
    # Find critical thresholds
    critical_utilization = None
    safe_utilization = None
    
    # Critical: First utilization level that shows Critical or High stress
    critical_data = df[df["stress_level"].isin(["Critical", "High"])]
    if len(critical_data) > 0:
        critical_utilization = critical_data["pool_utilization"].min()
    
    # Safe: Highest utilization with Low stress
    safe_data = df[df["stress_level"] == "Low"]
    if len(safe_data) > 0:
        safe_utilization = safe_data["pool_utilization"].max()
    
    return {
        "stress_level_thresholds": stress_thresholds,
        "critical_utilization_threshold": critical_utilization,
        "safe_utilization_threshold": safe_utilization,
        "recommended_utilization_cap": safe_utilization * 0.9 if safe_utilization else 0.5  # 10% safety buffer
    }


def generate_borrow_cap_recommendations(df: pd.DataFrame) -> Dict:
    """Generate specific borrow cap recommendations"""
    
    # Find scenarios where borrow caps are recommended
    cap_needed = df[df["borrow_cap_needed"] == True]
    cap_not_needed = df[df["borrow_cap_needed"] == False]
    
    recommendations = {}
    
    if len(cap_needed) > 0:
        # Conservative recommendation: Lowest cap where system shows stress
        min_stressed_utilization = cap_needed["pool_utilization"].min()
        conservative_cap = min_stressed_utilization * 0.8  # 20% safety buffer
        
        # Aggressive recommendation: Just below stress threshold
        max_safe_utilization = cap_not_needed["pool_utilization"].max() if len(cap_not_needed) > 0 else 0.5
        aggressive_cap = max_safe_utilization * 1.1  # 10% above safe threshold
        
        recommendations = {
            "conservative_cap": {
                "utilization_percentage": conservative_cap * 100,
                "reasoning": f"20% safety buffer below observed stress at {min_stressed_utilization:.1%} utilization"
            },
            "aggressive_cap": {
                "utilization_percentage": aggressive_cap * 100,
                "reasoning": f"10% buffer above safe threshold of {max_safe_utilization:.1%} utilization"
            },
            "recommended_approach": "conservative_cap",  # Default to conservative
            "monitoring_required": True
        }
    else:
        # No caps needed - system handles all tested loads
        max_tested_utilization = df["pool_utilization"].max()
        
        recommendations = {
            "no_cap_needed": True,
            "max_tested_utilization": max_tested_utilization * 100,
            "reasoning": f"System stable up to {max_tested_utilization:.1%} pool utilization",
            "monitoring_threshold": max_tested_utilization * 1.2 * 100,  # Monitor at 20% above tested
            "recommended_approach": "monitoring_only"
        }
    
    return recommendations


def analyze_stress_levels_by_utilization(df: pd.DataFrame) -> Dict:
    """Analyze how stress levels correlate with pool utilization"""
    
    utilization_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, float('inf')]
    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%", "100%+"]
    
    utilization_analysis = {}
    
    for i, (low, high) in enumerate(zip(utilization_bins[:-1], utilization_bins[1:])):
        bin_label = bin_labels[i]
        bin_data = df[(df["pool_utilization"] >= low) & (df["pool_utilization"] < high)]
        
        if len(bin_data) > 0:
            utilization_analysis[bin_label] = {
                "utilization_range": f"{low:.0%}-{high:.0%}",
                "scenario_count": len(bin_data),
                "avg_survival_rate": bin_data["survival_rate"].mean(),
                "avg_rebalancing_events": bin_data["rebalancing_events"].mean(),
                "avg_liquidation_rate": bin_data["liquidation_rate"].mean(),
                "stress_level_distribution": bin_data["stress_level"].value_counts().to_dict(),
                "dominant_stress_level": bin_data["stress_level"].mode().iloc[0] if not bin_data["stress_level"].mode().empty else "Unknown"
            }
    
    return utilization_analysis


def find_rebalancing_capacity_limits(df: pd.DataFrame) -> Dict:
    """Find the limits of rebalancing capacity under different loads"""
    
    # Analyze rebalancing efficiency across different loads
    efficiency_analysis = {}
    
    for agent_count in df["agent_count"].unique():
        count_data = df[df["agent_count"] == agent_count]
        
        avg_rebalancing = count_data["rebalancing_events"].mean()
        avg_liquidations = count_data["liquidation_rate"].mean()
        avg_utilization = count_data["pool_utilization"].mean()
        
        # Calculate efficiency metrics
        rebalancing_per_agent = avg_rebalancing / agent_count
        liquidation_prevention_ratio = avg_rebalancing / max(avg_liquidations, 0.1)
        
        efficiency_analysis[f"{agent_count}_agents"] = {
            "agent_count": agent_count,
            "avg_pool_utilization": avg_utilization,
            "avg_rebalancing_events": avg_rebalancing,
            "rebalancing_per_agent": rebalancing_per_agent,
            "avg_liquidation_rate": avg_liquidations,
            "liquidation_prevention_ratio": liquidation_prevention_ratio,
            "efficiency_rating": "High" if liquidation_prevention_ratio > 10 else "Medium" if liquidation_prevention_ratio > 3 else "Low"
        }
    
    # Find capacity limits
    high_efficiency_scenarios = [v for v in efficiency_analysis.values() if v["efficiency_rating"] == "High"]
    max_efficient_utilization = max([s["avg_pool_utilization"] for s in high_efficiency_scenarios]) if high_efficiency_scenarios else 0.5
    
    return {
        "efficiency_by_agent_count": efficiency_analysis,
        "max_efficient_utilization": max_efficient_utilization,
        "recommended_operational_limit": max_efficient_utilization * 0.9  # 10% safety buffer
    }


def analyze_risk_profile_pool_impact(df: pd.DataFrame) -> Dict:
    """Analyze how different risk profiles impact pool capacity"""
    
    profile_impact = {}
    
    for profile in df["profile"].unique():
        profile_data = df[df["profile"] == profile]
        
        profile_impact[profile] = {
            "avg_rebalancing_events": profile_data["rebalancing_events"].mean(),
            "avg_liquidation_rate": profile_data["liquidation_rate"].mean(),
            "avg_stress_events": profile_data["pool_stress_events"].mean(),
            "stress_level_distribution": profile_data["stress_level"].value_counts().to_dict(),
            "pool_impact_rating": calculate_pool_impact_rating(profile_data)
        }
    
    return profile_impact


def calculate_pool_impact_rating(profile_data: pd.DataFrame) -> str:
    """Calculate pool impact rating for a risk profile"""
    
    avg_stress = profile_data["pool_stress_events"].mean()
    avg_rebalancing = profile_data["rebalancing_events"].mean()
    
    if avg_stress > 2 or avg_rebalancing > 200:
        return "High Impact"
    elif avg_stress > 1 or avg_rebalancing > 100:
        return "Medium Impact"
    else:
        return "Low Impact"


def save_borrow_cap_results(analysis: Dict, results_matrix: List):
    """Save borrow cap analysis results to JSON"""
    
    output_dir = Path("tidal_protocol_sim/results/moet_yt_borrow_cap_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare comprehensive results
    final_results = {
        "analysis_metadata": {
            "analysis_type": "MOET_YT_Borrow_Cap_Analysis",
            "timestamp": datetime.now().isoformat(),
            "baseline_pool_size": 250_000,
            "agent_counts_tested": [20, 50, 100, 150, 200],
            "hf_scenarios_tested": 4,
            "total_scenarios": len(results_matrix)
        },
        "key_findings": analysis,
        "detailed_scenario_results": results_matrix
    }
    
    # Save JSON results
    results_path = output_dir / "borrow_cap_analysis_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"📁 Borrow cap analysis results saved to: {results_path}")
    
    # Save summary CSV
    csv_data = []
    for result in results_matrix:
        params = result["scenario_params"]
        ht = result["high_tide_metrics"]
        stress = result["pool_stress_analysis"]
        implications = result["borrow_cap_implications"]
        
        csv_data.append({
            "agent_count": params["agent_count"],
            "profile": params["profile"],
            "initial_hf": params["initial_hf"],
            "target_hf": params["target_hf"],
            "pool_utilization": params["pool_utilization"],
            "survival_rate": ht["survival_rate"],
            "rebalancing_events": ht["total_rebalancing_events"],
            "liquidation_rate": ht["liquidation_rate_percentage"],
            "stress_level": stress["stress_level"],
            "borrow_cap_needed": implications["borrow_cap_needed"],
            "recommended_cap_pct": implications.get("recommended_cap_percentage", 0)
        })
    
    df = pd.DataFrame(csv_data)
    csv_path = output_dir / "borrow_cap_summary.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"📊 Summary data saved to: {csv_path}")
    
    return results_path


def print_borrow_cap_summary(analysis: Dict):
    """Print summary of borrow cap analysis"""
    
    capacity_thresholds = analysis.get("capacity_thresholds", {})
    borrow_cap_recs = analysis.get("borrow_cap_recommendations", {})
    stress_analysis = analysis.get("stress_level_analysis", {})
    
    print("\n" + "=" * 80)
    print("MOET:YT POOL BORROW CAP ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Capacity thresholds
    critical_util = capacity_thresholds.get("critical_utilization_threshold")
    safe_util = capacity_thresholds.get("safe_utilization_threshold")
    recommended_cap = capacity_thresholds.get("recommended_utilization_cap")
    
    print(f"\n🏊 POOL CAPACITY THRESHOLDS:")
    if safe_util:
        print(f"   Safe utilization: ≤ {safe_util:.1%}")
    if critical_util:
        print(f"   Critical utilization: ≥ {critical_util:.1%}")
    if recommended_cap:
        print(f"   Recommended cap: {recommended_cap:.1%}")
    
    # Borrow cap recommendations
    print(f"\n📊 BORROW CAP RECOMMENDATIONS:")
    if borrow_cap_recs.get("no_cap_needed"):
        print(f"   ✅ No borrow cap needed")
        print(f"   📈 System stable up to {borrow_cap_recs['max_tested_utilization']:.1f}% utilization")
        print(f"   🔍 Monitor at {borrow_cap_recs['monitoring_threshold']:.1f}% utilization")
    else:
        conservative = borrow_cap_recs.get("conservative_cap", {})
        aggressive = borrow_cap_recs.get("aggressive_cap", {})
        
        if conservative:
            print(f"   🛡️  Conservative: {conservative['utilization_percentage']:.1f}% of pool liquidity")
            print(f"      {conservative['reasoning']}")
        
        if aggressive:
            print(f"   ⚡ Aggressive: {aggressive['utilization_percentage']:.1f}% of pool liquidity")
            print(f"      {aggressive['reasoning']}")
    
    # Stress level distribution
    print(f"\n🌡️  STRESS LEVEL ANALYSIS:")
    for utilization_range, data in stress_analysis.items():
        dominant_stress = data.get("dominant_stress_level", "Unknown")
        scenario_count = data.get("scenario_count", 0)
        avg_survival = data.get("avg_survival_rate", 0)
        
        print(f"   {utilization_range} utilization: {dominant_stress} stress ({scenario_count} scenarios, "
              f"{avg_survival:.1%} survival)")
    
    print("\n" + "=" * 80)


def main():
    """Main execution function"""
    try:
        results = run_borrow_cap_analysis()
        print("\n✅ MOET:YT borrow cap analysis completed successfully!")
        return results
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()