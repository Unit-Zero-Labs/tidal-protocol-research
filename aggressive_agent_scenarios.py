#!/usr/bin/env python3
"""
Aggressive Agent Scenarios Analysis

Tests very aggressive agent configurations with tight ranges between initial 
and target health factors to stress-test the rebalancing mechanism.

Scenarios:
- Initial HF: 1.1-1.2 (high LTV loans)  
- Target HF: 1.05 (aggressive rebalancing trigger)
- Tests the limits of the automated rebalancing system
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


def run_aggressive_scenarios_analysis():
    """Run aggressive agent scenarios to stress-test rebalancing limits"""
    
    print("=" * 80)
    print("AGGRESSIVE AGENT SCENARIOS ANALYSIS")
    print("=" * 80)
    print("Testing high-LTV loans with tight rebalancing triggers")
    print("Initial HF: 1.1-1.2, Target HF: 1.05")
    print()
    
    # Define aggressive scenarios
    aggressive_scenarios = [
        {"initial_hf": 1.20, "target_hf": 1.05, "buffer": 0.15},
        {"initial_hf": 1.15, "target_hf": 1.05, "buffer": 0.10},
        {"initial_hf": 1.12, "target_hf": 1.05, "buffer": 0.07},
        {"initial_hf": 1.10, "target_hf": 1.05, "buffer": 0.05},
        # Ultra-aggressive scenarios
        {"initial_hf": 1.08, "target_hf": 1.05, "buffer": 0.03},
        {"initial_hf": 1.07, "target_hf": 1.05, "buffer": 0.02}
    ]
    
    # Test with different pool stress levels
    pool_stress_scenarios = [
        {"label": "Standard Pool", "moet_yt_pool_size": 250_000, "stress_level": "low"},
        {"label": "Constrained Pool", "moet_yt_pool_size": 100_000, "stress_level": "medium"},
        {"label": "Minimal Pool", "moet_yt_pool_size": 50_000, "stress_level": "high"}
    ]
    
    all_results = []
    
    for pool_scenario in pool_stress_scenarios:
        print(f"\n🏊 Testing with {pool_scenario['label']} (MOET:YT = ${pool_scenario['moet_yt_pool_size']:,})")
        
        pool_results = []
        
        for scenario in aggressive_scenarios:
            result = run_aggressive_scenario(scenario, pool_scenario)
            pool_results.append(result)
            
            # Show immediate feedback
            ht_survival = result["high_tide_metrics"]["survival_rate"]
            ht_liquidations = result["high_tide_metrics"]["liquidation_frequency"]
            rebalancing_events = result["high_tide_metrics"]["total_rebalancing_events"]
            
            print(f"   HF {scenario['initial_hf']:.2f}→{scenario['target_hf']:.2f}: "
                  f"{ht_survival:.1%} survival, {ht_liquidations:.1%} liquidations, "
                  f"{rebalancing_events:.0f} rebalances")
        
        all_results.append({
            "pool_scenario": pool_scenario,
            "aggressive_results": pool_results
        })
    
    # Generate comprehensive analysis
    analysis = analyze_aggressive_scenarios(all_results)
    
    # Save results
    save_aggressive_scenarios_results(analysis, all_results)
    
    # Print summary
    print_aggressive_scenarios_summary(analysis)
    
    return analysis


def run_aggressive_scenario(scenario: Dict, pool_scenario: Dict) -> Dict:
    """Run a single aggressive scenario test"""
    
    initial_hf = scenario["initial_hf"]
    target_hf = scenario["target_hf"]
    moet_yt_pool_size = pool_scenario["moet_yt_pool_size"]
    
    monte_carlo_runs = 10  # Focused testing
    
    ht_results = []
    aave_results = []
    
    for run_num in range(monte_carlo_runs):
        # High Tide with aggressive agents
        ht_config = HighTideConfig()
        ht_config.num_high_tide_agents = 0  # Custom agents
        ht_config.btc_decline_duration = 60
        ht_config.moet_btc_pool_size = 250_000  # Standard liquidation pool
        ht_config.moet_yield_pool_size = moet_yt_pool_size  # Variable rebalancing pool
        
        # Create aggressive High Tide agents
        aggressive_ht_agents = create_aggressive_agents(
            initial_hf, target_hf, num_agents=20, agent_type="high_tide", run_num=run_num
        )
        
        ht_engine = HighTideSimulationEngine(ht_config)
        ht_engine.high_tide_agents = aggressive_ht_agents
        for agent in aggressive_ht_agents:
            ht_engine.agents[agent.agent_id] = agent
        
        ht_result = ht_engine.run_high_tide_simulation()
        ht_results.append(ht_result)
        
        # Matching Aave scenario
        aave_config = AaveConfig()
        aave_config.num_aave_agents = 0  # Custom agents
        aave_config.btc_decline_duration = 60
        aave_config.moet_btc_pool_size = 250_000
        aave_config.moet_yield_pool_size = moet_yt_pool_size
        
        # Create matching aggressive Aave agents
        aggressive_aave_agents = create_aggressive_agents(
            initial_hf, target_hf, num_agents=20, agent_type="aave", run_num=run_num
        )
        
        aave_engine = AaveSimulationEngine(aave_config)
        aave_engine.aave_agents = aggressive_aave_agents
        for agent in aggressive_aave_agents:
            aave_engine.agents[agent.agent_id] = agent
        
        aave_result = aave_engine.run_aave_simulation()
        aave_results.append(aave_result)
    
    # Aggregate results for this scenario
    return aggregate_aggressive_scenario_results(ht_results, aave_results, scenario, pool_scenario)


def create_aggressive_agents(initial_hf: float, target_hf: float, num_agents: int, 
                           agent_type: str, run_num: int) -> List:
    """Create aggressive agents with specified health factor parameters"""
    
    agents = []
    
    for i in range(num_agents):
        agent_id = f"aggressive_{agent_type}_{initial_hf}_{target_hf}_r{run_num}_a{i}"
        
        if agent_type == "high_tide":
            agent = HighTideAgent(agent_id, initial_hf, target_hf)
        else:  # aave
            agent = AaveAgent(agent_id, initial_hf, target_hf)
        
        # Override risk profile to aggressive
        agent.risk_profile = "aggressive"
        agent.color = "#DC143C"  # Crimson
        
        agents.append(agent)
    
    return agents


def aggregate_aggressive_scenario_results(ht_results: List, aave_results: List, 
                                        scenario: Dict, pool_scenario: Dict) -> Dict:
    """Aggregate results for a specific aggressive scenario"""
    
    # High Tide metrics
    ht_survival_rates = []
    ht_liquidations = []
    ht_rebalancing_events = []
    ht_costs = []
    
    for run in ht_results:
        survival_stats = run.get("survival_statistics", {})
        ht_survival_rates.append(survival_stats.get("survival_rate", 0.0))
        
        # Count emergency liquidations
        agent_outcomes = run.get("agent_outcomes", [])
        total_liquidations = sum(outcome.get("emergency_liquidations", 0) for outcome in agent_outcomes)
        ht_liquidations.append(total_liquidations)
        
        # Count rebalancing events
        rebalancing_activity = run.get("yield_token_activity", {})
        ht_rebalancing_events.append(rebalancing_activity.get("rebalancing_events", 0))
        
        # Calculate average cost
        cost_analysis = run.get("cost_analysis", {})
        ht_costs.append(cost_analysis.get("average_cost_per_agent", 0.0))
    
    # Aave metrics
    aave_survival_rates = []
    aave_liquidations = []
    aave_costs = []
    
    for run in aave_results:
        survival_stats = run.get("survival_statistics", {})
        aave_survival_rates.append(survival_stats.get("survival_rate", 0.0))
        
        # Count traditional liquidations
        liquidation_activity = run.get("liquidation_activity", {})
        aave_liquidations.append(liquidation_activity.get("total_liquidation_events", 0))
        
        # Calculate average cost
        cost_analysis = run.get("cost_analysis", {})
        aave_costs.append(cost_analysis.get("average_cost_per_agent", 0.0))
    
    # Calculate scenario metrics
    return {
        "scenario_params": scenario,
        "pool_params": pool_scenario,
        "high_tide_metrics": {
            "survival_rate": np.mean(ht_survival_rates),
            "survival_rate_std": np.std(ht_survival_rates),
            "liquidation_frequency": np.mean(ht_liquidations) / 20,  # Per agent
            "total_rebalancing_events": np.mean(ht_rebalancing_events),
            "average_cost": np.mean(ht_costs),
            "liquidation_rate_percentage": (np.mean(ht_liquidations) / 20) * 100
        },
        "aave_metrics": {
            "survival_rate": np.mean(aave_survival_rates),
            "survival_rate_std": np.std(aave_survival_rates),
            "liquidation_frequency": np.mean(aave_liquidations) / 20,  # Per agent
            "average_cost": np.mean(aave_costs),
            "liquidation_rate_percentage": (np.mean(aave_liquidations) / 20) * 100
        },
        "comparison": {
            "survival_improvement": ((np.mean(ht_survival_rates) - np.mean(aave_survival_rates)) / np.mean(aave_survival_rates) * 100) if np.mean(aave_survival_rates) > 0 else 0,
            "cost_reduction": ((np.mean(aave_costs) - np.mean(ht_costs)) / np.mean(aave_costs) * 100) if np.mean(aave_costs) > 0 else 0,
            "liquidation_reduction": ((np.mean(aave_liquidations) - np.mean(ht_liquidations)) / np.mean(aave_liquidations) * 100) if np.mean(aave_liquidations) > 0 else 0
        },
        "stress_test_results": {
            "rebalancing_effectiveness": "High" if np.mean(ht_liquidations) < np.mean(aave_liquidations) * 0.5 else "Medium" if np.mean(ht_liquidations) < np.mean(aave_liquidations) * 0.8 else "Low",
            "pool_stress_impact": pool_scenario["stress_level"],
            "system_breaking_point": np.mean(ht_liquidations) > 5  # More than 25% liquidation rate indicates system stress
        }
    }


def analyze_aggressive_scenarios(all_results: List[Dict]) -> Dict:
    """Analyze aggressive scenarios to find system limits"""
    
    analysis = {
        "breaking_point_analysis": find_breaking_points(all_results),
        "pool_stress_impact": analyze_pool_stress_impact(all_results),
        "rebalancing_limits": analyze_rebalancing_limits(all_results),
        "risk_recommendations": generate_risk_recommendations(all_results)
    }
    
    return analysis


def find_breaking_points(all_results: List[Dict]) -> Dict:
    """Find the breaking points where rebalancing starts failing"""
    
    breaking_points = {}
    
    for pool_results in all_results:
        pool_label = pool_results["pool_scenario"]["label"]
        stress_level = pool_results["pool_scenario"]["stress_level"]
        
        # Find scenarios where liquidation rate > 20% (system stress)
        stressed_scenarios = []
        safe_scenarios = []
        
        for result in pool_results["aggressive_results"]:
            ht_liquidation_rate = result["high_tide_metrics"]["liquidation_rate_percentage"]
            scenario_params = result["scenario_params"]
            
            if ht_liquidation_rate > 20:  # More than 20% liquidation rate
                stressed_scenarios.append({
                    "hf_buffer": scenario_params["buffer"],
                    "liquidation_rate": ht_liquidation_rate,
                    "survival_rate": result["high_tide_metrics"]["survival_rate"]
                })
            else:
                safe_scenarios.append({
                    "hf_buffer": scenario_params["buffer"],
                    "liquidation_rate": ht_liquidation_rate,
                    "survival_rate": result["high_tide_metrics"]["survival_rate"]
                })
        
        # Find the breaking point (smallest buffer that causes stress)
        breaking_point = None
        if stressed_scenarios:
            breaking_point = max(stressed_scenarios, key=lambda x: x["hf_buffer"])
        
        # Find the safest aggressive configuration
        safest_aggressive = None
        if safe_scenarios:
            safest_aggressive = min(safe_scenarios, key=lambda x: x["hf_buffer"])
        
        breaking_points[pool_label] = {
            "pool_stress_level": stress_level,
            "breaking_point": breaking_point,
            "safest_aggressive": safest_aggressive,
            "total_scenarios_tested": len(pool_results["aggressive_results"]),
            "stressed_scenarios_count": len(stressed_scenarios),
            "safe_scenarios_count": len(safe_scenarios)
        }
    
    return breaking_points


def analyze_pool_stress_impact(all_results: List[Dict]) -> Dict:
    """Analyze how pool size affects aggressive scenario performance"""
    
    pool_impact = {}
    
    # Compare same HF scenarios across different pool sizes
    for hf_buffer in [0.15, 0.10, 0.07, 0.05]:
        buffer_analysis = {}
        
        for pool_results in all_results:
            pool_label = pool_results["pool_scenario"]["label"]
            pool_size = pool_results["pool_scenario"]["moet_yt_pool_size"]
            
            # Find scenario with this buffer
            matching_scenarios = [
                r for r in pool_results["aggressive_results"] 
                if abs(r["scenario_params"]["buffer"] - hf_buffer) < 0.001
            ]
            
            if matching_scenarios:
                scenario = matching_scenarios[0]
                buffer_analysis[pool_label] = {
                    "pool_size": pool_size,
                    "liquidation_rate": scenario["high_tide_metrics"]["liquidation_rate_percentage"],
                    "survival_rate": scenario["high_tide_metrics"]["survival_rate"],
                    "rebalancing_events": scenario["high_tide_metrics"]["total_rebalancing_events"]
                }
        
        pool_impact[f"hf_buffer_{hf_buffer:.2f}"] = buffer_analysis
    
    return pool_impact


def analyze_rebalancing_limits(all_results: List[Dict]) -> Dict:
    """Analyze the limits of the rebalancing mechanism"""
    
    # Find scenarios with highest rebalancing activity
    high_activity_scenarios = []
    failed_scenarios = []
    
    for pool_results in all_results:
        for result in pool_results["aggressive_results"]:
            ht_metrics = result["high_tide_metrics"]
            scenario_params = result["scenario_params"]
            
            rebalancing_events = ht_metrics["total_rebalancing_events"]
            liquidation_rate = ht_metrics["liquidation_rate_percentage"]
            
            if rebalancing_events > 50:  # High rebalancing activity
                high_activity_scenarios.append({
                    "scenario": scenario_params,
                    "pool": result["pool_params"]["label"],
                    "rebalancing_events": rebalancing_events,
                    "liquidation_rate": liquidation_rate,
                    "effectiveness": "success" if liquidation_rate < 10 else "partial" if liquidation_rate < 25 else "failed"
                })
            
            if liquidation_rate > 30:  # Failed scenarios
                failed_scenarios.append({
                    "scenario": scenario_params,
                    "pool": result["pool_params"]["label"],
                    "liquidation_rate": liquidation_rate,
                    "rebalancing_events": rebalancing_events
                })
    
    return {
        "high_activity_scenarios": high_activity_scenarios,
        "failed_scenarios": failed_scenarios,
        "rebalancing_effectiveness_threshold": 50,  # Events needed for effectiveness
        "system_failure_threshold": 30  # Liquidation rate indicating system failure
    }


def generate_risk_recommendations(all_results: List[Dict]) -> Dict:
    """Generate risk management recommendations based on aggressive scenario results"""
    
    recommendations = {}
    
    # Analyze by pool stress level
    for pool_results in all_results:
        pool_label = pool_results["pool_scenario"]["label"]
        stress_level = pool_results["pool_scenario"]["stress_level"]
        
        # Find the most aggressive safe configuration for this pool
        safe_configs = []
        for result in pool_results["aggressive_results"]:
            if result["high_tide_metrics"]["liquidation_rate_percentage"] < 15:  # < 15% liquidation
                safe_configs.append({
                    "hf_buffer": result["scenario_params"]["buffer"],
                    "initial_hf": result["scenario_params"]["initial_hf"],
                    "target_hf": result["scenario_params"]["target_hf"],
                    "liquidation_rate": result["high_tide_metrics"]["liquidation_rate_percentage"],
                    "survival_rate": result["high_tide_metrics"]["survival_rate"]
                })
        
        # Find most aggressive safe config (smallest buffer)
        most_aggressive_safe = min(safe_configs, key=lambda x: x["hf_buffer"]) if safe_configs else None
        
        recommendations[pool_label] = {
            "pool_stress_level": stress_level,
            "pool_size": pool_results["pool_scenario"]["moet_yt_pool_size"],
            "most_aggressive_safe_config": most_aggressive_safe,
            "min_recommended_buffer": most_aggressive_safe["hf_buffer"] if most_aggressive_safe else 0.15,
            "max_recommended_initial_hf": most_aggressive_safe["initial_hf"] if most_aggressive_safe else 1.20,
            "total_scenarios_tested": len(pool_results["aggressive_results"]),
            "safe_scenarios_found": len(safe_configs)
        }
    
    return recommendations


def save_aggressive_scenarios_results(analysis: Dict, all_results: List):
    """Save aggressive scenarios analysis results to JSON"""
    
    output_dir = Path("tidal_protocol_sim/results/aggressive_agent_scenarios")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare comprehensive results
    final_results = {
        "analysis_metadata": {
            "analysis_type": "Aggressive_Agent_Scenarios",
            "timestamp": datetime.now().isoformat(),
            "scenarios_tested": [
                {"initial_hf": 1.20, "target_hf": 1.05, "buffer": 0.15},
                {"initial_hf": 1.15, "target_hf": 1.05, "buffer": 0.10},
                {"initial_hf": 1.12, "target_hf": 1.05, "buffer": 0.07},
                {"initial_hf": 1.10, "target_hf": 1.05, "buffer": 0.05},
                {"initial_hf": 1.08, "target_hf": 1.05, "buffer": 0.03},
                {"initial_hf": 1.07, "target_hf": 1.05, "buffer": 0.02}
            ],
            "pool_stress_levels": ["low", "medium", "high"],
            "monte_carlo_runs_per_scenario": 10
        },
        "analysis_findings": analysis,
        "detailed_results": all_results
    }
    
    # Save JSON results
    results_path = output_dir / "aggressive_scenarios_analysis.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"📁 Aggressive scenarios results saved to: {results_path}")
    
    # Save summary CSV
    csv_data = []
    for pool_results in all_results:
        for result in pool_results["aggressive_results"]:
            scenario = result["scenario_params"]
            pool = result["pool_params"]
            ht = result["high_tide_metrics"]
            aave = result["aave_metrics"]
            comp = result["comparison"]
            
            csv_data.append({
                "pool_label": pool["label"],
                "pool_size": pool["moet_yt_pool_size"],
                "initial_hf": scenario["initial_hf"],
                "target_hf": scenario["target_hf"],
                "hf_buffer": scenario["buffer"],
                "ht_survival_rate": ht["survival_rate"],
                "ht_liquidation_rate": ht["liquidation_rate_percentage"],
                "ht_rebalancing_events": ht["total_rebalancing_events"],
                "aave_survival_rate": aave["survival_rate"],
                "aave_liquidation_rate": aave["liquidation_rate_percentage"],
                "survival_improvement": comp["survival_improvement"],
                "liquidation_reduction": comp["liquidation_reduction"]
            })
    
    df = pd.DataFrame(csv_data)
    csv_path = output_dir / "aggressive_scenarios_summary.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"📊 Summary data saved to: {csv_path}")
    
    return results_path


def print_aggressive_scenarios_summary(analysis: Dict):
    """Print summary of aggressive scenarios analysis"""
    
    breaking_points = analysis.get("breaking_point_analysis", {})
    recommendations = analysis.get("risk_recommendations", {})
    rebalancing_limits = analysis.get("rebalancing_limits", {})
    
    print("\n" + "=" * 80)
    print("AGGRESSIVE SCENARIOS ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Breaking points by pool
    print(f"\n🚨 SYSTEM BREAKING POINTS:")
    for pool_label, bp_data in breaking_points.items():
        breaking_point = bp_data.get("breaking_point")
        safest_aggressive = bp_data.get("safest_aggressive")
        
        if breaking_point:
            print(f"   {pool_label}: HF buffer {breaking_point['hf_buffer']:.2f} → {breaking_point['liquidation_rate']:.1f}% liquidations")
        
        if safest_aggressive:
            print(f"   {pool_label} (Safe): HF buffer {safest_aggressive['hf_buffer']:.2f} → {safest_aggressive['liquidation_rate']:.1f}% liquidations")
    
    # Risk recommendations
    print(f"\n✅ RISK MANAGEMENT RECOMMENDATIONS:")
    for pool_label, rec in recommendations.items():
        safe_config = rec.get("most_aggressive_safe_config")
        if safe_config:
            print(f"   {pool_label}:")
            print(f"      Max Initial HF: {safe_config['initial_hf']:.2f}")
            print(f"      Min Target HF: {safe_config['target_hf']:.2f}")
            print(f"      Min HF Buffer: {safe_config['hf_buffer']:.2f}")
            print(f"      Expected liquidations: {safe_config['liquidation_rate']:.1f}%")
    
    # Rebalancing limits
    limits = rebalancing_limits
    high_activity = limits.get("high_activity_scenarios", [])
    failed_scenarios = limits.get("failed_scenarios", [])
    
    print(f"\n⚡ REBALANCING SYSTEM LIMITS:")
    print(f"   High activity scenarios: {len(high_activity)}")
    print(f"   Failed scenarios (>30% liquidations): {len(failed_scenarios)}")
    print(f"   Rebalancing effectiveness threshold: {limits.get('rebalancing_effectiveness_threshold', 0)} events")
    
    if failed_scenarios:
        print(f"\n⚠️  FAILED SCENARIOS:")
        for fail in failed_scenarios[:3]:  # Show top 3 failures
            scenario = fail["scenario"]
            print(f"      HF {scenario['initial_hf']:.2f}→{scenario['target_hf']:.2f} ({fail['pool']}): "
                  f"{fail['liquidation_rate']:.1f}% liquidations")
    
    print("\n" + "=" * 80)


def main():
    """Main execution function"""
    try:
        results = run_aggressive_scenarios_analysis()
        print("\n✅ Aggressive scenarios analysis completed successfully!")
        return results
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()