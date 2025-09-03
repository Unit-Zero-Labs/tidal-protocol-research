#!/usr/bin/env python3
"""
Target Health Factor Analysis

Tests different target health factors (1.01, 1.05, 1.1, 1.15) to determine
how low we can make the rebalancing trigger before agents begin getting liquidated frequently.

This analysis answers the key question: What is the optimal Target Health Factor 
that maintains the rebalancing mechanism without causing frequent liquidations?
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
from tidal_protocol_sim.agents.high_tide_agent import create_high_tide_agents
from tidal_protocol_sim.agents.aave_agent import create_aave_agents


def run_target_hf_analysis():
    """Run comprehensive Target Health Factor analysis"""
    
    print("=" * 80)
    print("TARGET HEALTH FACTOR ANALYSIS")
    print("=" * 80)
    print("Testing target health factors: 1.01, 1.05, 1.1, 1.15")
    print("Question: How low can Target HF go before frequent liquidations?")
    print()
    
    # Target health factors to test
    target_hfs = [1.01, 1.05, 1.1, 1.15]
    monte_carlo_runs = 20  # Sufficient for trend analysis
    
    results_matrix = []
    
    for target_hf in target_hfs:
        print(f"🎯 Testing Target Health Factor: {target_hf}")
        
        # Test multiple initial health factors with this target
        initial_hfs = [target_hf + 0.1, target_hf + 0.2, target_hf + 0.3]  # Close ranges
        
        for initial_hf in initial_hfs:
            result = run_target_hf_scenario(initial_hf, target_hf, monte_carlo_runs)
            result["scenario_params"] = {
                "initial_hf": initial_hf,
                "target_hf": target_hf,
                "hf_buffer": initial_hf - target_hf
            }
            results_matrix.append(result)
    
    # Generate comprehensive analysis
    analysis_results = analyze_target_hf_results(results_matrix)
    
    # Save results with JSON output
    save_target_hf_results(analysis_results, results_matrix)
    
    # Print summary
    print_target_hf_summary(analysis_results)
    
    return analysis_results


def run_target_hf_scenario(initial_hf: float, target_hf: float, monte_carlo_runs: int) -> Dict:
    """Run scenario testing specific initial and target health factors"""
    
    buffer = initial_hf - target_hf
    print(f"   Testing Initial HF: {initial_hf:.2f}, Target HF: {target_hf:.2f} (Buffer: {buffer:.2f})")
    
    # Store results from all runs
    ht_runs = []
    aave_runs = []
    
    for run_num in range(monte_carlo_runs):
        # High Tide simulation with custom agents
        ht_config = HighTideConfig()
        ht_config.num_high_tide_agents = 0  # We'll create custom agents
        ht_config.btc_decline_duration = 60
        ht_config.moet_btc_pool_size = 250_000  # Standard pool size
        ht_config.moet_yield_pool_size = 250_000  # Standard YT pool size
        
        # Create custom High Tide agents with specific HF parameters
        custom_ht_agents = create_custom_agents_for_hf_test(
            initial_hf, target_hf, num_agents=15, agent_type="high_tide"
        )
        
        ht_engine = HighTideSimulationEngine(ht_config)
        ht_engine.high_tide_agents = custom_ht_agents
        
        # Add agents to engine's agent dict
        for agent in custom_ht_agents:
            ht_engine.agents[agent.agent_id] = agent
        
        ht_results = ht_engine.run_high_tide_simulation()
        ht_runs.append(ht_results)
        
        # Aave simulation with matching agents
        aave_config = AaveConfig()
        aave_config.num_aave_agents = 0  # We'll create custom agents
        aave_config.btc_decline_duration = 60
        aave_config.moet_btc_pool_size = 250_000
        aave_config.moet_yield_pool_size = 250_000
        
        # Create matching Aave agents
        custom_aave_agents = create_custom_agents_for_hf_test(
            initial_hf, target_hf, num_agents=15, agent_type="aave"
        )
        
        aave_engine = AaveSimulationEngine(aave_config)
        aave_engine.aave_agents = custom_aave_agents
        
        # Add agents to engine's agent dict
        for agent in custom_aave_agents:
            aave_engine.agents[agent.agent_id] = agent
        
        aave_results = aave_engine.run_aave_simulation()
        aave_runs.append(aave_results)
    
    # Aggregate results
    scenario_analysis = aggregate_hf_scenario_results(ht_runs, aave_runs, initial_hf, target_hf)
    
    print(f"      High Tide: {scenario_analysis['high_tide_summary']['mean_survival_rate']:.1%} survival, "
          f"{scenario_analysis['high_tide_summary']['mean_liquidations']:.1f} liquidations")
    print(f"      Aave:      {scenario_analysis['aave_summary']['mean_survival_rate']:.1%} survival, "
          f"{scenario_analysis['aave_summary']['mean_liquidations']:.1f} liquidations")
    
    return scenario_analysis


def create_custom_agents_for_hf_test(initial_hf: float, target_hf: float, num_agents: int, agent_type: str) -> List:
    """Create custom agents with specific health factor parameters for testing"""
    from tidal_protocol_sim.agents.high_tide_agent import HighTideAgent
    from tidal_protocol_sim.agents.aave_agent import AaveAgent
    
    agents = []
    
    for i in range(num_agents):
        if agent_type == "high_tide":
            agent = HighTideAgent(
                f"hf_test_ht_{initial_hf}_{target_hf}_{i}",
                initial_hf,
                target_hf
            )
        else:  # aave
            agent = AaveAgent(
                f"hf_test_aave_{initial_hf}_{target_hf}_{i}",
                initial_hf,
                target_hf
            )
        
        agents.append(agent)
    
    return agents


def aggregate_hf_scenario_results(ht_runs: List, aave_runs: List, initial_hf: float, target_hf: float) -> Dict:
    """Aggregate results for a specific health factor scenario"""
    
    # Aggregate High Tide metrics
    ht_survival_rates = []
    ht_liquidations = []
    ht_rebalancing_events = []
    
    for run in ht_runs:
        survival_stats = run.get("survival_statistics", {})
        ht_survival_rates.append(survival_stats.get("survival_rate", 0.0))
        
        # Count liquidations (emergency liquidations in High Tide)
        agent_outcomes = run.get("agent_outcomes", [])
        total_liquidations = sum(outcome.get("emergency_liquidations", 0) for outcome in agent_outcomes)
        ht_liquidations.append(total_liquidations)
        
        # Count rebalancing events
        rebalancing_activity = run.get("yield_token_activity", {})
        ht_rebalancing_events.append(rebalancing_activity.get("rebalancing_events", 0))
    
    # Aggregate Aave metrics
    aave_survival_rates = []
    aave_liquidations = []
    
    for run in aave_runs:
        survival_stats = run.get("survival_statistics", {})
        aave_survival_rates.append(survival_stats.get("survival_rate", 0.0))
        
        # Count liquidations (traditional liquidations in Aave)
        liquidation_activity = run.get("liquidation_activity", {})
        aave_liquidations.append(liquidation_activity.get("total_liquidation_events", 0))
    
    # Calculate summary statistics
    return {
        "initial_hf": initial_hf,
        "target_hf": target_hf,
        "hf_buffer": initial_hf - target_hf,
        "high_tide_summary": {
            "mean_survival_rate": np.mean(ht_survival_rates),
            "survival_rate_std": np.std(ht_survival_rates),
            "mean_liquidations": np.mean(ht_liquidations),
            "mean_rebalancing_events": np.mean(ht_rebalancing_events),
            "liquidation_frequency": np.mean(ht_liquidations) / 15  # Per agent
        },
        "aave_summary": {
            "mean_survival_rate": np.mean(aave_survival_rates),
            "survival_rate_std": np.std(aave_survival_rates),
            "mean_liquidations": np.mean(aave_liquidations),
            "liquidation_frequency": np.mean(aave_liquidations) / 15  # Per agent
        },
        "comparison": {
            "survival_improvement": ((np.mean(ht_survival_rates) - np.mean(aave_survival_rates)) / np.mean(aave_survival_rates) * 100) if np.mean(aave_survival_rates) > 0 else 0,
            "liquidation_reduction": ((np.mean(aave_liquidations) - np.mean(ht_liquidations)) / np.mean(aave_liquidations) * 100) if np.mean(aave_liquidations) > 0 else 0
        },
        "raw_data": {
            "ht_survival_rates": ht_survival_rates,
            "aave_survival_rates": aave_survival_rates,
            "ht_liquidations": ht_liquidations,
            "aave_liquidations": aave_liquidations,
            "ht_rebalancing_events": ht_rebalancing_events
        }
    }


def analyze_target_hf_results(results_matrix: List[Dict]) -> Dict:
    """Analyze all target health factor results to find optimal parameters"""
    
    # Create DataFrame for analysis
    df_data = []
    for result in results_matrix:
        params = result["scenario_params"]
        ht_summary = result["high_tide_summary"]
        aave_summary = result["aave_summary"]
        comparison = result["comparison"]
        
        df_data.append({
            "initial_hf": params["initial_hf"],
            "target_hf": params["target_hf"],
            "hf_buffer": params["hf_buffer"],
            "ht_survival_rate": ht_summary["mean_survival_rate"],
            "ht_liquidation_frequency": ht_summary["liquidation_frequency"],
            "ht_rebalancing_events": ht_summary["mean_rebalancing_events"],
            "aave_survival_rate": aave_summary["mean_survival_rate"],
            "aave_liquidation_frequency": aave_summary["liquidation_frequency"],
            "survival_improvement": comparison["survival_improvement"],
            "liquidation_reduction": comparison["liquidation_reduction"]
        })
    
    df = pd.DataFrame(df_data)
    
    # Find optimal target health factors
    analysis = {
        "optimal_recommendations": find_optimal_target_hf(df),
        "liquidation_threshold_analysis": analyze_liquidation_thresholds(df),
        "rebalancing_efficiency": analyze_rebalancing_efficiency(df),
        "risk_vs_efficiency": analyze_risk_efficiency_tradeoff(df),
        "statistical_summary": generate_statistical_summary(df),
        "raw_results_matrix": results_matrix
    }
    
    return analysis


def find_optimal_target_hf(df: pd.DataFrame) -> Dict:
    """Find optimal target health factors based on liquidation frequency thresholds"""
    
    # Define thresholds for analysis
    low_liquidation_threshold = 0.05  # < 5% liquidation rate is "low"
    moderate_liquidation_threshold = 0.10  # < 10% liquidation rate is "moderate"
    
    recommendations = {}
    
    for target_hf in [1.01, 1.05, 1.1, 1.15]:
        target_data = df[df["target_hf"] == target_hf]
        
        if len(target_data) > 0:
            avg_liquidation_freq = target_data["ht_liquidation_frequency"].mean()
            avg_survival_rate = target_data["ht_survival_rate"].mean()
            avg_rebalancing = target_data["ht_rebalancing_events"].mean()
            
            # Categorize risk level
            if avg_liquidation_freq < low_liquidation_threshold:
                risk_level = "Low Risk"
            elif avg_liquidation_freq < moderate_liquidation_threshold:
                risk_level = "Moderate Risk"
            else:
                risk_level = "High Risk"
            
            recommendations[f"target_hf_{target_hf}"] = {
                "target_hf": target_hf,
                "avg_liquidation_frequency": avg_liquidation_freq,
                "avg_survival_rate": avg_survival_rate,
                "avg_rebalancing_events": avg_rebalancing,
                "risk_level": risk_level,
                "recommended": avg_liquidation_freq < moderate_liquidation_threshold
            }
    
    # Find most aggressive safe target HF
    safe_targets = [rec for rec in recommendations.values() if rec["recommended"]]
    most_aggressive_safe = min(safe_targets, key=lambda x: x["target_hf"]) if safe_targets else None
    
    return {
        "all_target_hf_analysis": recommendations,
        "most_aggressive_safe_target_hf": most_aggressive_safe,
        "analysis_criteria": {
            "low_liquidation_threshold": low_liquidation_threshold,
            "moderate_liquidation_threshold": moderate_liquidation_threshold
        }
    }


def analyze_liquidation_thresholds(df: pd.DataFrame) -> Dict:
    """Analyze liquidation frequency patterns across target health factors"""
    
    liquidation_analysis = {}
    
    for target_hf in df["target_hf"].unique():
        target_data = df[df["target_hf"] == target_hf]
        
        liquidation_analysis[f"target_hf_{target_hf}"] = {
            "target_hf": target_hf,
            "min_liquidation_frequency": target_data["ht_liquidation_frequency"].min(),
            "max_liquidation_frequency": target_data["ht_liquidation_frequency"].max(),
            "mean_liquidation_frequency": target_data["ht_liquidation_frequency"].mean(),
            "std_liquidation_frequency": target_data["ht_liquidation_frequency"].std(),
            "liquidation_frequency_by_buffer": []
        }
        
        # Analyze by HF buffer
        for _, row in target_data.iterrows():
            liquidation_analysis[f"target_hf_{target_hf}"]["liquidation_frequency_by_buffer"].append({
                "hf_buffer": row["hf_buffer"],
                "liquidation_frequency": row["ht_liquidation_frequency"],
                "survival_rate": row["ht_survival_rate"]
            })
    
    return liquidation_analysis


def analyze_rebalancing_efficiency(df: pd.DataFrame) -> Dict:
    """Analyze rebalancing efficiency vs liquidation frequency"""
    
    efficiency_analysis = {}
    
    for target_hf in df["target_hf"].unique():
        target_data = df[df["target_hf"] == target_hf]
        
        # Calculate efficiency metrics
        avg_rebalancing = target_data["ht_rebalancing_events"].mean()
        avg_liquidations = target_data["ht_liquidation_frequency"].mean()
        efficiency_ratio = avg_rebalancing / max(avg_liquidations, 0.001)  # Avoid division by zero
        
        efficiency_analysis[f"target_hf_{target_hf}"] = {
            "target_hf": target_hf,
            "avg_rebalancing_events": avg_rebalancing,
            "avg_liquidation_frequency": avg_liquidations,
            "rebalancing_efficiency_ratio": efficiency_ratio,
            "interpretation": "High" if efficiency_ratio > 10 else "Moderate" if efficiency_ratio > 3 else "Low"
        }
    
    return efficiency_analysis


def analyze_risk_efficiency_tradeoff(df: pd.DataFrame) -> Dict:
    """Analyze the tradeoff between risk (liquidations) and efficiency (aggressive HF)"""
    
    # Calculate risk score (higher = more risky)
    df["risk_score"] = df["ht_liquidation_frequency"] * 100  # Convert to percentage
    
    # Calculate efficiency score (lower target HF = more capital efficient)
    df["efficiency_score"] = 1.0 / df["target_hf"]  # Inverse relationship
    
    # Find Pareto frontier (optimal risk-efficiency combinations)
    pareto_optimal = []
    
    for target_hf in df["target_hf"].unique():
        target_data = df[df["target_hf"] == target_hf]
        best_scenario = target_data.loc[target_data["risk_score"].idxmin()]
        
        pareto_optimal.append({
            "target_hf": target_hf,
            "initial_hf": best_scenario["initial_hf"],
            "hf_buffer": best_scenario["hf_buffer"],
            "risk_score": best_scenario["risk_score"],
            "efficiency_score": best_scenario["efficiency_score"],
            "survival_rate": best_scenario["ht_survival_rate"],
            "liquidation_frequency": best_scenario["ht_liquidation_frequency"]
        })
    
    return {
        "pareto_optimal_scenarios": pareto_optimal,
        "overall_best_tradeoff": min(pareto_optimal, key=lambda x: x["risk_score"] + (1/x["efficiency_score"])),
        "most_aggressive_safe": min([p for p in pareto_optimal if p["risk_score"] < 5], key=lambda x: x["target_hf"]) if any(p["risk_score"] < 5 for p in pareto_optimal) else None
    }


def generate_statistical_summary(df: pd.DataFrame) -> Dict:
    """Generate statistical summary of all tested scenarios"""
    
    return {
        "total_scenarios_tested": len(df),
        "target_hf_range": [df["target_hf"].min(), df["target_hf"].max()],
        "hf_buffer_range": [df["hf_buffer"].min(), df["hf_buffer"].max()],
        "survival_rate_range": [df["ht_survival_rate"].min(), df["ht_survival_rate"].max()],
        "liquidation_frequency_range": [df["ht_liquidation_frequency"].min(), df["ht_liquidation_frequency"].max()],
        "correlation_analysis": {
            "target_hf_vs_liquidations": df["target_hf"].corr(df["ht_liquidation_frequency"]),
            "hf_buffer_vs_survival": df["hf_buffer"].corr(df["ht_survival_rate"]),
            "rebalancing_vs_survival": df["ht_rebalancing_events"].corr(df["ht_survival_rate"])
        }
    }


def save_target_hf_results(analysis_results: Dict, results_matrix: List):
    """Save Target Health Factor analysis results to JSON file"""
    
    # Create results directory
    output_dir = Path("tidal_protocol_sim/results/target_health_factor_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare comprehensive results
    final_results = {
        "analysis_metadata": {
            "analysis_type": "Target_Health_Factor_Analysis",
            "timestamp": datetime.now().isoformat(),
            "target_hfs_tested": [1.01, 1.05, 1.1, 1.15],
            "monte_carlo_runs_per_scenario": 20,
            "total_scenarios": len(results_matrix)
        },
        "key_findings": analysis_results,
        "detailed_scenario_results": results_matrix
    }
    
    # Save JSON results
    results_path = output_dir / "target_hf_analysis_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"📁 Target HF analysis results saved to: {results_path}")
    
    # Save summary CSV for easy analysis
    df_data = []
    for result in results_matrix:
        params = result["scenario_params"]
        ht_summary = result["high_tide_summary"]
        aave_summary = result["aave_summary"]
        comparison = result["comparison"]
        
        df_data.append({
            "initial_hf": params["initial_hf"],
            "target_hf": params["target_hf"],
            "hf_buffer": params["hf_buffer"],
            "ht_survival_rate": ht_summary["mean_survival_rate"],
            "ht_liquidation_frequency": ht_summary["liquidation_frequency"],
            "ht_rebalancing_events": ht_summary["mean_rebalancing_events"],
            "aave_survival_rate": aave_summary["mean_survival_rate"],
            "aave_liquidation_frequency": aave_summary["liquidation_frequency"],
            "survival_improvement": comparison["survival_improvement"],
            "liquidation_reduction": comparison["liquidation_reduction"]
        })
    
    df = pd.DataFrame(df_data)
    csv_path = output_dir / "target_hf_analysis_summary.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"📊 Summary data saved to: {csv_path}")
    
    return results_path


def print_target_hf_summary(analysis_results: Dict):
    """Print comprehensive summary of Target Health Factor analysis"""
    
    optimal_recs = analysis_results.get("optimal_recommendations", {})
    liquidation_analysis = analysis_results.get("liquidation_threshold_analysis", {})
    risk_efficiency = analysis_results.get("risk_vs_efficiency", {})
    
    print("\n" + "=" * 80)
    print("TARGET HEALTH FACTOR ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Optimal recommendations
    best_tradeoff = risk_efficiency.get("overall_best_tradeoff")
    most_aggressive_safe = risk_efficiency.get("most_aggressive_safe")
    
    if best_tradeoff:
        print(f"\n🎯 OPTIMAL TARGET HEALTH FACTOR RECOMMENDATION:")
        print(f"   Target HF: {best_tradeoff['target_hf']:.2f}")
        print(f"   Initial HF: {best_tradeoff['initial_hf']:.2f}")
        print(f"   Buffer: {best_tradeoff['hf_buffer']:.2f}")
        print(f"   Survival Rate: {best_tradeoff['survival_rate']:.1%}")
        print(f"   Liquidation Frequency: {best_tradeoff['liquidation_frequency']:.1%}")
    
    if most_aggressive_safe:
        print(f"\n⚡ MOST AGGRESSIVE SAFE TARGET HF:")
        print(f"   Target HF: {most_aggressive_safe['target_hf']:.2f}")
        print(f"   Risk Score: {most_aggressive_safe['risk_score']:.1f}% liquidation rate")
        print(f"   Survival Rate: {most_aggressive_safe['survival_rate']:.1%}")
    
    # Target HF recommendations
    all_recs = optimal_recs.get("all_target_hf_analysis", {})
    print(f"\n📊 TARGET HEALTH FACTOR ANALYSIS:")
    for target_hf in [1.01, 1.05, 1.1, 1.15]:
        rec = all_recs.get(f"target_hf_{target_hf}")
        if rec:
            status = "✅ SAFE" if rec["recommended"] else "⚠️  RISKY"
            print(f"   {target_hf:.2f}: {rec['risk_level']} - {rec['avg_liquidation_frequency']:.1%} liquidations {status}")
    
    # Statistical correlations
    stats = analysis_results.get("statistical_summary", {})
    correlations = stats.get("correlation_analysis", {})
    
    print(f"\n🔬 STATISTICAL CORRELATIONS:")
    print(f"   Target HF ↔ Liquidations: {correlations.get('target_hf_vs_liquidations', 0):.3f}")
    print(f"   HF Buffer ↔ Survival: {correlations.get('hf_buffer_vs_survival', 0):.3f}")
    print(f"   Rebalancing ↔ Survival: {correlations.get('rebalancing_vs_survival', 0):.3f}")
    
    print("\n" + "=" * 80)


def main():
    """Main execution function"""
    try:
        results = run_target_hf_analysis()
        print("\n✅ Target Health Factor analysis completed successfully!")
        return results
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()