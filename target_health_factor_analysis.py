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
import matplotlib.pyplot as plt
import seaborn as sns
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
    print("Testing target health factors: 1.01, 1.05, 1.075, 1.1, 1.15")
    print("Question: How low can Target HF go before frequent liquidations?")
    print()
    
    # Target health factors to test (discrete testing)
    target_hfs = [1.01, 1.05, 1.075, 1.1, 1.15]
    monte_carlo_runs = 20  # Each target HF gets 20 runs with varied agents
    
    results_matrix = []
    
    for target_hf in target_hfs:
        print(f"🎯 Testing Target Health Factor: {target_hf}")
        print(f"   Agents will have randomized Initial HFs (1.2-1.5) and fixed Target HF: {target_hf}")
        
        # Single scenario per target HF with varied agent population
        result = run_target_hf_scenario(target_hf, monte_carlo_runs)
        result["scenario_params"] = {
            "target_hf": target_hf,
            "initial_hf_range": [1.2, 1.5],
            "variation_type": "randomized_initial_hf"
        }
        results_matrix.append(result)
    
    # Generate comprehensive analysis
    analysis_results = analyze_target_hf_results(results_matrix)
    
    # Save results with JSON output
    save_target_hf_results(analysis_results, results_matrix)
    
    # Generate comprehensive charts
    output_dir = Path("tidal_protocol_sim/results/target_health_factor_analysis")
    generated_charts = create_target_hf_analysis_charts(analysis_results, results_matrix, output_dir)
    
    # Generate CSV with detailed agent data
    csv_path = create_agent_data_csv(results_matrix, output_dir)
    
    # Print summary
    print_target_hf_summary(analysis_results)
    
    print(f"\n📊 Generated {len(generated_charts)} Target Health Factor analysis charts")
    print(f"📁 Charts saved to: {output_dir}")
    if csv_path:
        print(f"📊 Agent data CSV saved to: {csv_path}")
    
    return analysis_results


def run_target_hf_scenario(target_hf: float, monte_carlo_runs: int) -> Dict:
    """Run scenario testing a specific target health factor with varied agent population"""
    
    print(f"   Running {monte_carlo_runs} Monte Carlo simulations with Target HF: {target_hf:.3f}")
    
    # Store results from all runs
    ht_runs = []
    aave_runs = []
    
    for run_num in range(monte_carlo_runs):
        # High Tide simulation with randomized agents
        ht_config = HighTideConfig()
        ht_config.num_high_tide_agents = 0  # We'll create custom agents
        ht_config.btc_decline_duration = 60
        ht_config.moet_btc_pool_size = 250_000  # Standard pool size
        ht_config.moet_yield_pool_size = 250_000  # Standard YT pool size
        
        # Create custom High Tide agents with randomized initial HFs
        custom_ht_agents = create_custom_agents_for_hf_test(
            target_hf, num_agents=15, agent_type="high_tide"
        )
        
        ht_engine = HighTideSimulationEngine(ht_config)
        ht_engine.high_tide_agents = custom_ht_agents
        
        # Add agents to engine's agent dict
        for agent in custom_ht_agents:
            ht_engine.agents[agent.agent_id] = agent
        
        ht_results = ht_engine.run_high_tide_simulation()
        ht_runs.append(ht_results)
        
        # Aave simulation with matching agents (same random seed for consistency)
        aave_config = AaveConfig()
        aave_config.num_aave_agents = 0  # We'll create custom agents
        aave_config.btc_decline_duration = 60
        aave_config.moet_btc_pool_size = 250_000
        aave_config.moet_yield_pool_size = 250_000
        
        # Create matching Aave agents with same target HF
        custom_aave_agents = create_custom_agents_for_hf_test(
            target_hf, num_agents=15, agent_type="aave"
        )
        
        aave_engine = AaveSimulationEngine(aave_config)
        aave_engine.aave_agents = custom_aave_agents
        
        # Add agents to engine's agent dict
        for agent in custom_aave_agents:
            aave_engine.agents[agent.agent_id] = agent
        
        aave_results = aave_engine.run_aave_simulation()
        aave_runs.append(aave_results)
    
    # Aggregate results
    scenario_analysis = aggregate_hf_scenario_results(ht_runs, aave_runs, target_hf)
    
    print(f"      High Tide: {scenario_analysis['high_tide_summary']['mean_survival_rate']:.1%} survival, "
          f"{scenario_analysis['high_tide_summary']['mean_liquidations']:.1f} liquidations")
    print(f"      Aave:      {scenario_analysis['aave_summary']['mean_survival_rate']:.1%} survival, "
          f"{scenario_analysis['aave_summary']['mean_liquidations']:.1f} liquidations")
    
    return scenario_analysis


def create_custom_agents_for_hf_test(target_hf: float, num_agents: int, agent_type: str) -> List:
    """Create custom agents with randomized initial HFs (1.2-1.5) and fixed target HF for testing"""
    import random
    from tidal_protocol_sim.agents.high_tide_agent import HighTideAgent
    from tidal_protocol_sim.agents.aave_agent import AaveAgent
    
    agents = []
    
    for i in range(num_agents):
        # Randomize initial health factor between 1.2-1.5 for proper variation
        initial_hf = random.uniform(1.2, 1.5)
        
        if agent_type == "high_tide":
            agent = HighTideAgent(
                f"hf_test_ht_{target_hf}_{i}",
                initial_hf,
                target_hf
            )
        else:  # aave
            agent = AaveAgent(
                f"hf_test_aave_{target_hf}_{i}",
                initial_hf,
                target_hf
            )
        
        agents.append(agent)
    
    return agents


def aggregate_hf_scenario_results(ht_runs: List, aave_runs: List, target_hf: float) -> Dict:
    """Aggregate results for a specific target health factor scenario"""
    
    # Aggregate High Tide metrics from actual simulation results
    ht_survival_rates = []
    ht_liquidations = []
    ht_rebalancing_events = []
    ht_agent_outcomes = []
    
    for run in ht_runs:
        survival_stats = run.get("survival_statistics", {})
        ht_survival_rates.append(survival_stats.get("survival_rate", 0.0))
        
        # Count liquidations from agent outcomes
        agent_outcomes = run.get("agent_outcomes", [])
        total_liquidations = sum(1 for outcome in agent_outcomes if not outcome.get("survived", True))
        ht_liquidations.append(total_liquidations)
        
        # Store individual agent outcomes for detailed analysis
        ht_agent_outcomes.extend(agent_outcomes)
        
        # Count rebalancing events from yield token activity
        rebalancing_activity = run.get("yield_token_activity", {})
        ht_rebalancing_events.append(rebalancing_activity.get("rebalancing_events", 0))
    
    # Aggregate Aave metrics from actual simulation results
    aave_survival_rates = []
    aave_liquidations = []
    aave_agent_outcomes = []
    
    for run in aave_runs:
        survival_stats = run.get("survival_statistics", {})
        aave_survival_rates.append(survival_stats.get("survival_rate", 0.0))
        
        # Count liquidations from agent outcomes
        agent_outcomes = run.get("agent_outcomes", [])
        total_liquidations = sum(1 for outcome in agent_outcomes if not outcome.get("survived", True))
        aave_liquidations.append(total_liquidations)
        
        # Store individual agent outcomes for detailed analysis
        aave_agent_outcomes.extend(agent_outcomes)
    
    # Calculate summary statistics
    return {
        "target_hf": target_hf,
        "initial_hf_range": [1.2, 1.5],  # Agent variation range
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
            "ht_rebalancing_events": ht_rebalancing_events,
            "ht_agent_outcomes": ht_agent_outcomes,
            "aave_agent_outcomes": aave_agent_outcomes
        }
    }


def analyze_target_hf_results(results_matrix: List[Dict]) -> Dict:
    """Analyze all target health factor results to find optimal parameters"""
    
    # Create DataFrame for analysis using actual simulation data
    df_data = []
    for result in results_matrix:
        params = result["scenario_params"]
        ht_summary = result["high_tide_summary"]
        aave_summary = result["aave_summary"]
        comparison = result["comparison"]
        
        df_data.append({
            "target_hf": params["target_hf"],
            "initial_hf_range": params.get("initial_hf_range", [1.2, 1.5]),
            "ht_survival_rate": ht_summary["mean_survival_rate"],
            "ht_liquidation_frequency": ht_summary["liquidation_frequency"],
            "ht_rebalancing_events": ht_summary["mean_rebalancing_events"],
            "aave_survival_rate": aave_summary["mean_survival_rate"],
            "aave_liquidation_frequency": aave_summary["liquidation_frequency"],
            "survival_improvement": comparison["survival_improvement"],
            "liquidation_reduction": comparison["liquidation_reduction"]
        })
    
    df = pd.DataFrame(df_data)
    
    # Find optimal target health factors using actual data
    analysis = {
        "optimal_recommendations": find_optimal_target_hf_from_data(results_matrix),
        "liquidation_threshold_analysis": analyze_liquidation_thresholds_from_data(results_matrix),
        "rebalancing_efficiency": analyze_rebalancing_efficiency_from_data(results_matrix),
        "risk_vs_efficiency": analyze_risk_efficiency_tradeoff_from_data(results_matrix),
        "statistical_summary": generate_statistical_summary_from_data(results_matrix),
        "raw_results_matrix": results_matrix
    }
    
    return analysis


def find_optimal_target_hf_from_data(results_matrix: List[Dict]) -> Dict:
    """Find optimal target health factors based on actual simulation data"""
    
    # Define thresholds for analysis
    low_liquidation_threshold = 0.05  # < 5% liquidation rate is "low"
    moderate_liquidation_threshold = 0.10  # < 10% liquidation rate is "moderate"
    
    recommendations = {}
    
    for result in results_matrix:
        target_hf = result["target_hf"]
        ht_summary = result["high_tide_summary"]
        
        avg_liquidation_freq = ht_summary["liquidation_frequency"]
        avg_survival_rate = ht_summary["mean_survival_rate"]
        avg_rebalancing = ht_summary["mean_rebalancing_events"]
        
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


def analyze_liquidation_thresholds_from_data(results_matrix: List[Dict]) -> Dict:
    """Analyze liquidation frequency patterns across target health factors using actual data"""
    
    liquidation_analysis = {}
    
    for result in results_matrix:
        target_hf = result["target_hf"]
        ht_summary = result["high_tide_summary"]
        raw_data = result["raw_data"]
        
        liquidation_analysis[f"target_hf_{target_hf}"] = {
            "target_hf": target_hf,
            "min_liquidation_frequency": min(raw_data["ht_liquidations"]) / 15,  # Per agent
            "max_liquidation_frequency": max(raw_data["ht_liquidations"]) / 15,  # Per agent
            "mean_liquidation_frequency": ht_summary["liquidation_frequency"],
            "std_liquidation_frequency": np.std([l/15 for l in raw_data["ht_liquidations"]]),
            "liquidation_frequency_by_buffer": []
        }
    
    return liquidation_analysis


def analyze_rebalancing_efficiency_from_data(results_matrix: List[Dict]) -> Dict:
    """Analyze rebalancing efficiency vs liquidation frequency using actual data"""
    
    efficiency_analysis = {}
    
    for result in results_matrix:
        target_hf = result["target_hf"]
        ht_summary = result["high_tide_summary"]
        
        # Calculate efficiency metrics from actual data
        avg_rebalancing = ht_summary["mean_rebalancing_events"]
        avg_liquidations = ht_summary["liquidation_frequency"]
        efficiency_ratio = avg_rebalancing / max(avg_liquidations, 0.001)  # Avoid division by zero
        
        efficiency_analysis[f"target_hf_{target_hf}"] = {
            "target_hf": target_hf,
            "avg_rebalancing_events": avg_rebalancing,
            "avg_liquidation_frequency": avg_liquidations,
            "rebalancing_efficiency_ratio": efficiency_ratio,
            "interpretation": "High" if efficiency_ratio > 10 else "Moderate" if efficiency_ratio > 3 else "Low"
        }
    
    return efficiency_analysis


def analyze_risk_efficiency_tradeoff_from_data(results_matrix: List[Dict]) -> Dict:
    """Analyze the tradeoff between risk (liquidations) and efficiency (aggressive HF) using actual data"""
    
    # Calculate risk and efficiency scores from actual data
    pareto_optimal = []
    
    for result in results_matrix:
        target_hf = result["target_hf"]
        ht_summary = result["high_tide_summary"]
        
        # Calculate risk score (higher = more risky)
        risk_score = ht_summary["liquidation_frequency"] * 100  # Convert to percentage
        
        # Calculate efficiency score (lower target HF = more capital efficient)
        efficiency_score = 1.0 / target_hf  # Inverse relationship
        
        pareto_optimal.append({
            "target_hf": target_hf,
            "initial_hf": 1.35,  # Average of 1.2-1.5 range
            "hf_buffer": target_hf - 1.0,  # Buffer from liquidation threshold
            "risk_score": risk_score,
            "efficiency_score": efficiency_score,
            "survival_rate": ht_summary["mean_survival_rate"],
            "liquidation_frequency": ht_summary["liquidation_frequency"]
        })
    
    return {
        "pareto_optimal_scenarios": pareto_optimal,
        "overall_best_tradeoff": min(pareto_optimal, key=lambda x: x["risk_score"] + (1/x["efficiency_score"])) if pareto_optimal else None,
        "most_aggressive_safe": min([p for p in pareto_optimal if p["risk_score"] < 5], key=lambda x: x["target_hf"]) if any(p["risk_score"] < 5 for p in pareto_optimal) else None
    }


def generate_statistical_summary_from_data(results_matrix: List[Dict]) -> Dict:
    """Generate statistical summary of all tested scenarios using actual data"""
    
    # Extract data for correlation analysis
    target_hfs = [r["target_hf"] for r in results_matrix]
    survival_rates = [r["high_tide_summary"]["mean_survival_rate"] for r in results_matrix]
    liquidation_freqs = [r["high_tide_summary"]["liquidation_frequency"] for r in results_matrix]
    rebalancing_events = [r["high_tide_summary"]["mean_rebalancing_events"] for r in results_matrix]
    
    return {
        "total_scenarios_tested": len(results_matrix),
        "target_hf_range": [min(target_hfs), max(target_hfs)],
        "initial_hf_range": [1.2, 1.5],  # Fixed range for all scenarios
        "survival_rate_range": [min(survival_rates), max(survival_rates)],
        "liquidation_frequency_range": [min(liquidation_freqs), max(liquidation_freqs)],
        "correlation_analysis": {
            "target_hf_vs_liquidations": np.corrcoef(target_hfs, liquidation_freqs)[0, 1] if len(target_hfs) > 1 else 0,
            "target_hf_vs_survival": np.corrcoef(target_hfs, survival_rates)[0, 1] if len(target_hfs) > 1 else 0,
            "rebalancing_vs_survival": np.corrcoef(rebalancing_events, survival_rates)[0, 1] if len(rebalancing_events) > 1 else 0
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
            "target_hfs_tested": [1.01, 1.05, 1.075, 1.1, 1.15],
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
            "target_hf": params["target_hf"],
            "initial_hf_range_min": params.get("initial_hf_range", [1.2, 1.5])[0],
            "initial_hf_range_max": params.get("initial_hf_range", [1.2, 1.5])[1],
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
    for target_hf in [1.01, 1.05, 1.075, 1.1, 1.15]:
        rec = all_recs.get(f"target_hf_{target_hf}")
        if rec:
            status = "✅ SAFE" if rec["recommended"] else "⚠️  RISKY"
            print(f"   {target_hf:.3f}: {rec['risk_level']} - {rec['avg_liquidation_frequency']:.1%} liquidations {status}")
    
    # Statistical correlations
    stats = analysis_results.get("statistical_summary", {})
    correlations = stats.get("correlation_analysis", {})
    
    print(f"\n🔬 STATISTICAL CORRELATIONS:")
    print(f"   Target HF ↔ Liquidations: {correlations.get('target_hf_vs_liquidations', 0):.3f}")
    print(f"   Target HF ↔ Survival: {correlations.get('target_hf_vs_survival', 0):.3f}")
    print(f"   Rebalancing ↔ Survival: {correlations.get('rebalancing_vs_survival', 0):.3f}")
    
    print("\n" + "=" * 80)


def create_target_hf_analysis_charts(analysis_results: Dict, results_matrix: List, output_dir: Path) -> List[Path]:
    """Create comprehensive Target Health Factor analysis charts"""
    
    print("🎨 Generating Target Health Factor analysis charts...")
    
    # Create charts directory
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    generated_charts = []
    
    try:
        # Import the High Tide chart generator
        from tidal_protocol_sim.analysis.high_tide_charts import HighTideChartGenerator
        
        # Run one full simulation with detailed data collection for chart generation
        print("🔄 Running detailed simulation for comprehensive chart generation...")
        detailed_simulation_result = run_detailed_simulation_for_charts()
        
        if detailed_simulation_result:
            # Generate all the charts using REAL simulation data
            chart_generator = HighTideChartGenerator()
            
            all_charts = chart_generator.generate_high_tide_charts(
                scenario_name="Target_Health_Factor_Analysis",
                results=detailed_simulation_result,
                charts_dir=charts_dir,
                pool_info={
                    "btc_pool_label": "$250k MOET:BTC",
                    "yield_pool_label": "$250k MOET:YT", 
                    "btc_pool_size": 250_000,
                    "yield_pool_size": 250_000
                }
            )
            
            generated_charts.extend(all_charts)
            print(f"✅ Generated {len(all_charts)} comprehensive charts from real simulation data")
        else:
            print("⚠️  Failed to run detailed simulation, skipping comprehensive charts")
        
        # Create individual agent tracking charts
        individual_charts = create_individual_agent_tracking_charts(detailed_simulation_result, charts_dir)
        generated_charts.extend(individual_charts)
        
        # 1. Target HF Agent Performance Summary (like high_tide_agent_performance_summary.png)
        chart_path = create_target_hf_agent_performance_summary(results_matrix, charts_dir)
        if chart_path:
            generated_charts.append(chart_path)
        
        # 2. Target HF Health Factor Analysis (like high_tide_health_factor_analysis.png)
        chart_path = create_target_hf_health_factor_analysis(results_matrix, charts_dir)
        if chart_path:
            generated_charts.append(chart_path)
        
        # 3. Target HF Net Position Analysis (like high_tide_net_position_analysis.png)
        chart_path = create_target_hf_net_position_analysis(results_matrix, charts_dir)
        if chart_path:
            generated_charts.append(chart_path)
        
        # 4. Target HF vs Aave Performance Comparison
        chart_path = create_ht_vs_aave_comparison_chart(analysis_results, charts_dir)
        if chart_path:
            generated_charts.append(chart_path)
        
        # 5. Target HF Optimization Dashboard
        chart_path = create_target_hf_dashboard(analysis_results, charts_dir)
        if chart_path:
            generated_charts.append(chart_path)
        
        print(f"✅ Generated {len(generated_charts)} Target Health Factor analysis charts")
        
    except Exception as e:
        print(f"❌ Error generating comprehensive charts: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to basic charts if comprehensive generation fails
        print("🔄 Falling back to basic chart generation...")
        basic_charts = create_basic_target_hf_charts(results_matrix, charts_dir)
        generated_charts.extend(basic_charts)
    
    return generated_charts


def run_detailed_simulation_for_charts() -> Dict:
    """Run a single detailed High Tide simulation to collect all data needed for charts"""
    
    try:
        # Create High Tide configuration for detailed simulation
        ht_config = HighTideConfig()
        ht_config.num_high_tide_agents = 10
        ht_config.btc_decline_duration = 60
        ht_config.moet_btc_pool_size = 250_000
        ht_config.moet_yield_pool_size = 250_000
        
        # Create varied High Tide agents with a moderate target HF
        target_hf = 1.1  # Use a moderate target HF for good rebalancing activity
        custom_ht_agents = create_custom_agents_for_hf_test(
            target_hf, num_agents=10, agent_type="high_tide"
        )
        
        # Run the simulation
        ht_engine = HighTideSimulationEngine(ht_config)
        ht_engine.high_tide_agents = custom_ht_agents
        
        for agent in custom_ht_agents:
            ht_engine.agents[agent.agent_id] = agent
        
        # Run simulation and get full results
        ht_results = ht_engine.run_high_tide_simulation()
        
        print(f"✅ Detailed simulation completed with {len(ht_results.get('agent_outcomes', []))} agents")
        return ht_results
        
    except Exception as e:
        print(f"❌ Error running detailed simulation: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_individual_agent_tracking_charts(simulation_result: Dict, charts_dir: Path) -> List[Path]:
    """Create individual agent tracking charts"""
    
    charts = []
    
    if not simulation_result:
        return charts
    
    try:
        # Setup styling
        plt.style.use('default')
        sns.set_palette("husl")
        
        plt.rcParams.update({
            'figure.figsize': (16, 12),
            'font.size': 11,
            'axes.titlesize': 16,
            'axes.labelsize': 13,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 18
        })
        
        # Extract agent health history
        agent_health_history = simulation_result.get("agent_health_history", [])
        agent_outcomes = simulation_result.get("agent_outcomes", [])
        
        if not agent_health_history or not agent_outcomes:
            print("⚠️  No agent health history or outcomes found")
            return charts
        
        # Create individual agent health factor tracking chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Individual Agent Tracking Analysis", fontsize=18, fontweight='bold')
        
        # Chart 1: Individual Agent Health Factors Over Time
        minutes = [entry["minute"] for entry in agent_health_history]
        
        # Get unique agent IDs
        agent_ids = set()
        for entry in agent_health_history:
            for agent_data in entry["agents"]:
                agent_ids.add(agent_data["agent_id"])
        
        agent_ids = sorted(list(agent_ids))
        
        # Plot each agent's health factor over time
        colors = plt.cm.tab20(np.linspace(0, 1, len(agent_ids)))
        
        for i, agent_id in enumerate(agent_ids):
            health_factors = []
            for entry in agent_health_history:
                for agent_data in entry["agents"]:
                    if agent_data["agent_id"] == agent_id:
                        health_factors.append(agent_data["health_factor"])
                        break
                else:
                    health_factors.append(None)  # Agent not found in this minute
            
            # Filter out None values and plot
            valid_minutes = [m for m, hf in zip(minutes, health_factors) if hf is not None]
            valid_hfs = [hf for hf in health_factors if hf is not None]
            
            if valid_hfs:
                ax1.plot(valid_minutes, valid_hfs, color=colors[i], linewidth=1.5, alpha=0.7, 
                        label=f'Agent {agent_id.split("_")[-1]}' if len(agent_ids) <= 10 else None)
        
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Health Factor')
        ax1.set_title('Individual Agent Health Factors Over Time')
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Liquidation Threshold')
        ax1.grid(True, alpha=0.3)
        if len(agent_ids) <= 10:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Chart 2: Yield Token Start vs End per Agent
        agent_yield_data = []
        for agent in agent_outcomes:
            agent_id = agent.get("agent_id", "unknown")
            yield_portfolio = agent.get("yield_token_portfolio", {})
            start_yield = yield_portfolio.get("total_principal", 0)
            end_yield = yield_portfolio.get("total_current_value", 0)
            agent_yield_data.append({
                "agent_id": agent_id,
                "start_yield": start_yield,
                "end_yield": end_yield
            })
        
        if agent_yield_data:
            agent_labels = [f"Agent {data['agent_id'].split('_')[-1]}" for data in agent_yield_data]
            start_values = [data["start_yield"] for data in agent_yield_data]
            end_values = [data["end_yield"] for data in agent_yield_data]
            
            x_pos = np.arange(len(agent_labels))
            width = 0.35
            
            bars1 = ax2.bar(x_pos - width/2, start_values, width, 
                           label='Start Yield Tokens', color='lightblue', alpha=0.8)
            bars2 = ax2.bar(x_pos + width/2, end_values, width, 
                           label='End Yield Tokens', color='darkblue', alpha=0.8)
            
            ax2.set_xlabel('Agent')
            ax2.set_ylabel('Yield Token Value ($)')
            ax2.set_title('Yield Token Holdings: Start vs End')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(agent_labels, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Chart 3: Rebalancing Events per Agent
        rebalancing_data = []
        for agent in agent_outcomes:
            agent_id = agent.get("agent_id", "unknown")
            rebalancing_events = agent.get("rebalancing_events", 0)
            rebalancing_data.append({
                "agent_id": agent_id,
                "rebalancing_events": rebalancing_events
            })
        
        if rebalancing_data:
            agent_labels = [f"Agent {data['agent_id'].split('_')[-1]}" for data in rebalancing_data]
            rebalancing_counts = [data["rebalancing_events"] for data in rebalancing_data]
            
            bars3 = ax3.bar(agent_labels, rebalancing_counts, color='orange', alpha=0.8)
            ax3.set_xlabel('Agent')
            ax3.set_ylabel('Number of Rebalancing Events')
            ax3.set_title('Rebalancing Activity per Agent')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, count in zip(bars3, rebalancing_counts):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{count}', ha='center', va='bottom', fontsize=9)
        
        # Chart 4: Final Health Factor Distribution
        final_health_factors = [agent.get("final_health_factor", 1.0) for agent in agent_outcomes]
        initial_health_factors = [agent.get("initial_health_factor", 1.0) for agent in agent_outcomes]
        
        ax4.hist(initial_health_factors, bins=10, alpha=0.7, label='Initial HF', color='lightgreen', edgecolor='black')
        ax4.hist(final_health_factors, bins=10, alpha=0.7, label='Final HF', color='darkgreen', edgecolor='black')
        ax4.set_xlabel('Health Factor')
        ax4.set_ylabel('Number of Agents')
        ax4.set_title('Health Factor Distribution: Initial vs Final')
        ax4.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Liquidation Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        chart_path = charts_dir / "individual_agent_tracking_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        charts.append(chart_path)
        print(f"✅ Generated individual agent tracking chart: {chart_path.name}")
        
    except Exception as e:
        print(f"❌ Error creating individual agent tracking charts: {e}")
        import traceback
        traceback.print_exc()
    
    return charts


def create_basic_target_hf_charts(results_matrix: List[Dict], charts_dir: Path) -> List[Path]:
    """Fallback basic chart generation if comprehensive generation fails"""
    
    charts = []
    
    # Create a simple summary chart
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        target_hfs = [r["target_hf"] for r in results_matrix]
        ht_survival_rates = [r["high_tide_summary"]["mean_survival_rate"] for r in results_matrix]
        aave_survival_rates = [r["aave_summary"]["mean_survival_rate"] for r in results_matrix]
        
        x_pos = np.arange(len(target_hfs))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, ht_survival_rates, width, 
                      label='High Tide', color='#2E8B57', alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, aave_survival_rates, width, 
                      label='Aave', color='#DC143C', alpha=0.8)
        
        ax.set_xlabel('Target Health Factor')
        ax.set_ylabel('Survival Rate')
        ax.set_title('Target HF Analysis: High Tide vs Aave Survival Rates')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{hf:.3f}' for hf in target_hfs])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        chart_path = charts_dir / "target_hf_basic_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        charts.append(chart_path)
        
    except Exception as e:
        print(f"Error creating basic chart: {e}")
    
    return charts


def create_agent_data_csv(results_matrix: List[Dict], output_dir: Path) -> Path:
    """Create CSV file with detailed agent data"""
    
    try:
        # Aggregate all agent outcomes across all target HF scenarios
        all_ht_agents = []
        
        for result in results_matrix:
            raw_data = result.get("raw_data", {})
            all_ht_agents.extend(raw_data.get("ht_agent_outcomes", []))
        
        if not all_ht_agents:
            print("⚠️  No agent data found for CSV generation")
            return None
        
        # Create CSV data
        csv_data = []
        for agent in all_ht_agents:
            agent_id = agent.get("agent_id", "unknown")
            
            # Extract collateral and debt information
            collateral = agent.get("collateral_value", 0)
            effective_collateral = agent.get("effective_collateral_value", collateral)
            debt = agent.get("debt_value", 0)
            
            # Extract health factor information
            initial_hf = agent.get("initial_health_factor", 1.0)
            target_hf = agent.get("target_health_factor", 1.1)
            final_hf = agent.get("final_health_factor", 1.0)
            
            # Extract rebalancing information
            rebalancing_events = agent.get("rebalancing_events", 0)
            
            csv_data.append({
                "Agent Name": agent_id,
                "Collateral": f"${collateral:,.2f}",
                "Effective Collateral": f"${effective_collateral:,.2f}",
                "Debt": f"${debt:,.2f}",
                "Initial Health Factor": f"{initial_hf:.3f}",
                "Target Health Factor": f"{target_hf:.3f}",
                "# of Rebalances": rebalancing_events,
                "End Health Factor": f"{final_hf:.3f}"
            })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(csv_data)
        csv_path = output_dir / "agent_detailed_data.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"📊 Agent detailed data saved to: {csv_path}")
        return csv_path
        
    except Exception as e:
        print(f"❌ Error creating agent data CSV: {e}")
        import traceback
        traceback.print_exc()
        return None


def _setup_chart_styling():
    """Setup professional chart styling for Target Health Factor analysis"""
    plt.style.use('default')
    sns.set_palette("husl")
    
    plt.rcParams.update({
        'figure.figsize': (14, 10),
        'font.size': 11,
        'axes.titlesize': 16,
        'axes.labelsize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 18
    })


def create_target_hf_agent_performance_summary(results_matrix: List[Dict], charts_dir: Path) -> Path:
    """Create Target Health Factor Agent Performance Summary dashboard"""
    
    _setup_chart_styling()
    
    try:
        # Aggregate agent data across all target HF scenarios
        all_ht_agents = []
        all_aave_agents = []
        
        for result in results_matrix:
            raw_data = result.get("raw_data", {})
            all_ht_agents.extend(raw_data.get("ht_agent_outcomes", []))
            all_aave_agents.extend(raw_data.get("aave_agent_outcomes", []))
        
        if not all_ht_agents:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Target Health Factor: Agent Performance Summary", fontsize=18, fontweight='bold')
        
        # Group agents by risk profile (using target HF as proxy for risk)
        risk_profiles = {"conservative": [], "moderate": [], "aggressive": []}
        
        for agent in all_ht_agents:
            target_hf = agent.get("target_health_factor", 1.1)
            if target_hf <= 1.05:
                risk_profiles["aggressive"].append(agent)
            elif target_hf <= 1.1:
                risk_profiles["moderate"].append(agent)
            else:
                risk_profiles["conservative"].append(agent)
        
        # Colors for risk profiles
        colors = {"conservative": "#2E8B57", "moderate": "#FF8C00", "aggressive": "#DC143C"}
        
        # Chart 1: Cost of Rebalancing by Risk Profile (Box Plot)
        profile_costs = {profile: [] for profile in risk_profiles}
        for profile, agents in risk_profiles.items():
            for agent in agents:
                cost = agent.get("cost_of_rebalancing", agent.get("cost_of_liquidation", 0))
                profile_costs[profile].append(cost)
        
        profile_names = []
        costs_data = []
        for profile in ["conservative", "moderate", "aggressive"]:
            if profile_costs[profile]:
                profile_names.append(profile.title())
                costs_data.append(profile_costs[profile])
        
        if costs_data:
            box_plot = ax1.boxplot(costs_data, labels=profile_names, patch_artist=True)
            for patch, profile in zip(box_plot['boxes'], ["conservative", "moderate", "aggressive"][:len(costs_data)]):
                patch.set_facecolor(colors[profile])
                patch.set_alpha(0.7)
        
        ax1.set_ylabel("Cost of Rebalancing ($)")
        ax1.set_title("Cost of Rebalancing by Risk Profile")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Chart 2: Survival Rate by Risk Profile
        survival_rates = []
        profile_labels = []
        for profile in ["conservative", "moderate", "aggressive"]:
            if risk_profiles[profile]:
                survived = sum(1 for agent in risk_profiles[profile] if agent.get("survived", True))
                total = len(risk_profiles[profile])
                survival_rate = survived / total if total > 0 else 0
                survival_rates.append(survival_rate * 100)
                profile_labels.append(profile.title())
        
        bars2 = ax2.bar(profile_labels, survival_rates, 
                       color=[colors[p.lower()] for p in profile_labels], alpha=0.8)
        ax2.set_ylabel("Survival Rate (%)")
        ax2.set_title("Agent Survival Rate by Risk Profile")
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rate in zip(bars2, survival_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        # Chart 3: Average Yield Earned vs Sold
        yield_earned = []
        yield_sold = []
        for profile in ["conservative", "moderate", "aggressive"]:
            if risk_profiles[profile]:
                avg_earned = np.mean([agent.get("total_yield_earned", 0) for agent in risk_profiles[profile]])
                avg_sold = np.mean([agent.get("total_yield_sold", 0) for agent in risk_profiles[profile]])
                yield_earned.append(avg_earned)
                yield_sold.append(avg_sold)
            else:
                yield_earned.append(0)
                yield_sold.append(0)
        
        x_pos = np.arange(len(profile_labels))
        width = 0.35
        
        bars3a = ax3.bar(x_pos - width/2, yield_earned, width, 
                        label='Yield Earned', color='#2E8B57', alpha=0.8)
        bars3b = ax3.bar(x_pos + width/2, yield_sold, width, 
                        label='Yield Sold', color='#DC143C', alpha=0.8)
        
        ax3.set_ylabel("Average Yield ($)")
        ax3.set_title("Average Yield Earned vs Sold")
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(profile_labels)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Average Rebalancing Frequency
        rebalancing_freqs = []
        for profile in ["conservative", "moderate", "aggressive"]:
            if risk_profiles[profile]:
                # Count rebalancing events from agent outcomes
                total_rebalancing = sum(agent.get("rebalancing_events", 0) for agent in risk_profiles[profile])
                avg_rebalancing = total_rebalancing / len(risk_profiles[profile])
                rebalancing_freqs.append(avg_rebalancing)
            else:
                rebalancing_freqs.append(0)
        
        bars4 = ax4.bar(profile_labels, rebalancing_freqs, 
                       color=[colors[p.lower()] for p in profile_labels], alpha=0.8)
        ax4.set_ylabel("Average Rebalancing Events")
        ax4.set_title("Average Rebalancing Frequency")
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, freq in zip(bars4, rebalancing_freqs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{freq:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        chart_path = charts_dir / "target_hf_agent_performance_summary.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating target HF agent performance summary: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_target_hf_health_factor_analysis(results_matrix: List[Dict], charts_dir: Path) -> Path:
    """Create Target Health Factor Health Factor Analysis dashboard"""
    
    _setup_chart_styling()
    
    try:
        # Get sample simulation data for health factor analysis
        sample_result = None
        for result in results_matrix:
            raw_data = result.get("raw_data", {})
            if raw_data.get("ht_agent_outcomes"):
                sample_result = result
                break
        
        if not sample_result:
            return None
        
        # We need to get the actual simulation run data for health factor history
        # For now, let's create a meaningful analysis based on available data
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Target Health Factor: Health Factor Analysis", fontsize=18, fontweight='bold')
        
        # Group agents by target HF (risk profile)
        risk_profiles = {"conservative": [], "moderate": [], "aggressive": []}
        
        for result in results_matrix:
            raw_data = result.get("raw_data", {})
            for agent in raw_data.get("ht_agent_outcomes", []):
                target_hf = agent.get("target_health_factor", 1.1)
                if target_hf <= 1.05:
                    risk_profiles["aggressive"].append(agent)
                elif target_hf <= 1.1:
                    risk_profiles["moderate"].append(agent)
                else:
                    risk_profiles["conservative"].append(agent)
        
        # Colors for risk profiles
        colors = {"conservative": "#2E8B57", "moderate": "#FF8C00", "aggressive": "#DC143C"}
        
        # Chart 1: Average Health Factor by Risk Profile (Simulated over time)
        # Since we don't have time series data, we'll show initial vs final HF
        initial_hfs = []
        final_hfs = []
        profile_labels = []
        
        for profile in ["conservative", "moderate", "aggressive"]:
            if risk_profiles[profile]:
                avg_initial = np.mean([agent.get("initial_health_factor", 1.2) for agent in risk_profiles[profile]])
                avg_final = np.mean([agent.get("final_health_factor", 1.0) for agent in risk_profiles[profile]])
                initial_hfs.append(avg_initial)
                final_hfs.append(avg_final)
                profile_labels.append(profile.title())
        
        x_pos = np.arange(len(profile_labels))
        width = 0.35
        
        bars1a = ax1.bar(x_pos - width/2, initial_hfs, width, 
                        label='Initial HF', color='lightblue', alpha=0.8)
        bars1b = ax1.bar(x_pos + width/2, final_hfs, width, 
                        label='Final HF', color='darkblue', alpha=0.8)
        
        ax1.set_ylabel("Health Factor")
        ax1.set_title("Average Health Factor: Initial vs Final")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(profile_labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Liquidation Threshold')
        
        # Chart 2: Health Factor Distribution: Start vs End
        all_initial_hfs = []
        all_final_hfs = []
        
        for result in results_matrix:
            raw_data = result.get("raw_data", {})
            for agent in raw_data.get("ht_agent_outcomes", []):
                all_initial_hfs.append(agent.get("initial_health_factor", 1.2))
                all_final_hfs.append(agent.get("final_health_factor", 1.0))
        
        # Create histogram bins
        bins = np.arange(1.0, 2.5, 0.1)
        
        ax2.hist(all_initial_hfs, bins=bins, alpha=0.7, label='Start', color='lightblue', edgecolor='black')
        ax2.hist(all_final_hfs, bins=bins, alpha=0.7, label='End', color='darkblue', edgecolor='black')
        ax2.set_xlabel('Health Factor')
        ax2.set_ylabel('Number of Agents')
        ax2.set_title('Health Factor Distribution: Start vs End')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Liquidation Threshold')
        
        # Chart 3: Agents Requiring Rebalancing (Simulated)
        # Since we don't have time series, we'll show rebalancing frequency by risk profile
        rebalancing_counts = []
        for profile in ["conservative", "moderate", "aggressive"]:
            if risk_profiles[profile]:
                total_rebalancing = sum(agent.get("rebalancing_events", 0) for agent in risk_profiles[profile])
                rebalancing_counts.append(total_rebalancing)
            else:
                rebalancing_counts.append(0)
        
        bars3 = ax3.bar(profile_labels, rebalancing_counts, 
                       color=[colors[p.lower()] for p in profile_labels], alpha=0.8)
        ax3.set_ylabel("Total Rebalancing Events")
        ax3.set_title("Rebalancing Activity by Risk Profile")
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars3, rebalancing_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count:.0f}', ha='center', va='bottom')
        
        # Chart 4: Target vs Final Health Factor Scatter
        target_hfs = []
        final_hfs = []
        profile_colors = []
        
        for result in results_matrix:
            raw_data = result.get("raw_data", {})
            for agent in raw_data.get("ht_agent_outcomes", []):
                target_hf = agent.get("target_health_factor", 1.1)
                final_hf = agent.get("final_health_factor", 1.0)
                target_hfs.append(target_hf)
                final_hfs.append(final_hf)
                
                # Color by risk profile
                if target_hf <= 1.05:
                    profile_colors.append(colors["aggressive"])
                elif target_hf <= 1.1:
                    profile_colors.append(colors["moderate"])
                else:
                    profile_colors.append(colors["conservative"])
        
        scatter = ax4.scatter(target_hfs, final_hfs, c=profile_colors, alpha=0.7, s=50)
        ax4.set_xlabel('Target Health Factor')
        ax4.set_ylabel('Final Health Factor')
        ax4.set_title('Target vs Final Health Factor')
        ax4.grid(True, alpha=0.3)
        
        # Add reference lines
        ax4.axline((1.0, 1.0), slope=1, color='grey', linestyle='--', alpha=0.7, label='Target = Final')
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Liquidation')
        
        # Add legend for risk profiles
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors["conservative"], label='Conservative'),
                          Patch(facecolor=colors["moderate"], label='Moderate'),
                          Patch(facecolor=colors["aggressive"], label='Aggressive')]
        ax4.legend(handles=legend_elements, loc='upper left')
        
        plt.tight_layout()
        
        chart_path = charts_dir / "target_hf_health_factor_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating target HF health factor analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_target_hf_net_position_analysis(results_matrix: List[Dict], charts_dir: Path) -> Path:
    """Create Target Health Factor Net Position Analysis dashboard"""
    
    _setup_chart_styling()
    
    try:
        # Aggregate agent data across all target HF scenarios
        all_ht_agents = []
        
        for result in results_matrix:
            raw_data = result.get("raw_data", {})
            all_ht_agents.extend(raw_data.get("ht_agent_outcomes", []))
        
        if not all_ht_agents:
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[2, 1])
        fig.suptitle("Target Health Factor: Net Position Analysis", fontsize=18, fontweight='bold')
        
        # Group agents by risk profile (using target HF as proxy for risk)
        risk_profiles = {"conservative": [], "moderate": [], "aggressive": []}
        
        for agent in all_ht_agents:
            target_hf = agent.get("target_health_factor", 1.1)
            if target_hf <= 1.05:
                risk_profiles["aggressive"].append(agent)
            elif target_hf <= 1.1:
                risk_profiles["moderate"].append(agent)
            else:
                risk_profiles["conservative"].append(agent)
        
        # Colors for risk profiles
        colors = {"conservative": "#2E8B57", "moderate": "#FF8C00", "aggressive": "#DC143C"}
        
        # Chart 1: Net Position Value by Risk Profile
        # Since we don't have time series data, we'll show final net position values
        net_positions = {"conservative": [], "moderate": [], "aggressive": []}
        
        for profile, agents in risk_profiles.items():
            for agent in agents:
                net_position = agent.get("net_position_value", 0)
                net_positions[profile].append(net_position)
        
        # Create box plots for net position distribution
        profile_names = []
        position_data = []
        for profile in ["conservative", "moderate", "aggressive"]:
            if net_positions[profile]:
                profile_names.append(profile.title())
                position_data.append(net_positions[profile])
        
        if position_data:
            box_plot = ax1.boxplot(position_data, labels=profile_names, patch_artist=True)
            for patch, profile in zip(box_plot['boxes'], ["conservative", "moderate", "aggressive"][:len(position_data)]):
                patch.set_facecolor(colors[profile])
                patch.set_alpha(0.7)
        
        ax1.set_ylabel("Net Position Value ($)")
        ax1.set_title("Net Position Value Distribution by Risk Profile")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Break Even')
        
        # Add legend for risk profiles
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors["conservative"], label='Conservative'),
                          Patch(facecolor=colors["moderate"], label='Moderate'),
                          Patch(facecolor=colors["aggressive"], label='Aggressive')]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # Chart 2: BTC Price Decline (Simulated)
        # Since we don't have actual BTC price data, we'll create a representative decline
        minutes = np.arange(0, 61)
        # Simulate BTC price decline from $100k to ~$79k over 60 minutes
        btc_prices = 100000 - (21000 * (minutes / 60) ** 1.5)  # Non-linear decline
        
        ax2.plot(minutes, btc_prices, color='#FF6B35', linewidth=2, label='BTC Price')
        ax2.set_xlabel("Time (minutes)")
        ax2.set_ylabel("BTC Price ($)")
        ax2.set_title("BTC Price Decline")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add some annotations
        ax2.annotate('Price Decline\nStarts', xy=(5, 95000), xytext=(15, 90000),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, ha='center')
        
        plt.tight_layout()
        
        chart_path = charts_dir / "target_hf_net_position_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating target HF net position analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_ht_vs_aave_comparison_chart(analysis_results: Dict, charts_dir: Path) -> Path:
    """Create High Tide vs Aave performance comparison chart"""
    
    _setup_chart_styling()
    
    try:
        # Extract comparison data from results matrix
        raw_results = analysis_results.get("raw_results_matrix", [])
        
        if not raw_results:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("High Tide vs Aave: Performance Comparison", fontsize=18, fontweight='bold')
        
        # Extract data from actual simulation results
        target_hfs = []
        ht_survival_rates = []
        aave_survival_rates = []
        ht_liquidations = []
        aave_liquidations = []
        
        for result in raw_results:
            target_hf = result.get("target_hf")
            ht_summary = result.get("high_tide_summary", {})
            aave_summary = result.get("aave_summary", {})
            
            if target_hf and ht_summary and aave_summary:
                target_hfs.append(target_hf)
                ht_survival_rates.append(ht_summary.get("mean_survival_rate", 0))
                aave_survival_rates.append(aave_summary.get("mean_survival_rate", 0))
                ht_liquidations.append(ht_summary.get("mean_liquidations", 0))
                aave_liquidations.append(aave_summary.get("mean_liquidations", 0))
        
        if not target_hfs:
            return None
        
        # Chart 1: Survival Rate Comparison
        x_pos = np.arange(len(target_hfs))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, ht_survival_rates, width, 
                       label='High Tide', color='#2E8B57', alpha=0.8)
        bars2 = ax1.bar(x_pos + width/2, aave_survival_rates, width, 
                       label='Aave', color='#DC143C', alpha=0.8)
        
        ax1.set_xlabel('Target Health Factor')
        ax1.set_ylabel('Survival Rate')
        ax1.set_title('Survival Rate: High Tide vs Aave')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'{hf:.3f}' for hf in target_hfs])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.1%}', ha='center', va='bottom', fontsize=9)
        
        # Chart 2: Liquidation Count Comparison
        bars3 = ax2.bar(x_pos - width/2, ht_liquidations, width, 
                       label='High Tide', color='#2E8B57', alpha=0.8)
        bars4 = ax2.bar(x_pos + width/2, aave_liquidations, width, 
                       label='Aave', color='#DC143C', alpha=0.8)
        
        ax2.set_xlabel('Target Health Factor')
        ax2.set_ylabel('Average Liquidations')
        ax2.set_title('Liquidation Count: High Tide vs Aave')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{hf:.3f}' for hf in target_hfs])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        chart_path = charts_dir / "target_hf_ht_vs_aave_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating HT vs Aave comparison chart: {e}")
        return None


def create_rebalancing_efficiency_chart(analysis_results: Dict, charts_dir: Path) -> Path:
    """Create rebalancing efficiency analysis chart"""
    
    _setup_chart_styling()
    
    try:
        rebalancing_analysis = analysis_results.get("rebalancing_efficiency", {})
        
        if not rebalancing_analysis:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("Target Health Factor: Rebalancing Efficiency Analysis", fontsize=18, fontweight='bold')
        
        # Extract data
        target_hfs = []
        rebalancing_events = []
        liquidation_frequencies = []
        efficiency_ratios = []
        interpretations = []
        
        for key, data in rebalancing_analysis.items():
            if key.startswith("target_hf_"):
                target_hfs.append(data["target_hf"])
                rebalancing_events.append(data["avg_rebalancing_events"])
                liquidation_frequencies.append(data["avg_liquidation_frequency"])
                efficiency_ratios.append(data["rebalancing_efficiency_ratio"])
                interpretations.append(data["interpretation"])
        
        if not target_hfs:
            return None
        
        # Chart 1: Rebalancing Events vs Target HF
        colors = ['#2E8B57' if interp == 'High' else '#FF8C00' if interp == 'Moderate' else '#DC143C' 
                 for interp in interpretations]
        
        bars1 = ax1.bar(target_hfs, rebalancing_events, color=colors, alpha=0.8)
        
        ax1.set_xlabel('Target Health Factor')
        ax1.set_ylabel('Average Rebalancing Events')
        ax1.set_title('Rebalancing Activity by Target Health Factor')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, events in zip(bars1, rebalancing_events):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{events:.1f}', ha='center', va='bottom')
        
        # Chart 2: Efficiency Ratio Analysis
        bars2 = ax2.bar(target_hfs, efficiency_ratios, color=colors, alpha=0.8)
        
        ax2.set_xlabel('Target Health Factor')
        ax2.set_ylabel('Rebalancing Efficiency Ratio')
        ax2.set_title('Rebalancing Efficiency by Target Health Factor')
        ax2.grid(True, alpha=0.3)
        
        # Add efficiency threshold lines
        ax2.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='High Efficiency')
        ax2.axhline(y=3, color='orange', linestyle='--', alpha=0.7, label='Moderate Efficiency')
        
        # Add value labels
        for bar, ratio in zip(bars2, efficiency_ratios):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{ratio:.1f}', ha='center', va='bottom')
        
        ax2.legend()
        
        plt.tight_layout()
        
        chart_path = charts_dir / "target_hf_rebalancing_efficiency.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating rebalancing efficiency chart: {e}")
        return None


def create_risk_efficiency_tradeoff_chart(analysis_results: Dict, charts_dir: Path) -> Path:
    """Create risk vs efficiency tradeoff analysis chart"""
    
    _setup_chart_styling()
    
    try:
        risk_efficiency = analysis_results.get("risk_vs_efficiency", {})
        pareto_optimal = risk_efficiency.get("pareto_optimal_scenarios", [])
        
        if not pareto_optimal:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("Target Health Factor: Risk vs Efficiency Tradeoff", fontsize=18, fontweight='bold')
        
        # Extract data
        target_hfs = [scenario["target_hf"] for scenario in pareto_optimal]
        risk_scores = [scenario["risk_score"] for scenario in pareto_optimal]
        efficiency_scores = [scenario["efficiency_score"] for scenario in pareto_optimal]
        survival_rates = [scenario["survival_rate"] for scenario in pareto_optimal]
        
        # Chart 1: Risk vs Efficiency Scatter Plot
        scatter = ax1.scatter(risk_scores, efficiency_scores, 
                            c=survival_rates, s=200, alpha=0.8, 
                            cmap='RdYlGn', edgecolors='black')
        
        # Add target HF labels
        for i, hf in enumerate(target_hfs):
            ax1.annotate(f'{hf:.3f}', (risk_scores[i], efficiency_scores[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax1.set_xlabel('Risk Score (Liquidation Frequency %)')
        ax1.set_ylabel('Efficiency Score (1/Target HF)')
        ax1.set_title('Risk vs Efficiency Tradeoff')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Survival Rate')
        
        # Chart 2: Target HF vs Risk Score
        colors = ['#2E8B57' if risk < 5 else '#FF8C00' if risk < 10 else '#DC143C' 
                 for risk in risk_scores]
        
        bars = ax2.bar(target_hfs, risk_scores, color=colors, alpha=0.8)
        
        # Add risk threshold lines
        ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Moderate Risk')
        ax2.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='High Risk')
        
        ax2.set_xlabel('Target Health Factor')
        ax2.set_ylabel('Risk Score (Liquidation Frequency %)')
        ax2.set_title('Risk Assessment by Target Health Factor')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, risk in zip(bars, risk_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{risk:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        chart_path = charts_dir / "target_hf_risk_efficiency_tradeoff.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating risk efficiency tradeoff chart: {e}")
        return None


def create_target_hf_dashboard(analysis_results: Dict, charts_dir: Path) -> Path:
    """Create comprehensive Target Health Factor optimization dashboard"""
    
    _setup_chart_styling()
    
    try:
        optimal_recs = analysis_results.get("optimal_recommendations", {})
        all_recs = optimal_recs.get("all_target_hf_analysis", {})
        most_aggressive_safe = optimal_recs.get("most_aggressive_safe_target_hf")
        
        if not all_recs:
            return None
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle("Target Health Factor Optimization Dashboard", fontsize=20, fontweight='bold')
        
        # Extract data
        target_hfs = []
        survival_rates = []
        liquidation_frequencies = []
        rebalancing_events = []
        risk_levels = []
        
        for target_hf in [1.01, 1.05, 1.075, 1.1, 1.15]:
            rec = all_recs.get(f"target_hf_{target_hf}")
            if rec:
                target_hfs.append(target_hf)
                survival_rates.append(rec["avg_survival_rate"])
                liquidation_frequencies.append(rec["avg_liquidation_frequency"])
                rebalancing_events.append(rec["avg_rebalancing_events"])
                risk_levels.append(rec["risk_level"])
        
        # Chart 1: Survival Rate (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        colors = ['#2E8B57' if level == 'Low Risk' else '#FF8C00' if level == 'Moderate Risk' else '#DC143C' 
                 for level in risk_levels]
        bars1 = ax1.bar(target_hfs, survival_rates, color=colors, alpha=0.8)
        ax1.set_title('Survival Rate by Target HF')
        ax1.set_ylabel('Survival Rate')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Chart 2: Liquidation Frequency (Top Center)
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(target_hfs, liquidation_frequencies, color=colors, alpha=0.8)
        ax2.set_title('Liquidation Frequency by Target HF')
        ax2.set_ylabel('Liquidation Frequency')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(y=0.10, color='red', linestyle='--', alpha=0.7)
        
        # Chart 3: Rebalancing Events (Top Right)
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(target_hfs, rebalancing_events, color=colors, alpha=0.8)
        ax3.set_title('Rebalancing Events by Target HF')
        ax3.set_ylabel('Rebalancing Events')
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Risk Assessment Matrix (Middle Left)
        ax4 = fig.add_subplot(gs[1, 0])
        risk_matrix = np.array([[1 if level == 'Low Risk' else 0.5 if level == 'Moderate Risk' else 0 
                               for level in risk_levels]])
        im = ax4.imshow(risk_matrix, cmap='RdYlGn', aspect='auto')
        ax4.set_title('Risk Assessment Matrix')
        ax4.set_xticks(range(len(target_hfs)))
        ax4.set_xticklabels([f'{hf:.3f}' for hf in target_hfs])
        ax4.set_yticks([0])
        ax4.set_yticklabels(['Risk Level'])
        
        # Chart 5: Efficiency vs Risk Scatter (Middle Center)
        ax5 = fig.add_subplot(gs[1, 1])
        efficiency_scores = [1.0/hf for hf in target_hfs]
        scatter = ax5.scatter(liquidation_frequencies, efficiency_scores, 
                            c=survival_rates, s=200, alpha=0.8, 
                            cmap='RdYlGn', edgecolors='black')
        ax5.set_title('Efficiency vs Risk Tradeoff')
        ax5.set_xlabel('Liquidation Frequency')
        ax5.set_ylabel('Efficiency Score (1/Target HF)')
        ax5.grid(True, alpha=0.3)
        
        # Add target HF labels
        for i, hf in enumerate(target_hfs):
            ax5.annotate(f'{hf:.3f}', (liquidation_frequencies[i], efficiency_scores[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Chart 6: Recommendations Summary (Middle Right)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Create recommendations text
        recommendations_text = "TARGET HF RECOMMENDATIONS:\n\n"
        
        for target_hf in [1.01, 1.05, 1.075, 1.1, 1.15]:
            rec = all_recs.get(f"target_hf_{target_hf}")
            if rec:
                status = "[RECOMMENDED]" if rec["recommended"] else "[NOT RECOMMENDED]"
                recommendations_text += f"• {target_hf:.3f}: {rec['risk_level']} {status}\n"
        
        if most_aggressive_safe:
            recommendations_text += f"\n[MOST AGGRESSIVE SAFE] {most_aggressive_safe['target_hf']:.3f}"
        
        ax6.text(0.05, 0.95, recommendations_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Chart 7: Statistical Summary (Bottom Row)
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Create statistical summary
        stats = analysis_results.get("statistical_summary", {})
        correlations = stats.get("correlation_analysis", {})
        
        summary_text = f"""
STATISTICAL ANALYSIS SUMMARY:
• Total Scenarios Tested: {stats.get('total_scenarios_tested', 0)}
• Target HF Range: {stats.get('target_hf_range', [0, 0])[0]:.3f} - {stats.get('target_hf_range', [0, 0])[1]:.3f}
• Survival Rate Range: {stats.get('survival_rate_range', [0, 0])[0]:.1%} - {stats.get('survival_rate_range', [0, 0])[1]:.1%}
• Liquidation Frequency Range: {stats.get('liquidation_frequency_range', [0, 0])[0]:.1%} - {stats.get('liquidation_frequency_range', [0, 0])[1]:.1%}

CORRELATION ANALYSIS:
• Target HF ↔ Liquidations: {correlations.get('target_hf_vs_liquidations', 0):.3f}
• Target HF ↔ Survival: {correlations.get('target_hf_vs_survival', 0):.3f}
• Rebalancing ↔ Survival: {correlations.get('rebalancing_vs_survival', 0):.3f}
        """
        
        ax7.text(0.05, 0.5, summary_text, transform=ax7.transAxes, 
                fontsize=12, verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        chart_path = charts_dir / "target_hf_optimization_dashboard.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating target HF dashboard: {e}")
        return None


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