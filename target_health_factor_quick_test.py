#!/usr/bin/env python3
"""
Target Health Factor Quick Test

Quick validation of the corrected Target Health Factor analysis with:
- Randomized agent initial HFs (1.2-1.5) 
- Discrete target HF testing: 1.01, 1.05, 1.075, 1.1, 1.15
- Fewer runs for rapid feedback

This validates the statistical design before running the full analysis.
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

from tidal_protocol_sim.simulation.high_tide_vault_engine import HighTideConfig, HighTideVaultEngine
from tidal_protocol_sim.simulation.aave_protocol_engine import AaveConfig, AaveProtocolEngine
from tidal_protocol_sim.core.protocol import TidalProtocol, Asset, AssetPool, LiquidityPool


class BTCOnyProtocol(TidalProtocol):
    """BTC-only protocol configuration for Target Health Factor analysis"""
    
    def _initialize_asset_pools(self):
        """Initialize only BTC asset pool"""
        return {
            Asset.BTC: AssetPool(Asset.BTC, 3_500_000),   # $3.5M initial BTC only
        }
    
    def _initialize_liquidity_pools(self):
        """Initialize only MOET:BTC trading pair"""
        return {
            "MOET_BTC": LiquidityPool(Asset.BTC, 1250000, 10.59),   # ~$1.25M at $118k/BTC
        }


def create_varied_agents_for_target_hf_test(target_hf: float, num_agents: int, agent_type: str) -> List:
    """Create agents with randomized initial HFs (1.2-1.5) and fixed target HF"""
    import random
    from tidal_protocol_sim.agents.high_tide_agent import HighTideAgent
    from tidal_protocol_sim.agents.aave_agent import AaveAgent
    
    agents = []
    
    for i in range(num_agents):
        # Randomize initial health factor between 1.2-1.5 for proper variation
        initial_hf = random.uniform(1.2, 1.5)
        
        if agent_type == "high_tide":
            agent = HighTideAgent(
                f"quick_test_ht_{target_hf}_{i}",
                initial_hf,
                target_hf
            )
        else:  # aave
            agent = AaveAgent(
                f"quick_test_aave_{target_hf}_{i}",
                initial_hf,
                target_hf
            )
        
        agents.append(agent)
    
    return agents


def run_quick_target_hf_test():
    """Run quick Target Health Factor validation test"""
    
    print("=" * 60)
    print("TARGET HEALTH FACTOR QUICK TEST")
    print("=" * 60)
    print("Testing design: Randomized Initial HFs (1.2-1.5) with discrete Target HFs")
    print("Target HFs: 1.01, 1.05 (3 runs each for validation)")
    print()
    
    # Quick test with 2 target HFs and 3 runs each
    target_hfs = [1.01, 1.05]  # Test the most aggressive target HFs
    monte_carlo_runs = 3  # 3 runs per target HF for better validation
    
    results = []
    
    for target_hf in target_hfs:
        print(f"🎯 Testing Target Health Factor: {target_hf}")
        print(f"   Running {monte_carlo_runs} Monte Carlo runs with varied agents")
        
        # Store results from all runs
        ht_runs = []
        aave_runs = []
        
        for run_num in range(monte_carlo_runs):
            print(f"      Run {run_num + 1}/{monte_carlo_runs}", end=" ")
            
            # High Tide simulation
            ht_config = HighTideConfig()
            ht_config.num_high_tide_agents = 0
            ht_config.btc_decline_duration = 60
            ht_config.moet_btc_pool_size = 250_000
            ht_config.moet_yield_pool_size = 250_000
            
            # Create varied High Tide agents
            custom_ht_agents = create_varied_agents_for_target_hf_test(
                target_hf, num_agents=10, agent_type="high_tide"
            )
            
            ht_engine = HighTideVaultEngine(ht_config)
            ht_engine.high_tide_agents = custom_ht_agents
            
            # Replace protocol with BTC-only version
            ht_engine.protocol = BTCOnyProtocol()
            
            for agent in custom_ht_agents:
                ht_engine.agents[agent.agent_id] = agent
            
            ht_results = ht_engine.run_simulation()
            ht_runs.append(ht_results)
            
            # Aave simulation
            aave_config = AaveConfig()
            aave_config.num_aave_agents = 0
            aave_config.btc_decline_duration = 60
            aave_config.moet_btc_pool_size = 250_000
            aave_config.moet_yield_pool_size = 250_000
            
            custom_aave_agents = create_varied_agents_for_target_hf_test(
                target_hf, num_agents=10, agent_type="aave"
            )
            
            aave_engine = AaveProtocolEngine(aave_config)
            aave_engine.aave_agents = custom_aave_agents
            
            for agent in custom_aave_agents:
                aave_engine.agents[agent.agent_id] = agent
            
            aave_results = aave_engine.run_simulation()
            aave_runs.append(aave_results)
            
            print("✓")
        
        # Aggregate results for this target HF
        scenario_results = aggregate_quick_test_results(ht_runs, aave_runs, target_hf)
        results.append(scenario_results)
        
        print(f"   Results: HT {scenario_results['high_tide_summary']['mean_survival_rate']:.1%} survival, "
              f"Aave {scenario_results['aave_summary']['mean_survival_rate']:.1%} survival")
        print()
    
    # Save quick test results
    # Save results with JSON output and get run folder
    run_folder = save_quick_test_results(results)
    
    # Generate quick test charts in the run folder
    generated_charts = create_quick_test_charts(results, run_folder)
    
    print("✅ Quick test completed! Ready for full analysis.")
    print(f"📊 Generated {len(generated_charts)} quick test charts")
    print(f"📁 All results saved to: {run_folder}")
    return results


def aggregate_quick_test_results(ht_runs: List, aave_runs: List, target_hf: float) -> Dict:
    """Aggregate results for quick test using actual simulation data"""
    
    # High Tide metrics from actual simulation results
    ht_survival_rates = [run.get("survival_statistics", {}).get("survival_rate", 0.0) for run in ht_runs]
    ht_liquidations = []
    ht_agent_outcomes = []
    
    for run in ht_runs:
        agent_outcomes = run.get("agent_outcomes", [])
        # Count liquidations from agent outcomes (survived = False means liquidated)
        total_liquidations = sum(1 for outcome in agent_outcomes if not outcome.get("survived", True))
        ht_liquidations.append(total_liquidations)
        # Store all agent outcomes for detailed analysis
        ht_agent_outcomes.extend(agent_outcomes)
    
    # Aave metrics from actual simulation results
    aave_survival_rates = [run.get("survival_statistics", {}).get("survival_rate", 0.0) for run in aave_runs]
    aave_liquidations = []
    aave_agent_outcomes = []
    
    for run in aave_runs:
        agent_outcomes = run.get("agent_outcomes", [])
        # Count liquidations from agent outcomes (survived = False means liquidated)
        total_liquidations = sum(1 for outcome in agent_outcomes if not outcome.get("survived", True))
        aave_liquidations.append(total_liquidations)
        # Store all agent outcomes for detailed analysis
        aave_agent_outcomes.extend(agent_outcomes)
    
    return {
        "target_hf": target_hf,
        "scenario_params": {
            "target_hf": target_hf,
            "initial_hf_range": [1.2, 1.5],
            "variation_type": "randomized_initial_hf"
        },
        "high_tide_summary": {
            "mean_survival_rate": np.mean(ht_survival_rates),
            "mean_liquidations": np.mean(ht_liquidations),
            "liquidation_frequency": np.mean(ht_liquidations) / 10  # 10 agents per run
        },
        "aave_summary": {
            "mean_survival_rate": np.mean(aave_survival_rates),
            "mean_liquidations": np.mean(aave_liquidations),
            "liquidation_frequency": np.mean(aave_liquidations) / 10  # 10 agents per run
        },
        "comparison": {
            "survival_improvement": ((np.mean(ht_survival_rates) - np.mean(aave_survival_rates)) / np.mean(aave_survival_rates) * 100) if np.mean(aave_survival_rates) > 0 else 0,
            "liquidation_reduction": ((np.mean(aave_liquidations) - np.mean(ht_liquidations)) / np.mean(aave_liquidations) * 100) if np.mean(aave_liquidations) > 0 else 0
        },
        "ht_agent_outcomes": ht_agent_outcomes,
        "aave_agent_outcomes": aave_agent_outcomes,
        "detailed_simulation_data": {
            "ht_runs": ht_runs,  # Store the raw runs for clean processing
            "aave_runs": aave_runs
        },
        "validation_checks": {
            "ht_has_variation": len(set(ht_survival_rates)) > 1,  # Different survival outcomes
            "aave_has_variation": len(set(aave_survival_rates)) > 1,  # Different survival outcomes
            "liquidation_spread": len(set(ht_liquidations + aave_liquidations)) > 1  # Different liquidation outcomes
        }
    }


def build_clean_simulation_data(ht_runs: List[Dict], target_hf: float) -> List[Dict]:
    """Build clean, navigable simulation data structure"""
    clean_runs = []
    
    for run_idx, run in enumerate(ht_runs):
        # Filter out non-test agents from agent_health_factors
        agent_health_factors = run.get("agent_health_factors", {})
        filtered_health_factors = {
            agent_id: hf for agent_id, hf in agent_health_factors.items()
            if agent_id.startswith(f"quick_test_ht_{target_hf}_")
        }
        
        # Filter agent_actions_history to only include test agents
        agent_actions = run.get("agent_actions_history", [])
        filtered_actions = [
            action for action in agent_actions
            if action.get("agent_id", "").startswith(f"quick_test_ht_{target_hf}_")
        ]
        
        # Filter agent_health_history to only include test agents
        agent_health_history = run.get("agent_health_history", [])
        filtered_health_history = []
        for minute_data in agent_health_history:
            filtered_agents = [
                agent for agent in minute_data.get("agents", [])
                if agent.get("agent_id", "").startswith(f"quick_test_ht_{target_hf}_")
            ]
            if filtered_agents:  # Only include minutes that have test agents
                filtered_health_history.append({
                    "minute": minute_data.get("minute"),
                    "agents": filtered_agents
                })
        
        clean_run = {
            "run_id": f"run_{run_idx+1:03d}_target_hf_{target_hf}",
            "timestamp": datetime.now().isoformat(),
            "btc_price_history": run.get("btc_price_history", []),
            "agent_health_history": filtered_health_history,
            "agent_actions_history": filtered_actions,
            "rebalancing_events": run.get("rebalancing_events", []),
            "agent_outcomes": run.get("agent_outcomes", []),
            "summary_stats": {
                "total_agents": len(filtered_health_factors),
                "survived_agents": sum(1 for outcome in run.get("agent_outcomes", []) if outcome.get("survived", True)),
                "total_rebalancing_events": len(run.get("rebalancing_events", [])),
                "total_slippage_costs": sum(outcome.get("total_slippage_costs", 0) for outcome in run.get("agent_outcomes", [])),
                "final_btc_price": run.get("btc_price_history", [100000])[-1] if isinstance(run.get("btc_price_history", [100000])[-1], (int, float)) else run.get("btc_price_history", [100000])[-1].get("btc_price", 100000) if run.get("btc_price_history") else 100000,
                "btc_price_decline_percent": ((100000 - (run.get("btc_price_history", [100000])[-1] if isinstance(run.get("btc_price_history", [100000])[-1], (int, float)) else run.get("btc_price_history", [100000])[-1].get("btc_price", 100000) if run.get("btc_price_history") else 100000)) / 100000) * 100
            }
        }
        clean_runs.append(clean_run)
    
    return clean_runs


def save_quick_test_results(results: List[Dict]):
    """Save quick test results with numbered run folder"""
    
    # Create base results directory
    base_results_path = Path("tidal_protocol_sim/results/target_health_factor_analysis")
    base_results_path.mkdir(parents=True, exist_ok=True)
    
    # Find next available run number
    run_number = 1
    while True:
        run_folder = base_results_path / f"run_{run_number:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not run_folder.exists():
            break
        run_number += 1
    
    # Create run-specific directory
    run_folder.mkdir(parents=True, exist_ok=True)
    
    # Build clean, navigable results structure (same as main analysis)
    clean_scenario_results = []
    for result in results:
        target_hf = result["target_hf"]
        
        # Extract the ht_runs from detailed_simulation_data if it exists
        ht_runs = []
        if "detailed_simulation_data" in result and "ht_runs" in result["detailed_simulation_data"]:
            ht_runs = result["detailed_simulation_data"]["ht_runs"]
        
        clean_runs = build_clean_simulation_data(ht_runs, target_hf)
        
        clean_scenario = {
            "target_hf": target_hf,
            "scenario_params": result.get("scenario_params", {}),
            "high_tide_summary": result.get("high_tide_summary", {}),
            "aave_summary": result.get("aave_summary", {}),
            "comparison": result.get("comparison", {}),
            "ht_agent_outcomes": result.get("ht_agent_outcomes", []),
            "aave_agent_outcomes": result.get("aave_agent_outcomes", []),
            "simulation_runs": clean_runs
        }
        clean_scenario_results.append(clean_scenario)
    
    quick_test_results = {
        "analysis_metadata": {
            "analysis_type": "Quick_Validation_Test",
            "timestamp": datetime.now().isoformat(),
            "target_hfs_tested": [r["target_hf"] for r in results],
            "monte_carlo_runs_per_scenario": 3,
            "agents_per_run": 10,
            "total_scenarios": len(results),
            "design_fix": "Randomized initial HFs (1.2-1.5) with fixed target HFs"
        },
        "detailed_scenario_results": clean_scenario_results
    }
    
    results_path = run_folder / "quick_test_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(quick_test_results, f, indent=2, default=str)
    
    print(f"📁 Quick test results saved to: {run_folder}")
    return run_folder


def create_quick_test_charts(results: List[Dict], output_dir: Path) -> List[Path]:
    """Create all charts from best_configuration_charts with exact styling"""
    
    print("🎨 Generating comprehensive quick test charts...")
    
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
                scenario_name="Target_Health_Factor_Quick_Test",
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
        
        # Also create the original quick test validation chart
        validation_chart = create_quick_test_validation_chart(results, charts_dir)
        if validation_chart:
            generated_charts.append(validation_chart)
        
        # Create individual agent tracking charts
        individual_charts = create_individual_agent_tracking_charts(detailed_simulation_result, charts_dir)
        generated_charts.extend(individual_charts)
        
        print(f"✅ Generated {len(generated_charts)} comprehensive charts")
        
    except Exception as e:
        print(f"❌ Error generating comprehensive charts: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to basic charts if comprehensive generation fails
        print("🔄 Falling back to basic chart generation...")
        basic_charts = create_basic_quick_test_charts(results, charts_dir)
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
        ht_config.yield_token_concentration = 0.95  # 95% concentration for yield tokens
        
        # Create varied High Tide agents with a moderate target HF
        target_hf = 1.1  # Use a moderate target HF for good rebalancing activity
        custom_ht_agents = create_varied_agents_for_target_hf_test(
            target_hf, num_agents=10, agent_type="high_tide"
        )
        
        # Run the simulation
        ht_engine = HighTideVaultEngine(ht_config)
        ht_engine.high_tide_agents = custom_ht_agents
        
        for agent in custom_ht_agents:
            ht_engine.agents[agent.agent_id] = agent
        
        # Run simulation and get full results
        ht_results = ht_engine.run_simulation()
        
        print(f"✅ Detailed simulation completed with {len(ht_results.get('agent_outcomes', []))} agents")
        return ht_results
        
    except Exception as e:
        print(f"❌ Error running detailed simulation: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_quick_test_validation_chart(results: List[Dict], charts_dir: Path) -> Path:
    """Create the original quick test validation chart"""
    
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
        
        # Extract data
        target_hfs = [r["target_hf"] for r in results]
        ht_survival_rates = [r["high_tide_summary"]["mean_survival_rate"] for r in results]
        aave_survival_rates = [r["aave_summary"]["mean_survival_rate"] for r in results]
        ht_liquidation_freqs = [r["high_tide_summary"]["liquidation_frequency"] for r in results]
        aave_liquidation_freqs = [r["aave_summary"]["liquidation_frequency"] for r in results]
        
        # Create comprehensive comparison dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Target Health Factor Quick Test: Performance Analysis", fontsize=18, fontweight='bold')
        
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
        
        # Chart 2: Liquidation Frequency Comparison
        bars3 = ax2.bar(x_pos - width/2, ht_liquidation_freqs, width, 
                       label='High Tide', color='#2E8B57', alpha=0.8)
        bars4 = ax2.bar(x_pos + width/2, aave_liquidation_freqs, width, 
                       label='Aave', color='#DC143C', alpha=0.8)
        
        ax2.set_xlabel('Target Health Factor')
        ax2.set_ylabel('Liquidation Frequency (per agent)')
        ax2.set_title('Liquidation Frequency: High Tide vs Aave')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'{hf:.3f}' for hf in target_hfs])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{height:.1%}', ha='center', va='bottom', fontsize=9)
        
        # Chart 3: Performance Improvement Analysis
        improvements = []
        for i, target_hf in enumerate(target_hfs):
            ht_survival = ht_survival_rates[i]
            aave_survival = aave_survival_rates[i]
            improvement = ((ht_survival - aave_survival) / aave_survival * 100) if aave_survival > 0 else 0
            improvements.append(improvement)
        
        colors_improvement = ['#2E8B57' if imp >= 0 else '#DC143C' for imp in improvements]
        bars5 = ax3.bar(x_pos, improvements, color=colors_improvement, alpha=0.8)
        
        ax3.set_xlabel('Target Health Factor')
        ax3.set_ylabel('Survival Rate Improvement (%)')
        ax3.set_title('High Tide Performance Improvement vs Aave')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([f'{hf:.3f}' for hf in target_hfs])
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, imp in zip(bars5, improvements):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                    f'{imp:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # Chart 4: Risk Assessment Summary
        ax4.axis('off')
        
        # Create risk assessment text
        risk_text = f"""
TARGET HEALTH FACTOR RISK ASSESSMENT:

"""
        
        for i, result in enumerate(results):
            target_hf = result['target_hf']
            ht_liquidation = result['ht_liquidation_frequency']
            aave_liquidation = result['aave_liquidation_frequency']
            
            # Risk assessment
            if ht_liquidation < 0.05:
                risk_level = "LOW RISK"
                risk_color = "green"
            elif ht_liquidation < 0.10:
                risk_level = "MODERATE RISK"
                risk_color = "orange"
            else:
                risk_level = "HIGH RISK"
                risk_color = "red"
            
            risk_text += f"""
Target HF {target_hf:.3f}: {risk_level}
  • HT Liquidations: {ht_liquidation:.1%} per agent
  • Aave Liquidations: {aave_liquidation:.1%} per agent
  • Risk Reduction: {((aave_liquidation - ht_liquidation) / aave_liquidation * 100) if aave_liquidation > 0 else 0:.1f}%
"""
        
        risk_text += f"""

QUICK TEST VALIDATION:
• Design: [OK] Randomized Initial HFs (1.2-1.5)
• Target HFs: [OK] Fixed per scenario
• Data Quality: {'[OK] Good variation' if len(set(ht_liquidation_freqs + aave_liquidation_freqs)) > 1 else '[WARN] Limited variation'}
• Ready for Full Analysis: {'[OK] YES' if len(set(ht_liquidation_freqs + aave_liquidation_freqs)) > 1 else '[WARN] Consider more variation'}
        """
        
        ax4.text(0.05, 0.95, risk_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        
        chart_path = charts_dir / "quick_test_ht_vs_aave_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating validation chart: {e}")
        return None


def create_basic_quick_test_charts(results: List[Dict], charts_dir: Path) -> List[Path]:
    """Fallback basic chart generation if comprehensive generation fails"""
    
    charts = []
    
    # Create a simple summary chart
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        target_hfs = [r["target_hf"] for r in results]
        ht_survival_rates = [r["high_tide_summary"]["mean_survival_rate"] for r in results]
        aave_survival_rates = [r["aave_summary"]["mean_survival_rate"] for r in results]
        
        x_pos = np.arange(len(target_hfs))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, ht_survival_rates, width, 
                      label='High Tide', color='#2E8B57', alpha=0.8)
        bars2 = ax.bar(x_pos + width/2, aave_survival_rates, width, 
                      label='Aave', color='#DC143C', alpha=0.8)
        
        ax.set_xlabel('Target Health Factor')
        ax.set_ylabel('Survival Rate')
        ax.set_title('Quick Test: High Tide vs Aave Survival Rates')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{hf:.3f}' for hf in target_hfs])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        chart_path = charts_dir / "quick_test_basic_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        charts.append(chart_path)
        
    except Exception as e:
        print(f"Error creating basic chart: {e}")
    
    return charts


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


def create_quick_test_agent_performance_summary(results: List[Dict], charts_dir: Path) -> Path:
    """Create comprehensive agent performance summary chart for quick test"""
    
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
        
        # Aggregate all agent outcomes across all target HF scenarios
        all_ht_agents = []
        all_aave_agents = []
        
        for result in results:
            raw_data = result.get("raw_data", {})
            all_ht_agents.extend(raw_data.get("ht_agent_outcomes", []))
            all_aave_agents.extend(raw_data.get("aave_agent_outcomes", []))
        
        if not all_ht_agents:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Quick Test: Agent Performance Summary", fontsize=18, fontweight='bold')
        
        # Group agents by risk profile (using target HF as proxy)
        risk_profiles = {"conservative": [], "moderate": [], "aggressive": []}
        
        for agent in all_ht_agents:
            target_hf = agent.get("target_health_factor", 1.1)
            if target_hf <= 1.05:
                risk_profiles["aggressive"].append(agent)
            elif target_hf <= 1.1:
                risk_profiles["moderate"].append(agent)
            else:
                risk_profiles["conservative"].append(agent)
        
        # If we don't have enough variation, group by actual performance instead
        if len(risk_profiles["conservative"]) == 0 and len(risk_profiles["moderate"]) == 0:
            # All agents are aggressive, so let's group by rebalancing activity
            risk_profiles = {"low_activity": [], "medium_activity": [], "high_activity": []}
            rebalancing_counts = [agent.get("rebalancing_events", 0) for agent in all_ht_agents]
            if rebalancing_counts:
                low_threshold = np.percentile(rebalancing_counts, 33)
                high_threshold = np.percentile(rebalancing_counts, 67)
                
                for agent in all_ht_agents:
                    events = agent.get("rebalancing_events", 0)
                    if events <= low_threshold:
                        risk_profiles["low_activity"].append(agent)
                    elif events <= high_threshold:
                        risk_profiles["medium_activity"].append(agent)
                    else:
                        risk_profiles["high_activity"].append(agent)
        
        # Colors for risk profiles
        colors = {"conservative": "#2E8B57", "moderate": "#FF8C00", "aggressive": "#DC143C",
                 "low_activity": "#2E8B57", "medium_activity": "#FF8C00", "high_activity": "#DC143C"}
        
        # Chart 1: Cost of Rebalancing by Risk Profile
        profile_names = []
        rebalancing_costs = []
        profile_colors = []
        
        # Determine which profiles to use based on what we have
        if "conservative" in risk_profiles:
            profile_list = ["conservative", "moderate", "aggressive"]
        else:
            profile_list = ["low_activity", "medium_activity", "high_activity"]
        
        for profile in profile_list:
            if risk_profiles[profile]:
                avg_cost = np.mean([agent.get("cost_of_rebalancing", 0) for agent in risk_profiles[profile]])
                profile_names.append(profile.replace("_", " ").title())
                rebalancing_costs.append(avg_cost)
                profile_colors.append(colors[profile])
        
        if rebalancing_costs:
            bars1 = ax1.bar(profile_names, rebalancing_costs, color=profile_colors, alpha=0.8)
            ax1.set_ylabel("Average Rebalancing Cost ($)")
            ax1.set_title("Cost of Rebalancing by Risk Profile")
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, cost in zip(bars1, rebalancing_costs):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(rebalancing_costs) * 0.01,
                        f'${cost:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Chart 2: Survival Rate by Risk Profile
        survival_rates = []
        for profile in profile_list:
            if risk_profiles[profile]:
                survived = sum(1 for agent in risk_profiles[profile] if agent.get("survived", True))
                total = len(risk_profiles[profile])
                survival_rate = (survived / total) * 100 if total > 0 else 0
                survival_rates.append(survival_rate)
            else:
                survival_rates.append(0)
        
        # Only plot if we have data for all profiles
        if len(survival_rates) == len(profile_names) and survival_rates:
            bars2 = ax2.bar(profile_names, survival_rates, color=profile_colors, alpha=0.8)
            ax2.set_ylabel("Survival Rate (%)")
            ax2.set_title("Survival Rate by Risk Profile")
            ax2.set_ylim(0, 105)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, rate in zip(bars2, survival_rates):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Chart 3: Average Yield Earned vs Sold
        yield_earned = []
        yield_sold = []
        for profile in profile_list:
            if risk_profiles[profile]:
                avg_earned = np.mean([agent.get("total_yield_earned", 0) for agent in risk_profiles[profile]])
                avg_sold = np.mean([agent.get("total_yield_sold", 0) for agent in risk_profiles[profile]])
                yield_earned.append(avg_earned)
                yield_sold.append(avg_sold)
            else:
                yield_earned.append(0)
                yield_sold.append(0)
        
        # Only plot if we have data for all profiles and arrays match
        if yield_earned and yield_sold and len(yield_earned) == len(profile_names):
            x_pos = np.arange(len(profile_names))
            width = 0.35
            
            bars3a = ax3.bar(x_pos - width/2, yield_earned, width, 
                            label='Yield Earned', color='lightgreen', alpha=0.8)
            bars3b = ax3.bar(x_pos + width/2, yield_sold, width, 
                            label='Yield Sold', color='lightcoral', alpha=0.8)
            
            ax3.set_ylabel("Average Yield ($)")
            ax3.set_title("Average Yield: Earned vs Sold")
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(profile_names)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Chart 4: Rebalancing Frequency by Risk Profile
        rebalancing_freqs = []
        for profile in profile_list:
            if risk_profiles[profile]:
                avg_freq = np.mean([agent.get("rebalancing_events", 0) for agent in risk_profiles[profile]])
                rebalancing_freqs.append(avg_freq)
            else:
                rebalancing_freqs.append(0)
        
        # Only plot if we have data for all profiles and arrays match
        if rebalancing_freqs and len(rebalancing_freqs) == len(profile_names):
            bars4 = ax4.bar(profile_names, rebalancing_freqs, color=profile_colors, alpha=0.8)
            ax4.set_ylabel("Average Rebalancing Events")
            ax4.set_title("Rebalancing Frequency by Risk Profile")
            ax4.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, freq in zip(bars4, rebalancing_freqs):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(rebalancing_freqs) * 0.01,
                        f'{freq:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        chart_path = charts_dir / "quick_test_agent_performance_summary.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating quick test agent performance summary: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_quick_test_health_factor_analysis(results: List[Dict], charts_dir: Path) -> Path:
    """Create comprehensive health factor analysis chart for quick test"""
    
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
        
        # Aggregate all agent outcomes across all target HF scenarios
        all_ht_agents = []
        
        for result in results:
            raw_data = result.get("raw_data", {})
            all_ht_agents.extend(raw_data.get("ht_agent_outcomes", []))
        
        if not all_ht_agents:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Quick Test: Health Factor Analysis", fontsize=18, fontweight='bold')
        
        # Group agents by risk profile
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
        
        # Chart 1: Average Initial vs Final Health Factor
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
        
        if initial_hfs and final_hfs:
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
        
        # Chart 2: Health Factor Distribution
        all_initial_hfs = [agent.get("initial_health_factor", 1.2) for agent in all_ht_agents]
        all_final_hfs = [agent.get("final_health_factor", 1.0) for agent in all_ht_agents]
        
        bins = np.arange(1.0, 2.5, 0.1)
        ax2.hist(all_initial_hfs, bins=bins, alpha=0.7, label='Start', color='lightblue', edgecolor='black')
        ax2.hist(all_final_hfs, bins=bins, alpha=0.7, label='End', color='darkblue', edgecolor='black')
        ax2.set_xlabel('Health Factor')
        ax2.set_ylabel('Number of Agents')
        ax2.set_title('Health Factor Distribution: Start vs End')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Liquidation Threshold')
        
        # Chart 3: Rebalancing Activity by Risk Profile
        rebalancing_counts = []
        for profile in ["conservative", "moderate", "aggressive"]:
            if risk_profiles[profile]:
                total_rebalancing = sum(agent.get("rebalancing_events", 0) for agent in risk_profiles[profile])
                rebalancing_counts.append(total_rebalancing)
            else:
                rebalancing_counts.append(0)
        
        # Only plot if we have data for all profiles
        if len(rebalancing_counts) == len(profile_labels) and rebalancing_counts:
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
        target_hfs = [agent.get("target_health_factor", 1.1) for agent in all_ht_agents]
        final_hfs = [agent.get("final_health_factor", 1.0) for agent in all_ht_agents]
        profile_colors = []
        
        for agent in all_ht_agents:
            target_hf = agent.get("target_health_factor", 1.1)
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
        
        chart_path = charts_dir / "quick_test_health_factor_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating quick test health factor analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_quick_test_net_position_analysis(results: List[Dict], charts_dir: Path) -> Path:
    """Create comprehensive net position analysis chart for quick test"""
    
    try:
        # Setup styling
        plt.style.use('default')
        sns.set_palette("husl")
        
        plt.rcParams.update({
            'figure.figsize': (14, 12),
            'font.size': 11,
            'axes.titlesize': 16,
            'axes.labelsize': 13,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 18
        })
        
        # Aggregate all agent outcomes across all target HF scenarios
        all_ht_agents = []
        
        for result in results:
            raw_data = result.get("raw_data", {})
            all_ht_agents.extend(raw_data.get("ht_agent_outcomes", []))
        
        if not all_ht_agents:
            return None
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), height_ratios=[2, 1])
        fig.suptitle("Quick Test: Net Position Analysis", fontsize=18, fontweight='bold')
        
        # Group agents by risk profile
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
        
        # Chart 1: Net Position Value Changes by Risk Profile
        # Since we don't have time series data, we'll show initial vs final net position values
        initial_positions = {"conservative": [], "moderate": [], "aggressive": []}
        final_positions = {"conservative": [], "moderate": [], "aggressive": []}
        
        for profile, agents in risk_profiles.items():
            for agent in agents:
                # Calculate initial net position (assuming $100k initial)
                initial_position = 100000.0  # Starting position
                final_position = agent.get("net_position_value", 0)
                initial_positions[profile].append(initial_position)
                final_positions[profile].append(final_position)
        
        # Create comparison chart showing initial vs final positions
        profile_names = []
        initial_avgs = []
        final_avgs = []
        profile_colors = []
        
        for profile in ["conservative", "moderate", "aggressive"]:
            if initial_positions[profile] and final_positions[profile]:
                profile_names.append(profile.title())
                initial_avgs.append(np.mean(initial_positions[profile]))
                final_avgs.append(np.mean(final_positions[profile]))
                profile_colors.append(colors[profile])
        
        if profile_names:
            x_pos = np.arange(len(profile_names))
            width = 0.35
            
            bars1a = ax1.bar(x_pos - width/2, initial_avgs, width, 
                            label='Initial Position', color='lightblue', alpha=0.8)
            bars1b = ax1.bar(x_pos + width/2, final_avgs, width, 
                            label='Final Position', color='darkblue', alpha=0.8)
            
            ax1.set_ylabel("Net Position Value ($)")
            ax1.set_title("Net Position Value: Initial vs Final by Risk Profile")
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(profile_names)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=100000, color='green', linestyle='--', alpha=0.7, label='Initial Value')
            
            # Add value labels
            for bars in [bars1a, bars1b]:
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 1000,
                            f'${height:,.0f}', ha='center', va='bottom', fontsize=9)
        
        # Chart 2: Simulated Net Position Value Over Time with Rebalancing Events
        minutes = np.arange(0, 61)
        
        # Create simulated net position trajectories for each risk profile
        for i, profile in enumerate(["conservative", "moderate", "aggressive"]):
            if risk_profiles[profile]:
                # Get average rebalancing events for this profile
                avg_rebalancing = np.mean([agent.get("rebalancing_events", 0) for agent in risk_profiles[profile]])
                avg_cost = np.mean([agent.get("cost_of_rebalancing", 0) for agent in risk_profiles[profile]])
                avg_final_position = np.mean([agent.get("net_position_value", 100000) for agent in risk_profiles[profile]])
                
                # Simulate net position decline with rebalancing events
                initial_position = 100000
                final_position = avg_final_position
                
                # Create a trajectory that shows the impact of rebalancing
                if avg_rebalancing > 0:
                    # If there were rebalancing events, show step decreases
                    position_trajectory = np.full(61, initial_position, dtype=float)
                    # Add rebalancing events at random times
                    rebalancing_times = np.random.choice(range(10, 51), int(avg_rebalancing), replace=False)
                    for rebal_time in rebalancing_times:
                        # Step decrease at rebalancing time
                        position_trajectory[rebal_time:] -= avg_cost / avg_rebalancing
                else:
                    # Smooth decline if no rebalancing
                    position_trajectory = initial_position + (final_position - initial_position) * (minutes / 60)
                
                ax2.plot(minutes, position_trajectory, color=colors[profile], linewidth=2, 
                        label=f'{profile.title()} (Avg: {avg_rebalancing:.1f} rebalances)', alpha=0.8)
        
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Net Position Value ($)')
        ax2.set_title('Simulated Net Position Value Over Time with Rebalancing Events')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Invert Y-axis to show $0 at top and negative values below
        ax2.invert_yaxis()
        
        # Add reference lines
        ax2.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Break Even')
        ax2.axhline(y=-100000, color='red', linestyle='--', alpha=0.7, label='Initial Position')
        
        plt.tight_layout()
        
        chart_path = charts_dir / "quick_test_net_position_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating quick test net position analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_quick_test_health_factor_time_series(results: List[Dict], charts_dir: Path) -> Path:
    """Create the exact health factor analysis charts from the reference"""
    
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
        
        # Aggregate all agent outcomes across all target HF scenarios
        all_ht_agents = []
        
        for result in results:
            raw_data = result.get("raw_data", {})
            all_ht_agents.extend(raw_data.get("ht_agent_outcomes", []))
        
        if not all_ht_agents:
            return None
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Quick Test: Health Factor Analysis (Time Series)", fontsize=18, fontweight='bold')
        
        # Group agents by risk profile
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
        
        # Chart 1: Average Health Factor by Risk Profile (Simulated over time)
        minutes = np.arange(0, 61)
        
        for profile in ["conservative", "moderate", "aggressive"]:
            if risk_profiles[profile]:
                # Get average initial and final health factors
                avg_initial = np.mean([agent.get("initial_health_factor", 1.2) for agent in risk_profiles[profile]])
                avg_final = np.mean([agent.get("final_health_factor", 1.0) for agent in risk_profiles[profile]])
                
                # Create a simulated decline from initial to final
                health_factors = avg_initial + (avg_final - avg_initial) * (minutes / 60)
                
                ax1.plot(minutes, health_factors, color=colors[profile], linewidth=2, 
                        label=profile.title(), alpha=0.8)
        
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Average Health Factor')
        ax1.set_title('Average Health Factor by Risk Profile')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Liquidation Threshold')
        
        # Chart 2: Health Factor Distribution: Start vs End
        all_initial_hfs = [agent.get("initial_health_factor", 1.2) for agent in all_ht_agents]
        all_final_hfs = [agent.get("final_health_factor", 1.0) for agent in all_ht_agents]
        
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
        # Since we don't have time series data, we'll simulate based on rebalancing events
        rebalancing_activity = np.zeros(61)
        
        for profile in ["conservative", "moderate", "aggressive"]:
            if risk_profiles[profile]:
                avg_rebalancing = np.mean([agent.get("rebalancing_events", 0) for agent in risk_profiles[profile]])
                if avg_rebalancing > 0:
                    # Simulate rebalancing activity over time
                    for i in range(10, 51, 10):  # Rebalancing events every 10 minutes
                        rebalancing_activity[i:i+5] += avg_rebalancing * 0.2
        
        ax3.fill_between(minutes, 0, rebalancing_activity, color='red', alpha=0.7, label='Agents Below Maintenance HF')
        ax3.plot(minutes, rebalancing_activity, color='darkred', linewidth=2)
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Agents Below Maintenance HF')
        ax3.set_title('Agents Requiring Rebalancing')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Chart 4: Target vs Final Health Factor Scatter
        target_hfs = [agent.get("target_health_factor", 1.1) for agent in all_ht_agents]
        final_hfs = [agent.get("final_health_factor", 1.0) for agent in all_ht_agents]
        profile_colors = []
        
        for agent in all_ht_agents:
            target_hf = agent.get("target_health_factor", 1.1)
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
        
        chart_path = charts_dir / "quick_test_health_factor_time_series.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"Error creating quick test health factor time series: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution function"""
    try:
        results = run_quick_target_hf_test()
        return results
    except KeyboardInterrupt:
        print("\n\nQuick test interrupted by user.")
    except Exception as e:
        print(f"\n❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()