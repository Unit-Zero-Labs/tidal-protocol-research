#!/usr/bin/env python3
"""
High Tide vs AAVE Comparison Scenarios

Implements Monte Carlo comparison analysis between High Tide active rebalancing
and AAVE-style traditional liquidation mechanisms.
"""

import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from ..simulation.high_tide_vault_engine import HighTideVaultEngine, HighTideConfig
from ..simulation.aave_protocol_engine import AaveProtocolEngine, AaveConfig
from ..analysis.results_manager import ResultsManager


class ComparisonConfig:
    """Configuration for High Tide vs AAVE comparison analysis"""
    
    def __init__(self):
        # Monte Carlo parameters
        self.num_monte_carlo_runs = 50  # Number of runs for statistical significance
        self.random_seed_base = 42  # Base seed for reproducibility
        
        # Shared simulation parameters (must be identical for fair comparison)
        self.btc_decline_duration = 60  # 60 minutes
        self.btc_initial_price = 100_000.0
        self.btc_final_price_range = (75_000.0, 85_000.0)  # 15-25% decline
        self.yield_apr = 0.10  # 10% APR for yield tokens
        self.moet_btc_pool_size = 250_000  # $250K each side
        
        # Agent parameters (same for both scenarios)
        self.num_agents_range = (20, 50)  # Range for Monte Carlo variation
        self.monte_carlo_agent_variation = True
        
        # Results configuration
        self.save_individual_runs = True
        self.generate_comparison_charts = True
        self.scenario_name = "High_Tide_vs_Aave_Comparison"


class HighTideVsAaveComparison:
    """Runs comparative Monte Carlo analysis between High Tide and AAVE strategies"""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.results_manager = ResultsManager()
        
        # Storage for all runs
        self.high_tide_runs = []
        self.aave_runs = []
        self.comparison_stats = {}
        
    def run_comparison_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive Monte Carlo comparison between High Tide and AAVE strategies
        """
        print("=" * 80)
        print("HIGH TIDE vs AAVE LIQUIDATION MECHANISM COMPARISON")
        print("=" * 80)
        print(f"Running {self.config.num_monte_carlo_runs} Monte Carlo iterations...")
        print(f"BTC decline: ${self.config.btc_initial_price:,.0f} → ${self.config.btc_final_price_range[0]:,.0f}-${self.config.btc_final_price_range[1]:,.0f}")
        print()
        
        # Run Monte Carlo simulations
        for run_id in range(self.config.num_monte_carlo_runs):
            print(f"Run {run_id + 1}/{self.config.num_monte_carlo_runs}")
            
            # Set seed for this run to ensure both scenarios use same parameters
            run_seed = self.config.random_seed_base + run_id
            
            # Run High Tide scenario
            high_tide_results = self._run_high_tide_scenario(run_id, run_seed)
            self.high_tide_runs.append(high_tide_results)
            
            # Run AAVE scenario with same parameters
            aave_results = self._run_aave_scenario(run_id, run_seed)
            self.aave_runs.append(aave_results)
            
            # Progress update
            if (run_id + 1) % 10 == 0:
                print(f"  Completed {run_id + 1} runs...")
        
        print("\nGenerating comparative analysis...")
        
        # Calculate comparison statistics
        self.comparison_stats = self._calculate_comparison_statistics()
        
        # Generate comprehensive results
        final_results = self._generate_comparison_results()
        
        # Generate comparison charts
        if self.config.generate_comparison_charts:
            self._generate_comparison_charts(final_results)
        
        # Save results
        if self.config.save_individual_runs:
            self._save_comparison_results(final_results)
        
        print("✅ Comparison analysis complete!")
        return final_results
    
    def _run_high_tide_scenario(self, run_id: int, seed: int) -> Dict[str, Any]:
        """Run a single High Tide scenario with specified seed"""
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Create High Tide configuration
        config = HighTideConfig()
        config.btc_decline_duration = self.config.btc_decline_duration
        config.btc_initial_price = self.config.btc_initial_price
        config.btc_final_price_range = self.config.btc_final_price_range
        config.yield_apr = self.config.yield_apr
        config.moet_btc_pool_size = self.config.moet_btc_pool_size
        config.monte_carlo_agent_variation = self.config.monte_carlo_agent_variation
        
        # Create and run engine
        engine = HighTideVaultEngine(config)
        results = engine.run_simulation()
        
        # Add run metadata
        results["run_metadata"] = {
            "run_id": run_id,
            "seed": seed,
            "strategy": "High_Tide",
            "num_agents": len(engine.high_tide_agents)
        }
        
        return results
    
    def _run_aave_scenario(self, run_id: int, seed: int) -> Dict[str, Any]:
        """Run a single AAVE scenario with specified seed"""
        # Set random seed for reproducibility (same as High Tide)
        random.seed(seed)
        np.random.seed(seed)
        
        # Create AAVE configuration
        config = AaveConfig()
        config.btc_decline_duration = self.config.btc_decline_duration
        config.btc_initial_price = self.config.btc_initial_price
        config.btc_final_price_range = self.config.btc_final_price_range
        config.yield_apr = self.config.yield_apr
        config.moet_btc_pool_size = self.config.moet_btc_pool_size
        config.monte_carlo_agent_variation = self.config.monte_carlo_agent_variation
        
        # Create and run engine
        engine = AaveProtocolEngine(config)
        results = engine.run_simulation()
        
        # Add run metadata
        results["run_metadata"] = {
            "run_id": run_id,
            "seed": seed,
            "strategy": "AAVE",
            "num_agents": len(engine.aave_agents)
        }
        
        return results
    
    def _calculate_comparison_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive comparison statistics"""
        
        # Extract key metrics from all runs
        high_tide_metrics = self._extract_metrics_from_runs(self.high_tide_runs, "High_Tide")
        aave_metrics = self._extract_metrics_from_runs(self.aave_runs, "AAVE")
        
        # Calculate statistical comparisons
        comparison_stats = {
            "survival_rate": self._compare_metrics(
                high_tide_metrics["survival_rates"], 
                aave_metrics["survival_rates"],
                "Survival Rate"
            ),
            "cost_per_agent": self._compare_metrics(
                high_tide_metrics["costs_per_agent"],
                aave_metrics["costs_per_agent"], 
                "Cost per Agent"
            ),
            "protocol_revenue": self._compare_metrics(
                high_tide_metrics["protocol_revenues"],
                aave_metrics["protocol_revenues"],
                "Protocol Revenue"
            ),
            "final_health_factor": self._compare_metrics(
                high_tide_metrics["avg_final_hf"],
                aave_metrics["avg_final_hf"],
                "Average Final Health Factor"
            ),
            "liquidation_events": self._compare_metrics(
                high_tide_metrics["liquidation_events"],
                aave_metrics["liquidation_events"],
                "Liquidation Events per Agent"
            )
        }
        
        # Add risk profile analysis
        comparison_stats["risk_profile_analysis"] = self._analyze_risk_profiles()
        
        # Add overall performance summary
        comparison_stats["performance_summary"] = self._generate_performance_summary(
            high_tide_metrics, aave_metrics
        )
        
        return comparison_stats
    
    def _extract_metrics_from_runs(self, runs: List[Dict], strategy: str) -> Dict[str, List]:
        """Extract key metrics from all runs of a strategy"""
        metrics = {
            "survival_rates": [],
            "costs_per_agent": [],
            "protocol_revenues": [],
            "avg_final_hf": [],
            "liquidation_events": [],
            "risk_profile_performance": {"conservative": [], "moderate": [], "aggressive": []}
        }
        
        for run in runs:
            survival_stats = run.get("survival_statistics", {})
            cost_analysis = run.get("cost_analysis", {})
            agent_outcomes = run.get("agent_outcomes", [])
            
            # Survival rate
            metrics["survival_rates"].append(survival_stats.get("survival_rate", 0.0))
            
            # Cost per agent - for AAVE, only include runs with liquidations
            cost_per_agent = cost_analysis.get("average_cost_per_agent", 0.0)
            if strategy == "AAVE":
                # Only include this run if there were actual liquidations (cost > 0)
                if cost_per_agent > 0:
                    metrics["costs_per_agent"].append(cost_per_agent)
            else:  # High Tide - include all runs (rebalancing can happen in any run)
                metrics["costs_per_agent"].append(cost_per_agent)
            
            # Protocol revenue (estimate based on activity)
            if strategy == "High_Tide":
                yield_activity = run.get("yield_token_activity", {})
                revenue = yield_activity.get("total_rebalancing_sales", 0) * 0.001  # 0.1% fee
            else:  # AAVE
                liquidation_activity = run.get("liquidation_activity", {})
                revenue = liquidation_activity.get("total_penalties_collected", 0)
            
            metrics["protocol_revenues"].append(revenue)
            
            # Average final health factor
            if agent_outcomes:
                avg_hf = np.mean([outcome["final_health_factor"] for outcome in agent_outcomes])
                metrics["avg_final_hf"].append(avg_hf)
            
            # Liquidation events per agent
            if strategy == "High_Tide":
                # High Tide: emergency liquidations
                total_emergencies = sum(outcome.get("emergency_liquidations", 0) for outcome in agent_outcomes)
                events_per_agent = total_emergencies / len(agent_outcomes) if agent_outcomes else 0
            else:  # AAVE
                # AAVE: traditional liquidations
                total_liquidations = sum(outcome.get("liquidation_events", 0) for outcome in agent_outcomes)
                events_per_agent = total_liquidations / len(agent_outcomes) if agent_outcomes else 0
            
            metrics["liquidation_events"].append(events_per_agent)
            
            # Risk profile performance
            for outcome in agent_outcomes:
                profile = outcome["risk_profile"]
                survival = 1.0 if outcome["survived"] else 0.0
                metrics["risk_profile_performance"][profile].append(survival)
        
        return metrics
    
    def _compare_metrics(self, high_tide_values: List[float], aave_values: List[float], metric_name: str) -> Dict[str, Any]:
        """Compare two sets of metric values with statistical analysis"""
        
        if not high_tide_values or not aave_values:
            return {}
        
        ht_array = np.array(high_tide_values)
        aave_array = np.array(aave_values)
        
        # Basic statistics
        comparison = {
            "metric_name": metric_name,
            "high_tide": {
                "mean": float(np.mean(ht_array)),
                "std": float(np.std(ht_array)),
                "median": float(np.median(ht_array)),
                "min": float(np.min(ht_array)),
                "max": float(np.max(ht_array)),
                "values": high_tide_values
            },
            "aave": {
                "mean": float(np.mean(aave_array)),
                "std": float(np.std(aave_array)),
                "median": float(np.median(aave_array)),
                "min": float(np.min(aave_array)),
                "max": float(np.max(aave_array)),
                "values": aave_values
            }
        }
        
        # Performance difference
        mean_diff = comparison["high_tide"]["mean"] - comparison["aave"]["mean"]
        improvement = (mean_diff / comparison["aave"]["mean"] * 100) if comparison["aave"]["mean"] != 0 else 0
        
        comparison["performance_difference"] = {
            "absolute_difference": mean_diff,
            "percentage_improvement": improvement,
            "high_tide_better": mean_diff > 0 if "cost" not in metric_name.lower() else mean_diff < 0
        }
        
        # Statistical significance (simple t-test approximation)
        pooled_std = np.sqrt((np.var(ht_array) + np.var(aave_array)) / 2)
        if pooled_std > 0:
            t_stat = abs(mean_diff) / (pooled_std * np.sqrt(2 / len(ht_array)))
            # Simplified significance test (t > 2 roughly corresponds to p < 0.05)
            comparison["statistical_significance"] = {
                "t_statistic": float(t_stat),
                "significant": t_stat > 2.0,
                "confidence_level": "95%" if t_stat > 2.0 else "Not significant"
            }
        
        return comparison
    
    def _analyze_risk_profiles(self) -> Dict[str, Any]:
        """Analyze performance by risk profile across both strategies"""
        analysis = {}
        
        for profile in ["conservative", "moderate", "aggressive"]:
            # Extract survival rates by profile from both strategies
            ht_profile_survival = []
            aave_profile_survival = []
            
            for ht_run, aave_run in zip(self.high_tide_runs, self.aave_runs):
                # High Tide survival by profile
                ht_outcomes = ht_run.get("agent_outcomes", [])
                ht_profile_agents = [o for o in ht_outcomes if o["risk_profile"] == profile]
                if ht_profile_agents:
                    ht_survival = sum(1 for a in ht_profile_agents if a["survived"]) / len(ht_profile_agents)
                    ht_profile_survival.append(ht_survival)
                
                # AAVE survival by profile
                aave_outcomes = aave_run.get("agent_outcomes", [])
                aave_profile_agents = [o for o in aave_outcomes if o["risk_profile"] == profile]
                if aave_profile_agents:
                    aave_survival = sum(1 for a in aave_profile_agents if a["survived"]) / len(aave_profile_agents)
                    aave_profile_survival.append(aave_survival)
            
            # Compare survival rates for this profile
            if ht_profile_survival and aave_profile_survival:
                analysis[profile] = self._compare_metrics(
                    ht_profile_survival, 
                    aave_profile_survival, 
                    f"{profile.title()} Survival Rate"
                )
        
        return analysis
    
    def _generate_performance_summary(self, ht_metrics: Dict, aave_metrics: Dict) -> Dict[str, Any]:
        """Generate overall performance summary"""
        
        # Calculate win rates (how often High Tide performs better)
        win_rates = {}
        
        for metric in ["survival_rates", "costs_per_agent", "protocol_revenues"]:
            if metric in ht_metrics and metric in aave_metrics:
                ht_values = ht_metrics[metric]
                aave_values = aave_metrics[metric]
                
                if metric == "costs_per_agent":
                    # For costs, lower is better
                    wins = sum(1 for ht, aave in zip(ht_values, aave_values) if ht < aave)
                else:
                    # For survival and revenue, higher is better
                    wins = sum(1 for ht, aave in zip(ht_values, aave_values) if ht > aave)
                
                win_rates[metric] = wins / len(ht_values) if ht_values else 0.0
        
        # Overall assessment
        avg_win_rate = np.mean(list(win_rates.values())) if win_rates else 0.0
        
        summary = {
            "win_rates": win_rates,
            "overall_win_rate": avg_win_rate,
            "high_tide_advantage": avg_win_rate > 0.5,
            "total_runs": len(self.high_tide_runs),
            "statistical_power": "High" if len(self.high_tide_runs) >= 30 else "Medium"
        }
        
        return summary
    
    def _generate_comparison_results(self) -> Dict[str, Any]:
        """Generate comprehensive comparison results"""
        
        return {
            "comparison_metadata": {
                "scenario_name": self.config.scenario_name,
                "num_monte_carlo_runs": self.config.num_monte_carlo_runs,
                "simulation_parameters": {
                    "btc_decline_duration": self.config.btc_decline_duration,
                    "btc_initial_price": self.config.btc_initial_price,
                    "btc_final_price_range": self.config.btc_final_price_range,
                    "yield_apr": self.config.yield_apr,
                    "agent_count_range": self.config.num_agents_range
                }
            },
            "comparison_statistics": self.comparison_stats,
            "high_tide_summary": self._summarize_strategy_results(self.high_tide_runs, "High_Tide"),
            "aave_summary": self._summarize_strategy_results(self.aave_runs, "AAVE"),
            "individual_runs": {
                "high_tide_runs": self.high_tide_runs if self.config.save_individual_runs else [],
                "aave_runs": self.aave_runs if self.config.save_individual_runs else []
            }
        }
    
    def _summarize_strategy_results(self, runs: List[Dict], strategy: str) -> Dict[str, Any]:
        """Summarize results for a strategy across all runs"""
        
        if not runs:
            return {}
        
        # Aggregate statistics
        survival_rates = [run.get("survival_statistics", {}).get("survival_rate", 0) for run in runs]
        costs_per_agent = [run.get("cost_analysis", {}).get("average_cost_per_agent", 0) for run in runs]
        
        summary = {
            "strategy": strategy,
            "total_runs": len(runs),
            "average_survival_rate": np.mean(survival_rates) if survival_rates else 0,
            "survival_rate_std": np.std(survival_rates) if survival_rates else 0,
            "average_cost_per_agent": np.mean(costs_per_agent) if costs_per_agent else 0,
            "cost_per_agent_std": np.std(costs_per_agent) if costs_per_agent else 0,
            "best_survival_rate": np.max(survival_rates) if survival_rates else 0,
            "worst_survival_rate": np.min(survival_rates) if survival_rates else 0,
            "lowest_cost": np.min(costs_per_agent) if costs_per_agent else 0,
            "highest_cost": np.max(costs_per_agent) if costs_per_agent else 0
        }
        
        return summary
    
    def _save_comparison_results(self, results: Dict[str, Any]):
        """Save comparison results to file system"""
        
        # Create results directory using the results manager's method
        run_dir = self.results_manager.create_run_directory(self.config.scenario_name)
        
        # Save results JSON
        results_path = run_dir / "comparison_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            import json
            
            def convert_asset_keys(obj):
                """Convert Asset enum keys to strings recursively"""
                from ..core.protocol import Asset
                
                if isinstance(obj, dict):
                    new_dict = {}
                    for key, value in obj.items():
                        # Convert Asset keys to strings
                        if isinstance(key, Asset):
                            new_key = key.name
                        else:
                            new_key = key
                        new_dict[new_key] = convert_asset_keys(value)
                    return new_dict
                elif isinstance(obj, list):
                    return [convert_asset_keys(item) for item in obj]
                elif isinstance(obj, Asset):
                    return obj.name
                else:
                    return obj
            
            # Convert the entire results structure
            serializable_results = convert_asset_keys(results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save metadata
        metadata = {
            "scenario_type": "Comparison_Analysis",
            "strategies_compared": ["High_Tide", "AAVE"],
            "monte_carlo_runs": self.config.num_monte_carlo_runs,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "btc_decline_duration": self.config.btc_decline_duration,
                "btc_price_range": self.config.btc_final_price_range,
                "yield_apr": self.config.yield_apr
            }
        }
        
        metadata_path = run_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"📁 Comparison results saved to: {run_dir}")
        
        return run_dir
    
    def _generate_comparison_charts(self, results: Dict[str, Any]):
        """Generate comprehensive comparison visualization charts"""
        try:
            from ..analysis.high_tide_charts import HighTideChartGenerator
            
            # Create results directory for charts
            scenario_dir = Path("tidal_protocol_sim/results") / self.config.scenario_name
            charts_dir = scenario_dir / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)
            
            # Use High Tide chart generator for comprehensive charts
            chart_generator = HighTideChartGenerator()
            
            # Generate charts for both High Tide and AAVE using representative samples
            generated_charts = []
            
            if self.high_tide_runs and self.aave_runs:
                # Use most recent runs as representative samples
                ht_sample = self.high_tide_runs[-1]
                aave_sample = self.aave_runs[-1]
                
                # Generate High Tide style charts for High Tide data
                print("📊 Generating High Tide analysis charts...")
                ht_charts = chart_generator.generate_high_tide_charts(
                    "High_Tide_Comparison", ht_sample, charts_dir / "high_tide", aave_sample
                )
                generated_charts.extend(ht_charts)
                
                # Generate AAVE analysis charts using the same format
                print("📊 Generating AAVE analysis charts...")
                aave_charts = self._generate_aave_analysis_charts(aave_sample, charts_dir / "aave")
                generated_charts.extend(aave_charts)
                
                # Generate comparison summary charts
                print("📊 Generating comparison summary charts...")
                comparison_charts = self._generate_comparison_summary_charts(results, charts_dir)
                generated_charts.extend(comparison_charts)
            
            print(f"📊 Generated {len(generated_charts)} total comparison charts")
            
        except ImportError:
            print("⚠️  High Tide chart generator not available - creating basic charts")
            self._generate_basic_comparison_charts(results)
        except Exception as e:
            print(f"⚠️  Chart generation failed: {e}")
            import traceback
            traceback.print_exc()
            self._generate_basic_comparison_charts(results)
    
    def _generate_basic_comparison_charts(self, results: Dict[str, Any]):
        """Generate basic comparison charts using matplotlib"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            scenario_dir = Path("tidal_protocol_sim/results") / self.config.scenario_name
            charts_dir = scenario_dir / "charts"
            charts_dir.mkdir(parents=True, exist_ok=True)
            
            # Get comparison statistics
            stats = results.get("comparison_statistics", {})
            
            # Chart 1: Survival Rate Comparison
            survival_comparison = stats.get("survival_rate", {})
            if survival_comparison:
                ht_survival = survival_comparison.get("high_tide", {}).get("mean", 0) * 100
                aave_survival = survival_comparison.get("aave", {}).get("mean", 0) * 100
                
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                strategies = ["High Tide", "AAVE"]
                survival_rates = [ht_survival, aave_survival]
                colors = ["#2E8B57", "#8B0000"]
                
                bars = ax.bar(strategies, survival_rates, color=colors, alpha=0.7)
                ax.set_ylabel("Survival Rate (%)")
                ax.set_title("Agent Survival Rate Comparison")
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, rate in zip(bars, survival_rates):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{rate:.1f}%', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(charts_dir / "survival_rate_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Chart 2: Cost Comparison
            cost_comparison = stats.get("cost_per_agent", {})
            if cost_comparison:
                ht_cost = cost_comparison.get("high_tide", {}).get("mean", 0)
                aave_cost = cost_comparison.get("aave", {}).get("mean", 0)
                
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                strategies = ["High Tide", "AAVE"]
                costs = [ht_cost, aave_cost]
                colors = ["#2E8B57", "#8B0000"]
                
                bars = ax.bar(strategies, costs, color=colors, alpha=0.7)
                ax.set_ylabel("Average Cost per Agent ($)")
                ax.set_title("Cost of Liquidation/Rebalancing Comparison")
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, cost in zip(bars, costs):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(costs)*0.02,
                           f'${cost:,.0f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(charts_dir / "cost_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"📊 Generated 2 basic comparison charts in {charts_dir}")
            
        except Exception as e:
            print(f"⚠️  Basic chart generation failed: {e}")
    
    def _generate_aave_analysis_charts(self, aave_results: Dict[str, Any], charts_dir: Path) -> List[Path]:
        """Generate AAVE analysis charts using High Tide chart format"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            charts_dir.mkdir(parents=True, exist_ok=True)
            generated_charts = []
            
            # 1. AAVE Agent Performance Summary (similar to High Tide)
            agent_outcomes = aave_results.get("agent_outcomes", [])
            if agent_outcomes:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                fig.suptitle("AAVE Agent Performance Summary", fontsize=16, fontweight='bold')
                
                # Net Position Values
                risk_profiles = ["conservative", "moderate", "aggressive"]
                colors = ["#2E8B57", "#FF8C00", "#DC143C"]
                
                profile_data = {profile: {"values": [], "liquidated": 0} for profile in risk_profiles}
                
                for outcome in agent_outcomes:
                    profile = outcome["risk_profile"]
                    if profile in profile_data:
                        profile_data[profile]["values"].append(outcome["net_position_value"])
                        if outcome["liquidation_events"] > 0:
                            profile_data[profile]["liquidated"] += 1
                
                # Plot net position values by risk profile
                positions = np.arange(len(risk_profiles))
                values = [np.mean(profile_data[p]["values"]) if profile_data[p]["values"] else 0 
                         for p in risk_profiles]
                
                bars = ax1.bar(positions, values, color=colors, alpha=0.7)
                ax1.set_title("Average Net Position Value by Risk Profile")
                ax1.set_xlabel("Risk Profile")
                ax1.set_ylabel("Net Position Value ($)")
                ax1.set_xticks(positions)
                ax1.set_xticklabels([p.title() for p in risk_profiles])
                ax1.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                           f'${value:,.0f}', ha='center', va='bottom')
                
                # Liquidation events by risk profile
                liquidated_counts = [profile_data[p]["liquidated"] for p in risk_profiles]
                bars2 = ax2.bar(positions, liquidated_counts, color=colors, alpha=0.7)
                ax2.set_title("Liquidated Agents by Risk Profile")
                ax2.set_xlabel("Risk Profile")
                ax2.set_ylabel("Number of Liquidated Agents")
                ax2.set_xticks(positions)
                ax2.set_xticklabels([p.title() for p in risk_profiles])
                ax2.grid(True, alpha=0.3)
                
                # Health factor distribution
                final_hfs = [outcome["final_health_factor"] for outcome in agent_outcomes 
                           if outcome["final_health_factor"] != float('inf')]
                
                if final_hfs:
                    ax3.hist(final_hfs, bins=20, alpha=0.7, color='#8B0000', edgecolor='black')
                    ax3.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Liquidation Threshold')
                    ax3.set_title("Final Health Factor Distribution")
                    ax3.set_xlabel("Health Factor")
                    ax3.set_ylabel("Number of Agents")
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
                
                # Liquidation penalties by agent
                liquidation_penalties = [outcome["liquidation_penalties"] for outcome in agent_outcomes 
                                       if outcome["liquidation_penalties"] > 0]
                
                if liquidation_penalties:
                    ax4.hist(liquidation_penalties, bins=15, alpha=0.7, color='#8B0000', edgecolor='black')
                    ax4.set_title("Liquidation Penalties Distribution")
                    ax4.set_xlabel("Liquidation Penalty ($)")
                    ax4.set_ylabel("Number of Agents")
                    ax4.grid(True, alpha=0.3)
                else:
                    ax4.text(0.5, 0.5, 'No Liquidations Occurred', ha='center', va='center', 
                           transform=ax4.transAxes, fontsize=14)
                    ax4.set_title("Liquidation Penalties Distribution")
                
                plt.tight_layout()
                chart_path = charts_dir / "aave_agent_performance_summary.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                generated_charts.append(chart_path)
            
            # 2. AAVE BTC Price vs Liquidation Events Timeline
            btc_history = aave_results.get("btc_price_history", [])
            liquidation_events = aave_results.get("liquidation_activity", {}).get("liquidation_events", [])
            
            if btc_history:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
                fig.suptitle("AAVE: BTC Price Decline vs Liquidation Events", fontsize=16, fontweight='bold')
                
                minutes = list(range(len(btc_history)))
                
                # BTC price timeline
                ax1.plot(minutes, btc_history, linewidth=2, color='#FF6B35', label='BTC Price')
                ax1.set_ylabel("BTC Price ($)")
                ax1.set_title("BTC Price Timeline")
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Format y-axis for currency
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                
                # Liquidation events timeline
                liquidation_minutes = [event["minute"] for event in liquidation_events]
                liquidation_amounts = [event["liquidation_bonus_value"] for event in liquidation_events]
                
                if liquidation_minutes:
                    ax2.scatter(liquidation_minutes, liquidation_amounts, 
                              color='red', s=100, alpha=0.7, label='Liquidation Events')
                    ax2.set_ylabel("Liquidation Bonus ($)")
                    ax2.set_title("Liquidation Events Timeline")
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                else:
                    ax2.text(0.5, 0.5, 'No Liquidation Events', ha='center', va='center', 
                           transform=ax2.transAxes, fontsize=14)
                    ax2.set_title("Liquidation Events Timeline")
                
                ax2.set_xlabel("Simulation Time (minutes)")
                
                plt.tight_layout()
                chart_path = charts_dir / "aave_btc_liquidation_timeline.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                generated_charts.append(chart_path)
            
            print(f"📊 Generated {len(generated_charts)} AAVE analysis charts")
            return generated_charts
            
        except Exception as e:
            print(f"⚠️  AAVE chart generation failed: {e}")
            return []
    
    def _generate_comparison_summary_charts(self, results: Dict[str, Any], charts_dir: Path) -> List[Path]:
        """Generate summary comparison charts"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            generated_charts = []
            
            # Get comparison statistics (if available)
            comparison_stats = results.get("comparison_statistics", {})
            
            if comparison_stats:
                # 1. Side-by-side survival and cost comparison
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                fig.suptitle("High Tide vs AAVE: Performance Comparison", fontsize=16, fontweight='bold')
                
                # Survival Rate Comparison
                survival_stats = comparison_stats.get("survival_rate", {})
                if survival_stats:
                    ht_survival = survival_stats.get("high_tide", {}).get("mean", 0) * 100
                    aave_survival = survival_stats.get("aave", {}).get("mean", 0) * 100
                    
                    strategies = ["High Tide", "AAVE"]
                    survival_rates = [ht_survival, aave_survival]
                    colors = ["#2E8B57", "#8B0000"]
                    
                    bars1 = ax1.bar(strategies, survival_rates, color=colors, alpha=0.7)
                    ax1.set_ylabel("Survival Rate (%)")
                    ax1.set_title("Agent Survival Rate Comparison")
                    ax1.set_ylim(0, 100)
                    ax1.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, rate in zip(bars1, survival_rates):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                # Cost Comparison
                cost_stats = comparison_stats.get("cost_per_agent", {})
                if cost_stats:
                    ht_cost = cost_stats.get("high_tide", {}).get("mean", 0)
                    aave_cost = cost_stats.get("aave", {}).get("mean", 0)
                    
                    strategies = ["High Tide", "AAVE"]
                    costs = [ht_cost, aave_cost]
                    colors = ["#2E8B57", "#8B0000"]
                    
                    bars2 = ax2.bar(strategies, costs, color=colors, alpha=0.7)
                    ax2.set_ylabel("Average Cost per Affected Agent ($)")
                    ax2.set_title("Cost Comparison (Liquidated/Rebalanced Agents Only)")
                    ax2.grid(True, alpha=0.3)
                    
                    # Add value labels
                    for bar, cost in zip(bars2, costs):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + max(costs)*0.02,
                               f'${cost:,.0f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                chart_path = charts_dir / "high_tide_vs_aave_summary.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                generated_charts.append(chart_path)
            
            print(f"📊 Generated {len(generated_charts)} comparison summary charts")
            return generated_charts
            
        except Exception as e:
            print(f"⚠️  Comparison summary chart generation failed: {e}")
            return []


# Convenience function for easy execution
def run_high_tide_vs_aave_comparison(num_runs: int = 50) -> Dict[str, Any]:
    """
    Convenience function to run High Tide vs AAVE comparison
    
    Args:
        num_runs: Number of Monte Carlo runs for statistical significance
        
    Returns:
        Comprehensive comparison results
    """
    config = ComparisonConfig()
    config.num_monte_carlo_runs = num_runs
    
    comparison = HighTideVsAaveComparison(config)
    return comparison.run_comparison_analysis()
