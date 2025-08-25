#!/usr/bin/env python3
"""
Stress Test Execution Engine

Runs stress tests with Monte Carlo simulation capabilities.
Enhanced with automatic results storage and visualization.
"""

import time
from typing import Dict, List, Optional
import numpy as np
from pathlib import Path
from .scenarios import TidalStressTestSuite
from .analyzer import StressTestAnalyzer
from ..simulation.engine import TidalSimulationEngine
from ..simulation.config import SimulationConfig
from ..core.protocol import Asset
from ..analysis.results_manager import ResultsManager, RunMetadata
from ..analysis.scenario_charts import ScenarioChartGenerator


class StressTestRunner:
    """Stress test execution engine with Monte Carlo capabilities and automatic results storage"""
    
    def __init__(self, config: SimulationConfig = None, auto_save: bool = True):
        self.config = config or SimulationConfig()
        self.test_suite = TidalStressTestSuite()
        self.analyzer = StressTestAnalyzer()
        self.results = {}
        
        # Results management components
        self.auto_save = auto_save
        self.results_manager = ResultsManager() if self.auto_save else None
        self.chart_generator = ScenarioChartGenerator()  # Always create chart generator
    
    def run_monte_carlo_stress_test(
        self, 
        scenario_name: str, 
        num_runs: int = 100,
        vary_params: bool = True
    ) -> Dict:
        """
        Run Monte Carlo stress test for a specific scenario
        
        Args:
            scenario_name: Name of stress scenario to run
            num_runs: Number of Monte Carlo runs
            vary_params: Whether to vary parameters across runs
            
        Returns:
            Aggregated results across all runs
        """
        
        print(f"Running Monte Carlo stress test: {scenario_name}")
        print(f"Number of runs: {num_runs}")
        print("=" * 50)
        
        runs_results = []
        start_time = time.time()
        
        for run in range(num_runs):
            try:
                # Create engine with potentially varied parameters
                config = self._create_varied_config() if vary_params else self.config
                engine = TidalSimulationEngine(config)
                
                # Run specific scenario
                result = self.test_suite.run_scenario(scenario_name, config)
                runs_results.append(result)
                
                if (run + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Completed {run + 1}/{num_runs} runs ({elapsed:.1f}s)")
                
            except Exception as e:
                print(f"Run {run} failed: {str(e)}")
                continue
        
        # Analyze results
        aggregated_results = self.analyzer.analyze_monte_carlo_results(
            scenario_name, runs_results
        )
        
        # Add a sample individual run for chart generation (use the last successful run)
        if runs_results:
            sample_run = runs_results[-1]  # Last run as sample
            aggregated_results["sample_scenario_results"] = sample_run
        
        total_time = time.time() - start_time
        print(f"Monte Carlo stress test completed in {total_time:.1f}s")
        
        # Auto-save results if enabled
        if self.auto_save:
            self._save_scenario_results(scenario_name, aggregated_results, total_time, num_runs)
        
        return aggregated_results
    
    def run_full_stress_test_suite(self, num_monte_carlo_runs: int = 100) -> Dict:
        """Run complete stress test suite with Monte Carlo analysis"""
        
        print("Running Full Tidal Protocol Stress Test Suite")
        print("=" * 60)
        
        suite_results = {}
        scenario_names = self.test_suite.get_scenario_names()
        
        for i, scenario_name in enumerate(scenario_names):
            print(f"\n[{i+1}/{len(scenario_names)}] Testing: {scenario_name}")
            
            try:
                # Run Monte Carlo for this scenario
                mc_results = self.run_monte_carlo_stress_test(
                    scenario_name, num_monte_carlo_runs
                )
                suite_results[scenario_name] = mc_results
                
            except Exception as e:
                print(f"Failed to run {scenario_name}: {str(e)}")
                suite_results[scenario_name] = {"error": str(e)}
        
        # Generate comprehensive analysis
        self.results = suite_results
        summary = self.analyzer.generate_suite_summary(suite_results)
        
        print("\n" + "=" * 60)
        print("STRESS TEST SUITE COMPLETED")
        print("=" * 60)
        
        return {
            "individual_results": suite_results,
            "suite_summary": summary
        }
    
    def run_targeted_scenario(
        self, 
        scenario_name: str, 
        custom_params: Optional[Dict] = None
    ) -> Dict:
        """
        Run a single targeted stress test scenario
        
        Args:
            scenario_name: Name of scenario to run
            custom_params: Custom parameters to override defaults
            
        Returns:
            Scenario results and analysis
        """
        
        print(f"Running targeted stress test: {scenario_name}")
        
        # Create config with custom params if provided
        config = self.config
        if custom_params:
            config = self._apply_custom_params(config, custom_params)
        
        # Run scenario
        engine = TidalSimulationEngine(config)
        results = self.test_suite.run_scenario(scenario_name, config)
        
        # Analyze results
        analysis = self.analyzer.analyze_single_scenario(scenario_name, results)
        
        final_results = {
            "scenario_results": results,
            "analysis": analysis
        }
        
        # Auto-save results if enabled
        if self.auto_save:
            self._save_scenario_results(scenario_name, final_results, 0, 1)
        
        return final_results
    
    def _create_varied_config(self) -> SimulationConfig:
        """Create configuration with parameter variations for Monte Carlo"""
        
        config = SimulationConfig()
        
        # Add some randomness to key parameters
        # Agent counts
        config.num_lenders += np.random.randint(-1, 2)  # ±1
        config.num_traders += np.random.randint(-1, 2)   # ±1
        config.num_liquidators = max(1, config.num_liquidators + np.random.randint(-1, 2))
        
        # Initial balances (±20% variation)
        balance_multiplier = np.random.uniform(0.8, 1.2)
        config.lender_initial_balance *= balance_multiplier
        config.trader_initial_balance *= balance_multiplier
        config.liquidator_initial_balance *= balance_multiplier
        
        # Volatility parameters (±50% variation)
        volatility_multiplier = np.random.uniform(0.5, 1.5)
        config.base_volatility *= volatility_multiplier
        config.flow_volatility *= volatility_multiplier
        
        return config
    
    def _apply_custom_params(self, config: SimulationConfig, custom_params: Dict) -> SimulationConfig:
        """Apply custom parameters to configuration"""
        
        for param, value in custom_params.items():
            if hasattr(config, param):
                setattr(config, param, value)
            else:
                print(f"Warning: Unknown parameter '{param}' ignored")
        
        return config
    
    def get_results_summary(self) -> Dict:
        """Get summary of all test results"""
        
        if not self.results:
            return {"message": "No test results available"}
        
        return self.analyzer.generate_suite_summary(self.results)
    
    def export_results(self, filepath: str):
        """Export results to file"""
        
        import json
        
        export_data = {
            "test_results": self.results,
            "summary": self.get_results_summary(),
            "config": {
                "num_lenders": self.config.num_lenders,
                "num_traders": self.config.num_traders,
                "num_liquidators": self.config.num_liquidators,
                "simulation_steps": self.config.simulation_steps
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Results exported to: {filepath}")
    
    def _save_scenario_results(
        self, 
        scenario_name: str, 
        results: Dict, 
        execution_time: float, 
        num_runs: int
    ) -> Optional[Path]:
        """Save scenario results with automatic directory management and chart generation"""
        
        if not self.auto_save or not self.results_manager:
            return None
        
        try:
            # Create run directory
            run_dir = self.results_manager.create_run_directory(scenario_name)
            
            # Create metadata
            metadata = RunMetadata(
                run_id=run_dir.name,
                scenario_name=scenario_name,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                parameters={
                    "num_monte_carlo_runs": num_runs,
                    "simulation_steps": self.config.simulation_steps,
                    "num_lenders": self.config.num_lenders,
                    "num_traders": self.config.num_traders,
                    "num_liquidators": self.config.num_liquidators
                },
                execution_time=execution_time
            )
            
            # Save main results
            results_file = self.results_manager.save_results(run_dir, results, metadata)
            
            # Generate scenario-specific charts
            charts_generated = []
            if hasattr(self, 'chart_generator') and self.chart_generator:
                try:
                    charts_dir = run_dir / "charts"
                    # Use the scenario results for chart generation - handle both single and Monte Carlo results
                    if "scenario_results" in results:
                        chart_data = results["scenario_results"]
                    elif "sample_scenario_results" in results:
                        chart_data = {"scenario_results": results["sample_scenario_results"]}
                    else:
                        chart_data = results
                    
                    charts_generated = self.chart_generator.generate_scenario_charts(
                        scenario_name, chart_data, charts_dir)
                except Exception as e:
                    print(f"Warning: Chart generation failed - {str(e)}")
            
            # Create summary report
            summary_data = {
                "metadata": metadata.__dict__,
                "key_metrics": self._extract_key_metrics(results),
                "risk_assessment": self._extract_risk_assessment(results),
                "charts_generated": [chart.name for chart in charts_generated]
            }
            self.results_manager.save_summary_report(run_dir, summary_data)
            
            # Print results location
            print(f"\n📁 Results saved to: {run_dir}")
            if charts_generated:
                print(f"📊 Charts generated: {len(charts_generated)} charts saved to charts/ subfolder")
            
            return run_dir
            
        except Exception as e:
            print(f"Warning: Could not save results - {str(e)}")
            return None
    
    def _extract_key_metrics(self, results: Dict) -> Dict:
        """Extract key metrics from results for summary"""
        key_metrics = {}
        
        # Try to extract from different result structures
        if "analysis" in results:
            analysis = results["analysis"]
            if "risk_metrics" in analysis:
                key_metrics.update(analysis["risk_metrics"])
            if "assessment" in analysis:
                assessment = analysis["assessment"]
                key_metrics["overall_score"] = assessment.get("overall_score", 0)
                key_metrics["risk_level"] = assessment.get("risk_level", "Unknown")
        
        # Extract from scenario results if available
        if "scenario_results" in results:
            scenario_results = results["scenario_results"]
            if "summary_statistics" in scenario_results:
                key_metrics.update(scenario_results["summary_statistics"])
        
        return key_metrics
    
    def _extract_risk_assessment(self, results: Dict) -> Dict:
        """Extract risk assessment from results"""
        risk_assessment = {}
        
        if "analysis" in results and "assessment" in results["analysis"]:
            assessment = results["analysis"]["assessment"]
            risk_assessment = {
                "risk_level": assessment.get("risk_level", "Unknown"),
                "risk_score": assessment.get("overall_score", 0),
                "key_concerns": assessment.get("key_concerns", [])
            }
        
        return risk_assessment
    
    def list_scenario_results(self, scenario_name: str) -> List[Dict]:
        """List all saved results for a scenario"""
        if not self.auto_save:
            return []
        
        return self.results_manager.list_scenario_runs(scenario_name)
    
    def list_all_scenarios(self) -> List[str]:
        """List all scenarios with saved results"""
        if not self.auto_save:
            return []
        
        return self.results_manager.list_all_scenarios()
    
    def load_scenario_results(self, scenario_name: str, run_id: str) -> Optional[Dict]:
        """Load results from a specific run"""
        if not self.auto_save:
            return None
        
        runs = self.list_scenario_results(scenario_name)
        target_run = next((run for run in runs if run["run_id"] == run_id), None)
        
        if not target_run:
            return None
        
        return self.results_manager.load_results(Path(target_run["path"]))


class QuickStressTest:
    """Quick stress tests for development and debugging"""
    
    @staticmethod
    def run_liquidation_test(steps: int = 100) -> Dict:
        """Quick liquidation efficiency test"""
        
        config = SimulationConfig()
        config.simulation_steps = steps
        engine = TidalSimulationEngine(config)
        
        # Apply ETH crash
        engine.state.apply_price_shock({Asset.ETH: -0.35})
        
        # Run simulation
        results = engine.run_simulation(steps)
        
        return {
            "liquidations": len(results["liquidation_events"]),
            "final_health_factors": [
                agent["health_factor"] 
                for agent in results["agent_states"].values()
            ],
            "protocol_treasury": results["final_protocol_state"]["protocol_treasury"]
        }
    
    @staticmethod
    def run_debt_cap_test(steps: int = 50) -> Dict:
        """Quick debt cap functionality test"""
        
        config = SimulationConfig()
        config.simulation_steps = steps
        engine = TidalSimulationEngine(config)
        
        initial_debt_cap = engine.protocol.calculate_debt_cap()
        
        # Apply market stress
        engine.state.apply_price_shock({
            Asset.ETH: -0.20,
            Asset.BTC: -0.20,
            Asset.FLOW: -0.30
        })
        
        # Run simulation
        results = engine.run_simulation(steps)
        
        final_debt_cap = engine.protocol.calculate_debt_cap()
        
        # Calculate total borrowed from agent states
        total_borrowed = 0.0
        for agent_state in results["agent_states"].values():
            total_borrowed += sum(agent_state.get("borrowed_balances", {}).values())
        
        return {
            "initial_debt_cap": initial_debt_cap,
            "final_debt_cap": final_debt_cap,
            "debt_cap_change": (final_debt_cap - initial_debt_cap) / initial_debt_cap if initial_debt_cap > 0 else 0,
            "total_borrowed": total_borrowed
        }