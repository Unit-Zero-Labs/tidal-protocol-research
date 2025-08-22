#!/usr/bin/env python3
"""
Stress Test Execution Engine

Runs stress tests with Monte Carlo simulation capabilities.
"""

import time
from typing import Dict, List, Optional
import numpy as np
from .scenarios import TidalStressTestSuite
from .analyzer import StressTestAnalyzer
from ..simulation.engine import TidalSimulationEngine
from ..simulation.config import SimulationConfig


class StressTestRunner:
    """Stress test execution engine with Monte Carlo capabilities"""
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.test_suite = TidalStressTestSuite()
        self.analyzer = StressTestAnalyzer()
        self.results = {}
    
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
        
        total_time = time.time() - start_time
        print(f"Monte Carlo stress test completed in {total_time:.1f}s")
        
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
        
        return {
            "scenario_results": results,
            "analysis": analysis
        }
    
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
        
        return {
            "initial_debt_cap": initial_debt_cap,
            "final_debt_cap": final_debt_cap,
            "debt_cap_change": (final_debt_cap - initial_debt_cap) / initial_debt_cap,
            "total_borrowed": results["final_protocol_state"]["total_borrowed"]
        }