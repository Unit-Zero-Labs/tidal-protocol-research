#!/usr/bin/env python3
"""
Tidal Protocol Stress Testing Module

This module focuses on stress testing the lending protocol and liquidation procedures
under various price shock scenarios and risk parameter changes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import copy

from ..core.simulation.factory import SimulationFactory
from ..config.schemas.tidal_config import create_default_config, AssetType
from .visualization.tidal_charts import TidalProtocolCharts


@dataclass
class StressTestScenario:
    """Definition of a stress test scenario"""
    name: str
    description: str
    price_shocks: Dict[AssetType, float]  # Asset -> percentage change
    duration_days: int = 30
    shock_timing: int = 15  # Day when shock occurs
    
    # Optional parameter changes
    collateral_factor_changes: Optional[Dict[AssetType, float]] = None
    liquidation_threshold_changes: Optional[Dict[AssetType, float]] = None
    target_health_factor_change: Optional[float] = None


class TidalStressTester:
    """Comprehensive stress testing for Tidal Protocol"""
    
    def __init__(self, base_config=None):
        """
        Initialize stress tester
        
        Args:
            base_config: Base configuration for stress tests
        """
        self.base_config = base_config or create_default_config()
        self.results = []
        
    def create_standard_scenarios(self) -> List[StressTestScenario]:
        """Create standard stress test scenarios for Tidal Protocol"""
        
        scenarios = [
            # Baseline scenario
            StressTestScenario(
                name="Baseline",
                description="Normal market conditions with no shocks",
                price_shocks={},
                duration_days=30
            ),
            
            # Single asset crashes
            StressTestScenario(
                name="ETH Crash -30%",
                description="ETH price drops 30% on day 15",
                price_shocks={AssetType.ETH: -0.30},
                duration_days=30,
                shock_timing=15
            ),
            
            StressTestScenario(
                name="BTC Crash -35%",
                description="BTC price drops 35% on day 15",
                price_shocks={AssetType.BTC: -0.35},
                duration_days=30,
                shock_timing=15
            ),
            
            StressTestScenario(
                name="FLOW Crash -50%",
                description="FLOW price drops 50% on day 15",
                price_shocks={AssetType.FLOW: -0.50},
                duration_days=30,
                shock_timing=15
            ),
            
            # Multi-asset scenarios
            StressTestScenario(
                name="Crypto Market Crash",
                description="All crypto assets drop significantly",
                price_shocks={
                    AssetType.ETH: -0.40,
                    AssetType.BTC: -0.35,
                    AssetType.FLOW: -0.60
                },
                duration_days=30,
                shock_timing=15
            ),
            
            StressTestScenario(
                name="Black Swan Event",
                description="Extreme market crash across all assets",
                price_shocks={
                    AssetType.ETH: -0.60,
                    AssetType.BTC: -0.55,
                    AssetType.FLOW: -0.80,
                    AssetType.USDC: -0.05  # Even USDC depegs slightly
                },
                duration_days=45,
                shock_timing=20
            ),
            
            # Gradual decline scenarios
            StressTestScenario(
                name="Bear Market Decline",
                description="Gradual 25% decline over 30 days",
                price_shocks={
                    AssetType.ETH: -0.25,
                    AssetType.BTC: -0.20,
                    AssetType.FLOW: -0.35
                },
                duration_days=30,
                shock_timing=1  # Gradual from day 1
            ),
            
            # USDC depeg scenario
            StressTestScenario(
                name="USDC Depeg",
                description="USDC loses peg and drops 10%",
                price_shocks={AssetType.USDC: -0.10},
                duration_days=30,
                shock_timing=10
            ),
            
            # Positive scenarios
            StressTestScenario(
                name="Bull Market Rally",
                description="All assets pump significantly",
                price_shocks={
                    AssetType.ETH: 0.50,
                    AssetType.BTC: 0.40,
                    AssetType.FLOW: 0.80
                },
                duration_days=30,
                shock_timing=15
            )
        ]
        
        return scenarios
    
    def run_stress_test(self, scenario: StressTestScenario, verbose: bool = False) -> Dict[str, Any]:
        """
        Run a single stress test scenario
        
        Args:
            scenario: Stress test scenario to run
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary containing stress test results
        """
        if verbose:
            print(f"\n🧪 Running Stress Test: {scenario.name}")
            print(f"Description: {scenario.description}")
            print(f"Duration: {scenario.duration_days} days")
            
        # Create modified configuration for this scenario
        config = copy.deepcopy(self.base_config)
        
        # Apply parameter changes if specified
        if scenario.collateral_factor_changes:
            tidal_market = None
            for market in config.simulation.markets:
                if market.market_id == "tidal_protocol":
                    tidal_market = market
                    break
            
            if tidal_market:
                for asset, new_factor in scenario.collateral_factor_changes.items():
                    tidal_market.collateral_factors[asset] = new_factor
        
        # Set simulation duration
        config.simulation.max_days = scenario.duration_days
        
        # Create and run simulation
        simulation = SimulationFactory.create_simulation(config)
        
        # Apply price shocks during simulation
        if scenario.price_shocks:
            simulation.schedule_price_shocks(scenario.price_shocks, scenario.shock_timing)
        
        results = simulation.run_simulation(max_days=scenario.duration_days, verbose=verbose)
        
        # Analyze results
        analysis = self._analyze_stress_test_results(results, scenario)
        
        if verbose:
            self._print_stress_test_summary(analysis)
        
        return {
            'scenario': scenario,
            'simulation_results': results,
            'analysis': analysis
        }
    
    def run_comprehensive_stress_tests(self, scenarios: List[StressTestScenario] = None, 
                                     verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Run comprehensive stress tests across multiple scenarios
        
        Args:
            scenarios: List of scenarios to test (uses standard scenarios if None)
            verbose: Whether to print detailed output
            
        Returns:
            List of stress test results
        """
        if scenarios is None:
            scenarios = self.create_standard_scenarios()
        
        print("🌊 TIDAL PROTOCOL COMPREHENSIVE STRESS TESTING")
        print("=" * 60)
        print(f"Running {len(scenarios)} stress test scenarios...")
        
        results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n[{i}/{len(scenarios)}] {scenario.name}")
            
            try:
                result = self.run_stress_test(scenario, verbose=False)
                results.append(result)
                
                # Quick summary
                analysis = result['analysis']
                print(f"  ✅ Completed - Final Debt Cap: ${analysis['final_debt_cap']:,.0f}")
                print(f"  📊 Liquidations: {analysis['total_liquidations']}")
                print(f"  💰 MOET Price: ${analysis['final_moet_price']:.4f}")
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue
        
        self.results = results
        
        if verbose:
            self._print_comprehensive_summary(results)
        
        return results
    
    def _analyze_stress_test_results(self, results: Dict[str, Any], 
                                   scenario: StressTestScenario) -> Dict[str, Any]:
        """
        Analyze stress test results and extract key metrics
        
        Args:
            results: Simulation results
            scenario: Stress test scenario
            
        Returns:
            Analysis dictionary
        """
        final_state = results.get('final_state', {})
        performance = results.get('performance_metrics', {})
        
        # Extract key metrics
        analysis = {
            'scenario_name': scenario.name,
            'final_debt_cap': final_state.get('debt_cap', 0),
            'final_moet_price': final_state.get('moet_price', 1.0),
            'final_protocol_treasury': final_state.get('protocol_treasury', 0),
            'final_total_liquidity': final_state.get('total_liquidity', 0),
            'total_return': performance.get('total_return', 0),
            'volatility': performance.get('volatility', 0),
            'max_price': performance.get('max_price', 1.0),
            'min_price': performance.get('min_price', 1.0),
            'total_liquidations': 0,  # Would extract from event history
            'liquidation_volume': 0,  # Would extract from event history
            'protocol_revenue': final_state.get('protocol_treasury', 0),
            'lp_rewards_distributed': final_state.get('lp_rewards', 0)
        }
        
        # Calculate risk metrics
        analysis['price_stability'] = self._calculate_price_stability(results)
        analysis['liquidation_efficiency'] = self._calculate_liquidation_efficiency(results)
        analysis['debt_cap_utilization'] = self._calculate_debt_cap_utilization(results)
        analysis['collateral_health'] = self._calculate_collateral_health(results)
        
        # Stress test specific metrics
        analysis['shock_impact_score'] = self._calculate_shock_impact_score(scenario, analysis)
        analysis['recovery_time'] = self._estimate_recovery_time(results, scenario)
        analysis['system_resilience'] = self._calculate_system_resilience(analysis)
        
        return analysis
    
    def _calculate_price_stability(self, results: Dict[str, Any]) -> float:
        """Calculate MOET price stability score (0-1, higher is better)"""
        final_state = results.get('final_state', {})
        moet_price = final_state.get('moet_price', 1.0)
        
        # Calculate deviation from $1.00 peg
        deviation = abs(moet_price - 1.0)
        
        # Score: 1.0 for perfect peg, decreases with deviation
        stability_score = max(0, 1.0 - deviation * 10)  # 10x penalty for deviation
        
        return stability_score
    
    def _calculate_liquidation_efficiency(self, results: Dict[str, Any]) -> float:
        """Calculate liquidation efficiency score (0-1, higher is better)"""
        # Placeholder - would analyze liquidation events from history
        # For now, assume efficient liquidations
        return 0.85
    
    def _calculate_debt_cap_utilization(self, results: Dict[str, Any]) -> float:
        """Calculate debt cap utilization ratio"""
        final_state = results.get('final_state', {})
        debt_cap = final_state.get('debt_cap', 1)
        total_debt = final_state.get('total_debt', 0)
        
        if debt_cap <= 0:
            return 0.0
        
        return min(1.0, total_debt / debt_cap)
    
    def _calculate_collateral_health(self, results: Dict[str, Any]) -> float:
        """Calculate overall collateral health score (0-1, higher is better)"""
        # Placeholder - would analyze individual position health factors
        # For now, assume reasonable health
        return 0.75
    
    def _calculate_shock_impact_score(self, scenario: StressTestScenario, 
                                    analysis: Dict[str, Any]) -> float:
        """
        Calculate how well the protocol handled the price shock
        (0-1, higher is better resilience)
        """
        # Base score
        score = 1.0
        
        # Penalize for MOET depeg
        moet_deviation = abs(analysis['final_moet_price'] - 1.0)
        score -= moet_deviation * 5  # 5x penalty for depeg
        
        # Penalize for low debt cap
        if analysis['final_debt_cap'] < 100000:  # Less than $100k
            score -= 0.3
        
        # Penalize for excessive liquidations (placeholder)
        if analysis['total_liquidations'] > 10:
            score -= 0.2
        
        return max(0, score)
    
    def _estimate_recovery_time(self, results: Dict[str, Any], 
                              scenario: StressTestScenario) -> int:
        """Estimate recovery time in days (placeholder)"""
        # Would analyze price and health factor recovery from history
        shock_magnitude = max([abs(shock) for shock in scenario.price_shocks.values()]) if scenario.price_shocks else 0
        
        # Simple heuristic: larger shocks take longer to recover
        base_recovery = 5  # 5 days base
        shock_penalty = shock_magnitude * 20  # 20 days per 100% shock
        
        return int(base_recovery + shock_penalty)
    
    def _calculate_system_resilience(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall system resilience score (0-1, higher is better)"""
        # Weighted combination of key metrics
        weights = {
            'price_stability': 0.3,
            'liquidation_efficiency': 0.2,
            'collateral_health': 0.2,
            'shock_impact_score': 0.3
        }
        
        resilience = 0.0
        for metric, weight in weights.items():
            resilience += analysis.get(metric, 0) * weight
        
        return resilience
    
    def _print_stress_test_summary(self, analysis: Dict[str, Any]):
        """Print summary of a single stress test"""
        print(f"\n📊 STRESS TEST RESULTS: {analysis['scenario_name']}")
        print("-" * 50)
        print(f"Final Debt Cap: ${analysis['final_debt_cap']:,.0f}")
        print(f"MOET Price: ${analysis['final_moet_price']:.4f}")
        print(f"Protocol Treasury: ${analysis['final_protocol_treasury']:,.0f}")
        print(f"Total Liquidations: {analysis['total_liquidations']}")
        print(f"Price Stability Score: {analysis['price_stability']:.2f}/1.00")
        print(f"System Resilience Score: {analysis['system_resilience']:.2f}/1.00")
        print(f"Estimated Recovery Time: {analysis['recovery_time']} days")
    
    def _print_comprehensive_summary(self, results: List[Dict[str, Any]]):
        """Print comprehensive summary of all stress tests"""
        print("\n🏆 COMPREHENSIVE STRESS TEST SUMMARY")
        print("=" * 60)
        
        # Create summary table
        summary_data = []
        for result in results:
            analysis = result['analysis']
            summary_data.append({
                'Scenario': analysis['scenario_name'],
                'Final Debt Cap': f"${analysis['final_debt_cap']:,.0f}",
                'MOET Price': f"${analysis['final_moet_price']:.4f}",
                'Liquidations': analysis['total_liquidations'],
                'Resilience': f"{analysis['system_resilience']:.2f}",
                'Recovery Days': analysis['recovery_time']
            })
        
        # Print table
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Overall statistics
        resilience_scores = [r['analysis']['system_resilience'] for r in results]
        
        if resilience_scores:
            print(f"\nOVERALL STATISTICS:")
            print(f"Average Resilience Score: {np.mean(resilience_scores):.2f}/1.00")
            print(f"Worst Case Resilience: {np.min(resilience_scores):.2f}/1.00")
            print(f"Best Case Resilience: {np.max(resilience_scores):.2f}/1.00")
        else:
            print(f"\nOVERALL STATISTICS:")
            print(f"No successful stress tests completed.")
        
        # Identify most vulnerable scenarios
        if results:
            worst_scenarios = sorted(results, key=lambda x: x['analysis']['system_resilience'])[:3]
            print(f"\nMOST VULNERABLE SCENARIOS:")
            for i, result in enumerate(worst_scenarios, 1):
                analysis = result['analysis']
                print(f"{i}. {analysis['scenario_name']} (Resilience: {analysis['system_resilience']:.2f})")
    
    def generate_stress_test_report(self, output_file: str = "tidal_stress_test_report.html"):
        """
        Generate comprehensive HTML report of stress test results
        
        Args:
            output_file: Output HTML file path
        """
        if not self.results:
            print("No stress test results available. Run stress tests first.")
            return
        
        # Create visualizations
        charts = TidalProtocolCharts([r['simulation_results'] for r in self.results])
        
        # Generate price shock scenarios for charts
        price_shocks = []
        for result in self.results:
            scenario = result['scenario']
            if scenario.price_shocks:
                avg_shock = np.mean(list(scenario.price_shocks.values()))
                price_shocks.append(avg_shock)
            else:
                price_shocks.append(0.0)
        
        # Create figures
        fig1 = charts.plot_liquidity_vs_debt_cap()
        fig2 = charts.plot_stress_test_results(price_shocks)
        fig3 = charts.create_comprehensive_dashboard(price_shocks)
        
        # Save figures
        fig1.savefig('stress_test_liquidity_debt_cap.png', dpi=300, bbox_inches='tight')
        fig2.savefig('stress_test_scenarios.png', dpi=300, bbox_inches='tight')
        fig3.savefig('stress_test_dashboard.png', dpi=300, bbox_inches='tight')
        
        print(f"📈 Stress test visualizations saved:")
        print(f"  - stress_test_liquidity_debt_cap.png")
        print(f"  - stress_test_scenarios.png") 
        print(f"  - stress_test_dashboard.png")
        
        # Note: Full HTML report generation would require additional HTML templating
        print(f"\n📋 For full HTML report generation, integrate with HTML templating library")


def run_tidal_stress_tests():
    """Main function to run Tidal Protocol stress tests"""
    
    # Initialize stress tester
    stress_tester = TidalStressTester()
    
    # Create scenarios
    scenarios = stress_tester.create_standard_scenarios()
    
    # Run comprehensive stress tests
    results = stress_tester.run_comprehensive_stress_tests(scenarios, verbose=True)
    
    # Generate visualizations
    stress_tester.generate_stress_test_report()
    
    return results


if __name__ == "__main__":
    run_tidal_stress_tests()
