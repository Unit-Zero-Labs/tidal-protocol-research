#!/usr/bin/env python3
"""
Results Analysis and Metrics

Analysis tools for stress test results focusing on protocol stability metrics.
"""

import numpy as np
from typing import Dict, List, Any
from ..core.protocol import Asset


class StressTestAnalyzer:
    """Results analysis and metrics calculation"""
    
    def analyze_monte_carlo_results(self, scenario_name: str, runs_results: List[Dict]) -> Dict:
        """Analyze Monte Carlo stress test results"""
        
        if not runs_results:
            return {"error": "No results to analyze"}
        
        # Extract key metrics across all runs
        metrics = {
            "protocol_treasury": [],
            "total_liquidations": [],
            "agent_health_factors": [],
            "debt_cap_utilization": [],
            "total_supplied": [],
            "total_borrowed": [],
            "moet_price_stability": []
        }
        
        for result in runs_results:
            if "error" in result:
                continue
            
            # Extract from summary statistics (more reliable)
            summary_stats = result.get("summary_statistics", {})
            final_state = result.get("final_protocol_state", {})
            
            # Protocol metrics
            metrics["protocol_treasury"].append(summary_stats.get("final_protocol_treasury", 0))
            
            # Liquidation metrics
            liquidation_events = result.get("liquidation_events", [])
            metrics["total_liquidations"].append(len(liquidation_events))
            
            # Agent health factors
            min_hf = summary_stats.get("min_health_factor", 1.0)
            avg_hf = summary_stats.get("avg_health_factor", 1.0)
            if avg_hf != float('inf') and avg_hf > 0:
                metrics["agent_health_factors"].append(avg_hf)
            
            # Debt cap utilization
            debt_cap = final_state.get("debt_cap", 1)
            total_borrowed = summary_stats.get("final_total_borrowed", 0)
            if debt_cap > 0:
                utilization = total_borrowed / debt_cap
                metrics["debt_cap_utilization"].append(min(utilization, 1.0))
            
            # Supply/borrow metrics
            metrics["total_supplied"].append(summary_stats.get("final_total_supplied", 0))
            metrics["total_borrowed"].append(total_borrowed)
            
            # MOET price stability
            metrics_history = result.get("metrics_history", [])
            if metrics_history:
                final_prices = metrics_history[-1].get("asset_prices", {})
                # Handle both string and Asset enum keys
                moet_price = final_prices.get("MOET", final_prices.get("Asset.MOET", 1.0))
                price_deviation = abs(moet_price - 1.0)
                metrics["moet_price_stability"].append(price_deviation)
        
        # Calculate statistics
        stats = {}
        for metric_name, values in metrics.items():
            if values:
                stats[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "percentile_5": np.percentile(values, 5),
                    "percentile_25": np.percentile(values, 25),
                    "percentile_75": np.percentile(values, 75),
                    "percentile_95": np.percentile(values, 95)
                }
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(metrics)
        
        # Generate scenario assessment
        assessment = self._assess_scenario_impact(scenario_name, stats, risk_metrics)
        
        return {
            "scenario_name": scenario_name,
            "num_runs": len(runs_results),
            "num_successful_runs": len([r for r in runs_results if "error" not in r]),
            "statistics": stats,
            "risk_metrics": risk_metrics,
            "assessment": assessment
        }
    
    def analyze_single_scenario(self, scenario_name: str, result: Dict) -> Dict:
        """Analyze single scenario result"""
        
        if "error" in result:
            return {"error": result["error"]}
        
        # Extract key information
        final_state = result.get("final_protocol_state", {})
        liquidation_events = result.get("liquidation_events", [])
        trade_events = result.get("trade_events", [])
        agent_states = result.get("agent_states", {})
        metrics_history = result.get("metrics_history", [])
        
        # Protocol health analysis
        protocol_health = self._analyze_protocol_health(final_state, metrics_history)
        
        # Liquidation analysis
        liquidation_analysis = self._analyze_liquidations(liquidation_events)
        
        # Agent analysis
        agent_analysis = self._analyze_agents(agent_states)
        
        # Market dynamics
        market_analysis = self._analyze_market_dynamics(metrics_history)
        
        # Overall assessment
        overall_score = self._calculate_scenario_score(
            protocol_health, liquidation_analysis, agent_analysis
        )
        
        return {
            "scenario_name": scenario_name,
            "protocol_health": protocol_health,
            "liquidation_analysis": liquidation_analysis,
            "agent_analysis": agent_analysis,
            "market_analysis": market_analysis,
            "overall_score": overall_score,
            "key_insights": self._generate_insights(scenario_name, result)
        }
    
    def generate_suite_summary(self, suite_results: Dict[str, Dict]) -> Dict:
        """Generate comprehensive summary of entire test suite"""
        
        if not suite_results:
            return {"message": "No results to summarize"}
        
        # Count successful vs failed scenarios
        successful = len([r for r in suite_results.values() if "error" not in r])
        failed = len(suite_results) - successful
        
        # Extract key metrics across all scenarios
        all_scores = []
        protocol_resilience = []
        liquidation_efficiency = []
        
        scenario_rankings = []
        
        for scenario_name, result in suite_results.items():
            if "error" in result:
                continue
            
            # Extract overall score if available
            if "assessment" in result and "overall_score" in result["assessment"]:
                score = result["assessment"]["overall_score"]
                all_scores.append(score)
                scenario_rankings.append((scenario_name, score))
            
            # Extract specific metrics
            if "risk_metrics" in result:
                risk_metrics = result["risk_metrics"]
                if "protocol_resilience" in risk_metrics:
                    protocol_resilience.append(risk_metrics["protocol_resilience"])
                if "liquidation_efficiency" in risk_metrics:
                    liquidation_efficiency.append(risk_metrics["liquidation_efficiency"])
        
        # Sort scenarios by risk level
        scenario_rankings.sort(key=lambda x: x[1])  # Lower score = higher risk
        
        # Generate recommendations
        recommendations = self._generate_recommendations(suite_results)
        
        return {
            "suite_statistics": {
                "total_scenarios": len(suite_results),
                "successful_runs": successful,
                "failed_runs": failed,
                "success_rate": successful / len(suite_results) if suite_results else 0
            },
            "overall_metrics": {
                "average_resilience_score": np.mean(all_scores) if all_scores else 0,
                "protocol_resilience": {
                    "mean": np.mean(protocol_resilience) if protocol_resilience else 0,
                    "min": np.min(protocol_resilience) if protocol_resilience else 0
                },
                "liquidation_efficiency": {
                    "mean": np.mean(liquidation_efficiency) if liquidation_efficiency else 0,
                    "min": np.min(liquidation_efficiency) if liquidation_efficiency else 0
                }
            },
            "scenario_rankings": {
                "highest_risk": scenario_rankings[:3] if scenario_rankings else [],
                "lowest_risk": scenario_rankings[-3:] if scenario_rankings else [],
                "all_scenarios": scenario_rankings
            },
            "recommendations": recommendations,
            "critical_findings": self._identify_critical_findings(suite_results)
        }
    
    def _calculate_risk_metrics(self, metrics: Dict[str, List]) -> Dict:
        """Calculate protocol-specific risk metrics"""
        
        risk_metrics = {}
        
        # Protocol resilience (based on treasury and liquidation efficiency)
        treasury_values = metrics.get("protocol_treasury", [])
        if treasury_values:
            # Higher treasury = more resilient
            treasury_score = np.mean(treasury_values) / 10000  # Normalize
            risk_metrics["protocol_resilience"] = min(treasury_score, 1.0)
        
        # Liquidation efficiency
        liquidations = metrics.get("total_liquidations", [])
        health_factors = metrics.get("agent_health_factors", [])
        
        if liquidations and health_factors:
            # More liquidations when health factors are low = efficient
            avg_liquidations = np.mean(liquidations)
            avg_health = np.mean(health_factors)
            
            if avg_health < 1.2:  # Stress condition
                efficiency = min(avg_liquidations / 10.0, 1.0)  # Normalize
            else:
                efficiency = 1.0  # No stress, no liquidations needed
            
            risk_metrics["liquidation_efficiency"] = efficiency
        
        # Debt cap safety
        debt_cap_utilizations = metrics.get("debt_cap_utilization", [])
        if debt_cap_utilizations:
            max_utilization = np.max(debt_cap_utilizations)
            # Lower utilization = safer
            debt_cap_safety = max(0.0, 1.0 - max_utilization)
            risk_metrics["debt_cap_safety"] = debt_cap_safety
        
        # MOET stability
        price_deviations = metrics.get("moet_price_stability", [])
        if price_deviations:
            avg_deviation = np.mean(price_deviations)
            # Lower deviation = more stable
            stability_score = max(0.0, 1.0 - (avg_deviation / 0.1))  # 10% deviation = 0 score
            risk_metrics["moet_stability"] = stability_score
        
        return risk_metrics
    
    def _assess_scenario_impact(self, scenario_name: str, stats: Dict, risk_metrics: Dict) -> Dict:
        """Assess overall impact of scenario"""
        
        # Calculate weighted overall score
        weights = {
            "protocol_resilience": 0.3,
            "liquidation_efficiency": 0.25,
            "debt_cap_safety": 0.25,
            "moet_stability": 0.2
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in risk_metrics:
                overall_score += risk_metrics[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            overall_score /= total_weight
        
        # Determine risk level
        if overall_score >= 0.8:
            risk_level = "LOW"
        elif overall_score >= 0.6:
            risk_level = "MEDIUM"
        elif overall_score >= 0.4:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        return {
            "overall_score": overall_score,
            "risk_level": risk_level,
            "key_concerns": self._identify_key_concerns(scenario_name, stats, risk_metrics)
        }
    
    def _analyze_protocol_health(self, final_state: Dict, metrics_history: List) -> Dict:
        """Analyze protocol health metrics"""
        
        return {
            "treasury_balance": final_state.get("protocol_treasury", 0),
            "debt_cap_utilization": final_state.get("total_borrowed", 0) / max(final_state.get("debt_cap", 1), 1),
            "total_supplied": final_state.get("total_supplied", 0),
            "total_borrowed": final_state.get("total_borrowed", 0),
            "utilization_rates": final_state.get("utilization_rates", {}),
            "health_status": "HEALTHY" if final_state.get("protocol_treasury", 0) > 0 else "AT_RISK"
        }
    
    def _analyze_liquidations(self, liquidation_events: List) -> Dict:
        """Analyze liquidation events"""
        
        if not liquidation_events:
            return {
                "total_liquidations": 0,
                "efficiency_score": 1.0,  # No liquidations needed = perfect
                "average_size": 0,
                "timing_analysis": "No liquidations occurred"
            }
        
        sizes = [event.get("repay_amount", 0) for event in liquidation_events]
        
        return {
            "total_liquidations": len(liquidation_events),
            "efficiency_score": min(len(liquidation_events) / 20.0, 1.0),  # Up to 20 = efficient
            "average_size": np.mean(sizes) if sizes else 0,
            "total_volume": sum(sizes),
            "timing_analysis": f"Liquidations occurred from step {liquidation_events[0].get('step', 0)} to {liquidation_events[-1].get('step', 0)}"
        }
    
    def _analyze_agents(self, agent_states: Dict) -> Dict:
        """Analyze agent states"""
        
        health_factors = []
        total_losses = 0
        healthy_agents = 0
        
        for agent in agent_states.values():
            hf = agent.get("health_factor", float('inf'))
            if hf != float('inf'):
                health_factors.append(hf)
            
            if hf >= 1.2:
                healthy_agents += 1
            
            pnl = agent.get("profit_loss", 0)
            if pnl < 0:
                total_losses += abs(pnl)
        
        return {
            "total_agents": len(agent_states),
            "healthy_agents": healthy_agents,
            "avg_health_factor": np.mean(health_factors) if health_factors else float('inf'),
            "min_health_factor": np.min(health_factors) if health_factors else float('inf'),
            "total_agent_losses": total_losses,
            "agent_survival_rate": healthy_agents / len(agent_states) if agent_states else 0
        }
    
    def _analyze_market_dynamics(self, metrics_history: List) -> Dict:
        """Analyze market dynamics over time"""
        
        if not metrics_history:
            return {"message": "No metrics history available"}
        
        # Track price movements
        initial_prices = metrics_history[0].get("asset_prices", {})
        final_prices = metrics_history[-1].get("asset_prices", {})
        
        price_changes = {}
        for asset, initial_price in initial_prices.items():
            final_price = final_prices.get(asset, initial_price)
            change = (final_price - initial_price) / initial_price if initial_price > 0 else 0
            price_changes[asset] = change
        
        return {
            "price_changes": price_changes,
            "max_price_drop": min(price_changes.values()) if price_changes else 0,
            "volatility_period": len(metrics_history),
            "market_stress_detected": any(abs(change) > 0.1 for change in price_changes.values())
        }
    
    def _calculate_scenario_score(self, protocol_health: Dict, liquidation_analysis: Dict, agent_analysis: Dict) -> float:
        """Calculate overall scenario score"""
        
        # Components of the score (0-1 scale)
        protocol_score = 1.0 if protocol_health.get("health_status") == "HEALTHY" else 0.5
        liquidation_score = liquidation_analysis.get("efficiency_score", 0.5)
        agent_score = agent_analysis.get("agent_survival_rate", 0.0)
        
        # Weighted average
        overall_score = (protocol_score * 0.4 + liquidation_score * 0.3 + agent_score * 0.3)
        
        return overall_score
    
    def _generate_insights(self, scenario_name: str, result: Dict) -> List[str]:
        """Generate key insights for scenario"""
        
        insights = []
        
        # Add scenario-specific insights based on results
        liquidations = len(result.get("liquidation_events", []))
        if liquidations > 10:
            insights.append(f"High liquidation activity ({liquidations} events) indicates effective liquidation mechanism")
        elif liquidations == 0:
            insights.append("No liquidations occurred - either no stress or ineffective liquidation system")
        
        # Treasury insights
        treasury = result.get("final_protocol_state", {}).get("protocol_treasury", 0)
        if treasury > 5000:
            insights.append(f"Protocol generated significant revenue (${treasury:.0f}) from stress event")
        elif treasury < 1000:
            insights.append("Low protocol revenue generation during stress event")
        
        return insights
    
    def _generate_recommendations(self, suite_results: Dict) -> List[str]:
        """Generate protocol recommendations based on all test results"""
        
        recommendations = []
        
        # Analyze patterns across all scenarios
        high_risk_scenarios = []
        liquidation_issues = []
        
        for scenario_name, result in suite_results.items():
            if "error" in result:
                continue
            
            if "assessment" in result:
                risk_level = result["assessment"].get("risk_level", "UNKNOWN")
                if risk_level in ["HIGH", "CRITICAL"]:
                    high_risk_scenarios.append(scenario_name)
        
        if high_risk_scenarios:
            recommendations.append(f"Priority: Address {len(high_risk_scenarios)} high-risk scenarios: {', '.join(high_risk_scenarios[:3])}")
        
        recommendations.append("Consider implementing additional safety mechanisms for extreme market conditions")
        recommendations.append("Monitor debt cap utilization closely during market stress")
        recommendations.append("Ensure liquidation bot incentives are sufficient for rapid response")
        
        return recommendations
    
    def _identify_critical_findings(self, suite_results: Dict) -> List[str]:
        """Identify critical findings across all tests"""
        
        findings = []
        
        # Look for systematic issues
        critical_scenarios = sum(1 for result in suite_results.values() 
                               if result.get("assessment", {}).get("risk_level") == "CRITICAL")
        
        if critical_scenarios > 0:
            findings.append(f"CRITICAL: {critical_scenarios} scenarios pose critical risk to protocol")
        
        # Check for MOET stability issues
        moet_issues = sum(1 for result in suite_results.values()
                         if result.get("risk_metrics", {}).get("moet_stability", 1.0) < 0.5)
        
        if moet_issues > len(suite_results) * 0.3:  # More than 30% of scenarios
            findings.append("WARNING: MOET peg stability at risk in multiple scenarios")
        
        return findings
    
    def _identify_key_concerns(self, scenario_name: str, stats: Dict, risk_metrics: Dict) -> List[str]:
        """Identify key concerns for specific scenario"""
        
        concerns = []
        
        # Check individual risk metrics
        if risk_metrics.get("protocol_resilience", 1.0) < 0.5:
            concerns.append("Low protocol resilience - insufficient treasury generation")
        
        if risk_metrics.get("liquidation_efficiency", 1.0) < 0.5:
            concerns.append("Poor liquidation efficiency - positions may become insolvent")
        
        if risk_metrics.get("debt_cap_safety", 1.0) < 0.3:
            concerns.append("Debt cap near maximum - high systemic risk")
        
        if risk_metrics.get("moet_stability", 1.0) < 0.5:
            concerns.append("MOET price instability - peg at risk")
        
        return concerns