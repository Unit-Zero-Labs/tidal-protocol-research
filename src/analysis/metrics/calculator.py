#!/usr/bin/env python3
"""
Comprehensive metrics calculator for the Tidal Protocol simulation.

This module calculates various metrics and statistics from simulation results,
following the analysis patterns from the architecture blueprint.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from ...core.simulation.primitives import Asset


@dataclass
class MetricsSummary:
    """Container for comprehensive metrics summary"""
    price_metrics: Dict[str, Any]
    protocol_metrics: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    agent_metrics: Dict[str, Any]
    market_metrics: Dict[str, Any]


class TokenomicsMetricsCalculator:
    """Calculate comprehensive tokenomics metrics from simulation results"""
    
    def __init__(self, simulation_results: List[Dict[str, Any]]):
        """
        Initialize with simulation results
        
        Args:
            simulation_results: List of simulation result dictionaries
        """
        self.results = simulation_results
        self.n_simulations = len(simulation_results)
        
        if self.n_simulations == 0:
            raise ValueError("No simulation results provided")
    
    def calculate_comprehensive_metrics(self) -> MetricsSummary:
        """Calculate all metrics categories"""
        return MetricsSummary(
            price_metrics=self.calculate_price_metrics(),
            protocol_metrics=self.calculate_protocol_metrics(),
            risk_metrics=self.calculate_risk_metrics(),
            agent_metrics=self.calculate_agent_metrics(),
            market_metrics=self.calculate_market_metrics()
        )
    
    def calculate_price_metrics(self) -> Dict[str, Any]:
        """Calculate price evolution, volatility, and appreciation scenarios"""
        # Extract price data
        final_prices = []
        price_changes = []
        volatilities = []
        max_prices = []
        min_prices = []
        
        for result in self.results:
            history = result.get('history', {})
            price_history = history.get('prices', [])
            
            if not price_history:
                continue
            
            # Extract MOET prices
            moet_prices = [
                prices.get(Asset.MOET, 1.0) for prices in price_history
            ]
            
            if len(moet_prices) < 2:
                continue
            
            # Final price
            final_price = moet_prices[-1]
            final_prices.append(final_price)
            
            # Price change from start to end
            initial_price = moet_prices[0]
            if initial_price > 0:
                price_change = (final_price - initial_price) / initial_price
                price_changes.append(price_change)
            
            # Volatility (standard deviation of daily returns)
            daily_returns = []
            for i in range(1, len(moet_prices)):
                if moet_prices[i-1] > 0:
                    daily_return = (moet_prices[i] - moet_prices[i-1]) / moet_prices[i-1]
                    daily_returns.append(daily_return)
            
            if daily_returns:
                volatility = np.std(daily_returns)
                volatilities.append(volatility)
            
            # Price extremes
            max_prices.append(max(moet_prices))
            min_prices.append(min(moet_prices))
        
        # Calculate statistics
        price_stats = self._calculate_distribution_stats(final_prices, "Final Price")
        volatility_stats = self._calculate_distribution_stats(volatilities, "Volatility")
        
        # Price appreciation scenarios
        positive_returns = [pc for pc in price_changes if pc > 0]
        negative_returns = [pc for pc in price_changes if pc < 0]
        
        appreciation_probability = len(positive_returns) / len(price_changes) if price_changes else 0
        
        return {
            'final_price_stats': price_stats,
            'price_volatility': volatility_stats,
            'price_appreciation': {
                'mean_change': np.mean(price_changes) if price_changes else 0,
                'median_change': np.median(price_changes) if price_changes else 0,
                'positive_return_probability': appreciation_probability,
                'mean_positive_return': np.mean(positive_returns) if positive_returns else 0,
                'mean_negative_return': np.mean(negative_returns) if negative_returns else 0
            },
            'price_extremes': {
                'max_price_mean': np.mean(max_prices) if max_prices else 0,
                'min_price_mean': np.mean(min_prices) if min_prices else 0,
                'max_drawdown': self._calculate_max_drawdown(price_changes)
            }
        }
    
    def calculate_protocol_metrics(self) -> Dict[str, Any]:
        """Calculate treasury health, market cap analysis"""
        final_treasuries = []
        market_caps = []
        total_liquidity = []
        protocol_revenues = []
        
        for result in self.results:
            final_state = result.get('final_state', {})
            
            # Treasury metrics
            treasury = final_state.get('protocol_treasury', 0)
            final_treasuries.append(treasury)
            
            # Market cap
            market_cap = final_state.get('market_cap', 0)
            market_caps.append(market_cap)
            
            # Total liquidity
            liquidity = final_state.get('total_liquidity', 0)
            total_liquidity.append(liquidity)
            
            # Protocol revenue (if available in history)
            history = result.get('history', {})
            metrics_history = history.get('metrics', [])
            
            if metrics_history:
                # Sum up protocol revenue over time
                total_revenue = sum(
                    metrics.get('protocol_treasury', 0) 
                    for metrics in metrics_history
                )
                protocol_revenues.append(total_revenue)
        
        # Treasury health analysis
        positive_treasury_count = sum(1 for t in final_treasuries if t > 0)
        treasury_success_rate = positive_treasury_count / len(final_treasuries) if final_treasuries else 0
        
        return {
            'treasury_health': {
                'final_balance_stats': self._calculate_distribution_stats(final_treasuries, "Treasury Balance"),
                'positive_balance_probability': treasury_success_rate,
                'bankruptcy_risk': 1 - treasury_success_rate
            },
            'market_cap_analysis': {
                'final_market_cap_stats': self._calculate_distribution_stats(market_caps, "Market Cap"),
                'growth_scenarios': self._calculate_growth_scenarios(market_caps)
            },
            'liquidity_analysis': {
                'total_liquidity_stats': self._calculate_distribution_stats(total_liquidity, "Total Liquidity")
            },
            'revenue_analysis': {
                'protocol_revenue_stats': self._calculate_distribution_stats(protocol_revenues, "Protocol Revenue")
            } if protocol_revenues else {}
        }
    
    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """Calculate VaR, price movement scenarios, stability ratios"""
        returns = []
        health_factors = []
        utilization_rates = []
        
        for result in self.results:
            # Extract returns
            performance = result.get('performance_metrics', {})
            total_return = performance.get('total_return', 0)
            returns.append(total_return)
            
            # Extract health factors and utilization from history
            history = result.get('history', {})
            metrics_history = history.get('metrics', [])
            
            for metrics in metrics_history:
                agent_metrics = metrics.get('agent_metrics', {})
                
                # This would need to be extracted from agent states
                # For now, use placeholder values
                health_factors.append(1.5)  # Placeholder
                utilization_rates.append(0.5)  # Placeholder
        
        # Value at Risk calculations
        var_95 = np.percentile(returns, 5) if returns else 0
        var_99 = np.percentile(returns, 1) if returns else 0
        
        # Extreme movement scenarios
        extreme_positive = [r for r in returns if r > 0.5]  # >50% gains
        extreme_negative = [r for r in returns if r < -0.2]  # >20% losses
        
        return {
            'value_at_risk': {
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall_95': np.mean([r for r in returns if r <= var_95]) if returns else 0
            },
            'price_movement_scenarios': {
                'extreme_positive_probability': len(extreme_positive) / len(returns) if returns else 0,
                'extreme_negative_probability': len(extreme_negative) / len(returns) if returns else 0,
                'stable_price_probability': len([r for r in returns if -0.1 <= r <= 0.1]) / len(returns) if returns else 0
            },
            'risk_distribution': {
                'return_stats': self._calculate_distribution_stats(returns, "Returns"),
                'downside_deviation': self._calculate_downside_deviation(returns),
                'sharpe_ratio': self._calculate_sharpe_ratio(returns)
            }
        }
    
    def calculate_agent_metrics(self) -> Dict[str, Any]:
        """Calculate agent behavior and performance metrics"""
        agent_counts = []
        total_values = []
        policy_distributions = {}
        
        for result in self.results:
            # Extract agent metrics from final state or history
            history = result.get('history', {})
            metrics_history = history.get('metrics', [])
            
            if metrics_history:
                final_metrics = metrics_history[-1]
                agent_metrics = final_metrics.get('agent_metrics', {})
                
                agent_counts.append(agent_metrics.get('total_agents', 0))
                total_values.append(agent_metrics.get('total_value_usd', 0))
                
                # Policy distribution
                policy_dist = agent_metrics.get('policy_distribution', {})
                for policy, count in policy_dist.items():
                    if policy not in policy_distributions:
                        policy_distributions[policy] = []
                    policy_distributions[policy].append(count)
        
        return {
            'population_stats': {
                'agent_count_stats': self._calculate_distribution_stats(agent_counts, "Agent Count"),
                'total_value_stats': self._calculate_distribution_stats(total_values, "Total Agent Value")
            },
            'policy_analysis': {
                policy: self._calculate_distribution_stats(counts, f"{policy} Count")
                for policy, counts in policy_distributions.items()
            },
            'behavior_patterns': {
                # Placeholder for more sophisticated agent behavior analysis
                'average_actions_per_agent': 0,  # Would need to calculate from event history
                'most_active_policy': max(policy_distributions.keys(), key=lambda k: np.mean(policy_distributions[k])) if policy_distributions else None
            }
        }
    
    def calculate_market_metrics(self) -> Dict[str, Any]:
        """Calculate market-specific performance metrics"""
        market_data = {}
        
        for result in self.results:
            # Extract market data from history
            history = result.get('history', {})
            metrics_history = history.get('metrics', [])
            
            for metrics in metrics_history:
                # Market-specific metrics would be extracted here
                # This is a placeholder structure
                pass
        
        return {
            'trading_volume': {
                'total_volume_stats': {},  # Placeholder
                'volume_trends': {}
            },
            'liquidity_metrics': {
                'pool_utilization': {},  # Placeholder
                'fee_generation': {}
            },
            'market_efficiency': {
                'price_discovery': {},  # Placeholder
                'slippage_analysis': {}
            }
        }
    
    def _calculate_distribution_stats(self, data: List[float], name: str) -> Dict[str, Any]:
        """Calculate comprehensive distribution statistics"""
        if not data:
            return {
                'count': 0,
                'mean': 0,
                'std': 0,
                'min': 0,
                'max': 0,
                'percentiles': {}
            }
        
        data_array = np.array(data)
        
        return {
            'count': len(data),
            'mean': np.mean(data_array),
            'std': np.std(data_array),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'percentiles': {
                '5th': np.percentile(data_array, 5),
                '25th': np.percentile(data_array, 25),
                '50th': np.percentile(data_array, 50),
                '75th': np.percentile(data_array, 75),
                '95th': np.percentile(data_array, 95)
            }
        }
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        if not returns:
            return 0.0
        
        cumulative_returns = np.cumprod([1 + r for r in returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        
        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    
    def _calculate_growth_scenarios(self, market_caps: List[float]) -> Dict[str, Any]:
        """Calculate market cap growth scenarios"""
        if not market_caps:
            return {}
        
        # Assume initial market cap of 1M (would be configurable)
        initial_market_cap = 1000000
        
        growth_rates = []
        for market_cap in market_caps:
            if initial_market_cap > 0:
                growth_rate = (market_cap - initial_market_cap) / initial_market_cap
                growth_rates.append(growth_rate)
        
        high_growth = len([g for g in growth_rates if g > 1.0])  # >100% growth
        moderate_growth = len([g for g in growth_rates if 0.2 <= g <= 1.0])  # 20-100% growth
        low_growth = len([g for g in growth_rates if 0 <= g < 0.2])  # 0-20% growth
        negative_growth = len([g for g in growth_rates if g < 0])  # Negative growth
        
        total = len(growth_rates)
        
        return {
            'high_growth_probability': high_growth / total if total > 0 else 0,
            'moderate_growth_probability': moderate_growth / total if total > 0 else 0,
            'low_growth_probability': low_growth / total if total > 0 else 0,
            'negative_growth_probability': negative_growth / total if total > 0 else 0,
            'mean_growth_rate': np.mean(growth_rates) if growth_rates else 0
        }
    
    def _calculate_downside_deviation(self, returns: List[float]) -> float:
        """Calculate downside deviation (volatility of negative returns)"""
        if not returns:
            return 0.0
        
        negative_returns = [r for r in returns if r < 0]
        return np.std(negative_returns) if negative_returns else 0.0
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return (mean_return - risk_free_rate) / std_return
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report of all metrics"""
        metrics = self.calculate_comprehensive_metrics()
        
        report = []
        report.append("=== TIDAL PROTOCOL SIMULATION METRICS SUMMARY ===\n")
        
        # Price metrics
        price_stats = metrics.price_metrics['final_price_stats']
        report.append("PRICE ANALYSIS:")
        report.append(f"  Final Price - Mean: ${price_stats['mean']:.4f}, Std: ${price_stats['std']:.4f}")
        report.append(f"  Price Range: ${price_stats['min']:.4f} - ${price_stats['max']:.4f}")
        report.append(f"  Appreciation Probability: {metrics.price_metrics['price_appreciation']['positive_return_probability']:.1%}")
        report.append("")
        
        # Protocol metrics
        treasury_stats = metrics.protocol_metrics['treasury_health']['final_balance_stats']
        report.append("PROTOCOL HEALTH:")
        report.append(f"  Treasury Balance - Mean: ${treasury_stats['mean']:,.0f}")
        report.append(f"  Positive Balance Probability: {metrics.protocol_metrics['treasury_health']['positive_balance_probability']:.1%}")
        report.append("")
        
        # Risk metrics
        risk_stats = metrics.risk_metrics['value_at_risk']
        report.append("RISK ANALYSIS:")
        report.append(f"  Value at Risk (95%): {risk_stats['var_95']:.2%}")
        report.append(f"  Value at Risk (99%): {risk_stats['var_99']:.2%}")
        report.append("")
        
        return "\n".join(report)
