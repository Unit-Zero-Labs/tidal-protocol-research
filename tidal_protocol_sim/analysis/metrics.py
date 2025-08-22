#!/usr/bin/env python3
"""
Protocol Stability Metrics

Essential metrics focusing on protocol stability as specified in refactoring requirements.
"""

import numpy as np
from typing import Dict, List, Tuple
from ..core.protocol import Asset, TidalProtocol


class TidalMetricsCalculator:
    """Protocol stability metrics calculator"""
    
    def __init__(self, protocol: TidalProtocol):
        self.protocol = protocol
    
    def calculate_protocol_health_score(self, current_prices: Dict[Asset, float]) -> float:
        """Calculate overall protocol health score (0-1)"""
        
        metrics = {
            "debt_cap_safety": self._debt_cap_safety_score(current_prices),
            "liquidation_readiness": self._liquidation_readiness_score(),
            "moet_peg_stability": self._moet_peg_stability_score(),
            "utilization_balance": self._utilization_balance_score(),
            "treasury_health": self._treasury_health_score()
        }
        
        # Weighted average
        weights = {
            "debt_cap_safety": 0.25,
            "liquidation_readiness": 0.25,
            "moet_peg_stability": 0.20,
            "utilization_balance": 0.15,
            "treasury_health": 0.15
        }
        
        health_score = sum(metrics[key] * weights[key] for key in metrics)
        
        return {
            "overall_health_score": health_score,
            "component_scores": metrics,
            "health_status": self._categorize_health(health_score)
        }
    
    def calculate_debt_cap_metrics(self, current_prices: Dict[Asset, float]) -> Dict:
        """Calculate debt cap related metrics"""
        
        debt_cap = self.protocol.calculate_debt_cap()
        
        # Calculate current total MOET debt
        total_debt = sum(pool.total_borrowed for pool in self.protocol.asset_pools.values() 
                        if hasattr(pool, 'total_borrowed'))  # Only MOET debt
        
        # Debt cap utilization
        utilization = total_debt / debt_cap if debt_cap > 0 else 0
        
        # Available debt capacity
        available_capacity = max(0, debt_cap - total_debt)
        
        # Safety buffer analysis
        safety_buffer = available_capacity / debt_cap if debt_cap > 0 else 0
        
        # Liquidation capacity breakdown
        liquidation_breakdown = self._analyze_liquidation_capacity()
        
        return {
            "debt_cap": debt_cap,
            "total_debt": total_debt,
            "utilization_rate": utilization,
            "available_capacity": available_capacity,
            "safety_buffer": safety_buffer,
            "utilization_status": self._categorize_debt_cap_utilization(utilization),
            "liquidation_breakdown": liquidation_breakdown
        }
    
    def calculate_liquidation_metrics(self, agent_states: List[Dict], current_prices: Dict[Asset, float]) -> Dict:
        """Calculate liquidation-related metrics"""
        
        liquidatable_positions = []
        at_risk_positions = []
        healthy_positions = []
        
        total_collateral_at_risk = 0.0
        total_debt_at_risk = 0.0
        
        for agent_state in agent_states:
            health_factor = agent_state.get("health_factor", float('inf'))
            collateral_value = agent_state.get("total_collateral_value", 0.0)
            debt_value = agent_state.get("total_debt_value", 0.0)
            
            if health_factor < 1.0:
                liquidatable_positions.append(agent_state)
                total_collateral_at_risk += collateral_value
                total_debt_at_risk += debt_value
            elif health_factor < 1.2:
                at_risk_positions.append(agent_state)
            else:
                healthy_positions.append(agent_state)
        
        # Calculate liquidation efficiency potential
        liquidation_capacity = self._calculate_total_liquidation_capacity()
        liquidation_coverage = liquidation_capacity / total_collateral_at_risk if total_collateral_at_risk > 0 else float('inf')
        
        return {
            "liquidatable_positions": len(liquidatable_positions),
            "at_risk_positions": len(at_risk_positions),
            "healthy_positions": len(healthy_positions),
            "total_positions": len(agent_states),
            "collateral_at_risk": total_collateral_at_risk,
            "debt_at_risk": total_debt_at_risk,
            "liquidation_capacity": liquidation_capacity,
            "liquidation_coverage_ratio": liquidation_coverage,
            "liquidation_efficiency": self._assess_liquidation_efficiency(liquidation_coverage)
        }
    
    def calculate_moet_stability_metrics(self) -> Dict:
        """Calculate MOET stablecoin stability metrics"""
        
        moet = self.protocol.moet_system
        
        # Price deviation from peg
        price_deviation = abs(moet.current_price - moet.target_price) / moet.target_price
        
        # Peg stability status
        is_stable = moet.is_peg_stable()
        stability_action = moet.calculate_stability_action()
        
        # Supply metrics
        total_supply = moet.total_supply
        
        # Stability pressure
        if stability_action:
            pressure_type, pressure_magnitude = moet.get_stability_pressure()
        else:
            pressure_type, pressure_magnitude = "stable", 0.0
        
        return {
            "current_price": moet.current_price,
            "target_price": moet.target_price,
            "price_deviation": price_deviation,
            "is_peg_stable": is_stable,
            "stability_action_needed": stability_action,
            "total_supply": total_supply,
            "stability_pressure_type": pressure_type,
            "stability_pressure_magnitude": pressure_magnitude,
            "stability_score": max(0, 1 - (price_deviation / 0.05))  # 5% deviation = 0 score
        }
    
    def calculate_utilization_metrics(self) -> Dict:
        """Calculate utilization metrics across all asset pools"""
        
        utilization_rates = {}
        supply_rates = {}
        borrow_rates = {}
        
        total_supplied_value = 0.0
        total_borrowed_value = 0.0
        
        for asset, pool in self.protocol.asset_pools.items():
            utilization_rates[asset.value] = pool.utilization_rate
            supply_rates[asset.value] = pool.calculate_supply_rate()
            borrow_rates[asset.value] = pool.calculate_borrow_rate()
            
            # Assume $1 for simplicity (would need actual prices)
            total_supplied_value += pool.total_supplied
            total_borrowed_value += pool.total_borrowed
        
        # Overall utilization
        overall_utilization = total_borrowed_value / total_supplied_value if total_supplied_value > 0 else 0
        
        # Interest rate efficiency (spread between borrow and supply rates)
        rate_spreads = {
            asset: borrow_rates.get(asset, 0) - supply_rates.get(asset, 0)
            for asset in borrow_rates.keys()
        }
        
        return {
            "utilization_rates": utilization_rates,
            "supply_rates": supply_rates,
            "borrow_rates": borrow_rates,
            "rate_spreads": rate_spreads,
            "overall_utilization": overall_utilization,
            "total_supplied_value": total_supplied_value,
            "total_borrowed_value": total_borrowed_value,
            "utilization_balance_score": self._calculate_utilization_balance_score(utilization_rates)
        }
    
    def calculate_protocol_revenue_metrics(self) -> Dict:
        """Calculate protocol revenue and treasury metrics"""
        
        treasury_balance = self.protocol.protocol_treasury
        
        # Estimate revenue sources
        interest_revenue = sum(
            pool.total_borrowed * pool.calculate_borrow_rate() * pool.reserve_factor / 12  # Monthly
            for pool in self.protocol.asset_pools.values()
        )
        
        # Revenue efficiency
        total_borrowed = sum(pool.total_borrowed for pool in self.protocol.asset_pools.values())
        revenue_per_dollar_borrowed = interest_revenue / total_borrowed if total_borrowed > 0 else 0
        
        return {
            "treasury_balance": treasury_balance,
            "estimated_monthly_revenue": interest_revenue,
            "revenue_per_dollar_borrowed": revenue_per_dollar_borrowed,
            "treasury_health_score": min(treasury_balance / 10000, 1.0),  # $10k = perfect
            "revenue_diversification": self._analyze_revenue_diversification()
        }
    
    def _debt_cap_safety_score(self, current_prices: Dict[Asset, float]) -> float:
        """Calculate debt cap safety score"""
        
        debt_cap_metrics = self.calculate_debt_cap_metrics(current_prices)
        utilization = debt_cap_metrics["utilization_rate"]
        
        if utilization > 0.9:
            return 0.1  # Critical
        elif utilization > 0.8:
            return 0.3  # High risk
        elif utilization > 0.6:
            return 0.6  # Medium risk
        else:
            return 1.0  # Safe
    
    def _liquidation_readiness_score(self) -> float:
        """Calculate liquidation system readiness score"""
        
        total_capacity = self._calculate_total_liquidation_capacity()
        
        # Score based on available liquidation capacity
        if total_capacity > 50000:  # $50k capacity
            return 1.0
        elif total_capacity > 25000:  # $25k capacity
            return 0.7
        elif total_capacity > 10000:  # $10k capacity
            return 0.4
        else:
            return 0.1
    
    def _moet_peg_stability_score(self) -> float:
        """Calculate MOET peg stability score"""
        
        moet = self.protocol.moet_system
        price_deviation = abs(moet.current_price - 1.0)
        
        if price_deviation < 0.01:  # Within 1%
            return 1.0
        elif price_deviation < 0.02:  # Within 2%
            return 0.8
        elif price_deviation < 0.05:  # Within 5%
            return 0.5
        else:
            return 0.1
    
    def _utilization_balance_score(self) -> float:
        """Calculate utilization balance score"""
        
        utilizations = [pool.utilization_rate for pool in self.protocol.asset_pools.values()]
        
        if not utilizations:
            return 1.0
        
        # Penalize both very high and very low utilization
        avg_utilization = np.mean(utilizations)
        utilization_std = np.std(utilizations)
        
        # Optimal range: 40-70% utilization
        if 0.4 <= avg_utilization <= 0.7:
            balance_score = 1.0 - (utilization_std * 2)  # Penalize high variance
        else:
            balance_score = 0.5 - abs(avg_utilization - 0.55) * 2
        
        return max(0.1, balance_score)
    
    def _treasury_health_score(self) -> float:
        """Calculate treasury health score"""
        
        treasury = self.protocol.protocol_treasury
        
        # Score based on treasury balance relative to protocol size
        total_supplied = sum(pool.total_supplied for pool in self.protocol.asset_pools.values())
        treasury_ratio = treasury / total_supplied if total_supplied > 0 else 0
        
        return min(treasury_ratio * 100, 1.0)  # 1% of TVL = perfect score
    
    def _categorize_health(self, health_score: float) -> str:
        """Categorize overall health score"""
        
        if health_score >= 0.8:
            return "EXCELLENT"
        elif health_score >= 0.6:
            return "GOOD"
        elif health_score >= 0.4:
            return "FAIR"
        elif health_score >= 0.2:
            return "POOR"
        else:
            return "CRITICAL"
    
    def _categorize_debt_cap_utilization(self, utilization: float) -> str:
        """Categorize debt cap utilization level"""
        
        if utilization > 0.95:
            return "CRITICAL"
        elif utilization > 0.85:
            return "HIGH"
        elif utilization > 0.70:
            return "MODERATE"
        elif utilization > 0.50:
            return "NORMAL"
        else:
            return "LOW"
    
    def _analyze_liquidation_capacity(self) -> Dict:
        """Analyze liquidation capacity breakdown by asset"""
        
        breakdown = {}
        total_capacity = 0.0
        
        for pool_key, lp_pool in self.protocol.liquidity_pools.items():
            capacity = lp_pool.get_liquidation_capacity()
            breakdown[pool_key] = capacity
            total_capacity += capacity
        
        return {
            "by_pool": breakdown,
            "total_capacity": total_capacity,
            "dominant_pool": max(breakdown, key=breakdown.get) if breakdown else None
        }
    
    def _calculate_total_liquidation_capacity(self) -> float:
        """Calculate total liquidation capacity across all pools"""
        
        return sum(
            pool.get_liquidation_capacity() 
            for pool in self.protocol.liquidity_pools.values()
        )
    
    def _assess_liquidation_efficiency(self, coverage_ratio: float) -> str:
        """Assess liquidation efficiency based on coverage ratio"""
        
        if coverage_ratio >= 2.0:
            return "EXCELLENT"
        elif coverage_ratio >= 1.5:
            return "GOOD"
        elif coverage_ratio >= 1.0:
            return "ADEQUATE"
        elif coverage_ratio >= 0.5:
            return "INSUFFICIENT"
        else:
            return "CRITICAL"
    
    def _calculate_utilization_balance_score(self, utilization_rates: Dict) -> float:
        """Calculate utilization balance score"""
        
        if not utilization_rates:
            return 1.0
        
        rates = list(utilization_rates.values())
        avg_rate = np.mean(rates)
        std_rate = np.std(rates)
        
        # Penalize extreme utilization rates and high variance
        balance_penalty = abs(avg_rate - 0.6) + std_rate
        
        return max(0.0, 1.0 - balance_penalty)
    
    def _analyze_revenue_diversification(self) -> Dict:
        """Analyze revenue source diversification"""
        
        revenue_sources = {}
        total_revenue = 0.0
        
        for asset, pool in self.protocol.asset_pools.items():
            asset_revenue = pool.total_borrowed * pool.calculate_borrow_rate() * pool.reserve_factor
            revenue_sources[asset.value] = asset_revenue
            total_revenue += asset_revenue
        
        # Calculate concentration (1 = perfectly diversified, 0 = single source)
        if total_revenue > 0:
            proportions = [rev / total_revenue for rev in revenue_sources.values()]
            herfindahl_index = sum(p**2 for p in proportions)
            diversification_score = 1 - herfindahl_index
        else:
            diversification_score = 1.0
        
        return {
            "revenue_by_asset": revenue_sources,
            "diversification_score": diversification_score,
            "dominant_revenue_source": max(revenue_sources, key=revenue_sources.get) if revenue_sources else None
        }