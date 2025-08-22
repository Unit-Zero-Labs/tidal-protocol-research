#!/usr/bin/env python3
"""
Tidal Protocol Specific Lender Policy

This policy implements sophisticated lending behavior specifically designed for
Tidal Protocol, including MOET borrowing strategies, health factor management,
and protocol-specific risk assessment.
"""

from typing import List
import random
from ..base import BasePolicy
from ...simulation.primitives import Action, ActionKind, MarketSnapshot, AgentState, Asset


class TidalLenderPolicy(BasePolicy):
    """Advanced lending policy specifically for Tidal Protocol"""
    
    def __init__(self,
                 action_frequency: float = 0.08,
                 min_supply_apy: float = 0.025,
                 target_health_factor: float = 1.2,
                 min_health_factor: float = 1.1,
                 max_utilization_rate: float = 0.85,
                 moet_borrowing_ratio: float = 0.6,
                 collateral_diversification: bool = True,
                 risk_tolerance: float = 0.5):
        """
        Initialize Tidal-specific lender policy
        
        Args:
            action_frequency: Probability of taking action each period
            min_supply_apy: Minimum APY to consider supplying
            target_health_factor: Target health factor to maintain
            min_health_factor: Minimum health factor before emergency actions
            max_utilization_rate: Maximum pool utilization to accept
            moet_borrowing_ratio: Percentage of borrowing power to use for MOET
            collateral_diversification: Whether to diversify collateral across assets
            risk_tolerance: Risk tolerance level (0-1)
        """
        self.action_frequency = action_frequency
        self.min_supply_apy = min_supply_apy
        self.target_health_factor = target_health_factor
        self.min_health_factor = min_health_factor
        self.max_utilization_rate = max_utilization_rate
        self.moet_borrowing_ratio = moet_borrowing_ratio
        self.collateral_diversification = collateral_diversification
        self.risk_tolerance = risk_tolerance
        
        # Tidal-specific preferences
        self.preferred_collateral_assets = [Asset.ETH, Asset.BTC, Asset.USDC, Asset.FLOW]
        self.collateral_asset_weights = {
            Asset.ETH: 0.4,    # 40% preference for ETH (high collateral factor, liquid)
            Asset.BTC: 0.3,    # 30% preference for BTC (high collateral factor, liquid)
            Asset.USDC: 0.2,   # 20% preference for USDC (highest collateral factor, stable)
            Asset.FLOW: 0.1    # 10% preference for FLOW (lower collateral factor, higher risk)
        }
        
        # Strategy state
        self.last_action_timestamp = 0
        self.emergency_mode = False
        self.position_history = []
    
    def decide(self, agent_state: AgentState, snapshot: MarketSnapshot) -> List[Action]:
        """
        Make Tidal Protocol specific lending decisions
        
        Args:
            agent_state: Current agent state
            snapshot: Current market snapshot
            
        Returns:
            List of Tidal-specific actions
        """
        actions = []
        
        # Check if we should act this period (be more aggressive initially)
        if random.random() > self.action_frequency:
            # Still have a chance to act even if random check fails (30% chance)
            if random.random() > 0.3:
                return [Action(ActionKind.HOLD, "", {})]
        
        # Get Tidal market data
        tidal_data = snapshot.markets.get('tidal_protocol', {})
        if not tidal_data:
            return [Action(ActionKind.HOLD, "", {})]
        
        # Calculate current health factor
        current_health_factor = self._calculate_health_factor(agent_state, snapshot, tidal_data)
        
        # Determine strategy based on health factor and market conditions
        if current_health_factor < self.min_health_factor:
            self.emergency_mode = True
            action = self._emergency_action(agent_state, snapshot, tidal_data)
        elif current_health_factor < self.target_health_factor:
            action = self._conservative_action(agent_state, snapshot, tidal_data)
        else:
            self.emergency_mode = False
            # Choose primary strategy
            strategy = self._choose_strategy(agent_state, snapshot, tidal_data)
            
            if strategy == "supply":
                action = self._supply_strategy(agent_state, snapshot, tidal_data)
            elif strategy == "borrow":
                action = self._borrow_strategy(agent_state, snapshot, tidal_data)
            elif strategy == "rebalance":
                action = self._rebalance_strategy(agent_state, snapshot, tidal_data)
            elif strategy == "optimize":
                action = self._optimize_strategy(agent_state, snapshot, tidal_data)
            else:
                action = None
        
        if action:
            actions.append(action)
            self.position_history.append({
                'timestamp': snapshot.timestamp,
                'action': action.kind,
                'health_factor': current_health_factor,
                'strategy': getattr(self, '_last_strategy', 'unknown')
            })
        
        return actions if actions else [Action(ActionKind.HOLD, "", {})]
    
    def _calculate_health_factor(self, agent_state: AgentState, snapshot: MarketSnapshot, tidal_data: dict) -> float:
        """Calculate agent's current health factor in Tidal Protocol"""
        total_collateral_value = 0.0
        total_debt_value = 0.0
        
        asset_pools = tidal_data.get('asset_pools', {})
        
        # Calculate effective collateral value
        for asset_str, pool_data in asset_pools.items():
            try:
                asset = Asset(asset_str)
                supplied_amount = agent_state.supplied_balances.get(asset, 0.0)
                
                if supplied_amount > 0:
                    asset_price = snapshot.token_prices.get(asset, 1.0)
                    collateral_factor = pool_data.get('collateral_factor', 0.0)
                    collateral_value = supplied_amount * asset_price * collateral_factor
                    total_collateral_value += collateral_value
            except (ValueError, KeyError):
                continue
        
        # Calculate debt value (only MOET in Tidal)
        moet_debt = agent_state.borrowed_balances.get(Asset.MOET, 0.0)
        if moet_debt > 0:
            moet_price = snapshot.token_prices.get(Asset.MOET, 1.0)
            total_debt_value = moet_debt * moet_price
        
        if total_debt_value == 0:
            return float('inf')
        
        return total_collateral_value / total_debt_value
    
    def _choose_strategy(self, agent_state: AgentState, snapshot: MarketSnapshot, tidal_data: dict) -> str:
        """Choose optimal strategy based on market conditions and agent state"""
        # Analyze market conditions
        asset_pools = tidal_data.get('asset_pools', {})
        moet_data = tidal_data.get('moet_stablecoin', {})
        
        # Check for high-yield supply opportunities
        high_yield_opportunities = []
        for asset_str, pool_data in asset_pools.items():
            supply_rate = pool_data.get('supply_rate', 0.0)
            utilization = pool_data.get('utilization_rate', 0.0)
            
            if supply_rate >= self.min_supply_apy and utilization <= self.max_utilization_rate:
                high_yield_opportunities.append((asset_str, supply_rate))
        
        # Check MOET stability and borrowing opportunities
        moet_stable = moet_data.get('is_peg_stable', True)
        current_health_factor = self._calculate_health_factor(agent_state, snapshot, tidal_data)
        
        # Strategy decision logic
        if high_yield_opportunities and len(high_yield_opportunities) > 0:
            # Prioritize high-yield supply if available
            self._last_strategy = "supply"
            return "supply"
        elif (moet_stable and 
              current_health_factor > self.target_health_factor * 1.2 and 
              self.moet_borrowing_ratio > 0):
            # Consider borrowing MOET if conditions are favorable
            self._last_strategy = "borrow"
            return "borrow"
        elif self._needs_rebalancing(agent_state, tidal_data):
            # Rebalance collateral if needed
            self._last_strategy = "rebalance"
            return "rebalance"
        elif current_health_factor > self.target_health_factor * 1.5:
            # Optimize position if very healthy
            self._last_strategy = "optimize"
            return "optimize"
        else:
            self._last_strategy = "hold"
            return "hold"
    
    def _supply_strategy(self, agent_state: AgentState, snapshot: MarketSnapshot, tidal_data: dict) -> Action:
        """Execute supply strategy for best yield opportunities"""
        asset_pools = tidal_data.get('asset_pools', {})
        
        # Find best supply opportunity
        best_asset = None
        best_rate = 0.0
        
        for asset_str, pool_data in asset_pools.items():
            try:
                asset = Asset(asset_str)
                supply_rate = pool_data.get('supply_rate', 0.0)
                utilization = pool_data.get('utilization_rate', 0.0)
                available_balance = agent_state.token_balances.get(asset, 0.0)
                
                if (supply_rate > best_rate and 
                    supply_rate >= self.min_supply_apy and
                    utilization <= self.max_utilization_rate and
                    available_balance > 0):
                    
                    best_asset = asset
                    best_rate = supply_rate
            except (ValueError, KeyError):
                continue
        
        if not best_asset:
            return None
        
        # Calculate supply amount
        available_balance = agent_state.token_balances.get(best_asset, 0.0)
        
        # Supply strategy: more aggressive if higher risk tolerance
        if self.risk_tolerance > 0.7:
            supply_ratio = 0.8  # Supply 80% of balance
        elif self.risk_tolerance > 0.4:
            supply_ratio = 0.6  # Supply 60% of balance
        else:
            supply_ratio = 0.4  # Supply 40% of balance (conservative)
        
        supply_amount = available_balance * supply_ratio
        
        # Minimum supply check
        asset_price = snapshot.token_prices.get(best_asset, 1.0)
        if supply_amount * asset_price < 100:  # Minimum $100 supply
            return None
        
        return Action(
            kind=ActionKind.SUPPLY,
            agent_id="",
            params={
                "market_id": "tidal_protocol",
                "asset": best_asset,
                "amount": supply_amount,
                "expected_apy": best_rate
            }
        )
    
    def _borrow_strategy(self, agent_state: AgentState, snapshot: MarketSnapshot, tidal_data: dict) -> Action:
        """Execute MOET borrowing strategy"""
        # Calculate available borrowing power
        total_collateral_value = 0.0
        asset_pools = tidal_data.get('asset_pools', {})
        
        for asset_str, pool_data in asset_pools.items():
            try:
                asset = Asset(asset_str)
                supplied_amount = agent_state.supplied_balances.get(asset, 0.0)
                
                if supplied_amount > 0:
                    asset_price = snapshot.token_prices.get(asset, 1.0)
                    collateral_factor = pool_data.get('collateral_factor', 0.0)
                    collateral_value = supplied_amount * asset_price * collateral_factor
                    total_collateral_value += collateral_value
            except (ValueError, KeyError):
                continue
        
        # Calculate current debt
        current_debt = agent_state.borrowed_balances.get(Asset.MOET, 0.0)
        moet_price = snapshot.token_prices.get(Asset.MOET, 1.0)
        current_debt_value = current_debt * moet_price
        
        # Calculate max safe borrow amount
        max_total_debt = total_collateral_value / self.target_health_factor
        max_additional_borrow = max_total_debt - current_debt_value
        
        if max_additional_borrow <= 0:
            return None
        
        # Apply borrowing ratio and risk tolerance
        target_borrow = max_additional_borrow * self.moet_borrowing_ratio * self.risk_tolerance
        
        # Convert to MOET amount
        borrow_amount = target_borrow / moet_price
        
        # Minimum borrow check
        if borrow_amount * moet_price < 50:  # Minimum $50 borrow
            return None
        
        # Check if MOET is stable enough to borrow
        moet_data = tidal_data.get('moet_stablecoin', {})
        if not moet_data.get('is_peg_stable', True):
            return None
        
        return Action(
            kind=ActionKind.BORROW,
            agent_id="",
            params={
                "market_id": "tidal_protocol",
                "asset": Asset.MOET,
                "amount": borrow_amount,
                "target_health_factor": self.target_health_factor
            }
        )
    
    def _emergency_action(self, agent_state: AgentState, snapshot: MarketSnapshot, tidal_data: dict) -> Action:
        """Take emergency action when health factor is too low"""
        # Priority 1: Repay debt if possible
        moet_balance = agent_state.token_balances.get(Asset.MOET, 0.0)
        moet_debt = agent_state.borrowed_balances.get(Asset.MOET, 0.0)
        
        if moet_balance > 0 and moet_debt > 0:
            repay_amount = min(moet_balance, moet_debt * 0.5)  # Repay up to 50% of debt
            
            return Action(
                kind=ActionKind.REPAY,
                agent_id="",
                params={
                    "market_id": "tidal_protocol",
                    "asset": Asset.MOET,
                    "amount": repay_amount
                }
            )
        
        # Priority 2: Supply more collateral if available
        for asset in [Asset.USDC, Asset.ETH, Asset.BTC, Asset.FLOW]:  # Order by stability/liquidity
            available_balance = agent_state.token_balances.get(asset, 0.0)
            if available_balance > 0:
                supply_amount = available_balance * 0.9  # Supply 90% in emergency
                
                return Action(
                    kind=ActionKind.SUPPLY,
                    agent_id="",
                    params={
                        "market_id": "tidal_protocol",
                        "asset": asset,
                        "amount": supply_amount
                    }
                )
        
        return None
    
    def _conservative_action(self, agent_state: AgentState, snapshot: MarketSnapshot, tidal_data: dict) -> Action:
        """Take conservative action when health factor is below target but not emergency"""
        # Focus on improving health factor
        
        # Option 1: Repay some debt
        moet_balance = agent_state.token_balances.get(Asset.MOET, 0.0)
        moet_debt = agent_state.borrowed_balances.get(Asset.MOET, 0.0)
        
        if moet_balance > 0 and moet_debt > 0:
            # Repay 25% of debt or available balance, whichever is smaller
            repay_amount = min(moet_balance, moet_debt * 0.25)
            
            if repay_amount * snapshot.token_prices.get(Asset.MOET, 1.0) >= 25:  # Minimum $25
                return Action(
                    kind=ActionKind.REPAY,
                    agent_id="",
                    params={
                        "market_id": "tidal_protocol",
                        "asset": Asset.MOET,
                        "amount": repay_amount
                    }
                )
        
        # Option 2: Supply additional collateral
        best_collateral_asset = self._find_best_collateral_to_supply(agent_state, tidal_data)
        if best_collateral_asset:
            available_balance = agent_state.token_balances.get(best_collateral_asset, 0.0)
            supply_amount = available_balance * 0.5  # Conservative 50%
            
            if supply_amount > 0:
                return Action(
                    kind=ActionKind.SUPPLY,
                    agent_id="",
                    params={
                        "market_id": "tidal_protocol",
                        "asset": best_collateral_asset,
                        "amount": supply_amount
                    }
                )
        
        return None
    
    def _needs_rebalancing(self, agent_state: AgentState, tidal_data: dict) -> bool:
        """Check if collateral portfolio needs rebalancing"""
        if not self.collateral_diversification:
            return False
        
        # Check current collateral distribution
        total_collateral_value = 0.0
        collateral_distribution = {}
        
        asset_pools = tidal_data.get('asset_pools', {})
        
        for asset_str, pool_data in asset_pools.items():
            try:
                asset = Asset(asset_str)
                supplied_amount = agent_state.supplied_balances.get(asset, 0.0)
                
                if supplied_amount > 0:
                    # Simplified: assume price = 1 for distribution calculation
                    collateral_distribution[asset] = supplied_amount
                    total_collateral_value += supplied_amount
            except (ValueError, KeyError):
                continue
        
        if total_collateral_value == 0:
            return False
        
        # Check if distribution deviates significantly from target weights
        for asset, target_weight in self.collateral_asset_weights.items():
            current_weight = collateral_distribution.get(asset, 0.0) / total_collateral_value
            weight_deviation = abs(current_weight - target_weight)
            
            if weight_deviation > 0.2:  # 20% deviation threshold
                return True
        
        return False
    
    def _find_best_collateral_to_supply(self, agent_state: AgentState, tidal_data: dict) -> Asset:
        """Find the best collateral asset to supply based on current conditions"""
        asset_pools = tidal_data.get('asset_pools', {})
        
        best_asset = None
        best_score = 0.0
        
        for asset_str, pool_data in asset_pools.items():
            try:
                asset = Asset(asset_str)
                available_balance = agent_state.token_balances.get(asset, 0.0)
                
                if available_balance <= 0:
                    continue
                
                # Scoring factors
                collateral_factor = pool_data.get('collateral_factor', 0.0)
                supply_rate = pool_data.get('supply_rate', 0.0)
                utilization = pool_data.get('utilization_rate', 0.0)
                preference_weight = self.collateral_asset_weights.get(asset, 0.1)
                
                # Calculate composite score
                utilization_penalty = max(0, utilization - self.max_utilization_rate) * 2
                score = (collateral_factor * 0.4 + 
                        supply_rate * 0.3 + 
                        preference_weight * 0.3 - 
                        utilization_penalty)
                
                if score > best_score:
                    best_score = score
                    best_asset = asset
            except (ValueError, KeyError):
                continue
        
        return best_asset
    
    def _rebalance_strategy(self, agent_state: AgentState, snapshot: MarketSnapshot, tidal_data: dict) -> Action:
        """Execute collateral rebalancing strategy"""
        # Find overweight asset to withdraw from
        # Find underweight asset to supply to
        # For simplicity, implement basic rebalancing logic
        
        # This is a placeholder for more sophisticated rebalancing
        # In practice, this would involve complex optimization
        
        return None
    
    def _optimize_strategy(self, agent_state: AgentState, snapshot: MarketSnapshot, tidal_data: dict) -> Action:
        """Optimize position when health factor is very good"""
        # Look for yield optimization opportunities
        # Could involve more aggressive borrowing or yield farming
        
        # Check if we can safely borrow more MOET for yield opportunities
        current_health_factor = self._calculate_health_factor(agent_state, snapshot, tidal_data)
        
        if current_health_factor > self.target_health_factor * 2:
            # Very healthy position, consider more aggressive borrowing
            return self._borrow_strategy(agent_state, snapshot, tidal_data)
        
        return None
    
    def can_execute_action(self, action: Action, agent_state: AgentState, snapshot: MarketSnapshot) -> bool:
        """Validate if a Tidal Protocol action can be executed"""
        if action.kind == ActionKind.HOLD:
            return True
        
        asset = action.params.get("asset")
        amount = action.params.get("amount", 0)
        
        if action.kind == ActionKind.SUPPLY:
            available_balance = agent_state.token_balances.get(asset, 0.0)
            return available_balance >= amount
        
        elif action.kind == ActionKind.BORROW:
            # Market will handle detailed collateral checks
            return asset == Asset.MOET  # Tidal only allows borrowing MOET
        
        elif action.kind == ActionKind.REPAY:
            available_balance = agent_state.token_balances.get(asset, 0.0)
            return available_balance >= amount
        
        return False
