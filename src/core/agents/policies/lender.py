#!/usr/bin/env python3
"""
Lender policy for the Tidal Protocol simulation.

This policy implements APY-driven lending behavior where agents supply
assets to earn interest and borrow against their collateral.
"""

from typing import List
import random
from ..base import BasePolicy
from ...simulation.primitives import Action, ActionKind, MarketSnapshot, AgentState, Asset


class LenderPolicy(BasePolicy):
    """Policy for lending protocol interaction"""
    
    def __init__(self,
                 action_frequency: float = 0.05,
                 min_supply_apy: float = 0.02,
                 max_borrow_rate: float = 0.15,
                 target_health_factor: float = 2.0,
                 supply_ratio: float = 0.8):
        """
        Initialize lender policy
        
        Args:
            action_frequency: Probability of taking action each period (0-1)
            min_supply_apy: Minimum APY to supply assets (0-1)
            max_borrow_rate: Maximum rate willing to pay for borrowing (0-1)
            target_health_factor: Target health factor to maintain
            supply_ratio: Percentage of balance to supply as collateral
        """
        self.action_frequency = action_frequency
        self.min_supply_apy = min_supply_apy
        self.max_borrow_rate = max_borrow_rate
        self.target_health_factor = target_health_factor
        self.supply_ratio = supply_ratio
        self.last_action_timestamp = 0
    
    def decide(self, agent_state: AgentState, snapshot: MarketSnapshot) -> List[Action]:
        """
        Make lending decisions based on APY and risk management
        
        Args:
            agent_state: Current agent state
            snapshot: Current market snapshot
            
        Returns:
            List of lending actions
        """
        actions = []
        
        # Check if we should act this period
        if random.random() > self.action_frequency:
            return [Action(ActionKind.HOLD, "", {})]
        
        # Get lending market data
        lending_data = snapshot.markets.get('compound_lending', {})
        if not lending_data:
            return [Action(ActionKind.HOLD, "", {})]
        
        # Decide on primary action type
        action_type = random.choice(['supply', 'borrow', 'manage_position'])
        
        if action_type == 'supply':
            action = self._create_supply_action(agent_state, snapshot, lending_data)
        elif action_type == 'borrow':
            action = self._create_borrow_action(agent_state, snapshot, lending_data)
        else:
            action = self._manage_position(agent_state, snapshot, lending_data)
        
        if action:
            actions.append(action)
        
        return actions if actions else [Action(ActionKind.HOLD, "", {})]
    
    def _create_supply_action(self, agent_state: AgentState, snapshot: MarketSnapshot, lending_data: dict) -> Action:
        """Create a supply action if APY is attractive"""
        # Find the best asset to supply
        best_asset = None
        best_apy = 0.0
        
        for asset in [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]:
            asset_data = lending_data.get(asset.value, {})
            supply_apy = asset_data.get('supply_apy', 0.0)
            
            if supply_apy > best_apy and supply_apy >= self.min_supply_apy:
                best_apy = supply_apy
                best_asset = asset
        
        if not best_asset:
            return None
        
        # Check available balance
        available_balance = agent_state.token_balances.get(best_asset, 0.0)
        if available_balance <= 0:
            return None
        
        # Calculate supply amount (percentage of balance)
        supply_amount = available_balance * self.supply_ratio
        
        # Minimum supply check
        asset_price = snapshot.token_prices.get(best_asset, 0.0)
        if supply_amount * asset_price < 100:  # Minimum $100 supply
            return None
        
        return Action(
            kind=ActionKind.SUPPLY,
            agent_id="",
            params={
                "market_id": "compound_lending",
                "asset": best_asset,
                "amount": supply_amount
            }
        )
    
    def _create_borrow_action(self, agent_state: AgentState, snapshot: MarketSnapshot, lending_data: dict) -> Action:
        """Create a borrow action if rates are acceptable"""
        # Check if we have supplied collateral
        total_supplied_value = sum(
            balance * snapshot.token_prices.get(asset, 0.0)
            for asset, balance in agent_state.supplied_balances.items()
        )
        
        if total_supplied_value <= 0:
            return None
        
        # Find acceptable borrow rates
        borrow_options = []
        for asset in [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]:
            asset_data = lending_data.get(asset.value, {})
            borrow_rate = asset_data.get('borrow_apy', 1.0)  # Default high rate
            
            if borrow_rate <= self.max_borrow_rate:
                borrow_options.append((asset, borrow_rate))
        
        if not borrow_options:
            return None
        
        # Choose asset with lowest borrow rate
        best_asset, _ = min(borrow_options, key=lambda x: x[1])
        
        # Calculate safe borrow amount based on health factor
        collateral_factors = {
            Asset.ETH: 0.75,
            Asset.BTC: 0.75,
            Asset.FLOW: 0.50,
            Asset.USDC: 0.90
        }
        
        effective_collateral_value = sum(
            balance * snapshot.token_prices.get(asset, 0.0) * collateral_factors.get(asset, 0.0)
            for asset, balance in agent_state.supplied_balances.items()
        )
        
        current_borrowed_value = sum(
            balance * snapshot.token_prices.get(asset, 0.0)
            for asset, balance in agent_state.borrowed_balances.items()
        )
        
        # Calculate max safe borrow amount
        max_borrow_value = (effective_collateral_value / self.target_health_factor) - current_borrowed_value
        
        if max_borrow_value <= 0:
            return None
        
        # Convert to asset amount
        asset_price = snapshot.token_prices.get(best_asset, 0.0)
        if asset_price <= 0:
            return None
        
        borrow_amount = (max_borrow_value * 0.5) / asset_price  # Borrow 50% of max safe amount
        
        # Minimum borrow check
        if borrow_amount * asset_price < 50:  # Minimum $50 borrow
            return None
        
        return Action(
            kind=ActionKind.BORROW,
            agent_id="",
            params={
                "market_id": "compound_lending",
                "asset": best_asset,
                "amount": borrow_amount
            }
        )
    
    def _manage_position(self, agent_state: AgentState, snapshot: MarketSnapshot, lending_data: dict) -> Action:
        """Manage existing position (repay if health factor is low)"""
        # Calculate current health factor
        collateral_factors = {
            Asset.ETH: 0.75,
            Asset.BTC: 0.75,
            Asset.FLOW: 0.50,
            Asset.USDC: 0.90
        }
        
        current_health_factor = agent_state.get_health_factor(
            snapshot.token_prices, collateral_factors
        )
        
        # If health factor is too low, repay some debt
        if current_health_factor < self.target_health_factor * 0.8:  # 20% buffer
            # Find asset with highest borrowed amount
            max_borrowed_asset = None
            max_borrowed_amount = 0.0
            
            for asset, balance in agent_state.borrowed_balances.items():
                if balance > max_borrowed_amount:
                    max_borrowed_amount = balance
                    max_borrowed_asset = asset
            
            if max_borrowed_asset and max_borrowed_amount > 0:
                # Check if we have tokens to repay
                available_balance = agent_state.token_balances.get(max_borrowed_asset, 0.0)
                repay_amount = min(available_balance, max_borrowed_amount * 0.3)  # Repay 30%
                
                if repay_amount > 0:
                    return Action(
                        kind=ActionKind.REPAY,
                        agent_id="",
                        params={
                            "market_id": "compound_lending",
                            "asset": max_borrowed_asset,
                            "amount": repay_amount
                        }
                    )
        
        return None
    
    def can_execute_action(self, action: Action, agent_state: AgentState, snapshot: MarketSnapshot) -> bool:
        """
        Validate if a lending action can be executed
        
        Args:
            action: Action to validate
            agent_state: Current agent state
            snapshot: Current market snapshot
            
        Returns:
            True if action can be executed
        """
        if action.kind == ActionKind.HOLD:
            return True
        
        asset = action.params.get("asset")
        amount = action.params.get("amount", 0)
        
        if action.kind == ActionKind.SUPPLY:
            # Check if agent has enough tokens to supply
            available_balance = agent_state.token_balances.get(asset, 0.0)
            return available_balance >= amount
        
        elif action.kind == ActionKind.BORROW:
            # Check if agent has enough collateral (simplified check)
            return True  # Market will handle detailed validation
        
        elif action.kind == ActionKind.REPAY:
            # Check if agent has tokens to repay
            available_balance = agent_state.token_balances.get(asset, 0.0)
            return available_balance >= amount
        
        return False
