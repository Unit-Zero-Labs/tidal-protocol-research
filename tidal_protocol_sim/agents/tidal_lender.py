#!/usr/bin/env python3
"""
Tidal Lender Agent

Simplified but effective lending agent focused on core lending behavior
as specified in the refactoring requirements.
"""

import random
from typing import Dict, Tuple
from .base_agent import BaseAgent, AgentAction, Asset


class TidalLender(BaseAgent):
    """Simplified but effective lending agent"""
    
    def __init__(self, agent_id: str, initial_balance: float = 100_000.0):
        super().__init__(agent_id, "tidal_lender", initial_balance)
        
        # Key parameters only (as per refactor spec)
        self.target_health_factor = 1.5
        self.min_supply_apy = 0.02  # 2% minimum APY to supply
        self.moet_borrowing_ratio = 0.6  # Borrow up to 60% of collateral value
        self.risk_tolerance = 0.5
        
        # Decision thresholds
        self.emergency_hf_threshold = 1.1
        self.conservative_hf_threshold = 1.5
        self.high_apy_threshold = 0.08  # 8% APY considered high
        
    def decide_action(self, protocol_state: dict, asset_prices: Dict[Asset, float]) -> Tuple[AgentAction, dict]:
        """
        Decision logic as specified in refactor requirements:
        1. Emergency repay if HF < 1.1
        2. Conservative action if HF < 1.5
        3. Supply if high APY opportunity
        4. Borrow if can do safely
        5. Hold otherwise
        """
        
        # Update health factor
        self.state.update_health_factor(asset_prices, self._get_collateral_factors())
        hf = self.state.health_factor
        
        # Emergency actions (HF < 1.1)
        if hf < self.emergency_hf_threshold:
            return self._emergency_repay()
        
        # Conservative actions (HF < 1.5)
        elif hf < self.conservative_hf_threshold:
            return self._conservative_action(protocol_state, asset_prices)
        
        # Check for high APY opportunities
        elif self._high_apy_opportunity(protocol_state):
            return self._supply_action(protocol_state, asset_prices)
        
        # Check if can borrow safely
        elif self._can_borrow_safely(asset_prices):
            return self._borrow_action(asset_prices)
        
        # Default: hold
        else:
            return AgentAction.HOLD, {}
    
    def _emergency_repay(self) -> Tuple[AgentAction, dict]:
        """Emergency repay to improve health factor"""
        moet_debt = self.state.borrowed_balances.get(Asset.MOET, 0.0)
        
        if moet_debt > 0:
            # Repay portion of debt to improve health factor
            repay_amount = min(moet_debt * 0.3, self.state.token_balances.get(Asset.MOET, 0.0))
            
            if repay_amount > 0:
                return AgentAction.REPAY, {"amount": repay_amount, "asset": Asset.MOET}
        
        # If no MOET to repay, try to supply more collateral
        best_asset = self._find_best_supply_asset()
        if best_asset and self.state.token_balances.get(best_asset, 0.0) > 0:
            supply_amount = self.state.token_balances[best_asset] * 0.5
            return AgentAction.SUPPLY, {"asset": best_asset, "amount": supply_amount}
        
        return AgentAction.HOLD, {}
    
    def _conservative_action(self, protocol_state: dict, asset_prices: Dict[Asset, float]) -> Tuple[AgentAction, dict]:
        """Conservative action when health factor is marginal"""
        # Prefer to supply additional collateral rather than risky borrowing
        best_asset = self._find_best_supply_asset()
        
        if best_asset and self.state.token_balances.get(best_asset, 0.0) > 0:
            # Supply conservatively (25% of available balance)
            supply_amount = self.state.token_balances[best_asset] * 0.25
            return AgentAction.SUPPLY, {"asset": best_asset, "amount": supply_amount}
        
        return AgentAction.HOLD, {}
    
    def _supply_action(self, protocol_state: dict, asset_prices: Dict[Asset, float]) -> Tuple[AgentAction, dict]:
        """Supply when high APY opportunity exists"""
        best_asset = self._find_highest_apy_asset(protocol_state)
        
        if best_asset and self.state.token_balances.get(best_asset, 0.0) > 0:
            # Supply significant portion when APY is attractive
            supply_amount = self.state.token_balances[best_asset] * 0.7
            return AgentAction.SUPPLY, {"asset": best_asset, "amount": supply_amount}
        
        return AgentAction.HOLD, {}
    
    def _borrow_action(self, asset_prices: Dict[Asset, float]) -> Tuple[AgentAction, dict]:
        """Borrow MOET against collateral"""
        collateral_value = self.state.get_total_collateral_value(asset_prices)
        current_debt = self.state.get_total_debt_value(asset_prices)
        
        # Calculate safe borrow amount
        max_total_debt = collateral_value / self.target_health_factor
        available_borrow = max_total_debt - current_debt
        
        if available_borrow > 100:  # Minimum $100 borrow
            borrow_amount = available_borrow * self.moet_borrowing_ratio
            return AgentAction.BORROW, {"amount": borrow_amount, "asset": Asset.MOET}
        
        return AgentAction.HOLD, {}
    
    def _high_apy_opportunity(self, protocol_state: dict) -> bool:
        """Check if there's a high APY opportunity"""
        supply_rates = protocol_state.get("supply_rates", {})
        
        for asset, rate in supply_rates.items():
            if isinstance(rate, (int, float)) and rate > self.high_apy_threshold:
                return True
        
        return False
    
    def _can_borrow_safely(self, asset_prices: Dict[Asset, float]) -> bool:
        """Check if can borrow safely maintaining target health factor"""
        collateral_value = self.state.get_total_collateral_value(asset_prices)
        current_debt = self.state.get_total_debt_value(asset_prices)
        
        if collateral_value <= 0:
            return False
        
        # Check if we can borrow more while maintaining target HF
        max_safe_debt = collateral_value / self.target_health_factor
        return max_safe_debt > current_debt + 100  # At least $100 room
    
    def _find_best_supply_asset(self) -> Asset:
        """Find best asset to supply based on available balance"""
        best_asset = None
        max_balance = 0
        
        for asset, balance in self.state.token_balances.items():
            if asset != Asset.MOET and balance > max_balance:
                max_balance = balance
                best_asset = asset
        
        return best_asset
    
    def _find_highest_apy_asset(self, protocol_state: dict) -> Asset:
        """Find asset with highest supply APY"""
        supply_rates = protocol_state.get("supply_rates", {})
        best_asset = None
        highest_rate = 0
        
        for asset_name, rate in supply_rates.items():
            if isinstance(rate, (int, float)) and rate > highest_rate:
                try:
                    asset = Asset(asset_name)
                    if asset != Asset.MOET and self.state.token_balances.get(asset, 0.0) > 0:
                        highest_rate = rate
                        best_asset = asset
                except ValueError:
                    continue
        
        return best_asset
    
    def _get_collateral_factors(self) -> Dict[Asset, float]:
        """Get collateral factors for health factor calculation"""
        return {
            Asset.ETH: 0.75,
            Asset.BTC: 0.75,
            Asset.FLOW: 0.50,
            Asset.USDC: 0.90
        }