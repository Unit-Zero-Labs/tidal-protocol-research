#!/usr/bin/env python3
"""
Liquidator Agent

Specialized agent for identifying and executing liquidation opportunities
in the Tidal Protocol.
"""

from typing import Dict, List, Tuple, Optional
from .base_agent import BaseAgent, AgentAction, Asset


class Liquidator(BaseAgent):
    """Liquidation bot for Tidal Protocol"""
    
    def __init__(self, agent_id: str, initial_balance: float = 200_000.0):
        super().__init__(agent_id, "liquidator", initial_balance)
        
        # Liquidation parameters
        self.min_profit_threshold = 50.0  # Minimum $50 profit to liquidate
        self.max_liquidation_size = 50_000.0  # Max $50k liquidation per transaction
        self.health_factor_threshold = 1.0  # Liquidate positions below 1.0 HF
        self.gas_cost_estimate = 20.0  # Estimated gas cost per liquidation
        
        # Risk management
        self.max_position_exposure = 0.3  # Max 30% of balance in liquidation positions
        self.liquidation_penalty = 0.08  # 8% liquidation penalty (profit margin)
        
        # State tracking
        self.liquidation_history = []
        self.monitored_positions = {}
        self.successful_liquidations = 0
        self.total_liquidation_profit = 0.0
    
    def decide_action(self, protocol_state: dict, asset_prices: Dict[Asset, float]) -> Tuple[AgentAction, dict]:
        """
        Liquidation decision logic:
        1. Scan for liquidatable positions
        2. Calculate profitability
        3. Execute most profitable liquidation
        4. Manage liquidation portfolio
        """
        
        # First, check if we have liquidated collateral to sell
        liquidated_assets = self._check_liquidated_assets()
        if liquidated_assets:
            return self._sell_liquidated_assets(liquidated_assets, asset_prices)
        
        # Look for liquidation opportunities
        liquidation_opportunity = self._find_best_liquidation(protocol_state, asset_prices)
        if liquidation_opportunity:
            return self._execute_liquidation(liquidation_opportunity)
        
        # Manage MOET balance for liquidations
        moet_management = self._manage_moet_balance(asset_prices)
        if moet_management[0] != AgentAction.HOLD:
            return moet_management
        
        return AgentAction.HOLD, {}
    
    def _find_best_liquidation(self, protocol_state: dict, asset_prices: Dict[Asset, float]) -> Optional[dict]:
        """Find the most profitable liquidation opportunity"""
        
        # In a real implementation, this would scan all agent positions
        # For simulation, we'll create some mock liquidatable positions
        liquidatable_positions = self._scan_liquidatable_positions(protocol_state, asset_prices)
        
        if not liquidatable_positions:
            return None
        
        best_opportunity = None
        max_profit = 0
        
        for position in liquidatable_positions:
            profit = self._calculate_liquidation_profit(position, asset_prices)
            
            if profit > max_profit and profit > self.min_profit_threshold:
                max_profit = profit
                best_opportunity = position
        
        return best_opportunity
    
    def _scan_liquidatable_positions(self, protocol_state: dict, asset_prices: Dict[Asset, float]) -> List[dict]:
        """Scan for positions that can be liquidated"""
        
        # Mock liquidatable positions for simulation
        # In reality, this would query the protocol for unhealthy positions
        mock_positions = []
        
        # Create some sample liquidatable positions based on market stress
        if self._is_market_stressed(asset_prices):
            # Generate mock positions that might be underwater
            for i in range(3):
                position = {
                    "agent_id": f"mock_agent_{i}",
                    "collateral_asset": [Asset.ETH, Asset.BTC, Asset.FLOW][i % 3],
                    "collateral_amount": 1000.0 + i * 500,
                    "debt_amount": 800.0 + i * 400,  # MOET debt
                    "health_factor": 0.85 + i * 0.05  # Below 1.0
                }
                mock_positions.append(position)
        
        return [p for p in mock_positions if p["health_factor"] < self.health_factor_threshold]
    
    def _calculate_liquidation_profit(self, position: dict, asset_prices: Dict[Asset, float]) -> float:
        """Calculate expected profit from liquidating a position"""
        
        collateral_asset = position["collateral_asset"]
        collateral_amount = position["collateral_amount"]
        debt_amount = position["debt_amount"]
        
        asset_price = asset_prices.get(collateral_asset, 0.0)
        
        if asset_price <= 0:
            return 0.0
        
        # Calculate liquidation amounts (max 50% close factor)
        max_repay = debt_amount * 0.5
        
        # Collateral to seize (with 8% penalty)
        collateral_value_to_seize = max_repay * (1 + self.liquidation_penalty)
        collateral_to_seize = collateral_value_to_seize / asset_price
        
        # Check if enough collateral available
        if collateral_to_seize > collateral_amount:
            collateral_to_seize = collateral_amount
            max_repay = (collateral_to_seize * asset_price) / (1 + self.liquidation_penalty)
        
        # Profit = liquidation penalty minus costs
        liquidation_bonus = max_repay * self.liquidation_penalty
        total_profit = liquidation_bonus - self.gas_cost_estimate
        
        return total_profit
    
    def _execute_liquidation(self, opportunity: dict) -> Tuple[AgentAction, dict]:
        """Execute a liquidation opportunity"""
        
        target_agent = opportunity["agent_id"]
        collateral_asset = opportunity["collateral_asset"]
        debt_amount = opportunity["debt_amount"]
        
        # Calculate optimal repay amount
        max_repay = min(debt_amount * 0.5, self.max_liquidation_size)
        
        # Check if we have enough MOET
        available_moet = self.state.token_balances.get(Asset.MOET, 0.0)
        
        if available_moet < max_repay:
            # Need to acquire more MOET first
            return self._acquire_moet_for_liquidation(max_repay - available_moet)
        
        return AgentAction.LIQUIDATE, {
            "target_agent_id": target_agent,
            "collateral_asset": collateral_asset,
            "repay_amount": max_repay
        }
    
    def _acquire_moet_for_liquidation(self, needed_amount: float) -> Tuple[AgentAction, dict]:
        """Acquire MOET needed for liquidation"""
        
        # Find best asset to sell for MOET
        best_asset = None
        max_balance_value = 0
        
        for asset in [Asset.USDC, Asset.ETH, Asset.BTC, Asset.FLOW]:
            balance = self.state.token_balances.get(asset, 0.0)
            if balance > max_balance_value:
                max_balance_value = balance
                best_asset = asset
        
        if best_asset and max_balance_value > needed_amount:
            trade_amount = min(needed_amount * 1.1, max_balance_value)  # 10% buffer for slippage
            
            return AgentAction.SWAP, {
                "asset_in": best_asset,
                "asset_out": Asset.MOET,
                "amount_in": trade_amount,
                "min_amount_out": needed_amount * 0.98
            }
        
        return AgentAction.HOLD, {}
    
    def _check_liquidated_assets(self) -> List[Asset]:
        """Check for liquidated collateral assets to sell"""
        liquidated = []
        
        for asset in [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]:
            balance = self.state.token_balances.get(asset, 0.0)
            
            # If we have significant balance that likely came from liquidations
            if balance > 1000.0:  # Threshold for liquidated assets
                liquidated.append(asset)
        
        return liquidated
    
    def _sell_liquidated_assets(self, assets: List[Asset], asset_prices: Dict[Asset, float]) -> Tuple[AgentAction, dict]:
        """Sell liquidated collateral for profit"""
        
        # Sell the most valuable asset first
        best_asset = None
        max_value = 0
        
        for asset in assets:
            balance = self.state.token_balances.get(asset, 0.0)
            price = asset_prices.get(asset, 0.0)
            value = balance * price
            
            if value > max_value:
                max_value = value
                best_asset = asset
        
        if best_asset:
            balance = self.state.token_balances.get(best_asset, 0.0)
            sell_amount = min(balance * 0.8, balance)  # Sell 80% to maintain some inventory
            
            return AgentAction.SWAP, {
                "asset_in": best_asset,
                "asset_out": Asset.USDC,  # Convert to stable asset
                "amount_in": sell_amount,
                "min_amount_out": sell_amount * asset_prices.get(best_asset, 0.0) * 0.95
            }
        
        return AgentAction.HOLD, {}
    
    def _manage_moet_balance(self, asset_prices: Dict[Asset, float]) -> Tuple[AgentAction, dict]:
        """Maintain optimal MOET balance for liquidations"""
        
        current_moet = self.state.token_balances.get(Asset.MOET, 0.0)
        target_moet = 10_000.0  # Target $10k MOET balance for liquidations
        
        if current_moet < target_moet * 0.5:  # Below 50% of target
            # Convert some USDC to MOET
            usdc_balance = self.state.token_balances.get(Asset.USDC, 0.0)
            needed_moet = target_moet - current_moet
            
            if usdc_balance > needed_moet:
                return AgentAction.SWAP, {
                    "asset_in": Asset.USDC,
                    "asset_out": Asset.MOET,
                    "amount_in": needed_moet,
                    "min_amount_out": needed_moet * 0.99
                }
        
        return AgentAction.HOLD, {}
    
    def _is_market_stressed(self, asset_prices: Dict[Asset, float]) -> bool:
        """Determine if market is under stress (more liquidation opportunities)"""
        
        # Simple heuristic: if any major asset dropped significantly
        initial_prices = {
            Asset.ETH: 4400.0,
            Asset.BTC: 118_000.0,
            Asset.FLOW: 0.40,
            Asset.USDC: 1.0
        }
        
        for asset, current_price in asset_prices.items():
            if asset in initial_prices:
                initial_price = initial_prices[asset]
                drop = (initial_price - current_price) / initial_price
                
                if drop > 0.15:  # 15% drop indicates stress
                    return True
        
        return False
    
    def record_liquidation(self, profit: float, position: dict):
        """Record successful liquidation for tracking"""
        self.successful_liquidations += 1
        self.total_liquidation_profit += profit
        self.liquidation_history.append({
            "profit": profit,
            "position": position,
            "timestamp": len(self.liquidation_history)
        })