#!/usr/bin/env python3
"""
Basic Trader Agent

Simple trading agent for providing market activity and liquidity in MOET pairs.
Focuses on basic trading patterns rather than complex strategies.
"""

import random
from typing import Dict, Tuple
from .base_agent import BaseAgent, AgentAction, Asset


class BasicTrader(BaseAgent):
    """Basic trading agent for market activity"""
    
    def __init__(self, agent_id: str, initial_balance: float = 50_000.0):
        super().__init__(agent_id, "basic_trader", initial_balance)
        
        # Trading parameters
        self.trade_frequency = 0.3  # 30% chance to trade each round
        self.max_trade_size_pct = 0.1  # Max 10% of balance per trade
        self.profit_target = 0.02  # 2% profit target
        self.stop_loss = 0.05  # 5% stop loss
        
        # MOET trading preferences
        self.moet_trade_threshold = 0.02  # Trade when MOET deviates 2% from peg
        self.arbitrage_threshold = 0.01  # 1% price difference for arbitrage
        
        # State tracking
        self.last_trade_price = {}
        self.trades_executed = 0
    
    def decide_action(self, protocol_state: dict, asset_prices: Dict[Asset, float]) -> Tuple[AgentAction, dict]:
        """
        Basic trading decision logic:
        1. Check for MOET peg trading opportunities
        2. Simple momentum/mean reversion trades
        3. Random market-making activity
        """
        
        # Skip trading based on frequency
        if random.random() > self.trade_frequency:
            return AgentAction.HOLD, {}
        
        # Check for MOET peg trading opportunities
        moet_price = asset_prices.get(Asset.MOET, 1.0)
        peg_deviation = abs(moet_price - 1.0)
        
        if peg_deviation > self.moet_trade_threshold:
            return self._trade_moet_peg(moet_price, asset_prices)
        
        # Simple momentum trading
        trade_action = self._simple_momentum_trade(asset_prices)
        if trade_action[0] != AgentAction.HOLD:
            return trade_action
        
        # Random market-making activity (small trades for liquidity)
        if random.random() < 0.1:  # 10% chance for market making
            return self._market_making_trade(asset_prices)
        
        return AgentAction.HOLD, {}
    
    def _trade_moet_peg(self, moet_price: float, asset_prices: Dict[Asset, float]) -> Tuple[AgentAction, dict]:
        """Trade based on MOET peg deviation"""
        
        if moet_price > 1.02:  # MOET overpriced, sell MOET for other assets
            moet_balance = self.state.token_balances.get(Asset.MOET, 0.0)
            if moet_balance > 100:  # Minimum trade size
                trade_amount = min(moet_balance * 0.5, moet_balance)
                target_asset = self._select_trade_target(exclude=Asset.MOET)
                
                if target_asset:
                    return AgentAction.SWAP, {
                        "asset_in": Asset.MOET,
                        "asset_out": target_asset,
                        "amount_in": trade_amount,
                        "min_amount_out": trade_amount * 0.98  # 2% slippage tolerance
                    }
        
        elif moet_price < 0.98:  # MOET underpriced, buy MOET with other assets
            target_asset = self._select_asset_to_sell()
            if target_asset:
                asset_balance = self.state.token_balances.get(target_asset, 0.0)
                asset_price = asset_prices.get(target_asset, 1.0)
                
                if asset_balance * asset_price > 100:  # Minimum $100 trade
                    trade_amount = min(asset_balance * 0.3, asset_balance)
                    
                    return AgentAction.SWAP, {
                        "asset_in": target_asset,
                        "asset_out": Asset.MOET,
                        "amount_in": trade_amount,
                        "min_amount_out": trade_amount * asset_price * 0.98
                    }
        
        return AgentAction.HOLD, {}
    
    def _simple_momentum_trade(self, asset_prices: Dict[Asset, float]) -> Tuple[AgentAction, dict]:
        """Simple momentum-based trading"""
        
        for asset in [Asset.ETH, Asset.BTC, Asset.FLOW]:
            current_price = asset_prices.get(asset, 0.0)
            last_price = self.last_trade_price.get(asset, current_price)
            
            if last_price > 0:
                price_change = (current_price - last_price) / last_price
                
                # Simple momentum: buy on upward momentum, sell on downward
                if abs(price_change) > 0.05:  # 5% price movement
                    asset_balance = self.state.token_balances.get(asset, 0.0)
                    
                    if price_change > 0 and asset_balance * current_price < 10000:  # Buy more if going up
                        # Buy with USDC or MOET
                        funding_asset = Asset.USDC if self.state.token_balances.get(Asset.USDC, 0.0) > 500 else Asset.MOET
                        funding_balance = self.state.token_balances.get(funding_asset, 0.0)
                        
                        if funding_balance > 500:
                            trade_amount = min(funding_balance * 0.2, 2000)  # Max $2k trade
                            
                            return AgentAction.SWAP, {
                                "asset_in": funding_asset,
                                "asset_out": asset,
                                "amount_in": trade_amount,
                                "min_amount_out": trade_amount / current_price * 0.95
                            }
                    
                    elif price_change < -0.05 and asset_balance > 0:  # Sell on significant drop
                        trade_amount = min(asset_balance * 0.3, asset_balance)
                        
                        return AgentAction.SWAP, {
                            "asset_in": asset,
                            "asset_out": Asset.USDC,
                            "amount_in": trade_amount,
                            "min_amount_out": trade_amount * current_price * 0.95
                        }
            
            # Update price tracking
            self.last_trade_price[asset] = current_price
        
        return AgentAction.HOLD, {}
    
    def _market_making_trade(self, asset_prices: Dict[Asset, float]) -> Tuple[AgentAction, dict]:
        """Small trades to provide market liquidity"""
        
        # Random small trade between assets
        asset_options = [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC, Asset.MOET]
        asset_in = random.choice(asset_options)
        asset_out = random.choice([a for a in asset_options if a != asset_in])
        
        balance_in = self.state.token_balances.get(asset_in, 0.0)
        price_in = asset_prices.get(asset_in, 1.0)
        
        if balance_in * price_in > 200:  # Minimum $200 for market making
            # Small trade (2-5% of balance)
            trade_pct = random.uniform(0.02, 0.05)
            trade_amount = balance_in * trade_pct
            
            return AgentAction.SWAP, {
                "asset_in": asset_in,
                "asset_out": asset_out,
                "amount_in": trade_amount,
                "min_amount_out": 0  # Market making accepts wider spreads
            }
        
        return AgentAction.HOLD, {}
    
    def _select_trade_target(self, exclude: Asset = None) -> Asset:
        """Select best asset to trade into"""
        options = [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]
        if exclude:
            options = [a for a in options if a != exclude]
        
        # Simple selection based on current balance (diversification)
        best_asset = None
        min_balance_value = float('inf')
        
        for asset in options:
            balance = self.state.token_balances.get(asset, 0.0)
            if balance < min_balance_value:
                min_balance_value = balance
                best_asset = asset
        
        return best_asset if best_asset else random.choice(options)
    
    def _select_asset_to_sell(self) -> Asset:
        """Select asset with largest balance to sell"""
        best_asset = None
        max_balance_value = 0
        
        for asset in [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]:
            balance = self.state.token_balances.get(asset, 0.0)
            if balance > max_balance_value:
                max_balance_value = balance
                best_asset = asset
        
        return best_asset