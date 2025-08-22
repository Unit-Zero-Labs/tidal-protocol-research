#!/usr/bin/env python3
"""
Trader policy for the Tidal Protocol simulation.

This policy implements price momentum-based trading behavior.
"""

from typing import List
import random
from ..base import BasePolicy
from ...simulation.primitives import Action, ActionKind, MarketSnapshot, AgentState, Asset


class TraderPolicy(BasePolicy):
    """Policy for price momentum-based trading"""
    
    def __init__(self, 
                 trading_frequency: float = 0.1,
                 momentum_threshold: float = 0.05,
                 max_trade_size_pct: float = 0.1,
                 risk_tolerance: float = 0.5):
        """
        Initialize trader policy
        
        Args:
            trading_frequency: Probability of taking action each period (0-1)
            momentum_threshold: Minimum price change to trigger trade (0-1)
            max_trade_size_pct: Maximum percentage of balance to trade (0-1)
            risk_tolerance: Risk tolerance level (0-1, higher = more risk)
        """
        self.trading_frequency = trading_frequency
        self.momentum_threshold = momentum_threshold
        self.max_trade_size_pct = max_trade_size_pct
        self.risk_tolerance = risk_tolerance
        self.price_history: List[float] = []
        self.last_trade_timestamp = 0
    
    def decide(self, agent_state: AgentState, snapshot: MarketSnapshot) -> List[Action]:
        """
        Make trading decisions based on price momentum
        
        Args:
            agent_state: Current agent state
            snapshot: Current market snapshot
            
        Returns:
            List of trading actions
        """
        actions = []
        
        # Check if we should trade this period
        if random.random() > self.trading_frequency:
            return [Action(ActionKind.HOLD, "", {})]
        
        # Update price history
        current_price = snapshot.token_prices.get(Asset.MOET, 1.0)
        self.price_history.append(current_price)
        
        # Keep only recent history (last 10 periods)
        if len(self.price_history) > 10:
            self.price_history = self.price_history[-10:]
        
        # Need at least 2 price points for momentum
        if len(self.price_history) < 2:
            return [Action(ActionKind.HOLD, "", {})]
        
        # Calculate price momentum
        price_change = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
        
        # Check if momentum exceeds threshold
        if abs(price_change) < self.momentum_threshold:
            return [Action(ActionKind.HOLD, "", {})]
        
        # Determine trade direction and size
        if price_change > 0:
            # Upward momentum - buy
            action = self._create_buy_action(agent_state, snapshot, price_change)
        else:
            # Downward momentum - sell
            action = self._create_sell_action(agent_state, snapshot, abs(price_change))
        
        if action:
            actions.append(action)
        
        return actions if actions else [Action(ActionKind.HOLD, "", {})]
    
    def _create_buy_action(self, agent_state: AgentState, snapshot: MarketSnapshot, momentum: float) -> Action:
        """Create a buy action based on upward momentum"""
        # Check available cash
        available_cash = agent_state.cash_balance
        
        if available_cash <= 0:
            return None
        
        # Calculate trade size based on momentum and risk tolerance
        trade_size_factor = min(momentum * self.risk_tolerance, self.max_trade_size_pct)
        trade_amount = available_cash * trade_size_factor
        
        # Minimum trade size check
        if trade_amount < 10:  # Minimum $10 trade
            return None
        
        return Action(
            kind=ActionKind.SWAP_BUY,
            agent_id="",  # Will be set by agent
            params={
                "market_id": "uniswap_v2",
                "asset_in": Asset.USDC,
                "asset_out": Asset.MOET,
                "amount_in": trade_amount,
                "min_amount_out": trade_amount * 0.95  # 5% slippage tolerance
            }
        )
    
    def _create_sell_action(self, agent_state: AgentState, snapshot: MarketSnapshot, momentum: float) -> Action:
        """Create a sell action based on downward momentum"""
        # Check available MOET balance
        moet_balance = agent_state.token_balances.get(Asset.MOET, 0.0)
        
        if moet_balance <= 0:
            return None
        
        # Calculate trade size based on momentum and risk tolerance
        trade_size_factor = min(momentum * self.risk_tolerance, self.max_trade_size_pct)
        trade_amount = moet_balance * trade_size_factor
        
        # Minimum trade size check
        current_price = snapshot.token_prices.get(Asset.MOET, 1.0)
        if trade_amount * current_price < 10:  # Minimum $10 trade value
            return None
        
        return Action(
            kind=ActionKind.SWAP_SELL,
            agent_id="",  # Will be set by agent
            params={
                "market_id": "uniswap_v2",
                "asset_in": Asset.MOET,
                "asset_out": Asset.USDC,
                "amount_in": trade_amount,
                "min_amount_out": trade_amount * current_price * 0.95  # 5% slippage tolerance
            }
        )
    
    def can_execute_action(self, action: Action, agent_state: AgentState, snapshot: MarketSnapshot) -> bool:
        """
        Validate if a trading action can be executed
        
        Args:
            action: Action to validate
            agent_state: Current agent state
            snapshot: Current market snapshot
            
        Returns:
            True if action can be executed
        """
        if action.kind == ActionKind.HOLD:
            return True
        
        if action.kind == ActionKind.SWAP_BUY:
            # Check if agent has enough cash
            required_amount = action.params.get("amount_in", 0)
            return agent_state.cash_balance >= required_amount
        
        elif action.kind == ActionKind.SWAP_SELL:
            # Check if agent has enough tokens
            asset_in = action.params.get("asset_in", Asset.MOET)
            required_amount = action.params.get("amount_in", 0)
            available_balance = agent_state.token_balances.get(asset_in, 0.0)
            return available_balance >= required_amount
        
        return False
