#!/usr/bin/env python3
"""
Yield Token System for High Tide Scenario

Implements yield-bearing tokens that automatically earn 10% APR and can be
traded back to MOET for position rebalancing.
"""

from typing import List, Dict, Optional
import time
from dataclasses import dataclass
from .protocol import Asset


@dataclass
class YieldToken:
    """Individual yield token with continuous yield accrual"""
    
    def __init__(self, principal_amount: float, apr: float = 0.10, creation_minute: int = 0):
        self.principal = principal_amount
        self.apr = apr
        self.creation_minute = creation_minute
        
    def get_current_value(self, current_minute: int) -> float:
        """Calculate current value including accrued yield"""
        if current_minute < self.creation_minute:
            return self.principal
            
        minutes_elapsed = current_minute - self.creation_minute
        # Convert APR to per-minute rate: APR / (365 * 24 * 60)
        minute_rate = self.apr / 525600
        return self.principal * (1 + minute_rate * minutes_elapsed)
        
    def get_accrued_yield(self, current_minute: int) -> float:
        """Get yield earned above principal amount"""
        return self.get_current_value(current_minute) - self.principal
        
    def get_sellable_value(self, current_minute: int) -> float:
        """Get value that can be sold (with minimal slippage)"""
        # Assume 0.1% slippage on yield token trading
        return self.get_current_value(current_minute) * 0.999


class YieldTokenManager:
    """Manages yield token portfolio for a High Tide agent"""
    
    def __init__(self):
        self.yield_tokens: List[YieldToken] = []
        self.total_principal_invested = 0.0
        
    def mint_yield_tokens(self, moet_amount: float, current_minute: int) -> List[YieldToken]:
        """Convert MOET to yield tokens at 1:1 ratio"""
        if moet_amount <= 0:
            return []
            
        # Create yield tokens at $1.00 each
        num_tokens = int(moet_amount)  # Floor to whole tokens
        new_tokens = []
        
        for _ in range(num_tokens):
            token = YieldToken(1.0, 0.10, current_minute)
            new_tokens.append(token)
            self.yield_tokens.append(token)
            
        self.total_principal_invested += num_tokens
        return new_tokens
        
    def sell_yield_tokens(self, amount_needed: float, current_minute: int) -> float:
        """
        Sell yield tokens to raise MOET, prioritizing highest yield first
        
        Returns:
            Amount of MOET obtained from selling tokens
        """
        if amount_needed <= 0 or not self.yield_tokens:
            return 0.0
            
        # Sort tokens by current value (highest yield first)
        self.yield_tokens.sort(
            key=lambda token: token.get_current_value(current_minute),
            reverse=True
        )
        
        moet_raised = 0.0
        tokens_to_remove = []
        
        for token in self.yield_tokens:
            if moet_raised >= amount_needed:
                break
                
            token_value = token.get_sellable_value(current_minute)
            moet_raised += token_value
            tokens_to_remove.append(token)
            self.total_principal_invested -= token.principal
            
        # Remove sold tokens
        for token in tokens_to_remove:
            self.yield_tokens.remove(token)
            
        return min(moet_raised, amount_needed)
        
    def sell_yield_above_principal(self, current_minute: int) -> float:
        """
        Sell only the accrued yield portion, keeping principal intact
        
        Returns:
            Amount of MOET obtained from selling accrued yield
        """
        if not self.yield_tokens:
            return 0.0
            
        total_yield_raised = 0.0
        
        for token in self.yield_tokens:
            accrued_yield = token.get_accrued_yield(current_minute)
            if accrued_yield > 0:
                # "Sell" just the yield portion (conceptually)
                # In practice, we adjust the token principal down
                sellable_yield = accrued_yield * 0.999  # 0.1% slippage
                total_yield_raised += sellable_yield
                
                # Reset token to current principal value minus yield sold
                token.principal = token.get_current_value(current_minute) - accrued_yield
                token.creation_minute = current_minute
                
        return total_yield_raised
        
    def calculate_total_value(self, current_minute: int) -> float:
        """Calculate total current value of all yield tokens"""
        return sum(
            token.get_current_value(current_minute) 
            for token in self.yield_tokens
        )
        
    def calculate_total_yield_accrued(self, current_minute: int) -> float:
        """Calculate total yield accrued across all tokens"""
        return sum(
            token.get_accrued_yield(current_minute)
            for token in self.yield_tokens
        )
        
    def get_portfolio_summary(self, current_minute: int) -> Dict[str, float]:
        """Get summary of yield token portfolio"""
        total_value = self.calculate_total_value(current_minute)
        total_yield = self.calculate_total_yield_accrued(current_minute)
        
        return {
            "num_tokens": len(self.yield_tokens),
            "total_principal": self.total_principal_invested,
            "total_current_value": total_value,
            "total_accrued_yield": total_yield,
            "yield_percentage": (total_yield / self.total_principal_invested * 100) if self.total_principal_invested > 0 else 0.0,
            "average_token_age_minutes": self._calculate_average_age(current_minute)
        }
        
    def _calculate_average_age(self, current_minute: int) -> float:
        """Calculate average age of tokens in minutes"""
        if not self.yield_tokens:
            return 0.0
            
        total_age = sum(
            max(0, current_minute - token.creation_minute)
            for token in self.yield_tokens
        )
        return total_age / len(self.yield_tokens)


class YieldTokenPool:
    """Global pool for MOET <-> Yield Token trading"""
    
    def __init__(self, initial_moet_reserve: float = 250_000.0):
        self.moet_reserve = initial_moet_reserve
        self.yield_token_reserve = initial_moet_reserve  # Start 1:1
        self.fee_rate = 0.001  # 0.1% trading fee (minimal slippage)
        
    def quote_yield_token_purchase(self, moet_amount: float) -> float:
        """Quote how many yield tokens can be purchased with MOET"""
        if moet_amount <= 0:
            return 0.0
            
        # Simple 1:1 exchange with minimal slippage
        fee = moet_amount * self.fee_rate
        return moet_amount - fee
        
    def quote_yield_token_sale(self, yield_token_value: float) -> float:
        """Quote how much MOET will be received for selling yield tokens"""
        if yield_token_value <= 0:
            return 0.0
            
        # Simple 1:1 exchange with minimal slippage
        fee = yield_token_value * self.fee_rate
        return yield_token_value - fee
        
    def execute_yield_token_purchase(self, moet_amount: float) -> float:
        """Execute purchase of yield tokens with MOET"""
        yield_tokens = self.quote_yield_token_purchase(moet_amount)
        
        # Update reserves
        self.moet_reserve += moet_amount
        self.yield_token_reserve -= yield_tokens
        
        return yield_tokens
        
    def execute_yield_token_sale(self, yield_token_value: float) -> float:
        """Execute sale of yield tokens for MOET"""
        moet_received = self.quote_yield_token_sale(yield_token_value)
        
        # Update reserves
        self.moet_reserve -= moet_received
        self.yield_token_reserve += yield_token_value
        
        return moet_received
        
    def get_pool_state(self) -> Dict[str, float]:
        """Get current state of the yield token pool"""
        return {
            "moet_reserve": self.moet_reserve,
            "yield_token_reserve": self.yield_token_reserve,
            "exchange_rate": self.yield_token_reserve / self.moet_reserve if self.moet_reserve > 0 else 1.0,
            "total_liquidity": self.moet_reserve + self.yield_token_reserve
        }
