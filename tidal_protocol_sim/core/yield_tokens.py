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
from .uniswap_v3_math import create_yield_token_pool, UniswapV3SlippageCalculator


@dataclass
class YieldToken:
    """Individual yield token with continuous yield accrual"""
    
    def __init__(self, principal_amount: float, apr: float = 0.05, creation_minute: int = 0):
        self.principal = principal_amount
        self.apr = apr
        self.creation_minute = creation_minute
        
    def get_current_value(self, current_minute: int) -> float:
        """Calculate current value including accrued yield"""
        if current_minute < self.creation_minute:
            return self.principal
            
        minutes_elapsed = current_minute - self.creation_minute
        # Convert APR to per-minute rate: (1 + APR)^(1/525600) - 1
        # For 10% APR over 60 minutes, this should be very small
        minute_rate = (1 + self.apr) ** (1 / 525600) - 1
        return self.principal * (1 + minute_rate) ** minutes_elapsed
        
    def get_accrued_yield(self, current_minute: int) -> float:
        """Get yield earned above principal amount"""
        return self.get_current_value(current_minute) - self.principal
        
    def get_sellable_value(self, current_minute: int) -> float:
        """Get value that can be sold (with minimal slippage)"""
        # Assume 0.1% slippage on yield token trading
        return self.get_current_value(current_minute) * 0.999


class YieldTokenManager:
    """Manages yield token portfolio for a High Tide agent"""
    
    def __init__(self, yield_token_pool: Optional['YieldTokenPool'] = None):
        self.yield_tokens: List[YieldToken] = []
        self.total_principal_invested = 0.0
        self.yield_token_pool = yield_token_pool
        
    def mint_yield_tokens(self, moet_amount: float, current_minute: int) -> List[YieldToken]:
        """Convert MOET to yield tokens using Uniswap V3 pool"""
        if moet_amount <= 0:
            return []
            
        # Use Uniswap V3 pool to get actual yield tokens for MOET
        if self.yield_token_pool:
            actual_yield_tokens_received = self.yield_token_pool.execute_yield_token_purchase(moet_amount)
        else:
            # Fallback to 1:1 ratio if no pool available
            actual_yield_tokens_received = moet_amount
            
        # Create yield tokens based on actual amount received from pool
        num_tokens = int(actual_yield_tokens_received)  # Floor to whole tokens
        new_tokens = []
        
        for _ in range(num_tokens):
            token = YieldToken(1.0, 0.10, current_minute)  # Keep hardcoded APR
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
                
            # Use real Uniswap V3 slippage calculation instead of hardcoded 0.999
            token_value = token.get_current_value(current_minute)
            real_moet_value = self._calculate_real_slippage(token_value)
            moet_raised += real_moet_value
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
                # Use real Uniswap V3 slippage calculation instead of hardcoded 0.999
                real_moet_value = self._calculate_real_slippage(accrued_yield)
                total_yield_raised += real_moet_value
                
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
    
    def _calculate_real_slippage(self, yield_token_value: float) -> float:
        """Calculate real slippage using Uniswap V3 math"""
        if not self.yield_token_pool or yield_token_value <= 0:
            return yield_token_value * 0.999  # Fallback to hardcoded if no pool
        
        # Use Uniswap V3 slippage calculator for real pricing
        swap_result = self.yield_token_pool.slippage_calculator.calculate_swap_slippage(
            amount_in=yield_token_value,
            token_in="Yield_Token",
            concentrated_range=self.yield_token_pool.concentration_range
        )
        
        # Return the actual MOET amount after slippage
        return swap_result.get("amount_out", yield_token_value * 0.999)


class YieldTokenPool:
    """
    Global pool for MOET <-> Yield Token trading using Uniswap V3 math
    
    This represents the internal protocol pool with tight concentration (95% at peg)
    since yield tokens are protocol-native assets designed to track MOET closely.
    Now leverages sophisticated Uniswap V3 concentrated liquidity mathematics.
    """
    
    def __init__(self, initial_moet_reserve: float = 250_000.0, btc_price: float = 100_000.0, concentration: float = 0.95):
        # Create the underlying Uniswap V3 pool
        pool_size_usd = initial_moet_reserve * 2  # Total pool size (both sides)
        self.uniswap_pool = create_yield_token_pool(
            pool_size_usd=pool_size_usd,
            btc_price=btc_price,
            concentration=concentration
        )
        
        # Create slippage calculator for accurate cost calculations
        self.slippage_calculator = UniswapV3SlippageCalculator(self.uniswap_pool)
        
        # Legacy interface properties for backward compatibility
        self._update_legacy_properties()
        
        # Store configuration
        self.btc_price = btc_price
        self.concentration = concentration
    
    def _update_legacy_properties(self):
        """Update legacy properties for backward compatibility"""
        # Get reserves from Uniswap V3 pool
        self.moet_reserve = self.uniswap_pool.token0_reserve or 0.0
        self.yield_token_reserve = self.uniswap_pool.token1_reserve or 0.0
        
        # Calculate effective fee rate from Uniswap V3
        self.fee_rate = self.uniswap_pool.fee_tier
        self.concentration_range = 1.0 - self.uniswap_pool.concentration  # Inverted for legacy compatibility
        
    def quote_yield_token_purchase(self, moet_amount: float) -> float:
        """Quote how many yield tokens can be purchased with MOET using Uniswap V3 math"""
        if moet_amount <= 0:
            return 0.0
            
        # Use Uniswap V3 slippage calculator for accurate pricing
        swap_result = self.slippage_calculator.calculate_swap_slippage(
            amount_in=moet_amount,
            token_in="MOET",
            concentrated_range=self.concentration_range
        )
        
        return swap_result.get("amount_out", 0.0)
        
    def quote_yield_token_sale(self, yield_token_value: float) -> float:
        """Quote how much MOET will be received for selling yield tokens using Uniswap V3 math"""
        if yield_token_value <= 0:
            return 0.0
            
        # Use Uniswap V3 slippage calculator for accurate pricing
        swap_result = self.slippage_calculator.calculate_swap_slippage(
            amount_in=yield_token_value,
            token_in="Yield_Token",
            concentrated_range=self.concentration_range
        )
        
        return swap_result.get("amount_out", 0.0)
        
    def execute_yield_token_purchase(self, moet_amount: float) -> float:
        """Execute purchase of yield tokens with MOET using Uniswap V3 swaps"""
        if moet_amount <= 0:
            return 0.0
            
        # Execute actual Uniswap V3 swap: MOET -> Yield Token
        # Convert USD to scaled amount for Uniswap V3 math
        amount_in_scaled = int(moet_amount * 1e6)
        
        try:
            from .uniswap_v3_math import MIN_SQRT_RATIO, MAX_SQRT_RATIO
            
            # MOET -> Yield Token, so zero_for_one = True
            amount_in_actual, amount_out_actual = self.uniswap_pool.swap(
                zero_for_one=True,  # MOET (token0) -> Yield Token (token1)
                amount_specified=amount_in_scaled,
                sqrt_price_limit_x96=MIN_SQRT_RATIO + 1  # No specific limit
            )
            
            # Convert back to USD amounts
            yield_tokens_received = amount_out_actual / 1e6
            
            # Update legacy properties for backward compatibility
            self._update_legacy_properties()
            
            return yield_tokens_received
            
        except (ValueError, ZeroDivisionError) as e:
            print(f"Warning: Yield token purchase failed: {e}")
            return 0.0
        
    def execute_yield_token_sale(self, yield_token_value: float) -> float:
        """Execute sale of yield tokens for MOET using Uniswap V3 swaps"""
        if yield_token_value <= 0:
            return 0.0
            
        # Execute actual Uniswap V3 swap: Yield Token -> MOET
        # Convert USD to scaled amount for Uniswap V3 math
        amount_in_scaled = int(yield_token_value * 1e6)
        
        try:
            from .uniswap_v3_math import MIN_SQRT_RATIO, MAX_SQRT_RATIO
            
            # Yield Token -> MOET, so zero_for_one = False
            amount_in_actual, amount_out_actual = self.uniswap_pool.swap(
                zero_for_one=False,  # Yield Token (token1) -> MOET (token0)
                amount_specified=amount_in_scaled,
                sqrt_price_limit_x96=MAX_SQRT_RATIO - 1  # No specific limit
            )
            
            # Convert back to USD amounts
            moet_received = amount_out_actual / 1e6
            
            # Update legacy properties for backward compatibility
            self._update_legacy_properties()
            
            return moet_received
            
        except (ValueError, ZeroDivisionError) as e:
            print(f"Warning: Yield token sale failed: {e}")
            return 0.0
        
    def get_pool_state(self) -> Dict[str, float]:
        """Get current state of the yield token pool using Uniswap V3 data"""
        # Update legacy properties to get current state
        self._update_legacy_properties()
        
        # Get current price from Uniswap V3 pool
        current_price = self.uniswap_pool.get_price()
        
        return {
            "moet_reserve": self.moet_reserve,
            "yield_token_reserve": self.yield_token_reserve,
            "exchange_rate": current_price,  # Use Uniswap V3 price
            "total_liquidity": self.uniswap_pool.get_total_active_liquidity(),
            "concentration": self.uniswap_pool.concentration,
            "fee_tier": self.uniswap_pool.fee_tier,
            "current_price": current_price,
            "tick_current": self.uniswap_pool.tick_current
        }
    
    def get_uniswap_pool(self):
        """Get the underlying Uniswap V3 pool for advanced operations"""
        return self.uniswap_pool
    
    def get_slippage_calculator(self):
        """Get the slippage calculator for cost analysis"""
        return self.slippage_calculator
