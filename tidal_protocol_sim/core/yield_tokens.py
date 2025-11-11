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


def calculate_true_yield_token_price(current_minute: int, apr: float = 0.10, initial_price: float = 1.0) -> float:
    """
    Calculate the true yield token price at any point in time based on APR
    
    This is the "oracle" price that pool rebalancers use as their target.
    Yield tokens start at $1.00 and appreciate continuously at the specified APR.
    
    Uses linear interest (simple APR): Price = Initial × (1 + APR × t)
    
    Args:
        current_minute: Current simulation minute
        apr: Annual Percentage Rate (default 0.10 for 10%)
        initial_price: Initial token price (default 1.0)
    
    Returns:
        True yield token price including accrued yield
    """
    if current_minute <= 0:
        return initial_price
        
    # Convert APR to per-minute rate: APR * (minutes_elapsed / minutes_per_year)
    minutes_per_year = 365 * 24 * 60  # 525,600 minutes per year
    minute_rate = apr * (current_minute / minutes_per_year)
    return initial_price * (1 + minute_rate)


@dataclass
class YieldToken:
    """
    Fungible yield token that uses global pricing.
    All YT tokens have the same price at any given moment, regardless of when they were created.
    """
    
    def __init__(self, quantity: float, apr: float = 0.10, creation_minute: int = 0):
        self.quantity = quantity  # Number of YT tokens held
        self.apr = apr  # APR for the global price function
        self.creation_minute = creation_minute  # For tracking only, not used in valuation
        
        # Legacy field for backward compatibility with accounting that tracks "initial_value"
        # This represents the QUANTITY at creation, which equals the value at creation time
        # since YT price starts at $1.00 at minute 0
        self.initial_value = quantity
        
    def get_current_value(self, current_minute: int) -> float:
        """Calculate current value using GLOBAL YT price function"""
        # Use the global price function - all YT tokens have same price at this moment
        global_yt_price = calculate_true_yield_token_price(current_minute, self.apr, initial_price=1.0)
        return self.quantity * global_yt_price
        
    def get_sellable_value(self, current_minute: int) -> float:
        """Get value that can be sold (with minimal slippage)"""
        # Assume 0.1% slippage on yield token trading
        return self.get_current_value(current_minute) * 0.999


class YieldTokenManager:
    """Manages yield token portfolio for a High Tide agent"""
    
    def __init__(self, yield_token_pool: Optional['YieldTokenPool'] = None):
        self.yield_tokens: List[YieldToken] = []
        self.total_initial_value_invested = 0.0  # Total initial value invested
        self.yield_token_pool = yield_token_pool
        
    def mint_yield_tokens(self, moet_amount: float, current_minute: int, use_direct_minting: bool = False) -> List[YieldToken]:
        """Convert MOET to yield tokens using 1:1 rate at minute 0 or Uniswap V3 pool"""
        if moet_amount <= 0:
            return []
            
        if not self.yield_token_pool:
            raise ValueError("YieldTokenManager requires a YieldTokenPool for minting yield tokens")
        
        # MINUTE 0: Use 1:1 rate to establish balanced pool
        if current_minute == 0 and use_direct_minting:
            # At minute 0, yield tokens are purchased at 1:1 rate
            # This establishes the initial $250k:$250k balanced pool
            actual_yield_tokens_received = moet_amount  # 1:1 rate
        else:
            # MINUTE > 0: Use Uniswap V3 pool with real slippage
            actual_yield_tokens_received = self.yield_token_pool.execute_yield_token_purchase(moet_amount)
        
        if actual_yield_tokens_received <= 0:
            raise ValueError(f"Failed to mint yield tokens for MOET amount {moet_amount}")
            
        # Create yield tokens based on actual amount received
        # Each token represents $1 of initial value, but will rebase over time
        # Handle fractional amounts by creating one token with the full amount
        new_tokens = []
        
        if actual_yield_tokens_received > 0:
            # Convert USD value received to YT quantity using global price
            current_yt_price = calculate_true_yield_token_price(current_minute, 0.10, 1.0)
            yt_quantity = actual_yield_tokens_received / current_yt_price
            
            # Create a token with the quantity (fungible tokens priced globally)
            # CRITICAL: Set initial_value to moet_amount for accounting consistency
            # This ensures that when we sell, we deduct the correct MOET cost basis
            token = YieldToken(yt_quantity, 0.10, current_minute)
            token.initial_value = moet_amount  # Override with MOET cost for accounting
            new_tokens.append(token)
            self.yield_tokens.append(token)
            
        # Track the MOET amount spent for accounting purposes
        # Note: actual_yield_tokens_received can differ from moet_amount due to Uniswap slippage
        # We track moet_amount to match debt accounting
        self.total_initial_value_invested += moet_amount
        return new_tokens
        
    def sell_yield_tokens(self, moet_amount_needed: float, current_minute: int) -> tuple[float, float]:
        """
        Sell yield tokens to raise the exact MOET amount needed
        Args:
            moet_amount_needed: Amount of MOET needed (not yield token value). Use float('inf') for emergency sale of all tokens.
        Returns:
            tuple: (Amount of MOET obtained, Actual yield token value sold)
        """
        if not self.yield_tokens:
            return 0.0, 0.0
        
        # Calculate total value of all tokens (including yield appreciation)
        total_token_value = sum(token.get_current_value(current_minute) for token in self.yield_tokens)
        
        if total_token_value <= 0:
            return 0.0, 0.0
        
        # Handle emergency sale (sell all tokens)
        if moet_amount_needed == float('inf'):
            yield_tokens_to_sell = total_token_value
        else:
            # Use simple calculation to find the amount of yield tokens to sell
            # Start with 1:1 assumption, let slippage handle reality
            yield_tokens_to_sell = self._calculate_yield_tokens_needed(moet_amount_needed)
            
            if yield_tokens_to_sell <= 0:
                return 0.0, 0.0
        
        # Handle emergency sale (sell all tokens)
        if moet_amount_needed == float('inf'):
            # Remove all tokens for emergency sale
            total_initial_value = sum(token.initial_value for token in self.yield_tokens)
            self.yield_tokens.clear()
            self.total_initial_value_invested = 0.0
        else:
            # Calculate how many complete tokens we can sell
            token_value = self.yield_tokens[0].get_current_value(current_minute)
            complete_tokens_needed = int(yield_tokens_to_sell / token_value)
            fractional_amount_needed = yield_tokens_to_sell - (complete_tokens_needed * token_value)
            
            # Sell complete tokens (remove them entirely)
            tokens_to_remove = min(complete_tokens_needed, len(self.yield_tokens))
            for _ in range(tokens_to_remove):
                token = self.yield_tokens.pop(0)  # Remove from front
                self.total_initial_value_invested -= token.initial_value
            
            # If we still need more and have tokens left, sell fractional amount
            if fractional_amount_needed > 0 and self.yield_tokens:
                fractional_fraction = fractional_amount_needed / token_value
                # Reduce both quantity (actual YT held) and initial_value (accounting cost) proportionally
                self.yield_tokens[0].quantity *= (1 - fractional_fraction)
                self.yield_tokens[0].initial_value *= (1 - fractional_fraction)
                # Also deduct the sold portion from total accounting
                self.total_initial_value_invested -= (self.yield_tokens[0].initial_value * fractional_fraction / (1 - fractional_fraction))
                # If the token becomes too small, remove it
                if self.yield_tokens[0].quantity < 0.01:
                    removed_token = self.yield_tokens.pop(0)
                    self.total_initial_value_invested -= removed_token.initial_value
        
        # Use real Uniswap V3 slippage calculation for the batch
        # This gives us the ACTUAL MOET amount based on pool math
        moet_raised = self._calculate_real_slippage(yield_tokens_to_sell)
        
        return moet_raised, yield_tokens_to_sell
    
    def _calculate_yield_tokens_needed(self, moet_amount_needed: float) -> float:
        """
        Simple calculation: assume 1:1 ratio, let slippage handle reality
        This replaces the complex binary search with a simple algebraic approach
        """
        return moet_amount_needed  # Start with 1:1 assumption
    
    def calculate_total_yield_earned(self, current_minute: int) -> float:
        """
        Calculate total yield earned from all yield tokens
        
        This is the actual profit from yield accrual, separate from
        the principal value of the tokens.
        """
        total_original_value = sum(token.initial_value for token in self.yield_tokens)
        total_current_value = sum(token.get_current_value(current_minute) for token in self.yield_tokens)
        return total_current_value - total_original_value
        
    def calculate_total_value(self, current_minute: int) -> float:
        """Calculate total current value of all yield tokens"""
        return sum(
            token.get_current_value(current_minute) 
            for token in self.yield_tokens
        )
        
    def calculate_total_yield_accrued(self, current_minute: int) -> float:
        """Calculate total yield accrued across all tokens (current value - initial value)"""
        return sum(
            token.get_current_value(current_minute) - token.initial_value
            for token in self.yield_tokens
        )
        
    def get_portfolio_summary(self, current_minute: int) -> Dict[str, float]:
        """Get summary of yield token portfolio"""
        total_value = self.calculate_total_value(current_minute)
        total_yield = self.calculate_total_yield_accrued(current_minute)
        
        return {
            "num_tokens": len(self.yield_tokens),
            "total_initial_value": self.total_initial_value_invested,
            "total_current_value": total_value,
            "total_accrued_yield": total_yield,
            "yield_percentage": (total_yield / self.total_initial_value_invested * 100) if self.total_initial_value_invested > 0 else 0.0,
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
        if not self.yield_token_pool:
            return 0.0
        
        if yield_token_value <= 0:
            return 0.0
        
        # Use Uniswap V3 slippage calculator for real pricing
        swap_result = self.yield_token_pool.slippage_calculator.calculate_swap_slippage(
            amount_in=yield_token_value,
            token_in="Yield_Token",
            concentrated_range=self.yield_token_pool.concentration_range
        )
        
        # Validate the swap result
        if "amount_out" not in swap_result or swap_result["amount_out"] is None:
            return 0.0
        
        # Return the actual MOET amount after slippage
        return swap_result["amount_out"]
    
    def _remove_yield_tokens(self, yield_tokens_to_remove: float, current_minute: int):
        """Remove yield tokens from inventory after successful sale"""
        if yield_tokens_to_remove <= 0 or not self.yield_tokens:
            return
            
        remaining_to_remove = yield_tokens_to_remove
        tokens_to_remove = []
        
        # Remove tokens from front of list (FIFO)
        for i, token in enumerate(self.yield_tokens):
            token_value = token.get_current_value(current_minute)
            
            if remaining_to_remove >= token_value:
                # Remove entire token
                tokens_to_remove.append(i)
                remaining_to_remove -= token_value
                self.total_initial_value_invested -= token.initial_value
            else:
                # Remove partial token - reduce both quantity and accounting cost
                fraction_to_remove = remaining_to_remove / token_value
                sold_initial_value = token.initial_value * fraction_to_remove
                token.quantity *= (1 - fraction_to_remove)
                token.initial_value *= (1 - fraction_to_remove)
                self.total_initial_value_invested -= sold_initial_value
                if token.quantity < 0.01:  # Remove if too small
                    tokens_to_remove.append(i)
                    self.total_initial_value_invested -= token.initial_value
                remaining_to_remove = 0
                break
        
        # Remove tokens in reverse order to maintain indices
        for i in reversed(tokens_to_remove):
            self.yield_tokens.pop(i)


class YieldTokenPool:
    """
    Global pool for MOET <-> Yield Token trading using Uniswap V3 math
    
    This represents the internal protocol pool with tight concentration (95% at peg)
    since yield tokens are protocol-native assets designed to track MOET closely.
    Now leverages sophisticated Uniswap V3 concentrated liquidity mathematics.
    """
    
    def __init__(self, total_pool_size: float, token0_ratio: float, concentration: float = 0.95):
        # Validate inputs
        if not (0.1 <= token0_ratio <= 0.9):
            raise ValueError(f"token0_ratio must be between 0.1 and 0.9, got {token0_ratio}")
        
        # Create the underlying Uniswap V3 pool with asymmetric ratio
        self.uniswap_pool = create_yield_token_pool(
            pool_size_usd=total_pool_size,
            concentration=concentration,
            token0_ratio=token0_ratio
        )
        
        # Create slippage calculator for accurate cost calculations
        self.slippage_calculator = UniswapV3SlippageCalculator(self.uniswap_pool)
        
        # Legacy interface properties for backward compatibility
        self._update_legacy_properties()
        
        # Store configuration
        self.concentration = concentration
        self.token0_ratio = token0_ratio
        self.total_pool_size = total_pool_size
    
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
            print(f"🚫 DEBUG: execute_yield_token_sale called with invalid amount: {yield_token_value}")
            return 0.0
            
        # Execute actual Uniswap V3 swap: Yield Token -> MOET
        # Convert USD to scaled amount for Uniswap V3 math
        amount_in_scaled = int(yield_token_value * 1e6)
        
        # Execute swap
        
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
            
            # Check if swap was successful (got meaningful output)
            if moet_received <= 0:
                print(f"🚨 Yield token sale failed: Pool liquidity exhausted")
                print(f"   Requested: ${yield_token_value:,.2f} YT -> MOET")
                print(f"   Received: ${moet_received:,.2f} MOET (insufficient liquidity)")
                return 0.0
            
            # Swap completed successfully
            
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
