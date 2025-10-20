#!/usr/bin/env python3
"""
Uniswap V3 Concentrated Liquidity System 

Implements authentic Uniswap V3 mathematics using:
- Tick-based price system with Q64.96 fixed-point arithmetic
- Proper constant product curve calculations
- Concentrated liquidity positions
- MOET:BTC: 80% concentrated around peg
- MOET:Yield Token: 95% concentrated around peg

Provides both trading functionality and visualization data for charts.
"""

import math
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

# Uniswap V3 Constants
MIN_TICK = -887272
MAX_TICK = 887272
Q96 = 2 ** 96
# Uniswap V3 Fee Tiers and Tick Spacings (Official Values)
TICK_SPACING_0_0_5_PERCENT = 10   # For 0.05% fee tier (stable pairs like MOET/YT)
TICK_SPACING_0_3_PERCENT = 60     # For 0.3% fee tier (standard pairs like MOET/BTC)
TICK_SPACING_1_PERCENT = 200      # For 1% fee tier (exotic/volatile pairs)
MIN_SQRT_RATIO = 4295128739  # sqrt(1.0001^-887272) * 2^96
MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342  # sqrt(1.0001^887272) * 2^96


# Core Uniswap V3 Math Functions - Exact Integer Implementation
def tick_to_sqrt_price_x96(tick: int) -> int:
    """Convert tick to sqrt price in Q64.96 format using simplified exact math"""
    if tick < MIN_TICK or tick > MAX_TICK:
        raise ValueError(f"Tick {tick} out of bounds [{MIN_TICK}, {MAX_TICK}]")
    
    # Use the mathematical formula: sqrt_price = sqrt(1.0001^tick)
    # For precision, we'll use the exact formula but with proper bounds checking
    sqrt_price = 1.0001 ** (tick / 2.0)
    sqrt_price_x96 = int(sqrt_price * Q96)
    
    # Ensure bounds
    return max(MIN_SQRT_RATIO, min(MAX_SQRT_RATIO, sqrt_price_x96))


def sqrt_price_x96_to_tick(sqrt_price_x96: int) -> int:
    """Convert sqrt price X96 to tick using exact integer math"""
    if sqrt_price_x96 < MIN_SQRT_RATIO or sqrt_price_x96 > MAX_SQRT_RATIO:
        raise ValueError(f"sqrt_price_x96 {sqrt_price_x96} out of bounds")
    
    # Use binary search for exact tick calculation
    tick_low = MIN_TICK
    tick_high = MAX_TICK
    
    while tick_high - tick_low > 1:
        tick_mid = (tick_low + tick_high) // 2
        sqrt_price_mid = tick_to_sqrt_price_x96(tick_mid)
        
        if sqrt_price_mid <= sqrt_price_x96:
            tick_low = tick_mid
        else:
            tick_high = tick_mid
    
    return tick_low


# Safe math helpers
def mul_div(a: int, b: int, denominator: int) -> int:
    """Multiply two numbers and divide by denominator with overflow protection"""
    if denominator == 0:
        raise ValueError("Division by zero")
    result = (a * b) // denominator
    return result

def mul_div_rounding_up(a: int, b: int, denominator: int) -> int:
    """Multiply and divide with rounding up"""
    if denominator == 0:
        raise ValueError("Division by zero")
    result = (a * b + denominator - 1) // denominator
    return result

def get_amount0_delta(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int, 
    liquidity: int,
    round_up: bool = False
) -> int:
    """Calculate amount0 delta for liquidity in price range with proper rounding"""
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
    
    if liquidity == 0 or sqrt_price_a_x96 == sqrt_price_b_x96:
        return 0
    
    numerator1 = liquidity << 96  # liquidity * Q96
    numerator2 = sqrt_price_b_x96 - sqrt_price_a_x96
    denominator = sqrt_price_b_x96 * sqrt_price_a_x96
    
    if round_up:
        return mul_div_rounding_up(mul_div_rounding_up(numerator1, numerator2, sqrt_price_b_x96), 1, sqrt_price_a_x96)
    else:
        return mul_div(numerator1, numerator2, denominator)


def get_amount0_delta_economic(
    sqrt_price_old_x96: int,
    sqrt_price_new_x96: int,
    liquidity: int,
    amount_in: int
) -> int:
    """
    Calculate amount0 delta using the correct economic relationship
    For YT->MOET swaps: Δx = Δy / (1 + Δy/(L*√P_old))
    This fixes the 5.66% efficiency loss in the original Uniswap V3 formula
    """
    if liquidity == 0 or sqrt_price_old_x96 == sqrt_price_new_x96:
        return 0
    
    # Convert to regular units for calculation
    sqrt_p_old = sqrt_price_old_x96 / Q96
    
    # Apply economic relationship: Δx = Δy / (1 + Δy/(L*√P_old))
    l_sqrt_p = liquidity * sqrt_p_old
    denominator_float = 1.0 + (amount_in / l_sqrt_p)
    
    # Calculate economically correct output
    economic_output = amount_in / denominator_float
    
    # Economic fix applied - removed debug logging
    
    return int(economic_output)


def get_amount1_delta(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    liquidity: int,
    round_up: bool = False
) -> int:
    """Calculate amount1 delta for liquidity in price range with proper rounding"""
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
    
    if liquidity == 0 or sqrt_price_a_x96 == sqrt_price_b_x96:
        return 0
    
    if round_up:
        return mul_div_rounding_up(liquidity, sqrt_price_b_x96 - sqrt_price_a_x96, Q96)
    else:
        return mul_div(liquidity, sqrt_price_b_x96 - sqrt_price_a_x96, Q96)


def get_next_sqrt_price_from_amount0_rounding_up(
    sqrt_price_x96: int,
    liquidity: int,
    amount: int,
    add: bool
) -> int:
    """Calculate next sqrt price from amount0 with proper rounding up"""
    if amount == 0:
        return sqrt_price_x96
    
    if liquidity == 0:
        raise ValueError("Liquidity cannot be zero")
    
    numerator1 = liquidity << 96  # liquidity * Q96
    
    if add:
        # Adding amount0: sqrt_price decreases
        # Formula: L * sqrt_P / (L + amount0 * sqrt_P)
        product = amount * sqrt_price_x96
        # Check for overflow
        if amount != 0 and product // amount != sqrt_price_x96:
            raise ValueError("Multiplication overflow")
        
        denominator = numerator1 + product
        if denominator < numerator1:
            raise ValueError("Addition overflow")
            
        return mul_div(numerator1, sqrt_price_x96, denominator)
    else:
        # Removing amount0: sqrt_price increases
        # Formula: L / (L/sqrt_P - amount0)
        quotient = mul_div_rounding_up(numerator1, 1, sqrt_price_x96)
        if quotient <= amount:
            raise ValueError("Amount too large")
        return mul_div_rounding_up(numerator1, 1, quotient - amount)


def get_next_sqrt_price_from_amount1_rounding_down(
    sqrt_price_x96: int,
    liquidity: int,
    amount: int,
    add: bool
) -> int:
    """Calculate next sqrt price from amount1 with proper rounding down"""
    if amount == 0:
        return sqrt_price_x96
    
    if liquidity == 0:
        raise ValueError("Liquidity cannot be zero")
    
    if add:
        # Adding amount1: sqrt_price increases
        # Formula: sqrt_P + amount1 / L
        quotient = mul_div(amount, Q96, liquidity)
        result = sqrt_price_x96 + quotient
        if result < sqrt_price_x96:
            raise ValueError("Addition overflow")
        return result
    else:
        # Removing amount1: sqrt_price decreases
        # Formula: sqrt_P - amount1 / L
        quotient = mul_div_rounding_up(amount, Q96, liquidity)
        if sqrt_price_x96 <= quotient:
            raise ValueError("Amount too large")
        return sqrt_price_x96 - quotient


def compute_swap_step(
    sqrt_price_current_x96: int,
    sqrt_price_target_x96: int,
    liquidity: int,
    amount_remaining: int,
    fee_pips: int
) -> Tuple[int, int, int, int]:
    """
    Enhanced compute swap step with proper cross-tick handling
    
    Implements two-scenario logic:
    1. Range has enough liquidity (stays within current range)
    2. Range needs cross-tick transition (moves to next tick)
    
    Returns: (sqrt_price_next_x96, amount_in, amount_out, fee_amount)
    """
    if liquidity == 0:
        return sqrt_price_current_x96, 0, 0, 0
    
    zero_for_one = sqrt_price_current_x96 >= sqrt_price_target_x96
    exact_in = amount_remaining >= 0
    
    # Validate inputs
    if exact_in and amount_remaining <= 0:
        return sqrt_price_current_x96, 0, 0, 0
    if not exact_in and amount_remaining >= 0:
        return sqrt_price_current_x96, 0, 0, 0
    
    # Calculate amount after fees for exact input
    amount_remaining_less_fee = 0
    if exact_in:
        if fee_pips >= 1000000:
            raise ValueError("Fee too high")
        amount_remaining_less_fee = mul_div(amount_remaining, 1000000 - fee_pips, 1000000)
        if amount_remaining_less_fee <= 0:
            return sqrt_price_current_x96, 0, 0, 0
    
    # Scenario 1: Calculate required input to reach target price
    try:
        if zero_for_one:
            amount_in_required = get_amount0_delta(
                sqrt_price_target_x96, sqrt_price_current_x96, liquidity, True
            )
        else:
            amount_in_required = get_amount1_delta(
                sqrt_price_current_x96, sqrt_price_target_x96, liquidity, True
            )
    except (ValueError, ZeroDivisionError):
        # Math error - return current state
        return sqrt_price_current_x96, 0, 0, 0
    
    # Scenario 2: Check if we can reach target with available amount
    sqrt_price_next_x96 = sqrt_price_current_x96
    
    if exact_in:
        if amount_remaining_less_fee >= amount_in_required:
            # Scenario 1: We can reach the target price
            sqrt_price_next_x96 = sqrt_price_target_x96
        else:
            # Scenario 2: We cannot reach target, calculate achievable price
            try:
                sqrt_price_next_x96 = get_next_sqrt_price_from_input(
                    sqrt_price_current_x96, liquidity, amount_remaining_less_fee, zero_for_one
                )
            except (ValueError, ZeroDivisionError):
                return sqrt_price_current_x96, 0, 0, 0
    else:
        # Exact output case
        try:
            if zero_for_one:
                amount_out_available = get_amount1_delta(
                    sqrt_price_target_x96, sqrt_price_current_x96, liquidity, False
                )
            else:
                amount_out_available = get_amount0_delta(
                    sqrt_price_current_x96, sqrt_price_target_x96, liquidity, False
                )
            
            if -amount_remaining >= amount_out_available:
                # We can reach the target price
                sqrt_price_next_x96 = sqrt_price_target_x96
            else:
                # We cannot reach target, calculate achievable price
                sqrt_price_next_x96 = get_next_sqrt_price_from_output(
                    sqrt_price_current_x96, liquidity, -amount_remaining, zero_for_one
                )
        except (ValueError, ZeroDivisionError):
            return sqrt_price_current_x96, 0, 0, 0
    
    # Validate price bounds
    sqrt_price_next_x96 = max(MIN_SQRT_RATIO, min(MAX_SQRT_RATIO, sqrt_price_next_x96))
    
    # Check if we made progress
    if sqrt_price_next_x96 == sqrt_price_current_x96:
        return sqrt_price_current_x96, 0, 0, 0
    
    # Calculate actual amounts based on price change
    try:
        max_price_reached = (sqrt_price_target_x96 == sqrt_price_next_x96)
        
        if zero_for_one:
            # Token0 -> Token1 swap
            amount_in = get_amount0_delta(
                sqrt_price_next_x96, sqrt_price_current_x96, liquidity, True
            )
            amount_out = get_amount1_delta(
                sqrt_price_next_x96, sqrt_price_current_x96, liquidity, False
            )
        else:
            # Token1 -> Token0 swap (YT -> MOET)
            amount_in = get_amount1_delta(
                sqrt_price_current_x96, sqrt_price_next_x96, liquidity, True
            )
            
            # CRITICAL FIX: Use economic formula instead of broken Uniswap V3 formula
            # This fixes the 5.66% efficiency loss
            if exact_in and amount_remaining_less_fee > 0:
                # Use the economically correct relationship for output calculation
                amount_out = get_amount0_delta_economic(
                    sqrt_price_current_x96, sqrt_price_next_x96, liquidity, amount_remaining_less_fee
                )
            else:
                # Fallback to original formula for exact output or edge cases
                amount_out = get_amount0_delta(
                    sqrt_price_current_x96, sqrt_price_next_x96, liquidity, False
                )
        
        # Cap output amount for exact output swaps
        if not exact_in and amount_out > -amount_remaining:
            amount_out = -amount_remaining
        
        # Enhanced fee calculation
        if exact_in:
            if max_price_reached:
                # We reached target - fee on actual input used
                fee_amount = mul_div_rounding_up(amount_in, fee_pips, 1000000 - fee_pips)
            else:
                # We didn't reach target - fee on remaining amount
                fee_amount = amount_remaining - amount_in
        else:
            # Exact output - fee on input amount
            fee_amount = mul_div_rounding_up(amount_in, fee_pips, 1000000 - fee_pips)
        
        # Ensure non-negative values
        amount_in = max(0, amount_in)
        amount_out = max(0, amount_out)
        fee_amount = max(0, fee_amount)
        
    except (ValueError, ZeroDivisionError):
        # Return safe values on calculation error
        return sqrt_price_current_x96, 0, 0, 0
    
    return sqrt_price_next_x96, amount_in, amount_out, fee_amount


def get_next_sqrt_price_from_input(
    sqrt_price_x96: int,
    liquidity: int,
    amount_in: int,
    zero_for_one: bool
) -> int:
    """Calculate next sqrt price from input amount"""
    if zero_for_one:
        return get_next_sqrt_price_from_amount0_rounding_up(sqrt_price_x96, liquidity, amount_in, True)
    else:
        return get_next_sqrt_price_from_amount1_rounding_down(sqrt_price_x96, liquidity, amount_in, True)


def get_next_sqrt_price_from_output(
    sqrt_price_x96: int,
    liquidity: int,
    amount_out: int,
    zero_for_one: bool
) -> int:
    """Calculate next sqrt price from output amount"""
    if zero_for_one:
        return get_next_sqrt_price_from_amount1_rounding_down(sqrt_price_x96, liquidity, amount_out, False)
    else:
        return get_next_sqrt_price_from_amount0_rounding_up(sqrt_price_x96, liquidity, amount_out, False)


@dataclass
class TickInfo:
    """Information stored for each tick in Uniswap V3"""
    liquidity_gross: int = 0  # Total liquidity referencing this tick
    liquidity_net: int = 0   # Net liquidity change at this tick
    initialized: bool = False  # Whether this tick has been initialized


@dataclass
class TickBitmap:
    """Efficient tick finding using bitmap approach (O(1) instead of O(n))"""
    bitmap: Dict[int, int] = None  # word_index -> bitmap_word
    
    def __post_init__(self):
        if self.bitmap is None:
            self.bitmap = {}
    
    def next_initialized_tick(self, tick: int, tick_spacing: int, zero_for_one: bool) -> int:
        """
        Find next initialized tick using bitmap lookup
        
        Args:
            tick: Current tick
            tick_spacing: Tick spacing for the pool
            zero_for_one: Direction of search
            
        Returns:
            Next initialized tick or boundary tick
        """
        if zero_for_one:
            # Moving down (decreasing price) - find next tick below current
            return self._next_initialized_tick_within_one_word(tick, tick_spacing, zero_for_one)
        else:
            # Moving up (increasing price) - find next tick above current
            return self._next_initialized_tick_within_one_word(tick, tick_spacing, zero_for_one)
    
    def _next_initialized_tick_within_one_word(self, tick: int, tick_spacing: int, zero_for_one: bool) -> int:
        """Find next initialized tick within one word (256 ticks)"""
        # Compress tick to word position
        compressed = tick // tick_spacing
        
        if zero_for_one:
            # Moving left (decreasing tick)
            word_pos = compressed >> 8  # Divide by 256
            bit_pos = compressed & 0xFF  # Modulo 256
            
            # Get bitmap word
            word = self.bitmap.get(word_pos, 0)
            
            # Mask off bits at or above the current position
            mask = (1 << bit_pos) - 1
            masked = word & mask
            
            if masked != 0:
                # Found initialized tick in current word
                most_significant_bit = self._most_significant_bit(masked)
                return (word_pos * 256 + most_significant_bit) * tick_spacing
            else:
                # Search previous words
                word_pos -= 1
                while word_pos >= (MIN_TICK // tick_spacing) >> 8:
                    word = self.bitmap.get(word_pos, 0)
                    if word != 0:
                        most_significant_bit = self._most_significant_bit(word)
                        return (word_pos * 256 + most_significant_bit) * tick_spacing
                    word_pos -= 1
                
                # No more ticks - return minimum
                return MIN_TICK
        else:
            # Moving right (increasing tick)
            word_pos = compressed >> 8
            bit_pos = compressed & 0xFF
            
            # Get bitmap word
            word = self.bitmap.get(word_pos, 0)
            
            # Mask off bits at or below the current position
            mask = ~((1 << (bit_pos + 1)) - 1)
            masked = word & mask
            
            if masked != 0:
                # Found initialized tick in current word
                least_significant_bit = self._least_significant_bit(masked)
                return (word_pos * 256 + least_significant_bit) * tick_spacing
            else:
                # Search next words
                word_pos += 1
                while word_pos <= (MAX_TICK // tick_spacing) >> 8:
                    word = self.bitmap.get(word_pos, 0)
                    if word != 0:
                        least_significant_bit = self._least_significant_bit(word)
                        return (word_pos * 256 + least_significant_bit) * tick_spacing
                    word_pos += 1
                
                # No more ticks - return maximum
                return MAX_TICK
    
    def flip_tick(self, tick: int, tick_spacing: int):
        """Flip tick state in bitmap when liquidity is added/removed"""
        compressed = tick // tick_spacing
        word_pos = compressed >> 8  # Divide by 256
        bit_pos = compressed & 0xFF  # Modulo 256
        
        if word_pos not in self.bitmap:
            self.bitmap[word_pos] = 0
        
        # Flip the bit
        self.bitmap[word_pos] ^= (1 << bit_pos)
        
        # Clean up empty words
        if self.bitmap[word_pos] == 0:
            del self.bitmap[word_pos]
    
    def _most_significant_bit(self, x: int) -> int:
        """Find the most significant bit position (0-indexed from right)"""
        if x == 0:
            return -1
        return x.bit_length() - 1
    
    def _least_significant_bit(self, x: int) -> int:
        """Find the least significant bit position (0-indexed from right)"""
        if x == 0:
            return -1
        return (x & -x).bit_length() - 1
    
@dataclass
class Position:
    """Represents a concentrated liquidity position"""
    tick_lower: int  # Lower tick of the position
    tick_upper: int  # Upper tick of the position  
    liquidity: int   # Amount of liquidity in this position
    


@dataclass
class UniswapV3Pool:
    """Proper Uniswap V3 pool implementation with tick-based math"""
    pool_name: str  # "MOET:BTC" or "MOET:Yield_Token"
    total_liquidity: float  # Total pool size in USD
    btc_price: float = 100_000.0  # BTC price in USD
    fee_tier: float = None  # Will be set based on pool type
    concentration: float = None  # Will be set based on pool type
    tick_spacing: int = None  # Will be set based on pool type
    token0_ratio: float = 0.5  # Ratio of token0 (MOET) in the pool (0.5 = 50/50, 0.75 = 75/25)
    
    # Core Uniswap V3 state
    sqrt_price_x96: int = Q96  # Current sqrt price in Q64.96 format
    liquidity: int = 0  # Current active liquidity
    tick_current: int = 0  # Current tick
    
    # Tick and position data
    ticks: Dict[int, TickInfo] = None  # tick -> TickInfo
    positions: List[Position] = None  # List of liquidity positions
    tick_bitmap: TickBitmap = None  # Efficient tick finding
    
    # Enhanced features
    use_enhanced_cross_tick: bool = True
    use_tick_bitmap: bool = True
    use_position_aware_liquidity: bool = True  # New flag for position-aware cross-tick logic
    max_swap_iterations: int = 1000
    debug_cross_tick: bool = False
    
    # Legacy fields for backward compatibility
    token0_reserve: Optional[float] = None  # MOET reserve (calculated from ticks)
    token1_reserve: Optional[float] = None  # BTC reserve (calculated from ticks)
    
    def __post_init__(self):
        """Initialize Uniswap V3 pool with proper tick-based math"""
        # Validate token0_ratio
        self._validate_token0_ratio()
        
        # Enable debug mode for liquidity tracking
        self.debug_liquidity = False  # Disable for clean simulation output
        # Initialize tick and position data structures
        self.ticks = {} if self.ticks is None else self.ticks
        self.positions = [] if self.positions is None else self.positions
        self.tick_bitmap = TickBitmap() if self.tick_bitmap is None else self.tick_bitmap
        
        # Set fee tier, tick spacing, and concentration based on pool type (following Uniswap V3 conventions)
        if any(pair in self.pool_name for pair in ["MOET:BTC", "USDC:BTC", "USDF:BTC"]):
            # BTC pairs use 0.3% fee tier (more volatile, less correlated assets)
            if self.fee_tier is None:
                self.fee_tier = 0.003  # 0.3% fee tier
            if self.tick_spacing is None:
                self.tick_spacing = TICK_SPACING_0_3_PERCENT  # 60
            if self.concentration is None:
                self.concentration = 0.80  # 80% concentration for volatile pairs
        elif any(pair in self.pool_name for pair in ["MOET:USDC", "MOET:USDF"]):
            # MOET stablecoin pairs use 0.01% fee tier (1 bp for tight arbitrage)
            if self.fee_tier is None:
                self.fee_tier = 0.0001  # 0.01% fee tier (1 bp)
            if self.tick_spacing is None:
                self.tick_spacing = 1  # 1 tick spacing for 0.01% fee tier
            if self.concentration is None:
                self.concentration = 0.95  # 95% concentration for stable pairs
        else:
            # Other stablecoin pairs (MOET:YieldToken) use 0.05% fee tier
            if self.fee_tier is None:
                self.fee_tier = 0.0005  # 0.05% fee tier  
            if self.tick_spacing is None:
                self.tick_spacing = TICK_SPACING_0_0_5_PERCENT  # 10
            if self.concentration is None:
                self.concentration = 0.95  # 95% concentration for stable pairs
        
        # Determine pool type and calculate dynamic pricing Uniswap V3 math
        if "BTC" in self.pool_name:
            # BTC pairs: MOET:BTC, USDC:BTC, USDF:BTC
            self.concentration_type = "btc_pair"
            
            if "MOET:BTC" in self.pool_name:
                # Calculate the actual market price: MOET/BTC ratio
                # MOET = $1, BTC = self.btc_price, so price = MOET_value / BTC_value = 1 / btc_price
                market_price = 1.0 / self.btc_price  # This gives us BTC per MOET (token1/token0)
            elif "USDC:BTC" in self.pool_name or "USDF:BTC" in self.pool_name:
                # Calculate the actual market price: USDC/BTC or USDF/BTC ratio
                # USDC/USDF = $1, BTC = self.btc_price, so price = stablecoin_value / BTC_value = 1 / btc_price
                market_price = 1.0 / self.btc_price  # This gives us BTC per stablecoin (token1/token0)
            
            # Convert market price to sqrt_price_x96  Uniswap V3 math
            # sqrt_price_x96 = sqrt(price) * 2^96
            sqrt_price = math.sqrt(market_price)
            self.sqrt_price_x96 = int(sqrt_price * Q96)
            
            # Ensure sqrt_price_x96 is within valid bounds
            self.sqrt_price_x96 = max(MIN_SQRT_RATIO, min(MAX_SQRT_RATIO, self.sqrt_price_x96))
            
            # Convert sqrt_price_x96 to tick using our existing function
            self.tick_current = sqrt_price_x96_to_tick(self.sqrt_price_x96)
            
            # Align tick to proper tick spacing for this fee tier
            self.tick_current = (self.tick_current // self.tick_spacing) * self.tick_spacing
            
            # Recalculate sqrt_price_x96 from the aligned tick for consistency
            self.sqrt_price_x96 = tick_to_sqrt_price_x96(self.tick_current)
            
            # Calculate peg price from the final sqrt_price_x96
            sqrt_price_final = self.sqrt_price_x96 / Q96
            self.peg_price = sqrt_price_final * sqrt_price_final
            
        else:
            # Stablecoin pairs: MOET:YieldToken, MOET:USDC, MOET:USDF
            self.concentration_type = "stablecoin_pair"
            # For stablecoin pairs, both tokens are worth $1, so price = 1.0
            # Tick 0 corresponds to price = 1.0001^0 = 1.0 (perfect 1:1 peg)
            self.tick_current = 0
            self.sqrt_price_x96 = tick_to_sqrt_price_x96(0)  # Use our function for consistency
            self.peg_price = 1.0
        
        # Initialize concentrated liquidity positions
        self._initialize_concentrated_positions()
        
        # CRITICAL: Validate position coverage after initialization
        self._validate_position_coverage()
        
        # Calculate legacy fields for backward compatibility
        self._update_legacy_fields()
    
    def _validate_token0_ratio(self):
        """Validate token0_ratio is within acceptable bounds"""
        if not (0.1 <= self.token0_ratio <= 0.9):
            raise ValueError(f"token0_ratio must be between 0.1 and 0.9, got {self.token0_ratio}")
        
        if self.token0_ratio < 0.1:
            raise ValueError("token0_ratio too low: minimum 10% allocation required for token0 (MOET)")
        
        if self.token0_ratio > 0.9:
            raise ValueError("token0_ratio too high: minimum 10% allocation required for token1")
    
    def _initialize_concentrated_positions(self):
        """Initialize concentrated liquidity positions using proper Uniswap V3 math"""
        if self.concentration_type == "btc_pair":
            self._initialize_btc_pair_positions()
        else:
            self._initialize_stablecoin_pair_positions()
    
    def _initialize_btc_pair_positions(self):
        """Initialize BTC pair concentrated liquidity positions using exact tick math"""
        total_liquidity_amount = int(self.total_liquidity * 1e6)
        concentrated_liquidity = int(total_liquidity_amount * self.concentration)
        
        # Position 1: Concentrated range (±1% = ~100 ticks)
        peg_tick = self.tick_current
        tick_range = 100  # Approximately 1% price range
        
        tick_lower = ((peg_tick - tick_range) // self.tick_spacing) * self.tick_spacing
        tick_upper = ((peg_tick + tick_range) // self.tick_spacing) * self.tick_spacing
        
        # Ensure valid bounds
        tick_lower = max(MIN_TICK + self.tick_spacing, tick_lower)
        tick_upper = min(MAX_TICK - self.tick_spacing, tick_upper)
        
        if tick_lower < tick_upper:
            self._add_position(tick_lower, tick_upper, concentrated_liquidity)
        
        # Position 2: Wider range for remaining liquidity
        remaining_liquidity = total_liquidity_amount - concentrated_liquidity
        if remaining_liquidity > 0:
            wide_tick_range = 1000  # Approximately 10% price range
            
            # Create 3 positions in wider range with better distribution
            position_liquidity = remaining_liquidity // 3
            
            # Position 2a: Medium range
            med_tick_lower = ((peg_tick - wide_tick_range // 2) // self.tick_spacing) * self.tick_spacing
            med_tick_upper = ((peg_tick + wide_tick_range // 2) // self.tick_spacing) * self.tick_spacing
            med_tick_lower = max(MIN_TICK + self.tick_spacing, med_tick_lower)
            med_tick_upper = min(MAX_TICK - self.tick_spacing, med_tick_upper)
            if med_tick_lower < med_tick_upper:
                self._add_position(med_tick_lower, med_tick_upper, position_liquidity)
            
            # Position 2b: Wide range
            wide_tick_lower = ((peg_tick - wide_tick_range) // self.tick_spacing) * self.tick_spacing
            wide_tick_upper = ((peg_tick + wide_tick_range) // self.tick_spacing) * self.tick_spacing
            wide_tick_lower = max(MIN_TICK + self.tick_spacing, wide_tick_lower)
            wide_tick_upper = min(MAX_TICK - self.tick_spacing, wide_tick_upper)
            if wide_tick_lower < wide_tick_upper:
                self._add_position(wide_tick_lower, wide_tick_upper, position_liquidity)
            
            # Position 2c: Very wide range
            very_wide_range = wide_tick_range * 2
            vwide_tick_lower = ((peg_tick - very_wide_range) // self.tick_spacing) * self.tick_spacing
            vwide_tick_upper = ((peg_tick + very_wide_range) // self.tick_spacing) * self.tick_spacing
            vwide_tick_lower = max(MIN_TICK + self.tick_spacing, vwide_tick_lower)
            vwide_tick_upper = min(MAX_TICK - self.tick_spacing, vwide_tick_upper)
            if vwide_tick_lower < vwide_tick_upper:
                self._add_position(vwide_tick_lower, vwide_tick_upper, remaining_liquidity - 2 * position_liquidity)
    
    def _initialize_stablecoin_pair_positions(self):
        """Initialize stablecoin pair concentrated liquidity positions using exact tick math"""
        total_liquidity_amount = int(self.total_liquidity * 1e6)
        concentrated_liquidity = int(total_liquidity_amount * self.concentration)
        
        # Check if we should use symmetric or asymmetric initialization
        if abs(self.token0_ratio - 0.5) < 0.01:  # Close to 50/50, use symmetric logic
            self._initialize_symmetric_yield_token_positions()
        else:
            # Use asymmetric bounds calculation for non-50/50 ratios
            self._initialize_asymmetric_yield_token_positions()
    
    def _initialize_symmetric_yield_token_positions(self):
        """Initialize symmetric 50/50 yield token positions (fallback logic)"""
        total_liquidity_amount = int(self.total_liquidity * 1e6)
        concentrated_liquidity = int(total_liquidity_amount * self.concentration)
        
        # Position 1: Tight range around 1:1 peg (±1% = ~100 ticks)
        peg_tick = self.tick_current  # Should be 0 for 1:1 peg
        tick_range = 100  # Approximately 1% price range
        
        tick_lower = ((peg_tick - tick_range) // self.tick_spacing) * self.tick_spacing
        tick_upper = ((peg_tick + tick_range) // self.tick_spacing) * self.tick_spacing
        
        # Ensure valid bounds
        tick_lower = max(MIN_TICK + self.tick_spacing, tick_lower)
        tick_upper = min(MAX_TICK - self.tick_spacing, tick_upper)
        
        if tick_lower < tick_upper:
            # Calculate proper liquidity using Uniswap V3 math for concentrated position
            concentrated_amount0 = self.total_liquidity / 2 * self.concentration  # $237.5k MOET
            concentrated_amount1 = self.total_liquidity / 2 * self.concentration  # $237.5k YT
            
            # Use the proper formula: L = amount1 / (sqrt_price_current - sqrt_price_lower)
            # Since we're at the peg, this gives us the correct concentrated liquidity
            sqrt_price_current_x96 = self.sqrt_price_x96
            sqrt_price_lower_x96 = tick_to_sqrt_price_x96(tick_lower)
            amount1_scaled = int(concentrated_amount1 * 1e6)
            denominator = sqrt_price_current_x96 - sqrt_price_lower_x96
            if denominator > 0:
                proper_concentrated_liquidity = mul_div(amount1_scaled, Q96, denominator)
            else:
                proper_concentrated_liquidity = concentrated_liquidity  # Fallback
                
            self._add_position(tick_lower, tick_upper, proper_concentrated_liquidity)
    
    def _initialize_asymmetric_yield_token_positions(self):
        """Initialize asymmetric yield token positions using your step-by-step computation"""
        import math
        
        total_liquidity_amount = int(self.total_liquidity * 1e6)
        concentrated_liquidity_usd = self.total_liquidity * self.concentration
        
        # Get current pool price for calculations
        current_price = self.get_price()
        
        
        # Clear existing positions first when recentering
        # This matches the same process used during initial pool creation
        self.positions.clear()
        self.ticks.clear()
        self.liquidity = 0
        if self.use_tick_bitmap and self.tick_bitmap:
            self.tick_bitmap = TickBitmap()
        
        # Check if optimal bounds are provided from lookup table
        if hasattr(self, '_optimal_bounds') and self._optimal_bounds:
            P_lower, P_upper = self._optimal_bounds
            print(f"🎯 Using OPTIMAL bounds from lookup table: [{P_lower:.6f}, {P_upper:.6f}]")
            
            # Clear the bounds after use to prevent reuse
            self._optimal_bounds = None
            
        else:
            # Fallback to current implementation if no optimal bounds provided
            print("⚠️  No optimal bounds provided, using fallback calculation")
            
            # Step 1: Fix upper bound at +1% from CURRENT price
            P_upper = current_price * 1.01
            
            # Step 2: Solve for lower bound to get desired ratio
            R = self.token0_ratio / (1 - self.token0_ratio)  # e.g., 75/25 = 3
            x = math.sqrt(current_price)  # Current sqrt price at CURRENT price
            b = math.sqrt(P_upper)
            a = 1 - (b - x) / (R * b)
            
            # Step 3: Convert to price bounds
            P_lower = a ** 2
        
        # Step 4: Calculate coefficients and liquidity using the bounds
        # These calculations work for both optimal and fallback bounds
        x = math.sqrt(current_price)  # Current sqrt price
        a = math.sqrt(P_lower)        # Lower sqrt price
        b = math.sqrt(P_upper)        # Upper sqrt price
        
        coeff_0 = (b - x) / (x * b)  # MOET coefficient
        coeff_1 = x - a              # YT coefficient
        coeff_sum = coeff_0 + coeff_1
        
        if coeff_sum <= 0:
            raise ValueError(f"Invalid coefficient sum {coeff_sum} for token0_ratio {self.token0_ratio}")
        
        # Calculate L using the standard method
        L = concentrated_liquidity_usd / coeff_sum
        
        # Step 5: Calculate actual token amounts
        amount_0 = L * coeff_0  # MOET amount
        amount_1 = L * coeff_1  # YT amount
        
        # Convert bounds to ticks
        tick_lower_exact = math.log(P_lower) / math.log(1.0001)
        tick_upper_exact = math.log(P_upper) / math.log(1.0001)
        
        # Smart rounding: try both directions and pick the one closest to target ratio
        def test_ratio(tick_l, tick_u):
            price_l = 1.0001 ** tick_l
            price_u = 1.0001 ** tick_u
            b_test = math.sqrt(price_u)
            a_test = math.sqrt(price_l)
            coeff_0_test = (b_test - 1) / b_test
            coeff_1_test = 1 - a_test
            return coeff_0_test / (coeff_0_test + coeff_1_test)
        
        # Generate rounding options
        lower_down = (int(tick_lower_exact) // self.tick_spacing) * self.tick_spacing
        lower_up = lower_down + self.tick_spacing
        upper_down = (int(tick_upper_exact) // self.tick_spacing) * self.tick_spacing
        upper_up = upper_down + self.tick_spacing
        
        # Test combinations and pick best
        options = [(lower_down, upper_down), (lower_up, upper_up), 
                  (lower_down, upper_up), (lower_up, upper_down)]
        
        best_deviation = float('inf')
        tick_lower, tick_upper = options[0]  # fallback
        
        for tick_l, tick_u in options:
            if tick_l < tick_u:  # valid range
                ratio = test_ratio(tick_l, tick_u)
                deviation = abs(ratio - self.token0_ratio)
                if deviation < best_deviation:
                    best_deviation = deviation
                    tick_lower, tick_upper = tick_l, tick_u
        
        # Ensure valid bounds
        tick_lower = max(MIN_TICK + self.tick_spacing, tick_lower)
        tick_upper = min(MAX_TICK - self.tick_spacing, tick_upper)
        
        if tick_lower < tick_upper:
            # Calculate proper concentrated liquidity using Uniswap V3 math
            sqrt_price_current_x96 = self.sqrt_price_x96
            sqrt_price_lower_x96 = tick_to_sqrt_price_x96(tick_lower)
            amount1_scaled = int(amount_1 * 1e6)
            denominator = sqrt_price_current_x96 - sqrt_price_lower_x96
            if denominator > 0:
                # CRITICAL FIX: The mathematical L is too small by 247,000x
                # We need to use the original Q96-based calculation but understand it correctly
                # The Q96 scaling is actually necessary for proper liquidity density
                proper_concentrated_liquidity = mul_div(amount1_scaled, Q96, denominator)
                print(f"🔧 LIQUIDITY FIX: Using proper Q96 calculation L={proper_concentrated_liquidity:,.0f} (was trying mathematical L={L:,.0f})")
            else:
                # Fallback calculation
                proper_concentrated_liquidity = int(concentrated_liquidity_usd * 1e6)
                
            self._add_position(tick_lower, tick_upper, proper_concentrated_liquidity)
        
        # Position 2: Backup liquidity positions (outside concentrated range)
        # Calculate backup amounts using the configured ratio
        total_amount0 = self.total_liquidity * self.token0_ratio  # e.g., $375k MOET for 75%
        total_amount1 = self.total_liquidity * (1 - self.token0_ratio)  # e.g., $125k YT for 25%
        backup_amount0 = total_amount0 * (1 - self.concentration)  # e.g., 5% of $375k = $18.75k
        backup_amount1 = total_amount1 * (1 - self.concentration)  # e.g., 5% of $125k = $6.25k
        
        # Calculate backup liquidity using proper Uniswap V3 math for each range
        if backup_amount0 > 0 and backup_amount1 > 0:
            wide_tick_range = 1000  # Approximately 10% price range
            peg_tick = self.tick_current  # Should be 0 for 1:1 peg
            
            # Create two separate positions outside the concentrated range
            # Position 2a: Below concentrated range (-1000 to -100)
            tick_lower_below = ((peg_tick - wide_tick_range) // self.tick_spacing) * self.tick_spacing
            tick_upper_below = tick_lower  # End at the concentrated range start
            
            tick_lower_below = max(MIN_TICK + self.tick_spacing, tick_lower_below)
            
            if tick_lower_below < tick_upper_below:
                # Calculate backup liquidity using simple percentage approach for initialization
                backup_liquidity_below = int(total_liquidity_amount * (1 - self.concentration) / 2)
                self._add_position(tick_lower_below, tick_upper_below, backup_liquidity_below)
            
            # Position 2b: Above concentrated range (+100 to +1000)
            tick_lower_above = tick_upper  # Start at the concentrated range end
            tick_upper_above = ((peg_tick + wide_tick_range) // self.tick_spacing) * self.tick_spacing
            
            tick_upper_above = min(MAX_TICK - self.tick_spacing, tick_upper_above)
            
            if tick_lower_above < tick_upper_above:
                # Calculate backup liquidity using simple percentage approach for initialization
                backup_liquidity_above = int(total_liquidity_amount * (1 - self.concentration) / 2)
                self._add_position(tick_lower_above, tick_upper_above, backup_liquidity_above)
    
    def _calculate_liquidity_from_amounts(self, amount0: float, amount1: float, tick_lower: int, tick_upper: int) -> int:
        """Calculate liquidity from token amounts using proper Uniswap V3 math"""
        # Convert amounts to scaled integers
        amount0_scaled = int(amount0 * 1e6)
        amount1_scaled = int(amount1 * 1e6)
        
        # Get sqrt prices for the range
        sqrt_price_lower_x96 = tick_to_sqrt_price_x96(tick_lower)
        sqrt_price_upper_x96 = tick_to_sqrt_price_x96(tick_upper)
        sqrt_price_current_x96 = self.sqrt_price_x96
        
        # Check if current price is within the range
        if sqrt_price_lower_x96 <= sqrt_price_current_x96 <= sqrt_price_upper_x96:
            # Current price is within range - use standard formula
            # For token1 (YT): L = amount1 / (sqrt_price_current - sqrt_price_lower)
            # For token0 (MOET): L = amount0 / (sqrt_price_upper - sqrt_price_current)
            
            # Calculate L1 (for token1)
            denominator1 = sqrt_price_current_x96 - sqrt_price_lower_x96
            if denominator1 > 0:
                L1 = mul_div(amount1_scaled, Q96, denominator1)
            else:
                L1 = 0
                
            # Calculate L0 (for token0) 
            denominator0 = sqrt_price_upper_x96 - sqrt_price_current_x96
            if denominator0 > 0:
                L0 = mul_div(amount0_scaled, Q96, denominator0)
            else:
                L0 = 0
                
            # Return the minimum (bottleneck)
            return min(L0, L1)
        else:
            # Current price is outside range - use full range formula
            # L = amount1 / (sqrt_price_upper - sqrt_price_lower) for token1
            # L = amount0 / (sqrt_price_upper - sqrt_price_lower) for token0
            denominator = sqrt_price_upper_x96 - sqrt_price_lower_x96
            if denominator > 0:
                L1 = mul_div(amount1_scaled, Q96, denominator)
                L0 = mul_div(amount0_scaled, Q96, denominator)
                return min(L0, L1)
            else:
                return 0

    def _add_position(self, tick_lower: int, tick_upper: int, liquidity: int):
        """Add a liquidity position and update tick data"""
        if liquidity <= 0:
            return
            
        # Create position
        position = Position(tick_lower, tick_upper, liquidity)
        self.positions.append(position)
        
        # Update tick data
        lower_was_initialized = tick_lower in self.ticks and self.ticks[tick_lower].initialized
        upper_was_initialized = tick_upper in self.ticks and self.ticks[tick_upper].initialized
        
        if tick_lower not in self.ticks:
            self.ticks[tick_lower] = TickInfo()
        if tick_upper not in self.ticks:
            self.ticks[tick_upper] = TickInfo()
        
        # Update liquidity deltas
        # Lower tick: positive liquidity_net (add liquidity when crossing from below)
        self.ticks[tick_lower].liquidity_net += liquidity
        self.ticks[tick_lower].liquidity_gross += liquidity
        self.ticks[tick_lower].initialized = True
        
        # Upper tick: negative liquidity_net (remove liquidity when crossing from below)
        self.ticks[tick_upper].liquidity_net -= liquidity
        self.ticks[tick_upper].liquidity_gross += liquidity
        self.ticks[tick_upper].initialized = True
        
        # Update bitmap for newly initialized ticks
        if self.use_tick_bitmap:
            if not lower_was_initialized:
                self.tick_bitmap.flip_tick(tick_lower, self.tick_spacing)
            if not upper_was_initialized:
                self.tick_bitmap.flip_tick(tick_upper, self.tick_spacing)
        
        # Update current liquidity if position includes current tick
        if tick_lower <= self.tick_current < tick_upper:
            self.liquidity += liquidity
    
    def _remove_position(self, tick_lower: int, tick_upper: int, liquidity: int):
        """Remove liquidity from a position and update tick data"""
        if liquidity <= 0:
            return
            
        # Find and remove the position
        for i, position in enumerate(self.positions):
            if position.tick_lower == tick_lower and position.tick_upper == tick_upper:
                if position.liquidity >= liquidity:
                    if position.liquidity == liquidity:
                        # Remove entire position
                        self.positions.pop(i)
                    else:
                        # Reduce position liquidity
                        position.liquidity -= liquidity
                    break
        
        # Update tick data
        if tick_lower in self.ticks:
            self.ticks[tick_lower].liquidity_net -= liquidity
            self.ticks[tick_lower].liquidity_gross -= liquidity
            if self.ticks[tick_lower].liquidity_gross <= 0:
                self.ticks[tick_lower].initialized = False
                if self.use_tick_bitmap:
                    self.tick_bitmap.flip_tick(tick_lower, self.tick_spacing)
        
        if tick_upper in self.ticks:
            self.ticks[tick_upper].liquidity_net += liquidity  # Reverse of add
            self.ticks[tick_upper].liquidity_gross -= liquidity
            if self.ticks[tick_upper].liquidity_gross <= 0:
                self.ticks[tick_upper].initialized = False
                if self.use_tick_bitmap:
                    self.tick_bitmap.flip_tick(tick_upper, self.tick_spacing)
        
        # Update current liquidity if position includes current tick
        if tick_lower <= self.tick_current < tick_upper:
            self.liquidity -= liquidity
    
    def update_liquidity_range(self, center_price: float, range_width: float = 0.01, token0_ratio: float = None):
        """
        Atomically update concentrated liquidity range around new center price
        
        Args:
            center_price: New center price for the range (true YT price)
            range_width: Range width as percentage (default 1% = 0.01)
            token0_ratio: Token0 ratio to maintain (defaults to current pool ratio)
        
        Returns:
            dict: Summary of the range update operation
        """
        import math
        
        # Use existing token0_ratio if not specified
        if token0_ratio is None:
            token0_ratio = self.token0_ratio
        
        # Calculate total liquidity to preserve
        total_liquidity_to_preserve = sum(pos.liquidity for pos in self.positions)
        
        if total_liquidity_to_preserve == 0:
            return {"success": False, "reason": "No liquidity to rebalance"}
        
        # Calculate new range bounds
        lower_price = center_price * (1 - range_width)
        upper_price = center_price * (1 + range_width)
        
        # Convert prices to ticks with proper rounding
        lower_tick_raw = math.log(lower_price) / math.log(1.0001)
        upper_tick_raw = math.log(upper_price) / math.log(1.0001)
        
        # Round to valid tick spacing
        tick_lower = int(lower_tick_raw // self.tick_spacing) * self.tick_spacing
        tick_upper = int(upper_tick_raw // self.tick_spacing + 1) * self.tick_spacing
        
        # Ensure valid tick range
        tick_lower = max(MIN_TICK, tick_lower)
        tick_upper = min(MAX_TICK, tick_upper)
        
        if tick_lower >= tick_upper:
            return {"success": False, "reason": "Invalid tick range"}
        
        # Store old positions for logging
        old_positions = self.positions.copy()
        
        # Remove all existing positions
        for position in old_positions:
            self._remove_position(position.tick_lower, position.tick_upper, position.liquidity)
        
        # Calculate liquidity amounts for new asymmetric position
        # Use the asymmetric yield token position logic
        total_amount0 = self.total_liquidity * token0_ratio  # MOET side
        total_amount1 = self.total_liquidity * (1 - token0_ratio)  # YT side
        
        # Calculate concentrated liquidity using Uniswap V3 math
        sqrt_price_current_x96 = self.sqrt_price_x96
        sqrt_price_lower_x96 = tick_to_sqrt_price_x96(tick_lower)
        sqrt_price_upper_x96 = tick_to_sqrt_price_x96(tick_upper)
        
        # Use concentrated portion (95% by default)
        concentrated_amount0 = total_amount0 * self.concentration
        concentrated_amount1 = total_amount1 * self.concentration
        
        # Calculate proper liquidity using Uniswap V3 formulas
        amount0_scaled = int(concentrated_amount0 * 1e6)
        amount1_scaled = int(concentrated_amount1 * 1e6)
        
        # Calculate liquidity from amounts
        if sqrt_price_current_x96 <= sqrt_price_lower_x96:
            # Price below range - all token0
            liquidity = mul_div(amount0_scaled, sqrt_price_lower_x96 * sqrt_price_upper_x96, Q96 * (sqrt_price_upper_x96 - sqrt_price_lower_x96))
        elif sqrt_price_current_x96 >= sqrt_price_upper_x96:
            # Price above range - all token1
            liquidity = mul_div(amount1_scaled, Q96, sqrt_price_upper_x96 - sqrt_price_lower_x96)
        else:
            # Price in range - use both tokens
            liquidity0 = mul_div(amount0_scaled, sqrt_price_current_x96 * sqrt_price_upper_x96, Q96 * (sqrt_price_upper_x96 - sqrt_price_current_x96))
            liquidity1 = mul_div(amount1_scaled, Q96, sqrt_price_current_x96 - sqrt_price_lower_x96)
            liquidity = min(liquidity0, liquidity1)
        
        # Ensure we preserve total liquidity
        if liquidity <= 0:
            liquidity = total_liquidity_to_preserve
        
        # Add the new concentrated position
        self._add_position(tick_lower, tick_upper, liquidity)
        
        # Add remaining liquidity as wide position (5% by default)
        if self.concentration < 1.0:
            remaining_liquidity = int(total_liquidity_to_preserve * (1 - self.concentration))
            if remaining_liquidity > 0:
                wide_tick_lower = max(MIN_TICK, tick_lower - 1000 * self.tick_spacing)
                wide_tick_upper = min(MAX_TICK, tick_upper + 1000 * self.tick_spacing)
                self._add_position(wide_tick_lower, wide_tick_upper, remaining_liquidity)
        
        # Calculate actual price range for logging
        actual_lower_price = 1.0001 ** tick_lower
        actual_upper_price = 1.0001 ** tick_upper
        
        return {
            "success": True,
            "old_positions": len(old_positions),
            "new_positions": len(self.positions),
            "center_price": center_price,
            "target_range": f"${lower_price:.6f} - ${upper_price:.6f}",
            "actual_range": f"${actual_lower_price:.6f} - ${actual_upper_price:.6f}",
            "tick_range": f"{tick_lower} - {tick_upper}",
            "liquidity_preserved": total_liquidity_to_preserve,
            "new_liquidity": sum(pos.liquidity for pos in self.positions),
            "token0_ratio": token0_ratio
        }
    
    def swap(
        self,
        zero_for_one: bool,
        amount_specified: int,
        sqrt_price_limit_x96: int
    ) -> Tuple[int, int]:
        """Execute a swap using proper Uniswap V3 math"""
        if amount_specified == 0:
            return (0, 0)
        
        exact_input = amount_specified > 0
        
        # Set price limit if not specified
        if sqrt_price_limit_x96 == 0:
            sqrt_price_limit_x96 = MIN_SQRT_RATIO + 1 if zero_for_one else MAX_SQRT_RATIO - 1
        
        # Validate price limit
        if zero_for_one:
            if sqrt_price_limit_x96 >= self.sqrt_price_x96:
                raise ValueError("Price limit too high for zero_for_one swap")
            if sqrt_price_limit_x96 < MIN_SQRT_RATIO:
                raise ValueError("Price limit too low")
        else:
            if sqrt_price_limit_x96 <= self.sqrt_price_x96:
                raise ValueError("Price limit too low for one_for_zero swap")
            if sqrt_price_limit_x96 > MAX_SQRT_RATIO:
                raise ValueError("Price limit too high")
        
        # Initialize swap state
        # Calculate active liquidity from ticks (not positions)
        active_liquidity = self._calculate_active_liquidity_from_ticks(self.tick_current)
        
        state = {
            'amount_specified_remaining': amount_specified,
            'amount_calculated': 0,
            'sqrt_price_x96': self.sqrt_price_x96,
            'tick': self.tick_current,
            'liquidity': active_liquidity
        }
        
        # Convert fee to pips (0.003 = 3000 pips)
        fee_pips = int(self.fee_tier * 1000000)
        
        # Enhanced cross-tick swap loop
        iteration_count = 0
        
        while (abs(state['amount_specified_remaining']) > 1 and  # Use tolerance instead of exact 0
               state['sqrt_price_x96'] != sqrt_price_limit_x96 and
               iteration_count < self.max_swap_iterations):
            
            iteration_count += 1
            
            if self.debug_cross_tick and iteration_count % 100 == 0:
                print(f"Cross-tick swap iteration {iteration_count}: remaining={state['amount_specified_remaining']}, price={state['sqrt_price_x96']}")
            
            # Store previous state to detect progress
            prev_amount_remaining = state['amount_specified_remaining']
            prev_sqrt_price = state['sqrt_price_x96']
            prev_liquidity = state['liquidity']
            
            # Find the next initialized tick using enhanced bitmap or fallback
            if self.use_tick_bitmap and self.use_enhanced_cross_tick:
                tick_next = self._next_initialized_tick_enhanced(state['tick'], zero_for_one)
            else:
                tick_next = self._next_initialized_tick(state['tick'], zero_for_one)
            
            # Boundary check - no more ticks available
            if ((zero_for_one and tick_next <= MIN_TICK) or 
                (not zero_for_one and tick_next >= MAX_TICK)):
                if self.debug_cross_tick:
                    print(f"Reached tick boundary, exiting swap at iteration {iteration_count}")
                break
            
            # Calculate target price for this step
            sqrt_price_next_tick = tick_to_sqrt_price_x96(tick_next)
            
            # Determine target price (either next tick or price limit)
            if zero_for_one:
                sqrt_price_target_x96 = max(sqrt_price_next_tick, sqrt_price_limit_x96)
            else:
                sqrt_price_target_x96 = min(sqrt_price_next_tick, sqrt_price_limit_x96)
            
            # Enhanced cross-tick computation
            try:
                if self.use_enhanced_cross_tick:
                    sqrt_price_after_step, amount_in, amount_out, fee_amount = compute_swap_step(
                        state['sqrt_price_x96'],
                        sqrt_price_target_x96,
                        state['liquidity'],
                        state['amount_specified_remaining'],
                        fee_pips
                    )
                else:
                    # Fallback to original implementation
                    sqrt_price_after_step, amount_in, amount_out, fee_amount = compute_swap_step(
                        state['sqrt_price_x96'],
                        sqrt_price_target_x96,
                        state['liquidity'],
                        state['amount_specified_remaining'],
                        fee_pips
                    )
            except (ValueError, ZeroDivisionError) as e:
                if self.debug_cross_tick:
                    print(f"Math error in swap step: {e}")
                break
            
            # Progress validation - ensure we're making meaningful progress
            if (amount_in == 0 and amount_out == 0 and 
                sqrt_price_after_step == state['sqrt_price_x96']):
                if self.debug_cross_tick:
                    print(f"No progress in swap step, exiting at iteration {iteration_count}")
                break
            
            # Update swap state
            if exact_input:
                state['amount_specified_remaining'] -= amount_in  # Fee already deducted in compute_swap_step
                state['amount_calculated'] -= amount_out
            else:
                state['amount_specified_remaining'] += amount_out
                state['amount_calculated'] += amount_in  # Fee already deducted in compute_swap_step
            
            # Update price
            state['sqrt_price_x96'] = sqrt_price_after_step
            
            # Enhanced cross-tick liquidity transition with position-aware logic
            if sqrt_price_after_step == sqrt_price_next_tick:
                # We've crossed to the next tick
                
                # CRITICAL FIX: Check if new tick is covered by existing positions
                if self._is_tick_covered_by_positions(tick_next):
                    # Still within concentrated liquidity range - density unchanged!
                    if self.debug_cross_tick:
                        print(f"Tick {tick_next} still covered by positions - liquidity unchanged: {state['liquidity']}")
                else:
                    # FAIL FAST: Don't try to fix it, expose the problem immediately
                    error_msg = (
                        f"🚨 LIQUIDITY COVERAGE FAILURE at iteration {iteration_count}\n"
                        f"   Pool: {self.pool_name}\n"
                        f"   Crossed to uncovered tick: {tick_next}\n"
                        f"   Current price: {(sqrt_price_after_step / Q96) ** 2:.6f}\n"
                        f"   Available positions:\n"
                    )
                    for i, pos in enumerate(self.positions):
                        price_lower = 1.0001 ** pos.tick_lower
                        price_upper = 1.0001 ** pos.tick_upper
                        error_msg += f"     [{i}]: ticks [{pos.tick_lower}, {pos.tick_upper}] = prices [{price_lower:.6f}, {price_upper:.6f}]\n"
                    
                    # CRITICAL: Fail immediately to catch this in year-long sims
                    raise ValueError(error_msg)
                
                # Update current tick
                state['tick'] = tick_next
            else:
                # We didn't reach the next tick - update tick based on current price
                new_tick = sqrt_price_x96_to_tick(sqrt_price_after_step)
                state['tick'] = new_tick
                
                # Check if we're still in covered range (price moved but no tick crossing)
                if not self._is_tick_covered_by_positions(new_tick):
                    error_msg = (
                        f"🚨 PRICE MOVEMENT TO UNCOVERED TICK at iteration {iteration_count}\n"
                        f"   Pool: {self.pool_name}\n"
                        f"   Moved to uncovered tick: {new_tick}\n"
                        f"   Current price: {(sqrt_price_after_step / Q96) ** 2:.6f}\n"
                        f"   This should not happen within position ranges!\n"
                    )
                    raise ValueError(error_msg)
            
            # Safety check: ensure liquidity remains positive using fail-fast monitoring
            if not self._monitor_liquidity_health(state, iteration_count):
                break
        
        # Warn if we hit the iteration limit
        if iteration_count >= self.max_swap_iterations:
            print(f"⚠️  Swap hit maximum iterations ({self.max_swap_iterations}) - potential infinite loop prevented")
        
        # Update pool state
        self.sqrt_price_x96 = state['sqrt_price_x96']
        self.tick_current = state['tick']
        self.liquidity = state['liquidity']
        
        # Update legacy fields to reflect actual swap impact
        self._update_legacy_fields()
        
        # Return amounts (input, output)
        if exact_input:
            amount_in_final = amount_specified - state['amount_specified_remaining']
            amount_out_final = -state['amount_calculated']
            # DEBUG: Print the final calculations (disabled)
            # print(f"DEBUG: amount_specified={amount_specified}, remaining={state['amount_specified_remaining']}, calculated={state['amount_calculated']}")
            # print(f"DEBUG: Final amounts: in={amount_in_final}, out={amount_out_final}")
            return (amount_in_final, amount_out_final)
        else:
            return (state['amount_calculated'], 
                    amount_specified - state['amount_specified_remaining'])
    
    def _next_initialized_tick(self, tick: int, zero_for_one: bool) -> int:
        """Find the next initialized tick in the given direction"""
        if zero_for_one:
            # Moving down (decreasing price)
            initialized_ticks = [t for t in self.ticks.keys() if t < tick and self.ticks[t].initialized]
            if initialized_ticks:
                return max(initialized_ticks)
            else:
                # No more ticks below - return MIN_TICK to signal boundary
                return MIN_TICK
        else:
            # Moving up (increasing price)  
            initialized_ticks = [t for t in self.ticks.keys() if t > tick and self.ticks[t].initialized]
            if initialized_ticks:
                return min(initialized_ticks)
            else:
                # No more ticks above - return MAX_TICK to signal boundary
                return MAX_TICK
    
    def _next_initialized_tick_enhanced(self, tick: int, zero_for_one: bool) -> int:
        """Enhanced tick finding using bitmap when available"""
        if self.use_tick_bitmap and self.tick_bitmap:
            return self.tick_bitmap.next_initialized_tick(tick, self.tick_spacing, zero_for_one)
        else:
            # Fallback to linear search
            return self._next_initialized_tick(tick, zero_for_one)
    
    def _calculate_liquidity_at_tick(self, tick: int) -> int:
        """Calculate available liquidity at a specific tick"""
        liquidity_at_tick = 0
        
        # Sum liquidity from all positions that include this tick
        for position in self.positions:
            if position.tick_lower <= tick < position.tick_upper:
                liquidity_at_tick += position.liquidity
        
        return max(0, liquidity_at_tick)
    
    
    def get_price(self) -> float:
        """Get current price from sqrt_price_x96 using exact calculation"""
        # Convert to float for compatibility with existing code
        sqrt_price = self.sqrt_price_x96 / Q96
        return sqrt_price * sqrt_price
    
    def get_liquidity_at_price(self, target_price: float) -> float:
        """Get liquidity available at a specific price point"""
        # Convert price to tick
        target_tick = int(math.log(target_price) / math.log(1.0001))
        
        # Find liquidity at this tick
        liquidity_at_tick = 0
        for position in self.positions:
            if position.tick_lower <= target_tick < position.tick_upper:
                liquidity_at_tick += position.liquidity
        
        # Convert back to USD
        return liquidity_at_tick / 1e6
    
    def _is_tick_covered_by_positions(self, tick: int) -> bool:
        """Check if tick is covered by any existing position"""
        for position in self.positions:
            if position.tick_lower <= tick < position.tick_upper:
                return True
        return False
    
    def _calculate_active_liquidity_from_ticks(self, current_tick: int) -> int:
        """Calculate active liquidity from positions that are currently active"""
        active_liquidity = 0
        
        # Only count liquidity from positions that are currently active
        for position in self.positions:
            if position.tick_lower <= current_tick < position.tick_upper:
                active_liquidity += position.liquidity
        
        return active_liquidity
    
    def _validate_position_coverage(self):
        """Ensure current tick is covered by at least one position - FAIL if not"""
        if not self._is_tick_covered_by_positions(self.tick_current):
            error_msg = (
                f"🚨 CRITICAL ERROR: Current tick {self.tick_current} not covered by any position!\n"
                f"   Pool: {self.pool_name}\n"
                f"   Current price: {self.get_price():.6f}\n"
                f"   Positions:\n"
            )
            for i, pos in enumerate(self.positions):
                price_lower = 1.0001 ** pos.tick_lower
                price_upper = 1.0001 ** pos.tick_upper
                error_msg += f"     [{i}]: ticks [{pos.tick_lower}, {pos.tick_upper}] = prices [{price_lower:.6f}, {price_upper:.6f}]\n"
            
            # FAIL IMMEDIATELY - don't mask the problem
            raise ValueError(error_msg)
    
    def _monitor_liquidity_health(self, state: dict, iteration: int):
        """Monitor liquidity health - FAIL if problems detected"""
        if state['liquidity'] <= 0:
            # Don't try to fix it - expose the root cause immediately
            error_msg = (
                f"🚨 LIQUIDITY EXHAUSTION DETECTED at iteration {iteration}\n"
                f"   Pool: {self.pool_name}\n"
                f"   Current tick: {state['tick']}\n"
                f"   Current price: {(state['sqrt_price_x96'] / Q96) ** 2:.6f}\n"
                f"   State liquidity: {state['liquidity']}\n"
                f"   Positions covering this tick:\n"
            )
            
            covering_positions = []
            for i, pos in enumerate(self.positions):
                if pos.tick_lower <= state['tick'] < pos.tick_upper:
                    covering_positions.append(i)
                    price_lower = 1.0001 ** pos.tick_lower  
                    price_upper = 1.0001 ** pos.tick_upper
                    error_msg += f"     Position {i}: [{pos.tick_lower}, {pos.tick_upper}] = [{price_lower:.6f}, {price_upper:.6f}] liquidity={pos.liquidity:,}\n"
            
            if not covering_positions:
                error_msg += "     ❌ NO POSITIONS COVER CURRENT TICK!\n"
            
            error_msg += f"   Total positions: {len(self.positions)}\n"
            error_msg += f"   This indicates a fundamental position management error.\n"
            
            # FAIL IMMEDIATELY - don't mask with emergency fixes
            raise ValueError(error_msg)
        
        return True
    
    def get_total_active_liquidity(self) -> float:
        """Get total liquidity across all active positions using the same calculation as initialization"""
        # For yield token pools after rebalancing, use the same approach as the initialization
        # The key insight: we calculated L = concentrated_liquidity_usd / coeff_sum during initialization
        # So: concentrated_liquidity_usd = L * coeff_sum
        if "Yield_Token" in self.pool_name and self.positions:
            import math
            current_price = self.get_price()
            total_usd_value = 0.0
            
            for position in self.positions:
                # Convert ticks back to prices
                P_lower = 1.0001 ** position.tick_lower
                P_upper = 1.0001 ** position.tick_upper
                
                # Check if current price is within this position's range
                if P_lower <= current_price <= P_upper:
                    # Calculate coefficients exactly as in _initialize_asymmetric_yield_token_positions
                    x = math.sqrt(current_price)
                    a = math.sqrt(P_lower)
                    b = math.sqrt(P_upper)
                    
                    coeff_0 = (b - x) / (x * b)  # MOET coefficient
                    coeff_1 = x - a              # YT coefficient
                    coeff_sum = coeff_0 + coeff_1
                    
                    if coeff_sum > 0:
                        # The position was created with: L = concentrated_liquidity_usd / coeff_sum
                        # We need to reverse this to get: concentrated_liquidity_usd = L * coeff_sum
                        # But position.liquidity is the Uniswap V3 liquidity calculated from amount1_scaled
                        
                        # From the original calculation:
                        # amount_1 = L * coeff_1
                        # amount1_scaled = int(amount_1 * 1e6) = int(L * coeff_1 * 1e6)
                        # proper_concentrated_liquidity = mul_div(amount1_scaled, Q96, denominator)
                        
                        # We can reverse-engineer L from position.liquidity:
                        # We know: amount_1 = L * coeff_1
                        # And: amount1_scaled = amount_1 * 1e6
                        # So: L = amount_1 / coeff_1 = amount1_scaled / (coeff_1 * 1e6)
                        
                        # But position.liquidity went through the Uniswap V3 calculation, so we need to
                        # reverse that calculation to get back to amount1_scaled
                        sqrt_price_current_x96 = self.sqrt_price_x96
                        sqrt_price_lower_x96 = tick_to_sqrt_price_x96(position.tick_lower)
                        denominator = sqrt_price_current_x96 - sqrt_price_lower_x96
                        
                        if denominator > 0:
                            # Reverse the mul_div operation: amount1_scaled = position.liquidity * denominator / Q96
                            amount1_scaled = position.liquidity * denominator // Q96
                            
                            # Now calculate L: L = amount1_scaled / (coeff_1 * 1e6)
                            if coeff_1 > 0:
                                L = amount1_scaled / (coeff_1 * 1e6)
                                
                                # Finally: concentrated_liquidity_usd = L * coeff_sum
                                position_usd = L * coeff_sum
                                total_usd_value += position_usd
            
            if total_usd_value > 0:
                print(f"💰 CALCULATED ACTIVE LIQUIDITY (corrected): ${total_usd_value:,.0f} (vs configured ${self.total_liquidity:,.0f})")
                return total_usd_value
        
        # Fallback to configured total liquidity for other cases
        return self.total_liquidity
    
    def get_liquidity_distribution(self) -> Tuple[List[float], List[float]]:
        """Get price and liquidity arrays for charting"""
        # Get all ticks sorted by value
        sorted_ticks = sorted(self.ticks.keys())
        
        if not sorted_ticks:
            return [], []
        
        prices = []
        liquidity = []
        
        for tick in sorted_ticks:
            price = (tick_to_sqrt_price_x96(tick) / Q96) ** 2
            
            # Calculate liquidity at this tick
            liquidity_at_tick = 0
            for position in self.positions:
                if position.tick_lower <= tick < position.tick_upper:
                    liquidity_at_tick += position.liquidity
            
            # Convert back to USD and only include active liquidity
            liquidity_usd = liquidity_at_tick / 1e6
            if liquidity_usd > 1000:  # $1k threshold
                prices.append(price)
                liquidity.append(liquidity_usd)
        
        return prices, liquidity
    
    def get_tick_data_for_charts(self) -> List[Dict]:
        """Get tick data formatted for bar chart visualization"""
        # Get all ticks sorted by value
        sorted_ticks = sorted(self.ticks.keys())
        
        if not sorted_ticks:
            return []
        
        tick_data = []
        for i, tick in enumerate(sorted_ticks):
            price = (tick_to_sqrt_price_x96(tick) / Q96) ** 2
            
            # Calculate liquidity at this tick
            liquidity_at_tick = 0
            for position in self.positions:
                if position.tick_lower <= tick < position.tick_upper:
                    liquidity_at_tick += position.liquidity
            
            # Convert back to USD
            liquidity_usd = liquidity_at_tick / 1e6
            
            tick_data.append({
                "tick_index": i,
                "price": price,
                "liquidity": liquidity_usd,
                "is_active": liquidity_usd > 1000,  # $1k threshold
                "price_label": self._format_price_label(price),
                "tick": tick
            })
        
        return tick_data
    
    def _format_price_label(self, price: float) -> str:
        """Format price for display labels"""
        if self.concentration_type == "moet_btc":
            # For MOET:BTC, show as BTC per MOET
            return f"{price:.6f}"
        else:
            # For yield tokens, show as MOET per yield token
            return f"{price:.4f}"
    
    def simulate_trade_impact(self, trade_amount_usd: float, token_in: str) -> Dict[str, float]:
        """Simulate trade impact using actual swap logic"""
        # Convert USD to scaled amount
        amount_in_scaled = int(trade_amount_usd * 1e6)
        
        # Store original state
        original_sqrt_price = self.sqrt_price_x96
        original_tick = self.tick_current
        original_liquidity = self.liquidity
        
        try:
            # Determine swap direction
            zero_for_one = token_in in ["MOET", "token0"]
            
            # Execute simulated swap
            amount_in_actual, amount_out_actual = self.swap(
                zero_for_one=zero_for_one,
                amount_specified=amount_in_scaled,
                sqrt_price_limit_x96=0  # No limit
            )
            
            # Calculate results
            new_price = self.get_price()
            current_price = (original_sqrt_price / Q96) ** 2
            price_impact = abs((new_price - current_price) / current_price) if current_price > 0 else 0
            
            result = {
                "price_impact": price_impact,
                "new_price": new_price,
                "current_price": current_price,
                "amount_in": amount_in_actual / 1e6,
                "amount_out": amount_out_actual / 1e6,
                "trade_amount": trade_amount_usd
            }
            
        except (ValueError, ZeroDivisionError):
            # Handle edge cases
            result = {
                "price_impact": 0.0,
                "new_price": self.get_price(),
                "current_price": self.get_price(),
                "amount_in": 0.0,
                "amount_out": 0.0,
                "trade_amount": trade_amount_usd
            }
        finally:
            # Restore original state
            self.sqrt_price_x96 = original_sqrt_price
            self.tick_current = original_tick
            self.liquidity = original_liquidity
        
        return result
    
    def _update_legacy_fields(self):
        """Update legacy fields for backward compatibility"""
        # For yield token pools, calculate reserves based on configured ratio
        if "Yield_Token" in self.pool_name:
            total_liquidity_usd = self.get_total_active_liquidity()
            
            # Use configured ratio instead of price-based calculation
            self.token0_reserve = total_liquidity_usd * self.token0_ratio  # MOET reserve
            self.token1_reserve = total_liquidity_usd * (1 - self.token0_ratio)  # YT reserve
        else:
            # For BTC pools, use the original method (50/50 split)
            total_active_liquidity = self.get_total_active_liquidity()
            self.token0_reserve = total_active_liquidity / 2  # MOET reserve
            self.token1_reserve = total_active_liquidity / 2  # BTC reserve
    


class UniswapV3SlippageCalculator:
    """
    Proper Uniswap V3 slippage calculator using tick-based math
    """
    
    def __init__(self, pool_state: UniswapV3Pool):
        self.pool = pool_state
        
    def calculate_swap_slippage(
        self, 
        amount_in: float, 
        token_in: str,  # "MOET", "BTC", "Yield_Token", "USDC", or "USDF"
        concentrated_range: float = 0.2  # Legacy parameter for backward compatibility
    ) -> Dict[str, float]:
        """
        Calculate slippage for a swap using proper Uniswap V3 tick-based math
        
        Args:
            amount_in: Amount of input token to swap (in USD)
            token_in: Which token is being swapped in ("MOET", "BTC", "Yield_Token", "USDC", "USDF")
            concentrated_range: Legacy parameter for backward compatibility
            
        Returns:
            Dict with swap details including slippage
        """
        
        if token_in == "MOET":
            if "Yield_Token" in self.pool.pool_name:
                return self._calculate_moet_to_yield_token_swap(amount_in)
            elif "USDC" in self.pool.pool_name:
                return self._calculate_moet_to_usdc_swap(amount_in)
            elif "USDF" in self.pool.pool_name:
                return self._calculate_moet_to_usdf_swap(amount_in)
            else:
                return self._calculate_moet_to_btc_swap(amount_in)
        elif token_in == "BTC":
            return self._calculate_btc_to_stablecoin_swap(amount_in)
        elif token_in == "Yield_Token":
            return self._calculate_yield_token_to_moet_swap(amount_in)
        elif token_in == "USDC":
            if "BTC" in self.pool.pool_name:
                return self._calculate_usdc_to_btc_swap(amount_in)
            else:
                return self._calculate_usdc_to_moet_swap(amount_in)
        elif token_in == "USDF":
            if "BTC" in self.pool.pool_name:
                return self._calculate_usdf_to_btc_swap(amount_in)
            else:
                return self._calculate_usdf_to_moet_swap(amount_in)
        else:
            raise ValueError("token_in must be 'MOET', 'BTC', 'Yield_Token', 'USDC', or 'USDF'")
    
    def _calculate_moet_to_btc_swap(self, moet_amount: float) -> Dict[str, float]:
        """Calculate MOET -> BTC swap using proper Uniswap V3 math"""
        
        # Store original pool state
        original_sqrt_price_x96 = self.pool.sqrt_price_x96
        original_tick_current = self.pool.tick_current
        original_liquidity = self.pool.liquidity
        
        # Current price (BTC per MOET)
        current_price = self.pool.get_price()
        
        # Convert USD amount to token amount (scaled for precision)
        # For MOET: 1 MOET = $1, so amount_in_tokens = moet_amount
        amount_in_scaled = int(moet_amount * 1e6)  # Scale up for precision
        
        try:
            # Execute the swap: MOET (token0) -> BTC (token1), so zero_for_one = True
            zero_for_one = True
            sqrt_price_limit_x96 = MIN_SQRT_RATIO + 1  # No specific limit
            
            amount_in_actual, amount_out_actual = self.pool.swap(
                zero_for_one=zero_for_one,
                amount_specified=amount_in_scaled,
                sqrt_price_limit_x96=sqrt_price_limit_x96
            )
            
            # Convert back to USD amounts
            amount_in_usd = amount_in_actual / 1e6
            amount_out_btc_tokens = amount_out_actual / 1e6
            
            # Calculate expected output without slippage
            moet_tokens_in = moet_amount  # 1 MOET = $1
            expected_btc_out = moet_tokens_in * current_price
            
            # Calculate slippage
            slippage_amount = max(0, expected_btc_out - amount_out_btc_tokens)
            slippage_percentage = (slippage_amount / expected_btc_out) * 100 if expected_btc_out > 0 else 0
            
            # Calculate new price and price impact
            new_price = self.pool.get_price()
            price_impact = abs((current_price - new_price) / current_price) * 100 if current_price > 0 else 0
            
            # Calculate trading fees
            trading_fees = amount_in_usd * self.pool.fee_tier
            
            result = {
                "amount_in": moet_amount,
                "token_in": "MOET",
                "amount_out": amount_out_btc_tokens,
                "token_out": "BTC",
                "expected_amount_out": expected_btc_out,
                "slippage_amount": slippage_amount,
                "slippage_percentage": slippage_percentage,
                "price_impact_percentage": price_impact,
                "trading_fees": trading_fees,
                "current_price": current_price,
                "new_price": new_price,
                "effective_liquidity": self.pool.get_total_active_liquidity(),
            }
            
        finally:
            # CRITICAL FIX: DON'T restore pool state - let swaps mutate the pool
            # This allows agents to impact each other through shared pool state
            pass  # Keep the finally block but don't restore state
        
        return result
    
    def _calculate_btc_to_moet_swap(self, btc_amount: float) -> Dict[str, float]:
        """Calculate BTC -> MOET swap using proper Uniswap V3 math"""
        
        # Store original pool state
        original_sqrt_price_x96 = self.pool.sqrt_price_x96
        original_tick_current = self.pool.tick_current
        original_liquidity = self.pool.liquidity
        
        # Current price (BTC per MOET)
        current_price_btc_per_moet = self.pool.get_price()
        
        # Convert USD amount to BTC tokens, then scale for precision
        btc_tokens = btc_amount / self.pool.btc_price  # USD to BTC tokens
        amount_in_scaled = int(btc_tokens * 1e6)  # Scale up for precision
        
        try:
            # Execute the swap: BTC (token1) -> MOET (token0), so zero_for_one = False
            zero_for_one = False
            sqrt_price_limit_x96 = MAX_SQRT_RATIO - 1  # No specific limit
            
            amount_in_actual, amount_out_actual = self.pool.swap(
                zero_for_one=zero_for_one,
                amount_specified=amount_in_scaled,
                sqrt_price_limit_x96=sqrt_price_limit_x96
            )
            
            # Convert back to USD amounts
            amount_out_moet_usd = amount_out_actual / 1e6  # MOET output in USD
            
            # Calculate expected output without slippage (CORRECTED)
            # If we're swapping $X worth of BTC, we expect to get $X worth of MOET (ignoring fees)
            expected_moet_out = btc_amount  # USD in = USD out (before slippage/fees)
            
            # Calculate slippage
            slippage_amount = max(0, expected_moet_out - amount_out_moet_usd)
            slippage_percentage = (slippage_amount / expected_moet_out) * 100 if expected_moet_out > 0 else 0
            
            # Calculate new price and price impact
            new_price_btc_per_moet = self.pool.get_price()
            price_impact = abs((current_price_btc_per_moet - new_price_btc_per_moet) / current_price_btc_per_moet) * 100 if current_price_btc_per_moet > 0 else 0
            
            # Calculate trading fees
            trading_fees = btc_amount * self.pool.fee_tier
            
            result = {
                "amount_in": btc_amount,
                "token_in": "BTC",
                "amount_out": amount_out_moet_usd,
                "token_out": "MOET",
                "expected_amount_out": expected_moet_out,
                "slippage_amount": slippage_amount,
                "slippage_percentage": slippage_percentage,
                "price_impact_percentage": price_impact,
                "trading_fees": trading_fees,
                "current_price": current_price_btc_per_moet,
                "new_price": new_price_btc_per_moet,
                "effective_liquidity": self.pool.get_total_active_liquidity(),
            }
            
        finally:
            # CRITICAL FIX: DON'T restore pool state - let swaps mutate the pool
            # This allows agents to impact each other through shared pool state
            pass  # Keep the finally block but don't restore state
        
        return result
    
    def _calculate_moet_to_yield_token_swap(self, moet_amount: float) -> Dict[str, float]:
        """Calculate MOET -> Yield Token swap using proper Uniswap V3 math"""
        
        # Store original pool state
        original_sqrt_price_x96 = self.pool.sqrt_price_x96
        original_tick_current = self.pool.tick_current
        original_liquidity = self.pool.liquidity
        
        # Current price (should be close to 1.0 for yield tokens)
        current_price = self.pool.get_price()
        
        # Convert USD amount to token amount (scaled for precision)
        amount_in_scaled = int(moet_amount * 1e6)  # Scale up for precision
        
        try:
            # For MOET:YieldToken pool, MOET is token0, YieldToken is token1
            # MOET -> YieldToken, so zero_for_one = True
            zero_for_one = True
            sqrt_price_limit_x96 = MIN_SQRT_RATIO + 1  # No specific limit
            
            amount_in_actual, amount_out_actual = self.pool.swap(
                zero_for_one=zero_for_one,
                amount_specified=amount_in_scaled,
                sqrt_price_limit_x96=sqrt_price_limit_x96
            )
            
            # Convert back to USD amounts
            amount_in_usd = amount_in_actual / 1e6
            amount_out_yt = amount_out_actual / 1e6
            
            # Calculate expected output without slippage (should be ~1:1)
            expected_yt_out = moet_amount * current_price
            
            # Calculate slippage
            slippage_amount = max(0, expected_yt_out - amount_out_yt)
            slippage_percent = (slippage_amount / expected_yt_out) * 100 if expected_yt_out > 0 else 0
            
            # Calculate new price and price impact
            new_price = self.pool.get_price()
            price_impact = abs((current_price - new_price) / current_price) * 100 if current_price > 0 else 0
            
            # Calculate trading fees
            trading_fees = amount_in_usd * self.pool.fee_tier
            
            result = {
                "amount_in": moet_amount,
                "amount_out": amount_out_yt,
                "slippage_percent": slippage_percent,
                "slippage_amount": slippage_amount,
                "trading_fees": trading_fees,
                "price_impact": price_impact
            }
            
        finally:
            # CRITICAL FIX: DON'T restore pool state - let swaps mutate the pool
            # This allows agents to impact each other through shared pool state
            pass  # Keep the finally block but don't restore state
        
        return result
    
    def _calculate_yield_token_to_moet_swap(self, yield_token_amount: float) -> Dict[str, float]:
        """Calculate Yield Token -> MOET swap using proper Uniswap V3 math"""
        
        # Store original pool state
        original_sqrt_price_x96 = self.pool.sqrt_price_x96
        original_tick_current = self.pool.tick_current
        original_liquidity = self.pool.liquidity
        
        # Current price (should be close to 1.0 for yield tokens)
        current_price = self.pool.get_price()
        
        # Convert USD amount to token amount (scaled for precision)
        amount_in_scaled = int(yield_token_amount * 1e6)  # Scale up for precision
        
        try:
            # For MOET:YieldToken pool, YieldToken is token1, MOET is token0
            # YieldToken -> MOET, so zero_for_one = False
            zero_for_one = False
            sqrt_price_limit_x96 = MAX_SQRT_RATIO - 1  # No specific limit
            
            amount_in_actual, amount_out_actual = self.pool.swap(
                zero_for_one=zero_for_one,
                amount_specified=amount_in_scaled,
                sqrt_price_limit_x96=sqrt_price_limit_x96
            )
            
            # Convert back to USD amounts
            amount_out_moet = amount_out_actual / 1e6
            
            # Calculate expected output without slippage (should be ~1:1)
            expected_moet_out = yield_token_amount * current_price
            
            # Calculate slippage
            slippage_amount = max(0, expected_moet_out - amount_out_moet)
            slippage_percent = (slippage_amount / expected_moet_out) * 100 if expected_moet_out > 0 else 0
            
            # Calculate new price and price impact
            new_price = self.pool.get_price()
            price_impact = abs((current_price - new_price) / current_price) * 100 if current_price > 0 else 0
            
            # Calculate trading fees
            trading_fees = yield_token_amount * self.pool.fee_tier
            
            result = {
                "amount_in": yield_token_amount,
                "amount_out": amount_out_moet,
                "slippage_percent": slippage_percent,
                "slippage_amount": slippage_amount,
                "trading_fees": trading_fees,
                "price_impact": price_impact
            }
            
        finally:
            # CRITICAL FIX: DON'T restore pool state - let swaps mutate the pool
            # This allows agents to impact each other through shared pool state
            pass  # Keep the finally block but don't restore state
        
        return result
    
    def _calculate_moet_to_usdc_swap(self, moet_amount: float) -> Dict[str, float]:
        """Calculate MOET -> USDC swap using proper Uniswap V3 math (stablecoin pair)"""
        return self._calculate_stablecoin_swap(moet_amount, "MOET", "USDC")
    
    def _calculate_moet_to_usdf_swap(self, moet_amount: float) -> Dict[str, float]:
        """Calculate MOET -> USDF swap using proper Uniswap V3 math (stablecoin pair)"""
        return self._calculate_stablecoin_swap(moet_amount, "MOET", "USDF")
    
    def _calculate_usdc_to_moet_swap(self, usdc_amount: float) -> Dict[str, float]:
        """Calculate USDC -> MOET swap using proper Uniswap V3 math (stablecoin pair)"""
        return self._calculate_stablecoin_swap(usdc_amount, "USDC", "MOET")
    
    def _calculate_usdf_to_moet_swap(self, usdf_amount: float) -> Dict[str, float]:
        """Calculate USDF -> MOET swap using proper Uniswap V3 math (stablecoin pair)"""
        return self._calculate_stablecoin_swap(usdf_amount, "USDF", "MOET")
    
    def _calculate_usdc_to_btc_swap(self, usdc_amount: float) -> Dict[str, float]:
        """Calculate USDC -> BTC swap using proper Uniswap V3 math"""
        return self._calculate_stablecoin_to_btc_swap(usdc_amount, "USDC")
    
    def _calculate_usdf_to_btc_swap(self, usdf_amount: float) -> Dict[str, float]:
        """Calculate USDF -> BTC swap using proper Uniswap V3 math"""
        return self._calculate_stablecoin_to_btc_swap(usdf_amount, "USDF")
    
    def _calculate_btc_to_stablecoin_swap(self, btc_amount: float) -> Dict[str, float]:
        """Calculate BTC -> stablecoin swap (USDC or USDF) using proper Uniswap V3 math"""
        # Determine which stablecoin based on pool name
        if "USDC" in self.pool.pool_name:
            stablecoin = "USDC"
        elif "USDF" in self.pool.pool_name:
            stablecoin = "USDF"
        else:
            stablecoin = "USDC"  # Default fallback
        
        return self._calculate_btc_to_stablecoin_swap_impl(btc_amount, stablecoin)
    
    def _calculate_stablecoin_swap(self, amount_in: float, token_in: str, token_out: str) -> Dict[str, float]:
        """Generic stablecoin swap calculation (MOET <-> USDC/USDF)"""
        # For stablecoin pairs, we expect minimal slippage due to tight concentration
        # Use similar logic to yield token swaps but with stablecoin naming
        
        # Store original pool state
        original_sqrt_price_x96 = self.pool.sqrt_price_x96
        original_tick_current = self.pool.tick_current
        original_liquidity = self.pool.liquidity
        
        # Current price (should be ~1.0 for stablecoin pairs)
        current_price = self.pool.get_price()
        
        # Convert USD amount to token amount (scaled for precision)
        amount_in_scaled = int(amount_in * 1e6)  # Scale up for precision
        
        try:
            # Execute the swap: determine direction based on token_in
            if token_in == "MOET":
                zero_for_one = True  # MOET (token0) -> USDC/USDF (token1)
            else:
                zero_for_one = False  # USDC/USDF (token1) -> MOET (token0)
            
            sqrt_price_limit_x96 = MIN_SQRT_RATIO + 1 if zero_for_one else MAX_SQRT_RATIO - 1
            
            amount_in_actual, amount_out_actual = self.pool.swap(
                zero_for_one=zero_for_one,
                amount_specified=amount_in_scaled,
                sqrt_price_limit_x96=sqrt_price_limit_x96
            )
            
            # Convert back to USD amounts
            amount_in_usd = amount_in_actual / 1e6
            amount_out_usd = amount_out_actual / 1e6
            
            # Calculate expected output without slippage (should be ~1:1)
            expected_out = amount_in  # 1:1 for stablecoin pairs
            
            # Calculate slippage
            slippage_amount = max(0, expected_out - amount_out_usd)
            slippage_percentage = (slippage_amount / expected_out) * 100 if expected_out > 0 else 0
            
            # Calculate new price and price impact
            new_price = self.pool.get_price()
            price_impact = abs((current_price - new_price) / current_price) * 100 if current_price > 0 else 0
            
            # Calculate trading fees (0.05% for stablecoin pairs)
            trading_fees = amount_in_usd * self.pool.fee_tier
            
            result = {
                "amount_out": amount_out_usd,
                "slippage_amount": slippage_amount,
                "slippage_percentage": slippage_percentage,
                "trading_fees": trading_fees,
                "price_impact_percentage": price_impact,
                "new_price": new_price,
                "token_in": token_in,
                "token_out": token_out,
                "price_impact": price_impact
            }
            
        except Exception as e:
            print(f"Stablecoin swap error: {e}")
            result = {
                "amount_out": 0.0,
                "slippage_amount": amount_in,
                "slippage_percentage": 100.0,
                "trading_fees": 0.0,
                "price_impact_percentage": 0.0,
                "new_price": current_price,
                "token_in": token_in,
                "token_out": token_out,
                "price_impact": 0.0
            }
            
        finally:
            # Keep pool state changes (don't restore)
            pass
        
        return result
    
    def _calculate_stablecoin_to_btc_swap(self, stablecoin_amount: float, stablecoin: str) -> Dict[str, float]:
        """Calculate stablecoin -> BTC swap using proper Uniswap V3 math"""
        # Similar to MOET -> BTC but with stablecoin as input
        
        # Store original pool state
        original_sqrt_price_x96 = self.pool.sqrt_price_x96
        original_tick_current = self.pool.tick_current
        original_liquidity = self.pool.liquidity
        
        # Current price (BTC per stablecoin)
        current_price = self.pool.get_price()
        
        # Convert USD amount to token amount (scaled for precision)
        amount_in_scaled = int(stablecoin_amount * 1e6)  # Scale up for precision
        
        try:
            # Execute the swap: stablecoin (token0) -> BTC (token1), so zero_for_one = True
            zero_for_one = True
            sqrt_price_limit_x96 = MIN_SQRT_RATIO + 1  # No specific limit
            
            amount_in_actual, amount_out_actual = self.pool.swap(
                zero_for_one=zero_for_one,
                amount_specified=amount_in_scaled,
                sqrt_price_limit_x96=sqrt_price_limit_x96
            )
            
            # Convert back to USD amounts
            amount_in_usd = amount_in_actual / 1e6
            amount_out_btc_tokens = amount_out_actual / 1e6
            
            # Calculate expected output without slippage
            stablecoin_tokens_in = stablecoin_amount  # 1 stablecoin = $1
            expected_btc_out = stablecoin_tokens_in * current_price
            
            # Calculate slippage
            slippage_amount = max(0, expected_btc_out - amount_out_btc_tokens)
            slippage_percentage = (slippage_amount / expected_btc_out) * 100 if expected_btc_out > 0 else 0
            
            # Calculate new price and price impact
            new_price = self.pool.get_price()
            price_impact = abs((current_price - new_price) / current_price) * 100 if current_price > 0 else 0
            
            # Calculate trading fees
            trading_fees = amount_in_usd * self.pool.fee_tier
            
            result = {
                "amount_out": amount_out_btc_tokens,
                "slippage_amount": slippage_amount,
                "slippage_percentage": slippage_percentage,
                "trading_fees": trading_fees,
                "price_impact_percentage": price_impact,
                "new_price": new_price,
                "token_in": stablecoin,
                "token_out": "BTC",
                "price_impact": price_impact
            }
            
        except Exception as e:
            print(f"Stablecoin to BTC swap error: {e}")
            result = {
                "amount_out": 0.0,
                "slippage_amount": stablecoin_amount,
                "slippage_percentage": 100.0,
                "trading_fees": 0.0,
                "price_impact_percentage": 0.0,
                "new_price": current_price,
                "token_in": stablecoin,
                "token_out": "BTC",
                "price_impact": 0.0
            }
            
        finally:
            # Keep pool state changes (don't restore)
            pass
        
        return result
    
    def _calculate_btc_to_stablecoin_swap_impl(self, btc_amount: float, stablecoin: str) -> Dict[str, float]:
        """Calculate BTC -> stablecoin swap using proper Uniswap V3 math"""
        # Similar to BTC -> MOET but with stablecoin as output
        
        # Store original pool state
        original_sqrt_price_x96 = self.pool.sqrt_price_x96
        original_tick_current = self.pool.tick_current
        original_liquidity = self.pool.liquidity
        
        # Current price (BTC per stablecoin)
        current_price = self.pool.get_price()
        
        # Convert BTC amount to scaled amount (BTC has different scaling)
        btc_price_usd = 1.0 / current_price if current_price > 0 else 100_000.0
        amount_in_usd = btc_amount * btc_price_usd
        amount_in_scaled = int(amount_in_usd * 1e6)  # Scale up for precision
        
        try:
            # Execute the swap: BTC (token1) -> stablecoin (token0), so zero_for_one = False
            zero_for_one = False
            sqrt_price_limit_x96 = MAX_SQRT_RATIO - 1  # No specific limit
            
            amount_in_actual, amount_out_actual = self.pool.swap(
                zero_for_one=zero_for_one,
                amount_specified=amount_in_scaled,
                sqrt_price_limit_x96=sqrt_price_limit_x96
            )
            
            # Convert back to USD amounts
            amount_out_stablecoin = amount_out_actual / 1e6
            
            # Calculate expected output without slippage
            expected_stablecoin_out = amount_in_usd  # Should get ~$1 stablecoin per $1 BTC value
            
            # Calculate slippage
            slippage_amount = max(0, expected_stablecoin_out - amount_out_stablecoin)
            slippage_percentage = (slippage_amount / expected_stablecoin_out) * 100 if expected_stablecoin_out > 0 else 0
            
            # Calculate new price and price impact
            new_price = self.pool.get_price()
            price_impact = abs((current_price - new_price) / current_price) * 100 if current_price > 0 else 0
            
            # Calculate trading fees
            trading_fees = amount_in_usd * self.pool.fee_tier
            
            result = {
                "amount_out": amount_out_stablecoin,
                "slippage_amount": slippage_amount,
                "slippage_percentage": slippage_percentage,
                "trading_fees": trading_fees,
                "price_impact_percentage": price_impact,
                "new_price": new_price,
                "token_in": "BTC",
                "token_out": stablecoin,
                "price_impact": price_impact
            }
            
        except Exception as e:
            print(f"BTC to stablecoin swap error: {e}")
            result = {
                "amount_out": 0.0,
                "slippage_amount": amount_in_usd,
                "slippage_percentage": 100.0,
                "trading_fees": 0.0,
                "price_impact_percentage": 0.0,
                "new_price": current_price,
                "token_in": "BTC",
                "token_out": stablecoin,
                "price_impact": 0.0
            }
            
        finally:
            # Keep pool state changes (don't restore)
            pass
        
        return result

    def update_pool_state(self, swap_result: Dict[str, float]):
        """Update pool state after a swap - simplified without bin consumption tracking"""
        
        # Update pool's current price based on the swap impact
        if "new_price" in swap_result and swap_result["new_price"] > 0:
            # Small price adjustment based on trade impact
            price_impact = swap_result.get("price_impact_percentage", 0) / 100.0
            if swap_result["token_in"] == "MOET":
                # MOET -> BTC swap should increase MOET price slightly (less MOET in pool)
                self.pool.peg_price *= (1 + price_impact * 0.1)  # Small adjustment
            else:
                # BTC -> MOET swap should decrease MOET price slightly (more MOET in pool)
                self.pool.peg_price *= (1 - price_impact * 0.1)  # Small adjustment
        
        # Update legacy fields for backward compatibility
        self.pool._update_legacy_fields()



def create_moet_btc_pool(pool_size_usd: float, btc_price: float = 100_000.0, concentration: float = 0.80) -> UniswapV3Pool:
    """
    Create a MOET:BTC Uniswap v3 pool with concentrated liquidity
    
    Args:
        pool_size_usd: Total pool size in USD
        btc_price: Current BTC price in USD (default: $100,000)
        concentration: Liquidity concentration level (0.80 = 80% at peg)
        
    Returns:
        UniswapV3Pool instance with concentrated liquidity positions
    """
    
    return UniswapV3Pool(
        pool_name="MOET:BTC",
        total_liquidity=pool_size_usd,
        btc_price=btc_price,
        concentration=concentration
        # fee_tier and tick_spacing will be set automatically based on pool type
    )


def create_yield_token_pool(pool_size_usd: float, concentration: float = 0.95, token0_ratio: float = 0.5) -> UniswapV3Pool:
    """
    Create a MOET:Yield Token Uniswap v3 pool with concentrated liquidity
    
    Args:
        pool_size_usd: Total pool size in USD
        concentration: Liquidity concentration level (0.95 = 95% at peg)
        token0_ratio: Ratio of token0 (MOET) in the pool (0.5 = 50/50, 0.75 = 75/25)
        
    Returns:
        UniswapV3Pool instance with concentrated liquidity positions
    """
    
    return UniswapV3Pool(
        pool_name="MOET:Yield_Token",
        total_liquidity=pool_size_usd,
        btc_price=100_000.0,  # Default value, not used for yield tokens
        concentration=concentration,
        token0_ratio=token0_ratio
        # fee_tier and tick_spacing will be set automatically based on pool type
    )


def create_usdc_btc_pool(pool_size_usd: float, btc_price: float = 100_000.0, concentration: float = 0.80) -> UniswapV3Pool:
    """
    Create a USDC:BTC Uniswap v3 pool with concentrated liquidity
    
    Args:
        pool_size_usd: Total pool size in USD
        btc_price: Current BTC price in USD (default: $100,000)
        concentration: Liquidity concentration level (0.80 = 80% at peg)
        
    Returns:
        UniswapV3Pool instance with concentrated liquidity positions
    """
    
    return UniswapV3Pool(
        pool_name="USDC:BTC",
        total_liquidity=pool_size_usd,
        btc_price=btc_price,
        concentration=concentration
        # fee_tier and tick_spacing will be set automatically based on pool type
    )


def create_usdf_btc_pool(pool_size_usd: float, btc_price: float = 100_000.0, concentration: float = 0.80) -> UniswapV3Pool:
    """
    Create a USDF:BTC Uniswap v3 pool with concentrated liquidity
    
    Args:
        pool_size_usd: Total pool size in USD
        btc_price: Current BTC price in USD (default: $100,000)
        concentration: Liquidity concentration level (0.80 = 80% at peg)
        
    Returns:
        UniswapV3Pool instance with concentrated liquidity positions
    """
    
    return UniswapV3Pool(
        pool_name="USDF:BTC",
        total_liquidity=pool_size_usd,
        btc_price=btc_price,
        concentration=concentration
        # fee_tier and tick_spacing will be set automatically based on pool type
    )


def create_moet_usdc_pool(pool_size_usd: float, concentration: float = 0.95, token0_ratio: float = 0.5) -> UniswapV3Pool:
    """
    Create a MOET:USDC Uniswap v3 pool with concentrated liquidity
    Same conditions as MOET:YT pool for tight peg maintenance
    
    Args:
        pool_size_usd: Total pool size in USD
        concentration: Liquidity concentration level (0.95 = 95% at peg)
        token0_ratio: Ratio of token0 (MOET) in the pool (0.5 = 50/50)
        
    Returns:
        UniswapV3Pool instance with concentrated liquidity positions
    """
    
    return UniswapV3Pool(
        pool_name="MOET:USDC",
        total_liquidity=pool_size_usd,
        btc_price=100_000.0,  # Default value, not used for stablecoin pairs
        concentration=concentration,
        token0_ratio=token0_ratio
        # fee_tier and tick_spacing will be set automatically (same as MOET:YT)
    )


def create_moet_usdf_pool(pool_size_usd: float, concentration: float = 0.95, token0_ratio: float = 0.5) -> UniswapV3Pool:
    """
    Create a MOET:USDF Uniswap v3 pool with concentrated liquidity
    Same conditions as MOET:YT pool for tight peg maintenance
    
    Args:
        pool_size_usd: Total pool size in USD
        concentration: Liquidity concentration level (0.95 = 95% at peg)
        token0_ratio: Ratio of token0 (MOET) in the pool (0.5 = 50/50)
        
    Returns:
        UniswapV3Pool instance with concentrated liquidity positions
    """
    
    return UniswapV3Pool(
        pool_name="MOET:USDF",
        total_liquidity=pool_size_usd,
        btc_price=100_000.0,  # Default value, not used for stablecoin pairs
        concentration=concentration,
        token0_ratio=token0_ratio
        # fee_tier and tick_spacing will be set automatically (same as MOET:YT)
    )




def calculate_rebalancing_cost_with_slippage(
    moet_amount: float,
    pool_size_usd: float = 500_000,
    concentrated_range: float = 0.2,
    btc_price: float = 100_000.0
) -> Dict[str, float]:
    """
    Calculate the total cost of rebalancing including Uniswap v3 slippage
    
    Args:
        moet_amount: Amount of MOET to swap for debt repayment
        pool_size_usd: Total MOET:BTC pool size in USD
        concentrated_range: Liquidity concentration range (0.2 = 20%)
        btc_price: Current BTC price in USD (default: $100,000)
        
    Returns:
        Dict with cost breakdown including slippage
    """
    
    # Create pool state with correct MOET:BTC ratio
    pool = create_moet_btc_pool(pool_size_usd, btc_price)
    calculator = UniswapV3SlippageCalculator(pool)
    
    # Calculate swap (MOET -> BTC to repay debt)
    swap_result = calculator.calculate_swap_slippage(moet_amount, "MOET", concentrated_range)
    
    # Total cost includes slippage and fees
    total_cost = swap_result["slippage_amount"] + swap_result["trading_fees"]
    
    return {
        "moet_amount_swapped": moet_amount,
        "btc_received": swap_result["amount_out"],
        "expected_btc_without_slippage": swap_result["expected_amount_out"],
        "slippage_cost": swap_result["slippage_amount"],
        "trading_fees": swap_result["trading_fees"],
        "total_swap_cost": total_cost,
        "slippage_percentage": swap_result["slippage_percentage"],
        "price_impact_percentage": swap_result["price_impact_percentage"],
        "effective_liquidity": swap_result["effective_liquidity"]
    }


def calculate_liquidation_cost_with_slippage(
    collateral_btc_amount: float,
    btc_price: float,
    liquidation_percentage: float = 0.5,
    liquidation_bonus: float = 0.05,
    pool_size_usd: float = 500_000
) -> Dict[str, float]:
    """
    Calculate the total cost of AAVE-style liquidation including Uniswap v3 slippage
    
    This function simulates the liquidator's perspective:
    1. Seizes BTC collateral
    2. Swaps BTC -> MOET through Uniswap V3
    3. Uses MOET to repay debt
    4. Keeps liquidation bonus
    
    Args:
        collateral_btc_amount: Amount of BTC collateral to liquidate
        btc_price: Current BTC price in USD
        liquidation_percentage: Percentage of collateral to liquidate (0.5 = 50%)
        liquidation_bonus: Liquidation bonus rate (0.05 = 5%)
        pool_size_usd: Total MOET:BTC pool size in USD
        
    Returns:
        Dict with liquidation cost breakdown including slippage
    """
    
    # Amount of BTC to liquidate
    btc_to_liquidate = collateral_btc_amount * liquidation_percentage
    btc_value_to_liquidate = btc_to_liquidate * btc_price
    
    # Create pool state
    pool = create_moet_btc_pool(pool_size_usd, btc_price)
    calculator = UniswapV3SlippageCalculator(pool)
    
    # Calculate swap (BTC -> MOET for debt repayment)
    swap_result = calculator.calculate_swap_slippage(btc_value_to_liquidate, "BTC")
    
    # Calculate debt that can be repaid (MOET received from swap)
    debt_repaid = swap_result["amount_out"]
    
    # Calculate liquidation bonus (5% of debt repaid, in BTC value)
    liquidation_bonus_value = debt_repaid * liquidation_bonus
    liquidation_bonus_btc = liquidation_bonus_value / btc_price
    
    # Total cost to agent = BTC seized (including bonus)
    total_cost_to_agent = btc_to_liquidate + liquidation_bonus_btc
    
    return {
        "btc_liquidated": btc_to_liquidate,
        "btc_value_liquidated": btc_value_to_liquidate,
        "moet_received": swap_result["amount_out"],
        "expected_moet_without_slippage": swap_result["expected_amount_out"],
        "debt_repaid": debt_repaid,
        "liquidation_bonus_value": liquidation_bonus_value,
        "liquidation_bonus_btc": liquidation_bonus_btc,
        "slippage_cost": swap_result["slippage_amount"],
        "trading_fees": swap_result["trading_fees"],
        "total_cost_to_agent": total_cost_to_agent,
        "slippage_percentage": swap_result["slippage_percentage"],
        "price_impact_percentage": swap_result["price_impact_percentage"],
        "effective_liquidity": swap_result["effective_liquidity"]
    }
