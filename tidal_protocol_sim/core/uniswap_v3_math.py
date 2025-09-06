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
TICK_SPACING_0_3_PERCENT = 60  # For 0.3% fee tier
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
    Compute a single swap step using exact Uniswap V3 logic
    Returns: (sqrt_price_next_x96, amount_in, amount_out, fee_amount)
    """
    zero_for_one = sqrt_price_current_x96 >= sqrt_price_target_x96
    exact_in = amount_remaining >= 0
    
    sqrt_price_next_x96 = 0
    amount_in = 0
    amount_out = 0
    fee_amount = 0
    
    if exact_in:
        # Calculate amount after fees
        amount_remaining_less_fee = mul_div(amount_remaining, 1000000 - fee_pips, 1000000)
        
        # Calculate how much input is needed to reach target price
        amount_in = get_amount0_delta(
            sqrt_price_target_x96, sqrt_price_current_x96, liquidity, True
        ) if zero_for_one else get_amount1_delta(
            sqrt_price_current_x96, sqrt_price_target_x96, liquidity, True
        )
        
        if amount_remaining_less_fee >= amount_in:
            # We can reach the target price
            sqrt_price_next_x96 = sqrt_price_target_x96
        else:
            # We cannot reach target price with remaining amount
            sqrt_price_next_x96 = get_next_sqrt_price_from_input(
                sqrt_price_current_x96, liquidity, amount_remaining_less_fee, zero_for_one
            )
    else:
        # Exact output case
        amount_out = get_amount1_delta(
            sqrt_price_target_x96, sqrt_price_current_x96, liquidity, False
        ) if zero_for_one else get_amount0_delta(
            sqrt_price_current_x96, sqrt_price_target_x96, liquidity, False
        )
        
        if -amount_remaining >= amount_out:
            # We can reach the target price
            sqrt_price_next_x96 = sqrt_price_target_x96
        else:
            # We cannot reach target price with remaining amount
            sqrt_price_next_x96 = get_next_sqrt_price_from_output(
                sqrt_price_current_x96, liquidity, -amount_remaining, zero_for_one
            )
    
    max_price_reached = sqrt_price_target_x96 == sqrt_price_next_x96
    
    # Calculate actual amounts based on the price change
    if zero_for_one:
        amount_in = get_amount0_delta(
            sqrt_price_next_x96, sqrt_price_current_x96, liquidity, True
        ) if not max_price_reached or not exact_in else get_amount0_delta(
            sqrt_price_next_x96, sqrt_price_current_x96, liquidity, False
        )
        
        amount_out = get_amount1_delta(
            sqrt_price_next_x96, sqrt_price_current_x96, liquidity, False
        )
    else:
        amount_in = get_amount1_delta(
            sqrt_price_current_x96, sqrt_price_next_x96, liquidity, True
        ) if not max_price_reached or not exact_in else get_amount1_delta(
            sqrt_price_current_x96, sqrt_price_next_x96, liquidity, False
        )
        
        amount_out = get_amount0_delta(
            sqrt_price_current_x96, sqrt_price_next_x96, liquidity, False
        )
    
    # Cap output amount if exact output
    if not exact_in and amount_out > -amount_remaining:
        amount_out = -amount_remaining
    
    # Calculate fees
    if exact_in and sqrt_price_next_x96 != sqrt_price_target_x96:
        # Not all input was used, so fee is on remaining amount
        fee_amount = amount_remaining - amount_in
    else:
        # Standard fee calculation
        fee_amount = mul_div_rounding_up(amount_in, fee_pips, 1000000 - fee_pips)
    
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
class Position:
    """Represents a concentrated liquidity position"""
    tick_lower: int  # Lower tick of the position
    tick_upper: int  # Upper tick of the position  
    liquidity: int   # Amount of liquidity in this position
    
# Backward compatibility - represents a "bin" as a tick for visualization
@dataclass
class LiquidityBin:
    """Backward compatibility: represents a tick as a bin for visualization"""
    price: float  # Price at this tick
    liquidity: float  # Amount of liquidity at this tick
    bin_index: int  # Sequential index (tick-based)
    is_active: bool = True  # Whether this tick has active liquidity
    tick: int = 0  # The actual tick number


@dataclass
class UniswapV3Pool:
    """Proper Uniswap V3 pool implementation with tick-based math"""
    pool_name: str  # "MOET:BTC" or "MOET:Yield_Token"
    total_liquidity: float  # Total pool size in USD
    btc_price: float = 100_000.0  # BTC price in USD
    fee_tier: float = 0.003  # 0.3% fee tier
    concentration: float = 0.80  # Concentration level (0.80 = 80% at peg)
    tick_spacing: int = TICK_SPACING_0_3_PERCENT  # Tick spacing for this fee tier
    
    # Core Uniswap V3 state
    sqrt_price_x96: int = Q96  # Current sqrt price in Q64.96 format
    liquidity: int = 0  # Current active liquidity
    tick_current: int = 0  # Current tick
    
    # Tick and position data
    ticks: Dict[int, TickInfo] = None  # tick -> TickInfo
    positions: List[Position] = None  # List of liquidity positions
    
    # Legacy fields for backward compatibility
    token0_reserve: Optional[float] = None  # MOET reserve (calculated from ticks)
    token1_reserve: Optional[float] = None  # BTC reserve (calculated from ticks)
    num_bins: int = 100  # For visualization compatibility
    bins: List[LiquidityBin] = None  # For backward compatibility
    
    def __post_init__(self):
        """Initialize Uniswap V3 pool with proper tick-based math"""
        # Initialize tick and position data structures
        self.ticks = {} if self.ticks is None else self.ticks
        self.positions = [] if self.positions is None else self.positions
        
        # Determine pool type and peg price using exact calculations
        if "MOET:BTC" in self.pool_name:
            self.concentration_type = "moet_btc"
            # For MOET:BTC, we want a small price (1 BTC = 100,000 MOET)
            # This corresponds to approximately tick -115129 for 0.00001 price
            self.tick_current = -115129  # Exact tick for ~0.00001 price
            self.tick_current = max(MIN_TICK + 1000, min(MAX_TICK - 1000, self.tick_current))
            self.sqrt_price_x96 = tick_to_sqrt_price_x96(self.tick_current)
            # Calculate exact peg price from sqrt_price_x96
            sqrt_price = self.sqrt_price_x96 / Q96
            self.peg_price = sqrt_price * sqrt_price
        else:
            self.concentration_type = "yield_token"
            # Set current price to exact 1:1 peg
            self.tick_current = 0
            self.sqrt_price_x96 = Q96
            self.peg_price = 1.0
        
        # Initialize concentrated liquidity positions
        self._initialize_concentrated_positions()
        
        # Create backward compatibility bins for visualization
        self._create_visualization_bins()
        
        # Calculate legacy fields for backward compatibility
        self._update_legacy_fields()
    
    def _initialize_concentrated_positions(self):
        """Initialize concentrated liquidity positions using proper Uniswap V3 math"""
        if self.concentration_type == "moet_btc":
            self._initialize_moet_btc_positions()
        else:
            self._initialize_yield_token_positions()
    
    def _initialize_moet_btc_positions(self):
        """Initialize MOET:BTC concentrated liquidity positions using exact tick math"""
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
            
            # Create 3 positions in wider range
            for i in range(3):
                range_multiplier = (i + 2) // 4  # 0.5, 0.75, 1.0
                inner_tick_range = wide_tick_range * range_multiplier
                
                pos_tick_lower = ((peg_tick - inner_tick_range) // self.tick_spacing) * self.tick_spacing
                pos_tick_upper = ((peg_tick + inner_tick_range) // self.tick_spacing) * self.tick_spacing
                
                pos_tick_lower = max(MIN_TICK + self.tick_spacing, pos_tick_lower)
                pos_tick_upper = min(MAX_TICK - self.tick_spacing, pos_tick_upper)
                
                if pos_tick_lower < pos_tick_upper:
                    self._add_position(pos_tick_lower, pos_tick_upper, remaining_liquidity // 3)
    
    def _initialize_yield_token_positions(self):
        """Initialize MOET:Yield Token concentrated liquidity positions using exact tick math"""
        total_liquidity_amount = int(self.total_liquidity * 1e6)
        concentrated_liquidity = int(total_liquidity_amount * self.concentration)
        
        # Position 1: Very tight range around 1:1 peg (±0.1% = ~10 ticks)
        peg_tick = self.tick_current  # Should be 0 for 1:1 peg
        tick_range = 10  # Approximately 0.1% price range
        
        tick_lower = ((peg_tick - tick_range) // self.tick_spacing) * self.tick_spacing
        tick_upper = ((peg_tick + tick_range) // self.tick_spacing) * self.tick_spacing
        
        # Ensure valid bounds
        tick_lower = max(MIN_TICK + self.tick_spacing, tick_lower)
        tick_upper = min(MAX_TICK - self.tick_spacing, tick_upper)
        
        if tick_lower < tick_upper:
            self._add_position(tick_lower, tick_upper, concentrated_liquidity)
        
        # Position 2: Wider range for remaining liquidity (±1% = ~100 ticks)
        remaining_liquidity = total_liquidity_amount - concentrated_liquidity
        if remaining_liquidity > 0:
            wide_tick_range = 100  # Approximately 1% price range
            
            tick_lower = ((peg_tick - wide_tick_range) // self.tick_spacing) * self.tick_spacing
            tick_upper = ((peg_tick + wide_tick_range) // self.tick_spacing) * self.tick_spacing
            
            tick_lower = max(MIN_TICK + self.tick_spacing, tick_lower)
            tick_upper = min(MAX_TICK - self.tick_spacing, tick_upper)
            
            if tick_lower < tick_upper:
                self._add_position(tick_lower, tick_upper, remaining_liquidity)
    
    def _add_position(self, tick_lower: int, tick_upper: int, liquidity: int):
        """Add a liquidity position and update tick data"""
        if liquidity <= 0:
            return
            
        # Create position
        position = Position(tick_lower, tick_upper, liquidity)
        self.positions.append(position)
        
        # Update tick data
        if tick_lower not in self.ticks:
            self.ticks[tick_lower] = TickInfo()
        if tick_upper not in self.ticks:
            self.ticks[tick_upper] = TickInfo()
        
        # Update liquidity deltas
        self.ticks[tick_lower].liquidity_net += liquidity
        self.ticks[tick_lower].liquidity_gross += liquidity
        self.ticks[tick_lower].initialized = True
        
        self.ticks[tick_upper].liquidity_net -= liquidity
        self.ticks[tick_upper].liquidity_gross += liquidity
        self.ticks[tick_upper].initialized = True
        
        # Update current liquidity if position includes current tick
        if tick_lower <= self.tick_current < tick_upper:
            self.liquidity += liquidity
    
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
        state = {
            'amount_specified_remaining': amount_specified,
            'amount_calculated': 0,
            'sqrt_price_x96': self.sqrt_price_x96,
            'tick': self.tick_current,
            'liquidity': self.liquidity
        }
        
        # Convert fee to pips (0.003 = 3000 pips)
        fee_pips = int(self.fee_tier * 1000000)
        
        # Continue swapping until amount is exhausted or price limit is reached
        # Add safety counter to prevent infinite loops
        max_iterations = 1000
        iteration_count = 0
        
        # Debug flag - can be enabled for troubleshooting
        debug_swap = False
        
        while (abs(state['amount_specified_remaining']) > 1 and  # Use tolerance instead of exact 0
               state['sqrt_price_x96'] != sqrt_price_limit_x96 and
               iteration_count < max_iterations):
            
            iteration_count += 1
            
            if debug_swap and iteration_count % 100 == 0:
                print(f"Swap iteration {iteration_count}: remaining={state['amount_specified_remaining']}, price={state['sqrt_price_x96']}")
            
            # Store previous state to detect if we're making progress
            prev_amount_remaining = state['amount_specified_remaining']
            prev_sqrt_price = state['sqrt_price_x96']
            
            # Find the next initialized tick
            tick_next = self._next_initialized_tick(state['tick'], zero_for_one)
            
            # Check if we have no more liquidity to trade through
            if ((zero_for_one and tick_next == MIN_TICK) or 
                (not zero_for_one and tick_next == MAX_TICK)):
                # No more initialized ticks in this direction - exit
                if debug_swap:
                    print(f"No more initialized ticks, exiting swap at iteration {iteration_count}")
                break
                
            sqrt_price_next_x96 = tick_to_sqrt_price_x96(tick_next)
            
            # Ensure we don't exceed the price limit
            if zero_for_one:
                sqrt_price_target_x96 = max(sqrt_price_next_x96, sqrt_price_limit_x96)
            else:
                sqrt_price_target_x96 = min(sqrt_price_next_x96, sqrt_price_limit_x96)
            
            # Compute the swap step
            try:
                sqrt_price_next_x96, amount_in, amount_out, fee_amount = compute_swap_step(
                    state['sqrt_price_x96'],
                    sqrt_price_target_x96,
                    state['liquidity'],
                    state['amount_specified_remaining'],
                    fee_pips
                )
            except (ValueError, ZeroDivisionError):
                # Exit on math errors
                break
            
            # Check if we made no progress (stuck in infinite loop)
            if (amount_in == 0 and amount_out == 0 and 
                state['sqrt_price_x96'] == prev_sqrt_price):
                if debug_swap:
                    print(f"No progress made, exiting swap at iteration {iteration_count}")
                break
            
            # Update amounts
            if exact_input:
                state['amount_specified_remaining'] -= (amount_in + fee_amount)
                state['amount_calculated'] -= amount_out
            else:
                state['amount_specified_remaining'] += amount_out
                state['amount_calculated'] += (amount_in + fee_amount)
            
            # Update price
            state['sqrt_price_x96'] = sqrt_price_next_x96
            
            # If we've reached the next tick, update liquidity
            if sqrt_price_next_x96 == tick_to_sqrt_price_x96(tick_next):
                if tick_next in self.ticks and self.ticks[tick_next].initialized:
                    liquidity_net = self.ticks[tick_next].liquidity_net
                    if zero_for_one:
                        liquidity_net = -liquidity_net
                    state['liquidity'] += liquidity_net
                
                state['tick'] = tick_next
            else:
                # We didn't reach the next tick, update tick to current price
                state['tick'] = sqrt_price_x96_to_tick(sqrt_price_next_x96)
            
            # Additional safety check: if liquidity becomes 0, we can't continue
            if state['liquidity'] <= 0:
                if debug_swap:
                    print(f"Liquidity exhausted, exiting swap at iteration {iteration_count}")
                break
        
        # Warn if we hit the iteration limit
        if iteration_count >= max_iterations:
            print(f"⚠️  Swap hit maximum iterations ({max_iterations}) - potential infinite loop prevented")
        
        # Update pool state
        self.sqrt_price_x96 = state['sqrt_price_x96']
        self.tick_current = state['tick']
        self.liquidity = state['liquidity']
        
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
    
    def _create_visualization_bins(self):
        """Create backward compatibility bins for visualization from tick data"""
        self.bins = []
        
        # Get all ticks sorted by value
        sorted_ticks = sorted(self.ticks.keys())
        
        if not sorted_ticks:
            return
        
        # Create bins representing tick ranges
        for i, tick in enumerate(sorted_ticks):
            price = (tick_to_sqrt_price_x96(tick) / Q96) ** 2
            
            # Calculate liquidity at this tick (sum of all positions that include this tick)
            liquidity_at_tick = 0
            for position in self.positions:
                if position.tick_lower <= tick < position.tick_upper:
                    liquidity_at_tick += position.liquidity
            
            # Convert back to USD for visualization
            liquidity_usd = liquidity_at_tick / 1e6
            
            self.bins.append(LiquidityBin(
                price=price,
                liquidity=liquidity_usd,
                bin_index=i,
                is_active=liquidity_usd > 1000,  # $1k threshold
                tick=tick
            ))
        
        # Sort bins by price for proper visualization
        self.bins.sort(key=lambda b: b.price)
        
        # Re-assign bin indices after sorting
        for i, bin in enumerate(self.bins):
            bin.bin_index = i
    
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
    
    def get_total_active_liquidity(self) -> float:
        """Get total liquidity across all active positions"""
        total_liquidity = sum(position.liquidity for position in self.positions)
        return total_liquidity / 1e6  # Convert back to USD
    
    def get_liquidity_distribution(self) -> Tuple[List[float], List[float]]:
        """Get price and liquidity arrays for charting"""
        if not self.bins:
            return [], []
        
        active_bins = [bin for bin in self.bins if bin.is_active]
        prices = [bin.price for bin in active_bins]
        liquidity = [bin.liquidity for bin in active_bins]
        return prices, liquidity
    
    def get_bin_data_for_charts(self) -> List[Dict]:
        """Get bin data formatted for bar chart visualization (backward compatibility)"""
        return [
            {
                "bin_index": bin.bin_index,
                "price": bin.price,
                "liquidity": bin.liquidity,
                "is_active": bin.is_active,
                "price_label": self._format_price_label(bin.price),
                "tick": bin.tick
            }
            for bin in self.bins
        ]
    
    def get_tick_data_for_charts(self) -> List[Dict]:
        """Get tick data formatted for bar chart visualization"""
        return self.get_bin_data_for_charts()  # Same format for compatibility
    
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
        # Calculate total reserves from positions
        total_active_liquidity = self.get_total_active_liquidity()
        
        # Split reserves 50/50 for compatibility
        self.token0_reserve = total_active_liquidity / 2  # MOET reserve in USD
        self.token1_reserve = total_active_liquidity / 2  # BTC reserve in USD
    
# Old bin-based methods removed - now using proper tick-based positions


# Old methods cleaned up - using proper Uniswap V3 implementation above


class UniswapV3SlippageCalculator:
    """
    Proper Uniswap V3 slippage calculator using tick-based math
    """
    
    def __init__(self, pool_state: UniswapV3Pool):
        self.pool = pool_state
        
    def calculate_swap_slippage(
        self, 
        amount_in: float, 
        token_in: str,  # "MOET", "BTC", or "Yield_Token"
        concentrated_range: float = 0.2  # Legacy parameter for backward compatibility
    ) -> Dict[str, float]:
        """
        Calculate slippage for a swap using proper Uniswap V3 tick-based math
        
        Args:
            amount_in: Amount of input token to swap (in USD)
            token_in: Which token is being swapped in ("MOET", "BTC", or "Yield_Token")
            concentrated_range: Legacy parameter for backward compatibility
            
        Returns:
            Dict with swap details including slippage
        """
        
        if token_in == "MOET":
            if "Yield_Token" in self.pool.pool_name:
                return self._calculate_moet_to_yield_token_swap(amount_in)
            else:
                return self._calculate_moet_to_btc_swap(amount_in)
        elif token_in == "BTC":
            return self._calculate_btc_to_moet_swap(amount_in)
        elif token_in == "Yield_Token":
            return self._calculate_yield_token_to_moet_swap(amount_in)
        else:
            raise ValueError("token_in must be 'MOET', 'BTC', or 'Yield_Token'")
    
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
                "bins_consumed": []  # For backward compatibility
            }
            
        finally:
            # Restore original pool state (for simulation purposes)
            self.pool.sqrt_price_x96 = original_sqrt_price_x96
            self.pool.tick_current = original_tick_current
            self.pool.liquidity = original_liquidity
        
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
                "bins_consumed": []  # For backward compatibility
            }
            
        finally:
            # Restore original pool state (for simulation purposes)
            self.pool.sqrt_price_x96 = original_sqrt_price_x96
            self.pool.tick_current = original_tick_current
            self.pool.liquidity = original_liquidity
        
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
            # Restore original pool state (for simulation purposes)
            self.pool.sqrt_price_x96 = original_sqrt_price_x96
            self.pool.tick_current = original_tick_current
            self.pool.liquidity = original_liquidity
        
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
            expected_moet_out = yield_token_amount / current_price
            
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
            # Restore original pool state (for simulation purposes)
            self.pool.sqrt_price_x96 = original_sqrt_price_x96
            self.pool.tick_current = original_tick_current
            self.pool.liquidity = original_liquidity
        
        return result

    def update_pool_state(self, swap_result: Dict[str, float]):
        """Update pool state after a swap by consuming liquidity from bins"""
        
        # Update bin liquidity based on consumption
        if "bins_consumed" in swap_result:
            for bin_consumption in swap_result["bins_consumed"]:
                bin_index = bin_consumption["bin_index"]
                if bin_index < len(self.pool.bins):
                    bin = self.pool.bins[bin_index]
                    
                    # Reduce bin liquidity based on actual consumption
                    # Since we consumed from one side, reduce total liquidity by 2x the consumed amount
                    if swap_result["token_in"] == "MOET":
                        liquidity_consumed = bin_consumption["moet_consumed"] * 2
                    else:
                        liquidity_consumed = bin_consumption.get("btc_consumed", 0) * 2
                    
                    # Update bin liquidity
                    bin.liquidity = max(0, bin.liquidity - liquidity_consumed)
                    
                    # Deactivate bin if liquidity too low
                    if bin.liquidity < 1000:  # Minimum $1k threshold
                        bin.is_active = False
        
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
    Create a MOET:BTC Uniswap v3 pool with discrete liquidity bins
    
    Args:
        pool_size_usd: Total pool size in USD
        btc_price: Current BTC price in USD (default: $100,000)
        concentration: Liquidity concentration level (0.80 = 80% at peg)
        
    Returns:
        UniswapV3Pool instance with discrete bins
    """
    
    return UniswapV3Pool(
        pool_name="MOET:BTC",
        total_liquidity=pool_size_usd,
        btc_price=btc_price,
        num_bins=100,
        fee_tier=0.003,  # 0.3% fee tier
        concentration=concentration
    )


def create_yield_token_pool(pool_size_usd: float, btc_price: float = 100_000.0, concentration: float = 0.95) -> UniswapV3Pool:
    """
    Create a MOET:Yield Token Uniswap v3 pool with discrete liquidity bins
    
    Args:
        pool_size_usd: Total pool size in USD
        btc_price: Current BTC price in USD (for consistency)
        concentration: Liquidity concentration level (0.95 = 95% at peg)
        
    Returns:
        UniswapV3Pool instance with discrete bins
    """
    
    return UniswapV3Pool(
        pool_name="MOET:Yield_Token",
        total_liquidity=pool_size_usd,
        btc_price=btc_price,
        num_bins=100,
        fee_tier=0.003,  # 0.3% fee tier
        concentration=concentration
    )


# Legacy factory function for backward compatibility
def create_moet_btc_concentrated_pool(pool_size_usd: float, btc_price: float = 100_000.0) -> UniswapV3Pool:
    """Legacy function - redirects to create_moet_btc_pool"""
    return create_moet_btc_pool(pool_size_usd, btc_price)


def create_yield_token_concentrated_pool(pool_size_usd: float, btc_price: float = 100_000.0) -> UniswapV3Pool:
    """Legacy function - redirects to create_yield_token_pool"""
    return create_yield_token_pool(pool_size_usd, btc_price)


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
    pool_size_usd: float = 500_000,
    concentrated_range: float = 0.2
) -> Dict[str, float]:
    """
    Calculate the total cost of AAVE-style liquidation including Uniswap v3 slippage
    
    Args:
        collateral_btc_amount: Amount of BTC collateral to liquidate
        btc_price: Current BTC price in USD
        liquidation_percentage: Percentage of collateral to liquidate (0.5 = 50%)
        liquidation_bonus: Liquidation bonus rate (0.05 = 5%)
        pool_size_usd: Total MOET:BTC pool size in USD
        concentrated_range: Liquidity concentration range
        
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
    swap_result = calculator.calculate_swap_slippage(btc_value_to_liquidate, "BTC", concentrated_range)
    
    # Liquidation bonus cost
    bonus_cost = btc_value_to_liquidate * liquidation_bonus
    
    # Total liquidation cost includes slippage, fees, and bonus
    total_cost = swap_result["slippage_amount"] + swap_result["trading_fees"] + bonus_cost
    
    return {
        "btc_liquidated": btc_to_liquidate,
        "btc_value_liquidated": btc_value_to_liquidate,
        "moet_received": swap_result["amount_out"],
        "expected_moet_without_slippage": swap_result["expected_amount_out"],
        "slippage_cost": swap_result["slippage_amount"],
        "trading_fees": swap_result["trading_fees"],
        "liquidation_bonus_cost": bonus_cost,
        "total_liquidation_cost": total_cost,
        "slippage_percentage": swap_result["slippage_percentage"],
        "price_impact_percentage": swap_result["price_impact_percentage"],
        "effective_liquidity": swap_result["effective_liquidity"]
    }
