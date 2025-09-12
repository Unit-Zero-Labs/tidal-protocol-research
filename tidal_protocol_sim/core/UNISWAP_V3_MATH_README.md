# Uniswap V3 Math Implementation - Technical Documentation

## Overview

This document provides a comprehensive breakdown of the `uniswap_v3_math.py` script, which implements a sophisticated Uniswap V3 concentrated liquidity system for the Tidal Protocol simulation. The implementation uses authentic Uniswap V3 mathematics with tick-based pricing, Q64.96 fixed-point arithmetic, and proper concentrated liquidity calculations. The system features within-range swap optimization, discrete liquidity ranges, and advanced cross-tick swap functionality when needed. It provides both trading functionality and visualization data for charts, with proper Uniswap V3 liquidity formulas and concentrated liquidity positions optimized for MOET:BTC (80% concentration) and MOET:Yield Token (95% concentration) pairs.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Proper Liquidity Calculations](#proper-liquidity-calculations)
4. [Core Math Functions](#core-math-functions)
5. [Pool Implementation](#pool-implementation)
6. [Within-Range vs Cross-Range Trading Logic](#within-range-vs-cross-range-trading-logic)
7. [Discrete Liquidity Ranges](#discrete-liquidity-ranges)
8. [Slippage Calculation](#slippage-calculation)
9. [Pool Types and Configuration](#pool-types-and-configuration)
10. [Usage Examples](#usage-examples)

## Core Concepts

### What is Uniswap V3?

Uniswap V3 is a decentralized exchange (DEX) that uses concentrated liquidity, allowing liquidity providers to concentrate their capital within specific price ranges. This increases capital efficiency compared to previous versions.

### Key Components

1. **Ticks**: Discrete price points that define liquidity ranges
2. **Sqrt Price**: Square root of price stored in Q64.96 fixed-point format
3. **Liquidity Positions**: Concentrated ranges of liquidity between two ticks
4. **Fee Tiers**: Different fee structures for different asset pairs (0.05%, 0.3%, 1%)
5. **Cross-Tick Swaps**: Sophisticated handling of swaps that move across multiple price ranges
6. **TickBitmap**: Efficient O(1) tick finding using bitmap approach instead of O(n) linear search
7. **Multiple Position Ranges**: Sophisticated liquidity distribution with multiple concentrated positions
8. **Chart Integration**: Built-in support for liquidity distribution visualization and tick data

## Mathematical Foundations

### Q64.96 Fixed-Point Arithmetic

Uniswap V3 uses Q64.96 format for price calculations:
- 64 bits for the integer part
- 96 bits for the fractional part
- Total: 160 bits for precise decimal calculations

```python
Q96 = 2 ** 96  # 79,228,162,514,264,337,593,543,950,336
```

### Tick System

Ticks represent discrete price points:
- Each tick corresponds to a price: `price = 1.0001^tick`
- Tick spacing varies by fee tier (10, 60, or 200)
- Price range: `1.0001^(-887272)` to `1.0001^(887272)`

### Price Conversion Functions

```python
def tick_to_sqrt_price_x96(tick: int) -> int:
    """Convert tick to sqrt price in Q64.96 format"""
    sqrt_price = 1.0001 ** (tick / 2.0)
    return int(sqrt_price * Q96)

def sqrt_price_x96_to_tick(sqrt_price_x96: int) -> int:
    """Convert sqrt price X96 to tick using binary search"""
    # Binary search implementation for precision
```

Note: In the implementation we clamp to Uniswap's bounds using `MIN_SQRT_RATIO` and `MAX_SQRT_RATIO` to ensure validity.

## Proper Liquidity Calculations

### Authentic Uniswap V3 Liquidity Formula

The implementation uses the official Uniswap V3 formulas for calculating liquidity from token amounts, as specified in the Uniswap V3 whitepaper and development book:

```python
def _calculate_liquidity_from_amounts(self, amount0: float, amount1: float, tick_lower: int, tick_upper: int) -> int:
    """Calculate liquidity using proper Uniswap V3 formulas with correct scaling"""
    
    # Get sqrt prices in Q96 format
    sqrt_price_lower_x96 = tick_to_sqrt_price_x96(tick_lower)
    sqrt_price_upper_x96 = tick_to_sqrt_price_x96(tick_upper)
    
    # Convert amounts to scaled integers (consistent with our swap scaling)
    amount0_scaled = int(amount0 * 1e6)
    amount1_scaled = int(amount1 * 1e6)
    
    # L0 = (amount0 × √P_upper × √P_lower) / (√P_upper - √P_lower)
    numerator0 = mul_div(
        mul_div(amount0_scaled, sqrt_price_upper_x96, Q96),
        sqrt_price_lower_x96, Q96
    )
    denominator = sqrt_price_upper_x96 - sqrt_price_lower_x96
    L0 = mul_div(numerator0, Q96, denominator)
    
    # L1 = amount1 / (√P_upper - √P_lower)
    L1 = mul_div(amount1_scaled, Q96, denominator)
    
    # Return the minimum (balanced liquidity)
    return min(L0, L1)
```

### Key Liquidity Principles

1. **Token0 Formula**: `L0 = (amount0 × √P_upper × √P_lower) / (√P_upper - √P_lower)`
2. **Token1 Formula**: `L1 = amount1 / (√P_upper - √P_lower)`  
3. **Final Liquidity**: `L = min(L0, L1)` ensures balanced liquidity provision
4. **Proper Scaling**: Uses consistent 1e6 scaling with `mul_div()` functions for precision

This approach ensures that liquidity calculations are mathematically correct and produce realistic price impact for trades within concentrated ranges.

## Core Math Functions

### 1. Safe Math Operations

```python
def mul_div(a: int, b: int, denominator: int) -> int:
    """Multiply two numbers and divide by denominator with overflow protection"""
    return (a * b) // denominator

def mul_div_rounding_up(a: int, b: int, denominator: int) -> int:
    """Multiply and divide with rounding up"""
    return (a * b + denominator - 1) // denominator
```

### 2. Liquidity Delta Calculations

These functions calculate how much of each token is needed for a given amount of liquidity:

```python
def get_amount0_delta(sqrt_price_a_x96, sqrt_price_b_x96, liquidity, round_up=False):
    """Calculate amount0 delta for liquidity in price range"""
    # Formula: L * (sqrt(P_b) - sqrt(P_a)) / (sqrt(P_a) * sqrt(P_b))
    
def get_amount1_delta(sqrt_price_a_x96, sqrt_price_b_x96, liquidity, round_up=False):
    """Calculate amount1 delta for liquidity in price range"""
    # Formula: L * (sqrt(P_b) - sqrt(P_a))
```

### 3. Price Update Functions

```python
def get_next_sqrt_price_from_amount0_rounding_up(sqrt_price_x96, liquidity, amount, add):
    """Calculate next sqrt price from amount0 with proper rounding up"""
    # Adding amount0: sqrt_price decreases
    # Formula: L * sqrt_P / (L + amount0 * sqrt_P)
    
def get_next_sqrt_price_from_amount1_rounding_down(sqrt_price_x96, liquidity, amount, add):
    """Calculate next sqrt price from amount1 with proper rounding down"""
    # Adding amount1: sqrt_price increases
    # Formula: sqrt_P + amount1 / L
```

## Pool Implementation

### UniswapV3Pool Class

The main pool class that manages:
- Current price and tick
- Liquidity positions with multiple concentrated ranges
- Tick data structure with efficient bitmap lookup
- Trading operations with enhanced cross-tick support
- Chart integration and visualization data

```python
@dataclass
class UniswapV3Pool:
    pool_name: str  # "MOET:BTC" or "MOET:Yield_Token"
    total_liquidity: float  # Total pool size in USD
    btc_price: float = None # Set up based on simulation
    fee_tier: float = None  # Will be set based on pool type
    concentration: float = None  # Will be set based on pool type
    tick_spacing: int = None  # Will be set based on pool type
    
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
    max_swap_iterations: int = 1000
    debug_cross_tick: bool = False
    
    # Legacy fields for backward compatibility
    token0_reserve: Optional[float] = None  # MOET reserve (calculated from ticks)
    token1_reserve: Optional[float] = None  # BTC reserve (calculated from ticks)
```

### Pool Types and Configuration

#### MOET:BTC Pools
- **Fee Tier**: 0.3% (3000 pips)
- **Tick Spacing**: 60
- **Concentration**: 80% around peg
- **Use Case**: Volatile asset pairs

#### MOET:Yield Token Pools
- **Fee Tier**: 0.05% (500 pips)
- **Tick Spacing**: 10
- **Concentration**: 95% around peg
- **Use Case**: Stable, highly correlated assets

### Liquidity Position Management

```python
@dataclass
class Position:
    tick_lower: int  # Lower tick of the position
    tick_upper: int  # Upper tick of the position  
    liquidity: int   # Amount of liquidity in this position

@dataclass
class TickInfo:
    liquidity_gross: int = 0  # Total liquidity referencing this tick
    liquidity_net: int = 0   # Net liquidity change at this tick
    initialized: bool = False  # Whether this tick has been initialized

@dataclass
class TickBitmap:
    """Efficient tick finding using bitmap approach (O(1) instead of O(n))"""
    bitmap: Dict[int, int] = None  # word_index -> bitmap_word
    
    def next_initialized_tick(self, tick: int, tick_spacing: int, zero_for_one: bool) -> int:
        """Find next initialized tick using bitmap lookup"""
        
    def flip_tick(self, tick: int, tick_spacing: int):
        """Flip tick state in bitmap when liquidity is added/removed"""
```

### Efficient Tick Finding with TickBitmap

The implementation includes a sophisticated `TickBitmap` class that provides O(1) tick finding instead of O(n) linear search:

- **Bitmap Structure**: Uses 256-bit words to represent tick states
- **Word-based Lookup**: Divides tick space into 256-tick words for efficient searching
- **Direction-aware Search**: Handles both increasing and decreasing price directions
- **Memory Efficient**: Only stores non-zero words to minimize memory usage
- **Automatic Cleanup**: Removes empty words when ticks are deactivated

## Within-Range vs Cross-Range Trading Logic

### Intelligent Swap Routing

The implementation features a sophisticated two-phase swap system that optimizes for capital efficiency:

**Phase 1: Within-Range Calculation**
- First attempts to execute swaps within the current liquidity range
- Uses the whitepaper formula `Δ√P = Δy / L` for price impact calculation
- Preserves liquidity values (no range crossing needed)
- Provides minimal slippage for small to medium trades

**Phase 2: Cross-Range Execution**
- Only activates when trades exceed current range capacity
- Transitions between discrete liquidity ranges
- Updates liquidity values when crossing range boundaries
- Handles large trades that require multiple liquidity sources

### Within-Range Price Impact Calculation

```python
def calculate_within_range_price_impact(self, amount_in_scaled: int, current_sqrt_price_x96: int, liquidity: int, zero_for_one: bool) -> int:
    """Calculate price impact staying within current range using whitepaper formula: Δ√P = Δy / L"""
    
    if liquidity <= 0:
        return current_sqrt_price_x96
    
    # Calculate delta sqrt price using proper Uniswap V3 math
    delta_sqrt_price_scaled = mul_div(amount_in_scaled, Q96, liquidity)
    
    # Calculate new sqrt price based on swap direction
    if zero_for_one:
        # MOET -> YT: price decreases
        new_sqrt_price_x96 = current_sqrt_price_x96 - delta_sqrt_price_scaled
    else:
        # YT -> MOET: price increases  
        new_sqrt_price_x96 = current_sqrt_price_x96 + delta_sqrt_price_scaled
    
    # Ensure price stays within valid bounds
    return max(MIN_SQRT_RATIO, min(MAX_SQRT_RATIO, new_sqrt_price_x96))
```

### Swap Implementation

The core swap function intelligently routes trades for optimal execution:

```python
def swap(self, zero_for_one: bool, amount_specified: int, sqrt_price_limit_x96: int):
    """
    Execute a swap using proper Uniswap V3 math with intelligent routing
    
    Features:
    - Within-range optimization for small trades
    - Cross-range handling for large trades
    - Proper liquidity management
    - Minimal slippage for concentrated ranges
    
    Args:
        zero_for_one: True if swapping token0 for token1
        amount_specified: Amount to swap (positive for exact input)
        sqrt_price_limit_x96: Maximum price change allowed
    """
```

### Swap Step Calculation

The `compute_swap_step` function implements sophisticated logic to handle both within-range and cross-tick scenarios:

```python
def compute_swap_step(sqrt_price_current_x96, sqrt_price_target_x96, liquidity, amount_remaining, fee_pips):
    """
    Compute a single swap step using advanced Uniswap V3 logic
    
    Handles two scenarios:
    1. Range has enough liquidity (stays within current range)
    2. Range needs cross-tick transition (moves to next tick)
    
    Returns: (sqrt_price_next_x96, amount_in, amount_out, fee_amount)
    """
```

### Key Trading Features

1. **Exact Input Swaps**: Specify input amount, get variable output
2. **Exact Output Swaps**: Specify output amount, get variable input
3. **Price Limits**: Prevent excessive price impact
4. **Fee Calculation**: Automatic fee deduction based on pool tier
5. **Liquidity Updates**: Dynamic liquidity adjustment as price crosses ticks
6. **Cross-Tick Support**: Seamless handling of swaps across multiple price ranges

## Discrete Liquidity Ranges

### Three-Tier Liquidity Architecture

The implementation uses a sophisticated discrete liquidity system with three non-overlapping ranges:

#### Range 1: Concentrated Core (-100 to +100 ticks)
- **Liquidity Allocation**: 95% of total tokens for yield token pairs, 80% for BTC pairs
- **Price Range**: ±1.005% around the peg (0.99005 to 1.01005)
- **Purpose**: Handles majority of trading volume with minimal slippage
- **Tick Spacing**: 10 for yield tokens, 60 for BTC pairs

#### Range 2: Lower Wide Range (-1000 to -100 ticks)
- **Liquidity Allocation**: 2.5% of total tokens (half of remaining liquidity)
- **Price Range**: Approximately -10% to -1.005% (0.9048 to 0.99005)
- **Purpose**: Provides liquidity for downward price movements
- **Characteristics**: Higher slippage due to lower liquidity density

#### Range 3: Upper Wide Range (+100 to +1000 ticks)
- **Liquidity Allocation**: 2.5% of total tokens (half of remaining liquidity)
- **Price Range**: Approximately +1.005% to +10% (1.01005 to 1.1052)
- **Purpose**: Provides liquidity for upward price movements
- **Characteristics**: Higher slippage due to lower liquidity density

### Discrete Range Benefits

1. **Capital Efficiency**: Most liquidity concentrated where trading occurs
2. **Predictable Slippage**: Clear slippage tiers based on trade size
3. **Range Isolation**: Each range operates independently with its own liquidity
4. **Optimal Routing**: Small trades stay in concentrated range, large trades cross ranges

### Pool Initialization with Discrete Ranges

```python
def _initialize_yield_token_positions(self):
    """Initialize MOET:Yield Token concentrated liquidity positions using proper Uniswap V3 math"""
    
    # Calculate token amounts (not arbitrary scaling)
    total_amount0 = self.total_liquidity / 2  # $250k MOET
    total_amount1 = self.total_liquidity / 2  # $250k YT
    
    # Range 1: Concentrated (95% of tokens in -100 to +100 ticks)
    concentrated_amount0 = total_amount0 * self.concentration  # $237.5k MOET
    concentrated_amount1 = total_amount1 * self.concentration  # $237.5k YT
    concentrated_liquidity = self._calculate_liquidity_from_amounts(
        concentrated_amount0, concentrated_amount1, -100, +100
    )
    
    # Range 2 & 3: Discrete wide ranges (5% of tokens split equally)
    remaining_amount0 = total_amount0 * (1 - self.concentration) / 2  # $6.25k each
    remaining_amount1 = total_amount1 * (1 - self.concentration) / 2  # $6.25k each
    
    # Create three discrete, non-overlapping positions
    self._add_position(-100, +100, concentrated_liquidity)    # Range 1
    self._add_position(-1000, -100, lower_liquidity)          # Range 2  
    self._add_position(+100, +1000, upper_liquidity)          # Range 3
```

## Cross-Range Swap Mechanics

### Overview

When trades exceed the capacity of the current liquidity range, the system seamlessly transitions to cross-range execution, following the patterns described in the [Uniswap V3 Development Book](https://uniswapv3book.com/milestone_3/cross-tick-swaps.html).

### Two-Scenario Logic

The swap step calculation implements proper two-scenario logic:

**Scenario 1: Within-Range Swaps**
- Trade stays within current discrete liquidity range
- Uses concentrated liquidity for minimal price impact
- Liquidity values remain constant (no range crossing)
- Optimal for small to medium trades (typical rebalancing operations)

**Scenario 2: Cross-Range Swaps**
- Trade exceeds current range capacity
- Transitions to next discrete liquidity range
- Liquidity values update when crossing range boundaries
- Higher slippage due to lower liquidity density in wide ranges

### Cross-Range Scenarios Handled

#### 1. Concentrated Range Swaps (Most Common)
- Small to medium trades stay within Range 1 (-100 to +100 ticks)
- Uses 95% of available liquidity for minimal slippage
- Price moves smoothly within concentrated bounds
- Typical for rebalancing operations ($1k-$10k trades)

#### 2. Concentrated to Wide Range Transitions
- Large trades exceed concentrated range capacity
- Seamless transition from Range 1 to Range 2 or Range 3
- Slippage increases as liquidity density decreases
- Automatic liquidity updates when crossing range boundaries

#### 3. Wide Range Operations
- Very large trades operate entirely in wide ranges
- Higher slippage due to 2.5% liquidity allocation
- Price movements can be significant (1-10% range)
- Emergency liquidation scenarios

#### 4. Multi-Range Traversal
- Extremely large trades may cross multiple ranges
- Complex liquidity dynamics across discrete ranges
- Progressive slippage increase as ranges are exhausted
- Rare scenarios for typical protocol operations

### Robust Error Handling

The implementation includes comprehensive error handling:

```python
# Input validation
if liquidity == 0:
    return sqrt_price_current_x96, 0, 0, 0

# Mathematical error handling
try:
    # Complex calculations
except (ValueError, ZeroDivisionError):
    return sqrt_price_current_x96, 0, 0, 0
```

### Advanced Fee Calculation

The implementation provides sophisticated fee calculations:

```python
# Advanced fee calculation
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
```


## Slippage Calculation

### UniswapV3SlippageCalculator Class

 All `calculate_swap_slippage` calls now execute real swaps that permanently mutate the pool state, ensuring agents compete for shared liquidity.

```python
class UniswapV3SlippageCalculator:
    def calculate_swap_slippage(self, amount_in: float, token_in: str, concentrated_range: float = 0.2):
        """
        Calculate slippage for a swap using proper Uniswap V3 tick-based math
        
        This method now executes REAL swaps that permanently alter pool state.
        All agents share the same liquidity pools and affect each other's trading costs.
        
        Returns:
            Dict with swap details including slippage, fees, and price impact
        """
```

### Supported Swap Types

1. **MOET → BTC**: For debt repayment scenarios
2. **BTC → MOET**: For collateral conversion
3. **MOET → Yield Token**: For yield token acquisition
4. **Yield Token → MOET**: For yield token redemption

### Slippage Metrics

- **Slippage Amount**: Absolute difference between expected and actual output
- **Slippage Percentage**: Relative slippage as percentage of expected output
- **Price Impact**: How much the price moved due to the trade
- **Trading Fees**: Fees charged by the pool
- **Effective Liquidity**: Available liquidity at current price

## Pool Types and Configuration

### MOET:BTC Pool Configuration

```python
def create_moet_btc_pool():
    """
    Create a MOET:BTC Uniswap v3 pool with concentrated liquidity
    
    - 80% liquidity concentrated around current BTC price
    - 0.3% fee tier for volatile pairs
    - Tick spacing of 60 for price granularity
    """
```

### Yield Token Pool Configuration

```python
def create_yield_token_pool():
    """
    Create a MOET:Yield Token Uniswap v3 pool with concentrated liquidity
    
    - 95% liquidity concentrated around 1:1 peg
    - 0.05% fee tier for stable pairs
    - Tick spacing of 10 for tight price control
    """
```

## Usage Examples

### Basic Pool Creation (these are overwritten by custom analysis scripts at root of the repo)

```python
# Create a $500k MOET:BTC pool
pool = create_moet_btc_pool(
    pool_size_usd=500_000,
    btc_price=100_000,
    concentration=0.80
)

# Create a $1M yield token pool
yt_pool = create_yield_token_pool(
    pool_size_usd=1_000_000,
    concentration=0.95
)
```

### Cross-Tick Swap Example

```python
# Execute a large swap that will cross multiple ticks
amount_in, amount_out = pool.swap(
    zero_for_one=True,  # MOET -> BTC
    amount_specified=100_000 * 1_000_000,  # $100k (scaled 1e6)
    sqrt_price_limit_x96=0  # No price limit
)

print(f"Swapped ${amount_in/1e6:.2f} MOET for {amount_out/1e6:.6f} BTC")
```

### Fees and Slippage Handling

The implementation provides sophisticated handling of fees and slippage that are deeply intertwined in the swap execution process:

#### Fee Structure and Calculation

**Fee Tiers by Pool Type:**
- **MOET:BTC Pools**: 0.3% (3000 pips) 
- **MOET:Yield Token Pools**: 0.05% (500 pips)

**Fee Calculation Process:**
```python
# Convert fee tier to pips for internal calculations
fee_pips = int(self.fee_tier * 1000000)  # 0.003 becomes 3000 pips

# In compute_swap_step function:
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
```

#### How Fees and Slippage Interact

**1. Fee Impact on Slippage:**
- Fees are deducted from the input amount before the swap calculation
- This reduces the effective amount available for the swap

**2. Slippage Impact on Fees:**
- Slippage affects the final price achieved
- Higher slippage means the swap moves further from the current price

**3. Cross-Tick Fee Accumulation:**
```python
# When crossing multiple ticks, fees accumulate
total_fees = 0
for step in swap_steps:
    total_fees += step.fee_amount
    # Each step may have different liquidity and fee calculations
```

#### Practical Example: Rebalancing Cost Analysis

```python
# Calculate total cost of rebalancing including both fees and slippage
def calculate_rebalancing_cost_with_slippage(
    moet_amount: float, 
    pool_size_usd: float = 500_000,
    concentrated_range: float = 0.2,
    btc_price: float = 100_000.0
) -> Dict[str, float]:
    """
    Calculate the total cost of rebalancing including:
    - Trading fees (0.3% for MOET:BTC pools)
    - Slippage costs (price impact)
    - Cross-tick execution costs
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

# Usage example
cost = calculate_rebalancing_cost_with_slippage(
    moet_amount=50_000,  # $50k of MOET
    pool_size_usd=500_000,
    btc_price=100_000
)

print(f"Total Swap Cost: ${cost['total_swap_cost']:.2f}")
print(f"Trading Fees: ${cost['trading_fees']:.2f}")
print(f"Slippage Cost: ${cost['slippage_cost']:.2f}")
print(f"Price Impact: {cost['price_impact_percentage']:.4f}%")
```

#### Liquidation Cost Analysis with Fees and Slippage

```python
# Calculate liquidation cost including both fees and slippage
def calculate_liquidation_cost_with_slippage(
    collateral_btc_amount: float,
    btc_price: float,
    liquidation_percentage: float = 0.5,
    liquidation_bonus: float = 0.05,
    pool_size_usd: float = 500_000
) -> Dict[str, float]:
    """
    Calculate total liquidation cost including:
    - Liquidation bonus (paid to liquidator)
    - Trading fees (paid to pool)
    - Slippage costs (price impact)
    """
    
    # Amount of BTC to liquidate
    btc_to_liquidate = collateral_btc_amount * liquidation_percentage
    btc_value_to_liquidate = btc_to_liquidate * btc_price
    
    # Create pool state
    pool = create_moet_btc_pool(pool_size_usd, btc_price)
    calculator = UniswapV3SlippageCalculator(pool)
    
    # Calculate swap (BTC -> MOET for debt repayment)
    swap_result = calculator.calculate_swap_slippage(btc_value_to_liquidate, "BTC")
    
    # Liquidation bonus is applied on debt repaid value
    bonus_value = debt_repaid * liquidation_bonus
    
    # Total liquidation cost includes slippage, fees, and bonus
    total_cost = swap_result["slippage_amount"] + swap_result["trading_fees"] + bonus_value
    
    return {
        "btc_liquidated": btc_to_liquidate,
        "btc_value_liquidated": btc_value_to_liquidate,
        "moet_received": swap_result["amount_out"],
        "expected_moet_without_slippage": swap_result["expected_amount_out"],
        "slippage_cost": swap_result["slippage_amount"],
        "trading_fees": swap_result["trading_fees"],
        "liquidation_bonus_cost": bonus_value,
        "total_liquidation_cost": total_cost,
        "slippage_percentage": swap_result["slippage_percentage"],
        "price_impact_percentage": swap_result["price_impact_percentage"],
        "effective_liquidity": swap_result["effective_liquidity"]
    }

# Usage example
liquidation_cost = calculate_liquidation_cost_with_slippage(
    collateral_btc_amount=1.0,  # 1 BTC
    btc_price=100_000,
    liquidation_percentage=0.5,  # 50% liquidation
    liquidation_bonus=0.05,  # 5% bonus
    pool_size_usd=500_000
)

print(f"Total Liquidation Cost: ${liquidation_cost['total_liquidation_cost']:.2f}")
print(f"Liquidation Bonus: ${liquidation_cost['liquidation_bonus_cost']:.2f}")
print(f"Trading Fees: ${liquidation_cost['trading_fees']:.2f}")
print(f"Slippage Cost: ${liquidation_cost['slippage_cost']:.2f}")
```

### Slippage Analysis

```python
# Calculate slippage for swapping $10k of MOET to BTC
calculator = UniswapV3SlippageCalculator(pool)
result = calculator.calculate_swap_slippage(10_000, "MOET")

print(f"Slippage: {result['slippage_percentage']:.4f}%")
print(f"Price Impact: {result['price_impact_percentage']:.4f}%")
print(f"Trading Fees: ${result['trading_fees']:.2f}")
```

## Key Features

### 1. Authentic Uniswap V3 Mathematics
- Proper liquidity calculations using official whitepaper formulas
- Exact tick-based calculations with Q64.96 fixed-point arithmetic
- Official fee tiers and tick spacings for different asset pairs

### 2. Intelligent Swap Routing
- Within-range optimization for small trades (minimal slippage)
- Cross-range execution for large trades (progressive slippage)
- Two-phase system optimizes capital efficiency
- Proper liquidity management across discrete ranges

### 3. Discrete Concentrated Liquidity
- Three-tier architecture with non-overlapping ranges
- 95% concentration for yield token pairs (±1% range)
- 80% concentration for MOET:BTC pairs
- Predictable slippage tiers based on trade size

### 4. Realistic Price Impact Modeling
- Small trades: ~0.01% price impact (within concentrated range)
- Medium trades: Progressive slippage as ranges are crossed
- Large trades: Higher slippage in wide ranges (2.5% liquidity)
- Emergency scenarios: Multi-range traversal with compound slippage

### 5. Production-Ready Pool State Management
- Pool state mutations persist after swaps
- Shared liquidity pools across all agents
- Proper liquidity calculations from token amounts
- Real-time price and tick updates

### 6. Comprehensive Analysis Integration
- **Liquidity Distribution Data**: `get_liquidity_distribution()` provides price and liquidity arrays for charting
- **Tick Data for Charts**: `get_tick_data_for_charts()` returns formatted tick data with price labels
- **Range Visualization**: Built-in support for discrete liquidity range visualization
- **Slippage Analysis**: Real-time slippage calculation with proper Uniswap V3 math
- **Trading Cost Breakdown**: Detailed fee and slippage reporting for protocol analysis


## Integration with Tidal Protocol

This Uniswap V3 implementation is specifically designed for the Tidal Protocol simulation, providing:

### Core Protocol Functions
- **MOET Token Trading**: Efficient pricing and trading with minimal slippage
- **BTC Collateral Management**: Realistic liquidation cost analysis
- **Yield Token Operations**: Optimized 1:1 peg maintenance with 95% concentration
- **Rebalancing Operations**: Small trade optimization within concentrated ranges

### Advanced Capabilities
- **Realistic Slippage Modeling**: Proper price impact for different trade sizes
- **Multi-Agent Competition**: Shared liquidity pools with persistent state changes
- **Emergency Scenarios**: Cross-range execution for large liquidations
- **Production Accuracy**: Authentic Uniswap V3 mathematics ensure realistic results

### Expected Performance
- **Small Rebalancing Trades** ($1k-$5k): ~0.01% price impact, ~$3 total cost
- **Medium Operations** ($10k-$50k): Progressive slippage as ranges are utilized
- **Large Liquidations** ($100k+): Cross-range execution with higher but predictable slippage
- **Pool Sustainability**: Proper liquidity calculations prevent artificial pool failures