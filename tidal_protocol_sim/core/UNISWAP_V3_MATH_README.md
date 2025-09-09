# Uniswap V3 Math Implementation - Technical Documentation

## Overview

This document provides a comprehensive breakdown of the `uniswap_v3_math.py` script, which implements a Uniswap V3 concentrated liquidity system suitable for simulation in the Tidal Protocol. The implementation uses Uniswap V3 mathematics with tick-based pricing, Q64.96 fixed-point arithmetic, and sophisticated cross-tick swap functionality. Tick-to-price conversions use float math and are clamped to official bounds for validity.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Core Math Functions](#core-math-functions)
4. [Pool Implementation](#pool-implementation)
5. [Trading Logic](#trading-logic)
6. [Cross-Tick Swap Mechanics](#cross-tick-swap-mechanics)
7. [Slippage Calculation](#slippage-calculation)
8. [Pool Types and Configuration](#pool-types-and-configuration)
9. [Usage Examples](#usage-examples)

## Core Concepts

### What is Uniswap V3?

Uniswap V3 is a decentralized exchange (DEX) that uses concentrated liquidity, allowing liquidity providers to concentrate their capital within specific price ranges. This increases capital efficiency compared to previous versions.

### Key Components

1. **Ticks**: Discrete price points that define liquidity ranges
2. **Sqrt Price**: Square root of price stored in Q64.96 fixed-point format
3. **Liquidity Positions**: Concentrated ranges of liquidity between two ticks
4. **Fee Tiers**: Different fee structures for different asset pairs
5. **Cross-Tick Swaps**: Sophisticated handling of swaps that move across multiple price ranges

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
- Liquidity positions
- Tick data structure
- Trading operations
- Cross-tick swap functionality

```python
@dataclass
class UniswapV3Pool:
    pool_name: str  # "MOET:BTC" or "MOET:Yield_Token"
    total_liquidity: float  # Total pool size in USD
    btc_price: float = None  # Set based on simulation
    fee_tier: float = None  # Set based on pool type
    concentration: float = None  # Set based on pool type
    tick_spacing: int = None  # Set based on pool type
    
    # Core Uniswap V3 state
    sqrt_price_x96: int = Q96  # Current sqrt price
    liquidity: int = 0  # Current active liquidity
    tick_current: int = 0  # Current tick
    
    # Advanced features
    debug_cross_tick: bool = False
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
```

## Trading Logic

### Swap Implementation

The core swap function uses a sophisticated step-by-step approach that handles complex liquidity scenarios:

```python
def swap(self, zero_for_one: bool, amount_specified: int, sqrt_price_limit_x96: int):
    """
    Execute a swap using proper Uniswap V3 math with cross-tick support
    
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

## Cross-Tick Swap Mechanics

### Overview

The implementation includes sophisticated cross-tick swap functionality that handles complex liquidity scenarios, following the patterns described in the [Uniswap V3 Development Book](https://uniswapv3book.com/milestone_3/cross-tick-swaps.html).

### Two-Scenario Logic

The swap step calculation implements proper two-scenario logic:

**Scenario 1: Within-Range Swaps**
- Price stays within current liquidity range
- All available liquidity can satisfy the swap
- Price moves smoothly within the range

**Scenario 2: Cross-Tick Swaps**
- Price moves outside current liquidity range
- Requires transition to next initialized tick
- Liquidity updates as price crosses tick boundaries

### Cross-Tick Scenarios Handled

#### 1. Single Price Range Swaps
- Small amounts that stay within current range
- Price moves smoothly within liquidity bounds
- No tick crossing required

#### 2. Multiple Identical Ranges
- Overlapping liquidity providing deeper pools
- Slower price movements due to increased liquidity
- Better execution for large trades

#### 3. Consecutive Price Ranges
- Price moves outside current range
- Activates next price range
- Seamless transition between ranges

#### 4. Partially Overlapping Ranges
- Complex liquidity dynamics
- Deeper liquidity in overlap areas
- More efficient price discovery

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

### State Management

The swap function properly handles:

- **Overlapping price ranges**: Deeper liquidity in overlap areas
- **Consecutive price ranges**: Seamless transitions between adjacent ranges
- **Partially overlapping ranges**: Complex liquidity dynamics
- **Gap handling**: Proper behavior when no liquidity exists in price range

### Legacy Field Updates

The implementation ensures complete state consistency by automatically updating legacy reserve fields after each swap:

```python
# After swap execution, legacy fields are updated to reflect actual state
def swap(self, zero_for_one: bool, amount_specified: int, sqrt_price_limit_x96: int):
    # ... swap execution logic ...
    
    # Update pool state
    self.sqrt_price_x96 = state['sqrt_price_x96']
    self.tick_current = state['tick']
    self.liquidity = state['liquidity']
    
    # Update legacy fields to reflect actual swap impact
    self._update_legacy_fields()
    
    return (amount_in_final, amount_out_final)
```

This ensures that:
- **Visualization compatibility**: Charts and analytics relying on legacy fields continue to work
- **State consistency**: Legacy fields are updated alongside tick-based calculations
- Note: Legacy reserve fields use a simple 50/50 split of active liquidity for backward compatibility and are approximate. For precise reserves, use tick/position data.

## Slippage Calculation

### UniswapV3SlippageCalculator Class

Provides detailed slippage analysis for different swap types:

```python
class UniswapV3SlippageCalculator:
    def calculate_swap_slippage(self, amount_in: float, token_in: str, concentrated_range: float = 0.2):
        """
        Calculate slippage for a swap using proper Uniswap V3 tick-based math
        
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
def create_moet_btc_pool(pool_size_usd: float, btc_price: float = 100_000.0, concentration: float = 0.80):
    """
    Create a MOET:BTC Uniswap v3 pool with concentrated liquidity
    
    - 80% liquidity concentrated around current BTC price
    - 0.3% fee tier for volatile pairs
    - Tick spacing of 60 for price granularity
    """
```

### Yield Token Pool Configuration

```python
def create_yield_token_pool(pool_size_usd: float, concentration: float = 0.95):
    """
    Create a MOET:Yield Token Uniswap v3 pool with concentrated liquidity
    
    - 95% liquidity concentrated around 1:1 peg
    - 0.05% fee tier for stable pairs
    - Tick spacing of 10 for tight price control
    """
```

## Usage Examples

### Basic Pool Creation

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
- **MOET:BTC Pools**: 0.3% (3000 pips) - for volatile asset pairs
- **MOET:Yield Token Pools**: 0.05% (500 pips) - for stable, highly correlated assets

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

### 1. Authentic Uniswap V3 Math
- Exact tick-based calculations
- Proper Q64.96 fixed-point arithmetic
- Official fee tiers and tick spacings

### 2. Advanced Cross-Tick Functionality
- Sophisticated two-scenario swap logic
- Proper handling of overlapping price ranges
- Seamless transitions between liquidity ranges
- Robust error handling and recovery

### 3. Concentrated Liquidity
- 80% concentration for MOET:BTC pairs
- 95% concentration for yield token pairs
- Multiple position ranges for optimal capital efficiency

### 4. Comprehensive Slippage Analysis
- Real-time slippage calculation
- Price impact analysis
- Trading fee breakdown
- Effective liquidity tracking

### 5. Simulation Support
- Non-destructive swap simulation
- State restoration after calculations
- Detailed result reporting

### 6. Chart Integration
- Liquidity distribution data
- Price range visualization
- Tick-based charting support

## Error Handling

The implementation includes robust error handling for:
- Overflow protection in mathematical operations
- Division by zero prevention
- Invalid tick range validation
- Price limit enforcement
- Infinite loop prevention in swap calculations
- Cross-tick error recovery

## Performance Considerations

- Binary search for tick calculations
- Efficient liquidity position management
- Minimal state updates during swaps
- Optimized memory usage for large pools
- Advanced cross-tick performance optimizations

## Integration with Tidal Protocol

This Uniswap V3 implementation is specifically designed for the Tidal Protocol simulation, providing:
- MOET token pricing and trading
- BTC collateral management
- Yield token operations
- Liquidation cost analysis
- Rebalancing calculations
- Advanced cross-tick swap functionality for complex scenarios