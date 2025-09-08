# Uniswap V3 Math Implementation - Technical Documentation

## Overview

This document provides a comprehensive breakdown of the `uniswap_v3_math.py` script, which implements a complete Uniswap V3 concentrated liquidity system for the Tidal Protocol simulation. The implementation uses authentic Uniswap V3 mathematics with tick-based pricing, Q64.96 fixed-point arithmetic, and concentrated liquidity positions.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Core Math Functions](#core-math-functions)
4. [Pool Implementation](#pool-implementation)
5. [Trading Logic](#trading-logic)
6. [Slippage Calculation](#slippage-calculation)
7. [Pool Types and Configuration](#pool-types-and-configuration)
8. [Usage Examples](#usage-examples)

## Core Concepts

### What is Uniswap V3?

Uniswap V3 is a decentralized exchange (DEX) that uses concentrated liquidity, allowing liquidity providers to concentrate their capital within specific price ranges. This increases capital efficiency compared to previous versions.

### Key Components

1. **Ticks**: Discrete price points that define liquidity ranges
2. **Sqrt Price**: Square root of price stored in Q64.96 fixed-point format
3. **Liquidity Positions**: Concentrated ranges of liquidity between two ticks
4. **Fee Tiers**: Different fee structures for different asset pairs

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

```python
@dataclass
class UniswapV3Pool:
    pool_name: str  # "MOET:BTC" or "MOET:Yield_Token"
    total_liquidity: float  # Total pool size in USD
    btc_price: float = 100_000.0  # BTC price in USD
    fee_tier: float = None  # Set based on pool type
    concentration: float = None  # Set based on pool type
    tick_spacing: int = None  # Set based on pool type
    
    # Core Uniswap V3 state
    sqrt_price_x96: int = Q96  # Current sqrt price
    liquidity: int = 0  # Current active liquidity
    tick_current: int = 0  # Current tick
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

The core swap function uses a step-by-step approach:

```python
def swap(self, zero_for_one: bool, amount_specified: int, sqrt_price_limit_x96: int):
    """
    Execute a swap using proper Uniswap V3 math
    
    Args:
        zero_for_one: True if swapping token0 for token1
        amount_specified: Amount to swap (positive for exact input)
        sqrt_price_limit_x96: Maximum price change allowed
    """
```

### Swap Step Calculation

```python
def compute_swap_step(sqrt_price_current_x96, sqrt_price_target_x96, liquidity, amount_remaining, fee_pips):
    """
    Compute a single swap step using exact Uniswap V3 logic
    Returns: (sqrt_price_next_x96, amount_in, amount_out, fee_amount)
    """
```

### Key Trading Features

1. **Exact Input Swaps**: Specify input amount, get variable output
2. **Exact Output Swaps**: Specify output amount, get variable input
3. **Price Limits**: Prevent excessive price impact
4. **Fee Calculation**: Automatic fee deduction based on pool tier
5. **Liquidity Updates**: Dynamic liquidity adjustment as price crosses ticks

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
def create_yield_token_pool(pool_size_usd: float, btc_price: float = 100_000.0, concentration: float = 0.95):
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

### Slippage Analysis

```python
# Calculate slippage for swapping $10k of MOET to BTC
calculator = UniswapV3SlippageCalculator(pool)
result = calculator.calculate_swap_slippage(10_000, "MOET")

print(f"Slippage: {result['slippage_percentage']:.4f}%")
print(f"Price Impact: {result['price_impact_percentage']:.4f}%")
print(f"Trading Fees: ${result['trading_fees']:.2f}")
```

### Rebalancing Cost Calculation

```python
# Calculate total cost of rebalancing including slippage
cost = calculate_rebalancing_cost_with_slippage(
    moet_amount=50_000,  # $50k of MOET
    pool_size_usd=500_000,
    btc_price=100_000
)

print(f"Total Swap Cost: ${cost['total_swap_cost']:.2f}")
print(f"Slippage Cost: ${cost['slippage_cost']:.2f}")
print(f"Trading Fees: ${cost['trading_fees']:.2f}")
```

### Liquidation Cost Analysis

```python
# Calculate liquidation cost with slippage
liquidation_cost = calculate_liquidation_cost_with_slippage(
    collateral_btc_amount=1.0,  # 1 BTC
    btc_price=100_000,
    liquidation_percentage=0.5,  # 50% liquidation
    liquidation_bonus=0.05,  # 5% bonus
    pool_size_usd=500_000
)

print(f"Total Liquidation Cost: ${liquidation_cost['total_liquidation_cost']:.2f}")
print(f"Liquidation Bonus: ${liquidation_cost['liquidation_bonus_cost']:.2f}")
```

## Key Features

### 1. Authentic Uniswap V3 Math
- Exact tick-based calculations
- Proper Q64.96 fixed-point arithmetic
- Official fee tiers and tick spacings

### 2. Concentrated Liquidity
- 80% concentration for MOET:BTC pairs
- 95% concentration for yield token pairs
- Multiple position ranges for optimal capital efficiency

### 3. Comprehensive Slippage Analysis
- Real-time slippage calculation
- Price impact analysis
- Trading fee breakdown
- Effective liquidity tracking

### 4. Simulation Support
- Non-destructive swap simulation
- State restoration after calculations
- Detailed result reporting

### 5. Chart Integration
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

## Performance Considerations

- Binary search for tick calculations
- Efficient liquidity position management
- Minimal state updates during swaps
- Optimized memory usage for large pools

## Integration with Tidal Protocol

This Uniswap V3 implementation is specifically designed for the Tidal Protocol simulation, providing:
- MOET token pricing and trading
- BTC collateral management
- Yield token operations
- Liquidation cost analysis
- Rebalancing calculations

The system integrates seamlessly with the broader Tidal Protocol simulation framework, providing accurate DEX functionality for comprehensive protocol analysis.
