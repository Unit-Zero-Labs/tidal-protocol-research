# Yield Token System - Technical Documentation

## Overview

This document provides a comprehensive breakdown of the `yield_tokens.py` script, which implements a sophisticated yield-bearing token system for the Tidal Protocol simulation. The system enables High Tide agents to convert MOET into yield-bearing tokens that automatically accrue 10% APR and can be traded back to MOET for position rebalancing using advanced Uniswap V3 concentrated liquidity mathematics.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Core Classes and Components](#core-classes-and-components)
4. [Yield Accrual System](#yield-accrual-system)
5. [Trading and Liquidity Management](#trading-and-liquidity-management)
6. [Portfolio Management](#portfolio-management)
7. [Integration with Uniswap V3](#integration-with-uniswap-v3)
8. [Usage Examples](#usage-examples)
9. [Performance Considerations](#performance-considerations)

## Core Concepts

### What are Yield Tokens?

Yield tokens are protocol-native assets that:
- Track MOET value closely (1:1 peg with minimal deviation)
- Automatically accrue yield at 10% APR
- Can be traded back to MOET for position rebalancing
- Provide capital efficiency for High Tide agents

### Key Components

1. **YieldToken**: Individual yield-bearing token with continuous accrual
2. **YieldTokenManager**: Portfolio management for individual agents
3. **YieldTokenPool**: Global trading pool using Uniswap V3 math
4. **Yield Accrual**: Continuous compound interest calculation
5. **Trading Interface**: MOET ↔ Yield Token conversions

## Mathematical Foundations

### Continuous Yield Accrual

Yield tokens use continuous compound interest with per-minute precision:

```python
def get_current_value(self, current_minute: int) -> float:
    """Calculate current value including accrued yield"""
    minutes_elapsed = current_minute - self.creation_minute
    # Convert APR to per-minute rate: (1 + APR)^(1/525600) - 1
    minute_rate = (1 + self.apr) ** (1 / 525600) - 1
    return self.principal * (1 + minute_rate) ** minutes_elapsed
```

### Key Mathematical Constants

- **APR**: 10% (0.10) annual percentage rate
- **Minutes per year**: 525,600 (365.25 days × 24 hours × 60 minutes)
- **Per-minute rate**: `(1 + 0.10)^(1/525600) - 1 ≈ 0.0000181%`
- **Slippage factor**: 0.1% (0.999) for trading operations

### Yield Calculation Formula

For a yield token with principal `P`, APR `r`, and time elapsed `t` minutes:

```
Value = P × (1 + r)^(t/525600)
Yield = Value - P
```

## Core Classes and Components

### 1. YieldToken Class

Individual yield-bearing token with continuous accrual:

```python
@dataclass
class YieldToken:
    """Individual yield token with continuous yield accrual"""
    
    def __init__(self, principal_amount: float, apr: float = 0.05, creation_minute: int = 0):
        self.principal = principal_amount
        self.apr = apr
        self.creation_minute = creation_minute
```

#### Key Methods

- `get_current_value(current_minute)`: Calculate total value including yield
- `get_accrued_yield(current_minute)`: Get yield earned above principal
- `get_sellable_value(current_minute)`: Get value after slippage

### 2. YieldTokenManager Class

Portfolio management for individual High Tide agents with Uniswap V3 integration:

```python
class YieldTokenManager:
    """Manages yield token portfolio for a High Tide agent"""
    
    def __init__(self, yield_token_pool: Optional['YieldTokenPool'] = None):
        self.yield_tokens: List[YieldToken] = []
        self.total_principal_invested = 0.0
        self.yield_token_pool = yield_token_pool
```

#### Core Portfolio Operations

- `mint_yield_tokens(moet_amount, current_minute)`: Convert MOET to yield tokens using Uniswap V3 pool
- `sell_yield_tokens(amount_needed, current_minute)`: Sell tokens to raise MOET with real slippage
- `sell_yield_above_principal(current_minute)`: Sell only accrued yield with real slippage
- `calculate_total_value(current_minute)`: Get total portfolio value
- `get_portfolio_summary(current_minute)`: Comprehensive portfolio analysis
- `_calculate_real_slippage(yield_token_value)`: Calculate real slippage using Uniswap V3 math

### 3. YieldTokenPool Class

Global trading pool using Uniswap V3 concentrated liquidity:

```python
class YieldTokenPool:
    """
    Global pool for MOET <-> Yield Token trading using Uniswap V3 math
    
    This represents the internal protocol pool with tight concentration (95% at peg).
    """
```

#### Pool Configuration

- **Concentration**: 95% liquidity at 1:1 peg
- **Fee Tier**: 0.05% (500 pips) for stable pairs
- **Tick Spacing**: 10 for tight price control
- **Price Range**: ±0.1% around 1:1 peg

## Yield Accrual System

### Continuous Compound Interest

The system implements true continuous compound interest with minute-level precision:

```python
def get_current_value(self, current_minute: int) -> float:
    """Calculate current value including accrued yield"""
    if current_minute < self.creation_minute:
        return self.principal
        
    minutes_elapsed = current_minute - self.creation_minute
    # Convert APR to per-minute rate: (1 + APR)^(1/525600) - 1
    minute_rate = (1 + self.apr) ** (1 / 525600) - 1
    return self.principal * (1 + minute_rate) ** minutes_elapsed
```

### Yield Accrual Examples

For a $1,000 yield token with 10% APR:

| Time Elapsed | Value | Yield Accrued | Yield % |
|--------------|-------|---------------|---------|
| 1 minute     | $1,000.00 | $0.00 | 0.000% |
| 1 hour       | $1,000.02 | $0.02 | 0.002% |
| 1 day        | $1,000.26 | $0.26 | 0.026% |
| 1 week       | $1,001.92 | $1.92 | 0.192% |
| 1 month      | $1,008.33 | $8.33 | 0.833% |
| 1 year       | $1,100.00 | $100.00 | 10.000% |

### Yield Selling Strategies

#### 1. Full Token Sale
Sell entire tokens to raise MOET using real Uniswap V3 slippage:
```python
def sell_yield_tokens(self, amount_needed: float, current_minute: int) -> float:
    """Sell yield tokens to raise MOET, prioritizing highest yield first"""
    # Sort by current value (highest yield first)
    self.yield_tokens.sort(
        key=lambda token: token.get_current_value(current_minute),
        reverse=True
    )
    
    # Use real Uniswap V3 slippage calculation instead of hardcoded values
    token_value = token.get_current_value(current_minute)
    real_moet_value = self._calculate_real_slippage(token_value)
```

#### 2. Yield-Only Sale
Sell only accrued yield, preserving principal with real Uniswap V3 slippage:
```python
def sell_yield_above_principal(self, current_minute: int) -> float:
    """Sell only the accrued yield portion, keeping principal intact"""
    for token in self.yield_tokens:
        accrued_yield = token.get_accrued_yield(current_minute)
        if accrued_yield > 0:
            # Use real Uniswap V3 slippage calculation instead of hardcoded values
            real_moet_value = self._calculate_real_slippage(accrued_yield)
            
            # Reset token to current principal value minus yield sold
            token.principal = token.get_current_value(current_minute) - accrued_yield
            token.creation_minute = current_minute
```

## Trading and Liquidity Management

### Uniswap V3 Integration

The yield token system leverages sophisticated Uniswap V3 mathematics throughout:

```python
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
```

The system ensures complete state consistency by using Uniswap V3 math for all trading operations, including:
- Real-time slippage calculation based on actual pool liquidity
- Accurate pricing that reflects current market conditions
- State updates that maintain consistency between agent portfolios and pool state

### Trading Operations

#### 1. MOET → Yield Token Purchase

```python
def execute_yield_token_purchase(self, moet_amount: float) -> float:
    """Execute purchase of yield tokens with MOET using Uniswap V3 swaps"""
    # Execute actual Uniswap V3 swap: MOET -> Yield Token
    amount_in_scaled = int(moet_amount * 1e6)
    
    amount_in_actual, amount_out_actual = self.uniswap_pool.swap(
        zero_for_one=True,  # MOET (token0) -> Yield Token (token1)
        amount_specified=amount_in_scaled,
        sqrt_price_limit_x96=MIN_SQRT_RATIO + 1
    )
    
    return amount_out_actual / 1e6
```

#### 2. Yield Token → MOET Sale

```python
def execute_yield_token_sale(self, yield_token_value: float) -> float:
    """Execute sale of yield tokens for MOET using Uniswap V3 swaps"""
    # Execute actual Uniswap V3 swap: Yield Token -> MOET
    amount_in_scaled = int(yield_token_value * 1e6)
    
    amount_in_actual, amount_out_actual = self.uniswap_pool.swap(
        zero_for_one=False,  # Yield Token (token1) -> MOET (token0)
        amount_specified=amount_in_scaled,
        sqrt_price_limit_x96=MAX_SQRT_RATIO - 1
    )
    
    return amount_out_actual / 1e6
```

### Slippage and Fee Management

The system includes sophisticated slippage calculation using real Uniswap V3 mathematics:

```python
def quote_yield_token_purchase(self, moet_amount: float) -> float:
    """Quote how many yield tokens can be purchased with MOET using Uniswap V3 math"""
    swap_result = self.slippage_calculator.calculate_swap_slippage(
        amount_in=moet_amount,
        token_in="MOET",
        concentrated_range=self.concentration_range
    )
    
    return swap_result.get("amount_out", 0.0)

def _calculate_real_slippage(self, yield_token_value: float) -> float:
    """Calculate real slippage using Uniswap V3 math"""
    if not self.yield_token_pool or yield_token_value <= 0:
        return yield_token_value * 0.999  # Fallback if no pool available
    
    # Use Uniswap V3 slippage calculator for real pricing
    swap_result = self.yield_token_pool.slippage_calculator.calculate_swap_slippage(
        amount_in=yield_token_value,
        token_in="Yield_Token",
        concentrated_range=self.yield_token_pool.concentration_range
    )
    
    return swap_result.get("amount_out", yield_token_value * 0.999)
```

### Both Directions Available:

#### 1. **MOET → Yield Token** (what you saw)
```python
def quote_yield_token_purchase(self, moet_amount: float) -> float:
    # How many yield tokens can I get for $X MOET?
    swap_result = self.slippage_calculator.calculate_swap_slippage(
        amount_in=moet_amount,
        token_in="MOET",  # ← Input token
        concentrated_range=self.concentration_range
    )
    return swap_result.get("amount_out", 0.0)  # ← Yield tokens out
```

#### 2. **Yield Token → MOET** (the reverse)
```python
def quote_yield_token_sale(self, yield_token_value: float) -> float:
    # How much MOET can I get for $X worth of yield tokens?
    swap_result = self.slippage_calculator.calculate_swap_slippage(
        amount_in=yield_token_value,
        token_in="Yield_Token",  # ← Input token
        concentrated_range=self.concentration_range
    )
    return swap_result.get("amount_out", 0.0)  # ← MOET out
```

## Portfolio Management

### Portfolio Analytics

The `YieldTokenManager` provides comprehensive portfolio analytics:

```python
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
```

### Portfolio Metrics

- **Total Principal**: Sum of all token principals
- **Total Current Value**: Sum of all token current values
- **Total Accrued Yield**: Sum of all accrued yield
- **Yield Percentage**: Total yield as percentage of principal
- **Average Token Age**: Average age of tokens in minutes
- **Number of Tokens**: Count of active yield tokens

## Integration with Uniswap V3

### Pool Configuration

The yield token pool uses Uniswap V3's most stable configuration:

```python
# Pool configuration for yield tokens
pool_config = {
    "fee_tier": 0.0005,      # 0.05% fee tier
    "tick_spacing": 10,       # 10 tick spacing
    "concentration": 0.95,    # 95% liquidity at peg
    "price_range": "±0.1%"    # Tight range around 1:1 peg
}
```

### Advanced Features

- **Concentrated Liquidity**: 95% of liquidity concentrated at 1:1 peg
- **Low Fees**: 0.05% fee tier for stable pairs
- **Tight Price Control**: 10 tick spacing for precise pricing
- **Cross-Tick Support**: Handles complex trading scenarios
- **Slippage Calculation**: Real-time cost analysis

## Usage Examples

### Basic Yield Token Operations

```python
# Create yield token pool and manager
pool = YieldTokenPool(initial_moet_reserve=500_000)
manager = YieldTokenManager(yield_token_pool=pool)

# Convert $10,000 MOET to yield tokens using Uniswap V3 pool
current_minute = 0
yield_tokens = manager.mint_yield_tokens(10_000, current_minute)

# Check portfolio after 1 hour (60 minutes)
portfolio = manager.get_portfolio_summary(60)
print(f"Total Value: ${portfolio['total_current_value']:.2f}")
print(f"Yield Accrued: ${portfolio['total_accrued_yield']:.2f}")
print(f"Yield Percentage: {portfolio['yield_percentage']:.4f}%")
```

### Advanced Portfolio Management

```python
# Sell only accrued yield (preserve principal) using real Uniswap V3 slippage
yield_raised = manager.sell_yield_above_principal(60)
print(f"Yield raised: ${yield_raised:.2f}")

# Sell tokens to raise specific amount using real Uniswap V3 slippage
moet_raised = manager.sell_yield_tokens(5_000, 60)
```

### Error Handling Example

```python
# The system will fail fast with clear error messages
try:
    # This will work - proper setup with pool
    pool = YieldTokenPool(initial_moet_reserve=500_000)
    manager = YieldTokenManager(yield_token_pool=pool)
    tokens = manager.mint_yield_tokens(1000, 0)
    
except ValueError as e:
    print(f"Error: {e}")
    # Handle the error appropriately

# This will fail - no pool provided
try:
    manager = YieldTokenManager()  # No pool!
    manager.mint_yield_tokens(1000, 0)
except ValueError as e:
    print(f"Expected error: {e}")
    # Output: "YieldTokenManager requires a YieldTokenPool for minting yield tokens"
```

### Pool Integration Example

```python
# Proper integration with High Tide agents
pool = YieldTokenPool(initial_moet_reserve=500_000)
manager = YieldTokenManager(yield_token_pool=pool)

# All operations now use real Uniswap V3 math
moet_raised = manager.sell_yield_tokens(5_000, 60)
print(f"MOET raised: ${moet_raised:.2f}")

# Get detailed portfolio analysis
summary = manager.get_portfolio_summary(60)
print(f"Portfolio Summary: {summary}")
```

### Pool Trading Operations

```python
# Create yield token pool
pool = YieldTokenPool(initial_moet_reserve=500_000)

# Quote yield token purchase
quote = pool.quote_yield_token_purchase(10_000)
print(f"Yield tokens for $10k MOET: {quote:.2f}")

# Execute purchase
yield_tokens_received = pool.execute_yield_token_purchase(10_000)
print(f"Yield tokens received: {yield_tokens_received:.2f}")

# Get pool state
state = pool.get_pool_state()
print(f"Pool State: {state}")
```

### Complete High Tide Agent Integration

- **Note** : Base simulation assumes MOET:YieldToken LP Pool is balanced at peg after yield tokens are initially purchased via High Tide; thus, mint_yield_tokens is created for simplicity. 

```python
class HighTideAgent:
    def __init__(self):
        self.pool = YieldTokenPool()
        self.yield_manager = YieldTokenManager(yield_token_pool=self.pool)
    
    def invest_surplus_moet(self, moet_amount: float, current_minute: int):
        """Invest surplus MOET in yield tokens"""
        # Convert MOET to yield tokens
        yield_tokens = self.yield_manager.mint_yield_tokens(moet_amount, current_minute)
        
        # Log investment
        print(f"Invested ${moet_amount} in {len(yield_tokens)} yield tokens")
        
        return yield_tokens
    
    def rebalance_position(self, moet_needed: float, current_minute: int):
        """Rebalance position by selling yield tokens"""
        # Try to raise MOET from yield tokens
        moet_raised = self.yield_manager.sell_yield_tokens(moet_needed, current_minute)
        
        if moet_raised < moet_needed:
            # If not enough, sell only yield to preserve principal
            yield_raised = self.yield_manager.sell_yield_above_principal(current_minute)
            total_raised = moet_raised + yield_raised
            
            print(f"Raised ${total_raised:.2f} MOET from yield tokens")
            return total_raised
        
        print(f"Raised ${moet_raised:.2f} MOET from yield tokens")
        return moet_raised
```

## Performance Considerations

### Computational Efficiency

- **Minute-level precision**: Efficient per-minute yield calculations
- **Uniswap V3 integration**: Leverages efficient concentrated liquidity math
- **Memory management**: Efficient token storage and cleanup

### Scalability Features

- **Batch operations**: Efficient handling of multiple tokens
- **Lazy evaluation**: Calculations performed only when needed
- **State caching**: Cached portfolio summaries for performance
- **Error handling**: Robust error recovery for trading operations

## Error Handling

The system implements **fail-fast error handling** to ensure data integrity and prevent silent failures:

### Core Principle: Fail Fast, Fail Clear

Instead of using hardcoded fallbacks, the system throws descriptive errors when Uniswap V3 math fails:

```python
def _calculate_real_slippage(self, yield_token_value: float) -> float:
    """Calculate real slippage using Uniswap V3 math"""
    if not self.yield_token_pool:
        raise ValueError("YieldTokenManager requires a YieldTokenPool for accurate slippage calculations")
    
    if yield_token_value <= 0:
        raise ValueError(f"Invalid yield token value for slippage calculation: {yield_token_value}")
    
    # Use Uniswap V3 slippage calculator for real pricing
    swap_result = self.yield_token_pool.slippage_calculator.calculate_swap_slippage(...)
    
    # Validate the swap result
    if "amount_out" not in swap_result or swap_result["amount_out"] is None:
        raise ValueError(f"Uniswap V3 slippage calculation failed for yield token value {yield_token_value}: {swap_result}")
    
    return swap_result["amount_out"]
```

### Error Scenarios That Cause System Failure

- **Missing YieldTokenPool**: `YieldTokenManager` requires a pool for all operations
- **Invalid input values**: Zero or negative amounts are rejected
- **Uniswap V3 calculation failures**: Invalid swap results cause immediate failure
- **State inconsistencies**: Missing or corrupted pool state triggers errors

### Benefits of Fail-Fast Approach

- **Data Integrity**: Prevents incorrect calculations from propagating
- **Clear Debugging**: Descriptive error messages pinpoint exact issues
- **No Silent Failures**: All problems are immediately visible
- **Consistent State**: Ensures all operations use valid Uniswap V3 math

## Integration with Tidal Protocol

This yield token system is specifically designed for the Tidal Protocol simulation, providing:

- **High Tide agent support**: Yield-bearing token portfolio management
- **Position rebalancing**: Flexible MOET raising strategies
- **Capital efficiency**: 10% APR on idle MOET
- **Risk management**: Principal preservation options
- **Protocol integration**: Seamless Uniswap V3 trading

The system integrates seamlessly with the broader Tidal Protocol simulation framework, providing sophisticated yield token functionality for comprehensive protocol analysis and realistic market simulations.

## Key Features Summary

### 1. Continuous Yield Accrual
- True compound interest with minute-level precision
- 10% APR with automatic accrual
- Flexible yield selling strategies

### 2. Advanced Portfolio Management
- Individual token tracking
- Portfolio analytics and reporting
- Optimal token selection algorithms

### 3. Complete Uniswap V3 Integration
- Sophisticated concentrated liquidity mathematics throughout the system
- Real-time slippage calculation based on actual pool state
- Cross-tick trading support for complex scenarios
- State consistency between agent portfolios and pool state
- **Fail-fast error handling** ensures data integrity and prevents silent failures

### 4. Flexible Trading Operations
- MOET ↔ Yield Token conversions using real Uniswap V3 swaps
- Quote and execute operations with accurate pricing
- **Fail-fast error handling** with descriptive error messages

### 5. High Tide Agent Support
- Surplus MOET investment
- Position rebalancing capabilities
- Capital efficiency optimization
