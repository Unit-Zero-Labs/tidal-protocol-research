# Yield Token System - Technical Documentation

## Overview

This document provides a comprehensive breakdown of the `yield_tokens.py` script, which implements a sophisticated yield-bearing token system for the Tidal Protocol simulation. The system enables High Tide agents to convert MOET into yield-bearing tokens that automatically accrue 10% APR and can be traded back to MOET for position rebalancing using advanced Uniswap V3 concentrated liquidity mathematics.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [System Architecture](#system-architecture)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Core Classes and Components](#core-classes-and-components)
5. [Yield Accrual System](#yield-accrual-system)
6. [Trading and Liquidity Management](#trading-and-liquidity-management)
7. [Portfolio Management](#portfolio-management)
8. [Integration with Uniswap V3](#integration-with-uniswap-v3)
9. [Usage Examples](#usage-examples)
10. [Performance Considerations](#performance-considerations)

## Core Concepts

### What are Yield Tokens?

Yield tokens are protocol-native rebasing assets that:
- Track MOET value closely (1:1 peg with minimal deviation)
- Automatically accrue yield at 10% APR through rebasing (same quantity, increasing value)
- Can be traded back to MOET for position rebalancing via engine-level real swaps
- Provide capital efficiency for High Tide agents

### Key Components

1. **YieldToken**: Individual rebasing yield token with continuous value appreciation
2. **YieldTokenManager**: Portfolio tracking and yield calculation for individual agents
3. **YieldTokenPool**: Global trading pool using Uniswap V3 math for real swap execution
4. **Engine Integration**: Real swap execution happens at the engine level, not portfolio level
5. **Rebasing Mechanism**: Continuous value increase without quantity change

## System Architecture

### Two-Layer Design

The yield token system operates on two distinct layers:

#### **Layer 1: Portfolio Management (YieldTokenManager)**
- **Purpose**: Track agent's yield token holdings and calculate portfolio values
- **Operations**: Minting tokens, calculating yields, portfolio analytics
- **Key Point**: Does NOT execute real swaps - only manages portfolio state

#### **Layer 2: Real Swap Execution (Engine + YieldTokenPool)**
- **Purpose**: Execute actual trades that affect shared liquidity pools
- **Operations**: Real Uniswap V3 swaps with slippage, fees, and pool state mutations
- **Key Point**: This is where actual trading happens with real economic impact

### Data Flow

```
Agent Needs MOET
       ↓
Engine._execute_yield_token_sale()
       ↓
Agent.execute_yield_token_sale()
       ↓
YieldTokenPool.execute_yield_token_sale()
       ↓
Uniswap V3 Pool Swap (Real)
       ↓
Pool State Updated + MOET Received
```

## Mathematical Foundations

### Continuous Yield Accrual

Yield tokens use continuous compound interest with per-minute precision:

```python
def get_current_value(self, current_minute: int) -> float:
    """Calculate current value including accrued yield (rebasing)"""
    if current_minute < self.creation_minute:
        return self.initial_value
        
    minutes_elapsed = current_minute - self.creation_minute
    # Convert APR to per-minute rate: APR * (minutes_elapsed / minutes_per_year)
    # For 10% APR over 60 minutes: 0.10 * (60 / 525600) = 0.0000114
    minutes_per_year = 365 * 24 * 60  # 525,600 minutes per year
    minute_rate = self.apr * (minutes_elapsed / minutes_per_year)
    return self.initial_value * (1 + minute_rate)
```

### Key Mathematical Constants

- **APR**: 10% (0.10) annual percentage rate
- **Minutes per year**: 525,600 (365 days × 24 hours × 60 minutes)
- **Per-minute rate**: `0.10 * (minutes_elapsed / 525600)` for linear approximation
- **Pool concentration**: 95% liquidity at 1:1 peg for minimal slippage

### Yield Calculation Formula

For a yield token with initial value `initial_value`, APR `r`, and time elapsed `t` minutes:

```
Value = initial_value × (1 + r × t/525600)
Yield = Value - initial_value
```

This uses a linear approximation for per-minute yield accrual, which is computationally efficient while maintaining accuracy for short time periods.

## Core Classes and Components

### 1. YieldToken Class

Individual rebasing yield token with continuous value appreciation:

```python
@dataclass
class YieldToken:
    """Individual yield token with continuous yield accrual (rebasing)"""
    
    def __init__(self, initial_value: float, apr: float = 0.10, creation_minute: int = 0):
        self.initial_value = initial_value  # Value at creation time
        self.apr = apr
        self.creation_minute = creation_minute
```

#### Key Methods

- `get_current_value(current_minute)`: Calculate current value including accrued yield (rebasing)
- `get_sellable_value(current_minute)`: Get value after minimal slippage estimate

### 2. YieldTokenManager Class

Portfolio tracking for individual High Tide agents:

```python
class YieldTokenManager:
    """Manages yield token portfolio for a High Tide agent"""
    
    def __init__(self, yield_token_pool: Optional['YieldTokenPool'] = None):
        self.yield_tokens: List[YieldToken] = []
        self.total_initial_value_invested = 0.0
        self.yield_token_pool = yield_token_pool
```

#### Core Portfolio Operations

- `mint_yield_tokens(moet_amount, current_minute, use_direct_minting)`: Convert MOET to yield tokens with flexible creation methods
- `calculate_total_value(current_minute)`: Get total portfolio value including accrued yield
- `get_portfolio_summary(current_minute)`: Comprehensive portfolio analysis
- `_remove_yield_tokens(yield_tokens_to_remove, current_minute)`: Remove tokens from portfolio after successful sales

**Important**: The YieldTokenManager does NOT execute real swaps. It only tracks portfolio state. Real swaps happen at the engine level.

### 3. YieldTokenPool Class

Global trading pool using Uniswap V3 concentrated liquidity for real swap execution:

```python
class YieldTokenPool:
    """
    Global pool for MOET <-> Yield Token trading using Uniswap V3 math
    
    This represents the internal protocol pool with tight concentration (95% at peg)
    and configurable token ratios for optimal capital efficiency.
    """
```

#### Pool Configuration

- **Concentration**: 95% liquidity at 1:1 peg
- **Token Ratio**: Configurable MOET:YT ratios (e.g., 75:25, 50:50)
- **Fee Tier**: 0.05% (500 pips) for stable pairs
- **Tick Spacing**: 10 for tight price control
- **Price Range**: Asymmetric bounds optimized for target ratios

#### Real Swap Execution Methods

- `execute_yield_token_purchase(moet_amount)`: Execute real MOET → Yield Token swaps
- `execute_yield_token_sale(yield_token_value)`: Execute real Yield Token → MOET swaps
- `quote_yield_token_purchase(moet_amount)`: Get quote for MOET → Yield Token
- `quote_yield_token_sale(yield_token_value)`: Get quote for Yield Token → MOET

## Yield Accrual System

### Continuous Compound Interest

The system implements true continuous compound interest with minute-level precision. Yield tokens automatically increase in value over time without changing quantity (rebasing).

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

### Portfolio Value Calculation

```python
def calculate_total_value(self, current_minute: int) -> float:
    """Calculate total current value of all yield tokens"""
    return sum(
        token.get_current_value(current_minute) 
        for token in self.yield_tokens
    )
```

The portfolio manager tracks all yield tokens and calculates their total current value including accrued yield.

## Trading and Liquidity Management

### Real Swap Execution Architecture

All actual trading happens through the engine-pool integration:

1. **Agent requests swap** → Engine receives request
2. **Engine calls agent method** → Agent calculates portfolio impact  
3. **Agent calls pool execution** → Real Uniswap V3 swap occurs
4. **Pool state updates** → Liquidity and prices adjust
5. **Agent portfolio updates** → Tokens removed from portfolio

### Flexible Yield Token Creation

The system supports two distinct creation methods for yield tokens:

#### 1. Direct Minting (Minute 0)
For establishing balanced pool states without affecting liquidity:

```python
def mint_yield_tokens(self, moet_amount: float, current_minute: int, use_direct_minting: bool = False):
    """Convert MOET to yield tokens using 1:1 rate at minute 0 or Uniswap V3 pool"""
    if current_minute == 0 and use_direct_minting:
        # At minute 0, yield tokens are purchased at 1:1 rate
        # This establishes the initial $250k:$250k balanced pool
        actual_yield_tokens_received = moet_amount  # 1:1 rate
    else:
        # MINUTE > 0: Use Uniswap V3 pool with real slippage
        actual_yield_tokens_received = self.yield_token_pool.execute_yield_token_purchase(moet_amount)
```

#### 2. Uniswap V3 Pool Trading
For realistic market-based pricing with slippage and fees:

```python
# All subsequent purchases use Uniswap V3 pool
actual_yield_tokens_received = self.yield_token_pool.execute_yield_token_purchase(moet_amount)
```

### Uniswap V3 Integration

The yield token system leverages sophisticated Uniswap V3 mathematics throughout:

```python
def __init__(self, initial_moet_reserve: float = 250_000.0, concentration: float = 0.95):
    # Create the underlying Uniswap V3 pool
    pool_size_usd = initial_moet_reserve * 2  # Total pool size (both sides)
    self.uniswap_pool = create_yield_token_pool(
        pool_size_usd=pool_size_usd,
        concentration=concentration
    )
    
    # Create slippage calculator for accurate cost calculations
    self.slippage_calculator = UniswapV3SlippageCalculator(self.uniswap_pool)
```

### Real Trading Operations

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
def quote_yield_token_sale(self, yield_token_value: float) -> float:
    """Quote how much MOET will be received for selling yield tokens using Uniswap V3 math"""
    swap_result = self.slippage_calculator.calculate_swap_slippage(
        amount_in=yield_token_value,
        token_in="Yield_Token",
        concentrated_range=self.concentration_range
    )
    
    return swap_result.get("amount_out", 0.0)
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
        "total_initial_value": self.total_initial_value_invested,
        "total_current_value": total_value,
        "total_accrued_yield": total_yield,
        "yield_percentage": (total_yield / self.total_initial_value_invested * 100) if self.total_initial_value_invested > 0 else 0.0,
        "average_token_age_minutes": self._calculate_average_age(current_minute)
    }
```

### Portfolio Metrics

- **Total Initial Value**: Sum of all token initial values
- **Total Current Value**: Sum of all token current values including accrued yield
- **Total Accrued Yield**: Sum of all accrued yield across tokens
- **Yield Percentage**: Total yield as percentage of initial value
- **Average Token Age**: Average age of tokens in minutes
- **Number of Tokens**: Count of active yield tokens

### Pool State Tracking

`YieldTokenPool.get_pool_state()` returns comprehensive pool information:

```python
{
    "moet_reserve": float,           # Current MOET reserve
    "yield_token_reserve": float,    # Current Yield Token reserve
    "exchange_rate": float,          # Current exchange rate (equals current_price)
    "total_liquidity": float,        # Total active liquidity
    "concentration": float,          # Liquidity concentration (0.95)
    "fee_tier": float,              # Fee tier (0.0005 = 0.05%)
    "current_price": float,         # Current price from Uniswap V3
    "tick_current": int             # Current tick position
}
```

## Integration with Uniswap V3

### Pool Configuration

The yield token pool uses Uniswap V3's most stable configuration optimized for 1:1 peg assets with flexible token ratios:

```python
# Pool configuration for yield tokens
pool_config = {
    "fee_tier": 0.0005,      # 0.05% fee tier (lowest available)
    "tick_spacing": 10,       # 10 tick spacing for precision
    "concentration": 0.95,    # 95% liquidity concentrated at peg
    "token0_ratio": 0.75,     # 75% MOET, 25% YT (configurable)
    "price_range": "±1%"      # Optimized range around 1:1 peg
}
```

### Advanced Features

- **Concentrated Liquidity**: 95% of liquidity within ±0.1% of 1:1 peg
- **Cross-Tick Trading**: Support for complex scenarios with tick boundary crossings
- **Real-Time State Updates**: Pool state updates immediately after each swap
- **Slippage Calculation**: Accurate slippage based on actual pool liquidity

## Usage Examples

### Basic Yield Token Operations

```python
# Create yield token pool and manager with asymmetric 75/25 ratio
pool = YieldTokenPool(
    total_pool_size=500_000,
    token0_ratio=0.75,  # 75% MOET, 25% YT
    concentration=0.95
)
manager = YieldTokenManager(yield_token_pool=pool)

# Convert $10,000 MOET to yield tokens using direct minting (minute 0)
current_minute = 0
yield_tokens = manager.mint_yield_tokens(10_000, current_minute, use_direct_minting=True)

# Check portfolio after 1 hour (60 minutes)
portfolio = manager.get_portfolio_summary(60)
print(f"Total Value: ${portfolio['total_current_value']:.2f}")
print(f"Yield Accrued: ${portfolio['total_accrued_yield']:.2f}")
print(f"Yield Percentage: {portfolio['yield_percentage']:.4f}%")

# Subsequent purchases use Uniswap V3 pool with real slippage
yield_tokens_2 = manager.mint_yield_tokens(5_000, 30, use_direct_minting=False)
```

### Engine-Level Swap Execution

```python
# This is how real swaps happen in the system
class HighTideAgent:
    def execute_yield_token_sale(self, moet_amount_needed: float, current_minute: int) -> float:
        """Execute yield token sale for rebalancing using REAL pool execution"""
        
        # Calculate how much yield token value to sell
        yield_tokens_to_sell = self._calculate_yield_tokens_needed(moet_amount_needed)
        
        if yield_tokens_to_sell <= 0:
            return 0.0
        
        # Execute REAL swap through the pool (this affects shared liquidity)
        moet_raised = self.state.yield_token_manager.yield_token_pool.execute_yield_token_sale(
            yield_tokens_to_sell
        )
        
        if moet_raised > 0:
            # Remove sold tokens from portfolio
            self.state.yield_token_manager._remove_yield_tokens(yield_tokens_to_sell, current_minute)
            
            # Update agent's MOET balance and debt
            self.state.token_balances[Asset.MOET] += moet_raised
            
        return moet_raised
```

### Complete High Tide Agent Integration

```python
class HighTideAgent:
    def __init__(self, agent_id: str, initial_hf: float, target_hf: float):
        self.agent_id = agent_id
        self.target_hf = target_hf
        self.engine = None  # Set by engine during initialization
        
        # Initialize yield token manager (portfolio tracking only)
        self.state.yield_token_manager = YieldTokenManager()
    
    def invest_surplus_moet(self, moet_amount: float, current_minute: int):
        """Invest surplus MOET in yield tokens"""
        # Use direct minting at minute 0, Uniswap V3 trading afterward
        use_direct_minting = (current_minute == 0)
        
        yield_tokens = self.state.yield_token_manager.mint_yield_tokens(
            moet_amount, current_minute, use_direct_minting
        )
        
        print(f"Agent {self.agent_id}: Invested ${moet_amount} in {len(yield_tokens)} yield tokens")
        return yield_tokens
    
    def rebalance_position(self, current_minute: int):
        """Rebalance position by selling yield tokens via engine"""
        if not self.engine:
            print(f"⚠️  WARNING: Agent {self.agent_id} has no engine reference!")
            return 0.0
        
        # Calculate MOET needed for rebalancing
        moet_needed = self._calculate_moet_needed_for_target_hf()
        
        # Execute real swap through engine
        success, swap_data = self.engine._execute_yield_token_sale(
            self, 
            {"moet_needed": moet_needed, "swap_type": "rebalancing"}, 
            current_minute
        )
        
        if success and swap_data:
            moet_received = swap_data.get("moet_received", 0.0)
            print(f"Agent {self.agent_id}: Raised ${moet_received:.2f} MOET from yield tokens")
            return moet_received
        
        return 0.0
```

### Pool Trading Operations

```python
# Create yield token pool with configurable ratio
pool = YieldTokenPool(
    total_pool_size=500_000,
    token0_ratio=0.75,  # 75% MOET, 25% YT
    concentration=0.95
)

# Quote yield token purchase
quote = pool.quote_yield_token_purchase(10_000)
print(f"Yield tokens for $10k MOET: {quote:.2f}")

# Execute purchase (real swap with pool state mutation)
yield_tokens_received = pool.execute_yield_token_purchase(10_000)
print(f"Yield tokens received: {yield_tokens_received:.2f}")

# Quote yield token sale
sale_quote = pool.quote_yield_token_sale(5_000)
print(f"MOET for $5k yield tokens: ${sale_quote:.2f}")

# Execute sale (real swap with pool state mutation)
moet_received = pool.execute_yield_token_sale(5_000)
print(f"MOET received: ${moet_received:.2f}")

# Get updated pool state
state = pool.get_pool_state()
print(f"Pool State: {state}")
```

## Key Features Summary

### 1. Rebasing Yield Accrual
- True compound interest with minute-level precision
- 10% APR with automatic value appreciation (rebasing)
- Tokens maintain same quantity but increase in value over time

### 2. Two-Layer Architecture
- **Portfolio Layer**: Tracks holdings and calculates yields (YieldTokenManager)
- **Execution Layer**: Performs real swaps with economic impact (Engine + YieldTokenPool)

### 3. Complete Uniswap V3 Integration
- Sophisticated concentrated liquidity mathematics
- Real-time slippage calculation based on actual pool state
- Cross-tick trading support for complex scenarios
- Shared liquidity pools that agents compete for

### 4. Real Economic Impact
- All swaps affect shared liquidity pools
- Agents impact each other through pool state mutations
- Realistic slippage progression as liquidity depletes
- Authentic DeFi trading costs and constraints

### 5. Flexible Creation Methods
- **Direct minting** for establishing balanced pool states (minute 0)
- **Uniswap V3 trading** for realistic market-based pricing (minute > 0)
- **Configuration-driven** behavior for different simulation scenarios

### 6. High Tide Agent Support
- Surplus MOET investment with yield generation
- Position rebalancing via real swap execution
- Capital efficiency optimization through automated yield accrual
- Integration with engine-level swap recording and cost tracking

This system provides a realistic and economically accurate simulation of yield-bearing token trading within the Tidal Protocol ecosystem, with proper separation between portfolio management and actual swap execution.