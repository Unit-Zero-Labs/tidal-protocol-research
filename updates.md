BELOW ARE INSTRUCTIONS FOR YOU ON IMPROVEMENTS RELATED TO THE UNISWAP MATH IN OUR TOKENOMICS SIMULATION SYSTEM. THIS DOCUMENT HAS SOME INFORMATION ABOUT THE SYSTEM  AND THEN IT DIVES INTO DETAILED STEPS AROUND WHERE WE NEED TO FIX THE INCORRECT UNISWAP MATH. READ THIS FILE THOROUGHLY:

Core components to understand:

High Tide is an automated yield aggregation layer built atop the Tidal Protocol, designed to enable crypto holders to deposit assets (BTC, ETH, FLOW, stables) and earn optimized in-kind yield without manual management. Unlike passive lending protocols that rely on forced liquidations, Tidal Protocol leverages active position management: automated rebalancing  proactively defend user health factors during market downturns, minimizing principal loss and maximizing capital efficiency. This allows High Tide to take advantage of this engine while offering users market-leading yield

Our simulation evaluates how different liquidity pool configurations—both external (MOET:BTC) and internal (MOET:Yield Token)—impact the protocol’s ability to execute rebalancing operations. We focus on three key metrics:

## **Technical Specification of High Tide Protocol**

### **2.1 Foundation: Tidal Protocol**

High Tide builds on the Tidal Protocol, an actively managed lending engine for the Flow blockchain. Core features include:

- **Collateral Supply:** Users deposit BTC as collateral
- **Stablecoin Borrowing:** MOET stablecoins are borrowed against collateral
- **Active Management:** Automated callbacks and health factor thresholds trigger position rebalancing, leveraging Flow’s scheduled transaction infrastructure

Tidal’s kinked interest rate model increases borrowing rates sharply above 80% pool utilization, incentivizing agents to maintain healthy positions and avoid costly liquidations.

### **2.2 High Tide Enhancement: Yield Token Strategy**

High Tide introduces an automated yield token strategy:

- **Immediate Yield Token Purchase:** Borrowed MOET is converted to yield-bearing tokens at 10% APR
- **Continuous Compounding:** Yield tokens accrue interest every minute
- **Health Factor Monitoring:** The system tracks each agent’s health factor (collateral value / debt value)
- **Active Rebalancing:** Yield tokens are automatically sold to repay debt when health factors fall below maintenance thresholds

**Liquidation as Last Resort:** Forced liquidation only occurs if yield token sales cannot restore solvency. Forced liquidations seize collateral and swap it for MOET to pay down DEBT



## **Structural Things:**

We need all of our simulations to create a simulation results json file that is used to create all of our charts. All charts must have the right data in the json file to meet our needs. No hallucination of data into the charts can occur. No hardcoded values or mock data should be implemented in order to achieve desirable results. The visualizations MUST use the system-generated simulation results.

## Comprehensive list of tasks to implement uniswap math improvements

I'll review the visualization methodologies in the target health factor analysis script and then provide a comprehensive system prompt for fixing the Uniswap V3 math implementation.

[3 tools called]

Now let me examine the core architecture dependencies:

[1 tool called]

Now I'll create a comprehensive system prompt for fixing the Uniswap V3 implementation:

## **System Prompt for Uniswap V3 Math Implementation Fix**

You are tasked with completely rewriting the Uniswap V3 math implementation in `tidal_protocol_sim/core/uniswap_v3_math.py` to use **actual Uniswap V3 mathematics** instead of the current discrete bin approximation. This is a critical fix that affects the entire simulation's accuracy.

### **Current Implementation Issues**
The existing code uses a **fundamentally flawed "discrete bins" approach** instead of Uniswap V3's continuous tick system with Q64.96 fixed-point arithmetic. This leads to inaccurate slippage calculations and misleading simulation results.

### **Required Implementation Changes**

#### **1. Replace Discrete Bins with Proper Tick System**
```python
# REMOVE: Current LiquidityBin class and discrete bin logic
# IMPLEMENT: Proper Uniswap V3 tick-based system

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Constants from Uniswap V3
MIN_TICK = -887272
MAX_TICK = 887272
Q96 = 2 ** 96
TICK_SPACING_0_3_PERCENT = 60  # For 0.3% fee tier

def tick_to_sqrt_price_x96(tick: int) -> int:
    """Convert tick to sqrt price in Q64.96 format"""
    sqrt_price = 1.0001 ** (tick / 2)
    return int(sqrt_price * Q96)

def sqrt_price_x96_to_tick(sqrt_price_x96: int) -> int:
    """Convert sqrt price X96 to tick"""
    sqrt_price = sqrt_price_x96 / Q96
    return int(math.log(sqrt_price ** 2) / math.log(1.0001))

def get_amount0_delta(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int, 
    liquidity: int
) -> int:
    """Calculate amount0 delta for liquidity in price range"""
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
    
    return int(liquidity * Q96 * (sqrt_price_b_x96 - sqrt_price_a_x96) // 
               (sqrt_price_b_x96 * sqrt_price_a_x96))

def get_amount1_delta(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    liquidity: int
) -> int:
    """Calculate amount1 delta for liquidity in price range"""
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96
    
    return int(liquidity * (sqrt_price_b_x96 - sqrt_price_a_x96) // Q96)
```

#### **2. Implement Proper Pool State Management**
```python
@dataclass
class UniswapV3Pool:
    """Proper Uniswap V3 pool implementation"""
    pool_name: str
    fee: int  # Fee in hundredths of a bip (3000 = 0.3%)
    tick_spacing: int  # Tick spacing for this fee tier
    
    # Core pool state
    sqrt_price_x96: int  # Current sqrt price in Q64.96
    liquidity: int  # Current active liquidity
    tick_current: int  # Current tick
    
    # Concentrated liquidity positions
    ticks: Dict[int, int]  # tick -> net liquidity delta
    
    def __post_init__(self):
        # Initialize based on pool type
        if "MOET:BTC" in self.pool_name:
            # 1 BTC = 100,000 MOET, so price = 0.00001
            price = 0.00001
            self.tick_current = int(math.log(price) / math.log(1.0001))
            self.sqrt_price_x96 = tick_to_sqrt_price_x96(self.tick_current)
        else:
            # 1:1 for yield tokens
            self.tick_current = 0
            self.sqrt_price_x96 = Q96
    
    def swap(
        self,
        zero_for_one: bool,
        amount_specified: int,
        sqrt_price_limit_x96: int
    ) -> Tuple[int, int]:
        """Execute a swap using proper Uniswap V3 math"""
        # Implement the actual Uniswap V3 swap logic
        # This is complex and requires implementing the full constant product curve
        pass
```

#### **3. Fix Slippage Calculations**
```python
class UniswapV3SlippageCalculator:
    """Proper Uniswap V3 slippage calculator"""
    
    def __init__(self, pool: UniswapV3Pool):
        self.pool = pool
    
    def calculate_swap_slippage(
        self,
        amount_in: int,  # Use integers for precision
        token_in: str,
        sqrt_price_limit_x96: Optional[int] = None
    ) -> Dict[str, any]:
        """Calculate swap with proper Uniswap V3 math"""
        
        # Determine swap direction
        zero_for_one = token_in in ["MOET", "Yield_Token"]
        
        # Set price limit if not provided
        if sqrt_price_limit_x96 is None:
            if zero_for_one:
                sqrt_price_limit_x96 = tick_to_sqrt_price_x96(MIN_TICK + 1)
            else:
                sqrt_price_limit_x96 = tick_to_sqrt_price_x96(MAX_TICK - 1)
        
        # Execute swap using proper math
        amount_out, sqrt_price_after = self.pool.swap(
            zero_for_one, amount_in, sqrt_price_limit_x96
        )
        
        # Calculate expected output without slippage (at current price)
        current_price = (self.pool.sqrt_price_x96 / Q96) ** 2
        expected_out = amount_in * current_price  # Simplified
        
        # Calculate slippage
        slippage = max(0, expected_out - amount_out)
        slippage_percent = (slippage / expected_out) * 100 if expected_out > 0 else 0
        
        return {
            "amount_in": amount_in,
            "amount_out": amount_out,
            "expected_amount_out": expected_out,
            "slippage_amount": slippage,
            "slippage_percentage": slippage_percent,
            "sqrt_price_after": sqrt_price_after
        }
```

### **Visualization System Updates Required**

#### **1. Update LP Curve Analysis (`lp_curve_analysis.py`)**
- Replace `get_bin_data_for_charts()` with `get_tick_data_for_charts()`
- Update bar charts to show tick-based liquidity distribution
- Fix price range calculations to use proper tick spacing

#### **2. Update Chart Generation (`target_health_factor_analysis.py`)**
- Replace all references to `bins` with `ticks`
- Update slippage cost extraction to use new data structures
- Fix CSV generation to use proper tick-based calculations

#### **3. Update Agent Cost Calculations**
All agent classes need updated slippage cost calculations:
```python
# In HighTideAgent and AaveAgent
def calculate_slippage_costs(self):
    # Use proper Uniswap V3 calculations instead of bin approximations
    pass
```

### **Backward Compatibility Requirements**

#### **1. Maintain Public API**
Keep these functions working but with proper implementations:
- `create_moet_btc_pool()`
- `create_yield_token_pool()`
- `calculate_rebalancing_cost_with_slippage()`
- `calculate_liquidation_cost_with_slippage()`

#### **2. Preserve Data Structures**
Ensure simulation engines can still access:
- Pool reserves (calculate from tick data)
- Current price (derive from sqrt_price_x96)
- Liquidity metrics (aggregate from tick data)

#### **3. Update Legacy Fields**
```python
def _update_legacy_fields(self):
    """Update legacy fields for backward compatibility"""
    # Calculate reserves from tick data
    self.token0_reserve = self._calculate_token0_reserve()
    self.token1_reserve = self._calculate_token1_reserve()
    
    # Price from sqrt_price_x96
    price = (self.sqrt_price_x96 / Q96) ** 2
    
    # Total liquidity from active positions
    self.liquidity = self._calculate_total_liquidity()
```

### **Testing Requirements**

#### **1. Unit Tests**
Create comprehensive tests for:
- Tick to price conversions
- Liquidity calculations
- Swap math accuracy
- Slippage calculations

#### **2. Integration Tests**
Verify that:
- Simulation engines still work
- Chart generation functions properly
- Agent cost calculations are accurate
- CSV outputs contain correct data

#### **3. Accuracy Validation**
Compare results against known Uniswap V3 behaviors:
- Price impact for various trade sizes
- Slippage in different liquidity conditions
- Fee calculations

### **Implementation Priority**

1. **High Priority**: Core math functions (tick conversions, swap calculations)
2. **Medium Priority**: Pool state management and slippage calculator
3. **Low Priority**: Visualization updates and legacy compatibility

### **Success Criteria**

Your implementation will be successful when:
1. ✅ Uses proper Uniswap V3 tick-based math
2. ✅ Produces accurate slippage calculations
3. ✅ Maintains backward compatibility with existing simulations
4. ✅ Updates all visualization components correctly
5. ✅ Passes comprehensive test suite

### **Resources**

Reference the official Uniswap V3 implementation:
- [Uniswap V3 Core Repository](https://github.com/Uniswap/v3-core)
- [Uniswap V3 Whitepaper](https://uniswap.org/whitepaper-v3.pdf)
- Focus on `UniswapV3Pool.sol` and `TickMath.sol`

**Remember**: This fix is critical for the accuracy of the High Tide vs AAVE analysis. Incorrect slippage calculations could lead to wrong conclusions about protocol performance.