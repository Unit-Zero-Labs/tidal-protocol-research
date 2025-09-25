# Pool Rebalancer Implementation Guide

## Overview

This document outlines the implementation of two specialized pool rebalancing agents designed to maintain MOET:YT liquidity pool price accuracy during Tidal Protocol simulations.

## Architecture

### 1. Pool Rebalancer Agents

Two new agents have been created in `tidal_protocol_sim/agents/pool_rebalancer.py`:

#### ALM Rebalancer
- **Type**: Time-based rebalancing
- **Trigger**: Fixed time intervals (default: 12 hours / 720 minutes)
- **Behavior**: Rebalances pool regardless of price deviation magnitude
- **Purpose**: Asset Liability Management - systematic pool maintenance

#### Algo Rebalancer  
- **Type**: Threshold-based rebalancing
- **Trigger**: Price deviation ≥ 50 basis points (0.5%)
- **Behavior**: Only rebalances when significant price deviations occur
- **Purpose**: Algorithmic arbitrage - profit from price inefficiencies

### 2. Key Features

**Shared Liquidity Pool**: Both agents share $500,000 total liquidity (all in MOET initially)

**External YT Sales**: When agents acquire YT through rebalancing, they "sell" it externally at true price to replenish MOET reserves. This process can be configured with an optional arbitrage delay to simulate real-world settlement times.

**Arbitrage Delay Mechanism**: Configurable delay period (default: 1 hour, auto-converted based on simulation time scale) that prevents immediate external YT sales, creating realistic capital constraints and testing system resilience under liquidity pressure.

**True Price Oracle**: Uses `calculate_true_yield_token_price()` function to determine the target price based on 10% APR yield accrual.

## Implementation Details

### 3. Core Components

#### PoolRebalancerState
Enhanced agent state tracking:
- MOET and YT balances
- Rebalancing activity metrics
- Profit tracking from arbitrage
- Arbitrage delay configuration and pending YT sales queue
- Automatic time scale detection for delay conversion

#### PoolRebalancerManager
Unified interface coordinating both rebalancers:
- Enable/disable functionality
- Pool reference management
- Event processing and logging
- Arbitrage delay configuration for both rebalancers
- Processing of pending YT sales based on time delays

### 4. Integration Points

#### Simulation Scripts
Added `enable_pool_arbing` configuration flag to:
- `balanced_scenario_monte_carlo.py`
- `tri_health_factor_analysis.py` 
- `rebalance_liquidity_test.py`
- `yield_token_pool_capacity_analysis.py`

**Default**: `False` (disabled) for backward compatibility

#### Long-Term Analysis
New script `longterm_scenario_analysis.py` provides:
- 12-month simulation capability
- Hourly BTC price updates using Geometric Brownian Motion
- Comprehensive pool arbitrage testing
- Bull/bear cycle simulation
- Flash crash events (optional)

### 5. Yield Token Pricing

Enhanced `tidal_protocol_sim/core/yield_tokens.py` with:

#### Global Price Function
```python
def calculate_true_yield_token_price(current_minute: int, apr: float = 0.10, initial_price: float = 1.0) -> float:
    """Calculate true yield token price at any simulation time"""
```

**Formula**: `price = initial_price * (1 + apr * (minutes_elapsed / minutes_per_year))`

**Usage**: Provides consistent "oracle" price for rebalancing decisions

## Usage Examples

### 6. Enabling Pool Arbitrage

#### Short-Term Simulations
```python
# In balanced_scenario_monte_carlo.py
config = ComprehensiveComparisonConfig()
config.enable_pool_arbing = True
config.alm_rebalance_interval_minutes = 720  # 12 hours
config.algo_deviation_threshold_bps = 50.0   # 50 basis points
```

#### Long-Term Analysis
```python
# In longterm_scenario_analysis.py  
config = LongTermSimulationConfig()
config.enable_pool_arbing = True
config.simulation_duration_months = 12
```

### 7. Rebalancer Configuration

#### ALM Rebalancer Parameters
- `rebalance_interval_minutes`: Time between rebalances (default: 720)
- `min_rebalance_amount`: Minimum rebalance size ($1,000)
- `max_single_rebalance`: Maximum single rebalance ($50,000)

#### Algo Rebalancer Parameters  
- `deviation_threshold_bps`: Trigger threshold (default: 50 bps)
- `min_rebalance_amount`: Minimum rebalance size ($1,000)
- `max_single_rebalance`: Maximum single rebalance ($50,000)

## Rebalancing Logic

### 8. Decision Process

1. **Price Comparison**: Compare pool YT price vs true YT price
2. **Direction Determination**: 
   - Pool price > true price → Sell YT to pool (get MOET) *[requires YT inventory]*
   - Pool price < true price → Buy YT from pool (with MOET) *[requires MOET balance]*
3. **Amount Calculation**: Scale rebalance size with price deviation magnitude
4. **Balance Check**: Ensure sufficient MOET or YT balance for operation
5. **Execution**: Perform Uniswap V3 swap through pool
6. **External Sale**: Sell acquired YT at true price externally (immediate or delayed based on configuration)
7. **Delay Processing**: If arbitrage delay is enabled, queue YT sales for future execution after delay period
8. **Inventory Management**: Build YT inventory through underpriced purchases for future overpriced sales

### 9. Profit Mechanism

**Arbitrage Profit** = (Pool Price - True Price) × Amount Traded

**Example**: 
- Pool YT price: $1.02
- True YT price: $1.01  
- Sell $10,000 YT to pool → Receive $10,200 MOET
- Sell $10,000 YT externally → Receive $10,100 MOET
- **Total Profit**: $10,200 + $10,100 - $10,000 = $300

## Testing and Validation

### 10. Verification Steps

1. **Price Tracking**: Verify `calculate_true_yield_token_price()` matches `YieldToken.get_current_value()`
2. **Pool Integration**: Confirm rebalancers can access and modify YT pool state
3. **Balance Management**: Test MOET/YT balance updates after rebalancing
4. **Event Logging**: Validate rebalancing events are properly recorded
5. **Profit Calculation**: Verify arbitrage profit calculations are accurate

### 11. Simulation Scenarios

#### Backward Compatibility Test
- Run existing simulations with `enable_pool_arbing = False`
- Confirm identical results to previous versions

#### Arbitrage Effectiveness Test  
- Enable pool arbitrage in long-term simulation
- Monitor pool price accuracy vs true YT price
- Measure total arbitrage profits and frequency

#### Stress Testing
- Large rebalancing events during high volatility
- Pool liquidity exhaustion scenarios
- Multiple simultaneous rebalancing triggers

## Configuration Reference

### 12. Default Parameters

```python
# Pool Arbitrage Configuration
enable_pool_arbing = False                    # Disabled by default
alm_rebalance_interval_minutes = 720          # 12 hours
algo_deviation_threshold_bps = 50.0           # 50 basis points

# Arbitrage Delay Configuration
enable_arb_delay = False                      # Disabled by default
arb_delay_time_units = 60                     # 1 hour (auto-converted based on simulation time scale)

# Rebalancer Liquidity
total_rebalancer_liquidity = 500_000          # $500k total
moet_balance = 500_000                        # $500k MOET (all liquidity)
yt_balance = 0                                # No initial YT holdings

# Rebalancing Limits
min_rebalance_amount = 1_000                  # $1k minimum
max_single_rebalance = 50_000                 # $50k maximum
```

### 13. Pool Configuration

```python
# MOET:YT Pool (for rebalancing)
moet_yt_pool_config = {
    "size": 500_000,                          # $250k each side
    "concentration": 0.95,                    # 95% at 1:1 peg
    "token0_ratio": 0.75,                     # 75% MOET, 25% YT
    "fee_tier": 0.0005                        # 0.05% fee tier
}
```

## Future Enhancements

### 14. Potential Improvements

1. **Dynamic Thresholds**: Adjust rebalancing parameters based on market volatility
2. **Multi-Pool Support**: Extend to other token pairs beyond MOET:YT
3. **Machine Learning**: Predictive rebalancing based on historical patterns
4. **Gas Cost Modeling**: Include transaction costs in rebalancing decisions
5. **Cross-DEX Arbitrage**: Utilize multiple liquidity sources for optimal pricing

### 15. Integration Opportunities

- **Real-Time Monitoring**: Dashboard for pool health and arbitrage opportunities
- **Alert System**: Notifications for large price deviations or rebalancing events
- **Performance Analytics**: Detailed profitability analysis and optimization recommendations

## Conclusion

The pool rebalancer implementation provides a comprehensive framework for maintaining MOET:YT pool price accuracy while generating arbitrage profits. The dual-agent approach (ALM + Algo) ensures both systematic maintenance and opportunistic profit capture, making the system robust across various market conditions.

The implementation maintains full backward compatibility while adding powerful new capabilities for long-term protocol analysis and optimization.
