
# Tidal Protocol Simulation System - Complete Architecture Overview

## Executive Summary

This document provides a comprehensive breakdown of the Tidal Protocol simulation system, including all protocol engines, agent architectures, mathematical models, and analysis frameworks. The system is designed for stress testing lending protocols, comparing liquidation mechanisms, and analyzing DeFi protocol stability under various market conditions.

## System Architecture Overview

The simulation system follows a modular, layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Entry Points & CLI                       │
│  main.py, run_*.py scripts, comprehensive_*.py             │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Simulation Engines                        │
│  TidalProtocolEngine, HighTideVaultEngine,                 │
│  AaveProtocolEngine, BaseLendingEngine                     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 Agent System & Policies                    │
│  HighTideAgent, AaveAgent, TidalLender,                   │
│  Liquidator, BasicTrader, BaseAgent                        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              Core Protocol Mathematics                      │
│  TidalProtocol, UniswapV3Math, MoetStablecoin,            │
│  YieldTokens, AssetPools, LiquidityPools                   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│          Analysis & Stress Testing Framework               │
│  Metrics, Charts, Stress Scenarios, Results Management     │
└─────────────────────────────────────────────────────────────┘
```

## Core Protocol Mathematics (`tidal_protocol_sim/core/`)

### 1. `protocol.py` - Tidal Protocol Engine
**Purpose**: Core lending protocol with kinked interest rate model and debt cap calculations

**Key Components**:
- **Asset Enum**: Supports ETH, BTC, FLOW, USDC, MOET
- **AssetPool Class**: Individual asset pools with utilization tracking
  - Kinked interest rate model (base rate, multiplier, jump rate at 80% kink)
  - Collateral factors: ETH/BTC (75%), FLOW (50%), USDC (90%)
  - Liquidation penalty: 8%
- **LiquidityPool Class**: MOET trading pairs with concentrated liquidity
- **TidalProtocol Class**: Main protocol coordinator
  - Debt cap calculation: `A × B × C` (liquidation capacity × DEX allocation × underwater collateral)
  - Target health factor: 1.2
  - Liquidation close factor: 50%

**Mathematical Models**:
```python
# Kinked Interest Rate Model
if utilization <= kink (0.80):
    rate = base_rate + (utilization × multiplier_per_block)
else:
    rate = base_rate + (kink × multiplier) + ((utilization - kink) × jump_rate)

# Debt Cap Formula (Ebisu-style)
debt_cap = total_liquidation_capacity × dex_allocation × weighted_underwater_percentage
```

### 2. `moet.py` - MOET Stablecoin System
**Purpose**: Fee-less stablecoin with ±2% stability bands

**Key Features**:
- 1:1 minting/burning (no fees as per refactor spec)
- $1.00 target peg with (0.98, 1.02) stability bands
- Stability pressure detection and action recommendations

### 3. `yield_tokens.py` - High Tide Yield System
**Purpose**: Yield-bearing tokens with 10% APR for High Tide strategy

**Components**:
- **YieldToken Class**: Individual tokens with continuous 10% APR accrual
- **YieldTokenManager Class**: Portfolio management with priority selling (highest yield first)
- **YieldTokenPool Class**: MOET ↔ Yield Token trading with 95% concentration at peg

**Mathematical Model**:
```python
# Continuous yield accrual
minute_rate = (1 + APR)^(1/525600) - 1
current_value = principal × (1 + minute_rate)^minutes_elapsed
```

### 4. `uniswap_v3_math.py` - Authentic Uniswap V3 Implementation
**Purpose**: Proper tick-based concentrated liquidity with Q64.96 fixed-point arithmetic

**Key Features**:
- **Tick System**: MIN_TICK (-887272) to MAX_TICK (887272)
- **Concentrated Liquidity**: 
  - MOET:BTC pools: 80% concentration around peg
  - MOET:Yield Token pools: 95% concentration around peg
- **Exact Swap Logic**: Multi-step swaps with proper slippage calculation
- **Position Management**: Tick-based liquidity positions with proper math

**Core Functions**:
```python
# Tick to price conversion
sqrt_price_x96 = int(1.0001^(tick/2) × 2^96)

# Liquidity delta calculations
amount0_delta = (liquidity × Q96 × (sqrt_price_b - sqrt_price_a)) / (sqrt_price_b × sqrt_price_a)
amount1_delta = (liquidity × (sqrt_price_b - sqrt_price_a)) / Q96
```

## Agent System & Policies (`tidal_protocol_sim/agents/`)

### 1. `base_agent.py` - Agent Foundation
**Purpose**: Base class with realistic position initialization

**AgentState Features**:
- Asset-specific position initialization by agent type
- Health factor calculation with collateral factors
- Portfolio value tracking and profit/loss calculation

### 2. `high_tide_agent.py` - Active Rebalancing Strategy
**Purpose**: Agents that automatically rebalance using yield tokens when health factors decline

**Key Strategy**:
1. **Initial Setup**: Deposit 1 BTC, borrow MOET based on initial HF, buy yield tokens
2. **Monitoring**: Track health factor vs. target threshold
3. **Rebalancing**: Sell yield tokens (yield first, then principal) to repay debt
4. **Emergency**: Sell all remaining yield tokens if HF ≤ 1.0

**Risk Profiles**:
- **Conservative (30%)**: Initial HF 2.1-2.4, target buffer 0.05-0.15
- **Moderate (40%)**: Initial HF 1.5-1.8, target buffer 0.15-0.25  
- **Aggressive (30%)**: Initial HF 1.3-1.5, target buffer 0.15-0.4

**Rebalancing Formula**:
```python
debt_reduction_needed = current_debt - (effective_collateral_value / initial_health_factor)
```

### 3. `aave_agent.py` - Traditional Liquidation Strategy
**Purpose**: Agents that hold positions until liquidation (no rebalancing)

**Key Differences from High Tide**:
- Same initial setup (1 BTC collateral, yield token purchase)
- **NO rebalancing** - positions held until liquidation
- Traditional AAVE liquidation: 50% collateral seizure + 5% bonus

### 4. `tidal_lender.py` - Lending-Focused Agent
**Purpose**: Traditional lending behavior with supply/borrow optimization

**Decision Logic**:
1. Emergency repay if HF < 1.1
2. Conservative action if HF < 1.5
3. Supply if high APY opportunity (>8%)
4. Borrow if can do safely (target HF 1.5)

### 5. `liquidator.py` - Liquidation Bot
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
 THIS SHOULD BE BUILT INTO THE AAVE ENGINE AND THE TIDAL ENGINE RESPECTIVELY, SINCE EACH HAVE DIFFERENT LIQUIDATION MECHANICS. WE DO NOT WANT A LIQUIDATOR AGENT, BUT RATHER THE LIQUIDATION IS BUILT INTO THE AAVE ENGINE AND THE TIDAL ENGINE
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
**Purpose**: Identifies and executes liquidation opportunities

**Strategy**:
- Scan for positions with HF < 1.0
- Calculate profitability (8% penalty - gas costs)
- Execute most profitable liquidations
- Manage MOET inventory for liquidations

### 6. `trader.py` - Market Making Agent
**Purpose**: Provides trading activity and liquidity

**Behaviors**:
- MOET peg trading (±2% deviation threshold)
- Simple momentum trading (5% price movements)
- Random market-making (small trades for liquidity)

## Simulation Engines (`tidal_protocol_sim/simulation/`)

### 1. `base_lending_engine.py` - Common Simulation Framework
**Purpose**: Abstract base for all lending protocol simulations

**Core Features**:
- Agent action processing loop
- Common liquidation and swap execution
- Health factor monitoring and updates
- Comprehensive metrics recording

### 2. `tidal_engine.py` - Base Tidal Protocol Simulation
**Purpose**: Standard Tidal Protocol with Uniswap V3 mathematics

**Features**:
- Full Uniswap V3 integration for MOET:BTC swaps
- Sophisticated slippage calculation
- Multi-agent ecosystem (lenders, traders, liquidators)

### 3. `high_tide_vault_engine.py` - High Tide Scenario Engine
**Purpose**: BTC decline scenario with active rebalancing

**Scenario Design**:
- **Duration**: 60 minutes
- **BTC Decline**: 15-25% (from $100k to $75k-$85k)
- **Decline Pattern**: Historical volatility rates with occasional spikes
- **Agent Behavior**: Active yield token rebalancing

**Key Components**:
- **BTCPriceDeclineManager**: Manages realistic BTC price decline
- **Yield Token Integration**: Full yield token trading and rebalancing
- **Position Tracking**: Minute-by-minute agent analysis

### 4. `aave_protocol_engine.py` - AAVE Comparison Engine  
**Purpose**: Traditional AAVE-style liquidation for comparison

**Key Differences**:
- **No Rebalancing**: Agents hold positions until liquidation
- **AAVE Liquidation**: 50% collateral + 5% bonus when HF ≤ 1.0
- **Same Market Conditions**: Identical BTC decline for fair comparison

### 5. `config.py` & `state.py` - Configuration and State Management
**Purpose**: Simulation parameters and state tracking

**Configuration Options**:
- Agent counts and initial balances
- Market volatility parameters
- Stress test scenario definitions
- Protocol parameter overrides

## Analysis Framework (`tidal_protocol_sim/analysis/`)

### 1. `metrics.py` - Protocol Health Metrics
**Purpose**: Comprehensive protocol stability analysis

**Key Metrics**:
- **Protocol Health Score**: Weighted composite of 5 components
- **Debt Cap Analysis**: Utilization, safety buffers, liquidation capacity
- **Liquidation Metrics**: Risk assessment, coverage ratios
- **MOET Stability**: Peg deviation, stability pressure
- **Utilization Balance**: Cross-asset utilization optimization

### 2. `high_tide_charts.py` - High Tide Visualization
**Purpose**: Specialized charts for High Tide analysis

**Chart Types**:
- Agent health factor evolution
- Rebalancing event timeline
- Cost of rebalancing analysis
- Yield token activity tracking

### 3. `lp_curve_analysis.py` - Liquidity Pool Analysis
**Purpose**: Uniswap V3 liquidity distribution analysis

**Features**:
- LP curve evolution tracking
- Concentration analysis
- Trade impact visualization

### 4. `agent_position_tracker.py` - Individual Agent Analysis
**Purpose**: Detailed position tracking for specific agents

**Tracking Data**:
- Minute-by-minute health factors
- Rebalancing decisions and outcomes
- Yield token portfolio evolution
- Cost analysis over time

## Stress Testing Framework (`tidal_protocol_sim/stress_testing/`)

### 1. `scenarios.py` - Stress Test Definitions
**Purpose**: Comprehensive stress test scenario library

**Scenario Categories**:
- **Single Asset Shocks**: ETH (-30%), BTC (-35%), FLOW (-50%)
- **Multi-Asset Crashes**: Crypto winter scenarios
- **Liquidity Crises**: MOET depeg, pool liquidity drain
- **Parameter Sensitivity**: Collateral factors, liquidation thresholds
- **Extreme Events**: Black swan, cascading liquidations

### 2. `runner.py` - Test Execution Engine
**Purpose**: Orchestrates stress test execution and result management

**Features**:
- Monte Carlo simulation support
- Result persistence and comparison
- Performance benchmarking
- Automated report generation

## System Data Flow

### 1. High Tide Simulation Flow
```
1. Initialize High Tide agents with 1 BTC collateral each
2. Agents borrow MOET based on initial health factor
3. Agents purchase yield tokens with borrowed MOET
4. BTC price begins gradual decline over 60 minutes
5. Each minute:
   - Update BTC price with realistic volatility
   - Accrue interest on debt
   - Agents check health factors
   - Execute rebalancing if HF < target threshold
   - Record all metrics and position changes
6. Generate comprehensive results and analysis
```

### 2. AAVE Comparison Flow
```
1. Initialize AAVE agents with identical setup to High Tide
2. Agents purchase yield tokens but NO rebalancing capability
3. Same BTC decline pattern as High Tide
4. Each minute:
   - Update prices and accrue interest
   - Check for liquidation conditions (HF ≤ 1.0)
   - Execute AAVE-style liquidations (50% + 5% bonus)
   - Record liquidation events and costs
5. Compare final outcomes with High Tide results
```

### 3. Stress Test Flow
```
1. Load stress test scenario definition
2. Initialize appropriate simulation engine
3. Apply scenario-specific shocks/conditions
4. Run simulation for specified duration
5. Collect comprehensive metrics
6. Generate comparative analysis
7. Store results for historical comparison
```

## Key Mathematical Relationships

### 1. Health Factor Calculation
```python
health_factor = (collateral_value × collateral_factor) / debt_value
```

### 2. Debt Cap Formula (Ebisu Methodology)
```python
debt_cap = Σ(liquidation_capacity_i) × dex_allocation × weighted_underwater_percentage
```

### 3. Yield Token Value Evolution
```python
current_value = principal × (1 + (APR/525600))^minutes_elapsed
```

### 4. Uniswap V3 Swap Calculation
```python
# Multi-step process through ticks
for each_tick_in_range:
    amount_in, amount_out = compute_swap_step(current_price, target_price, liquidity)
    update_price_and_liquidity()
```

### 5. Rebalancing Cost Calculation
```python
# High Tide: Direct MOET debt repayment
cost = max(0, final_debt - yield_token_value) + trading_fees

# AAVE: Liquidation penalties + slippage
cost = liquidation_penalties + swap_slippage_costs
```

## Usage Examples

### 1. Run High Tide vs AAVE Comparison
```python
python run_high_tide_vs_aave_comparison.py
```

### 2. Execute Stress Test Suite
```python
python main.py --full-suite --monte-carlo 100
```

### 3. Analyze Specific Scenario
```python
python main.py --scenario ETH_Flash_Crash --detailed-analysis
```

### 4. Generate Target Health Factor Analysis
```python
python target_health_factor_analysis.py
```

## Results and Output Structure

The system generates comprehensive results including:

1. **Agent Outcomes**: Individual agent performance, survival rates, costs
2. **Protocol Metrics**: Utilization, stability, liquidation efficiency
3. **Comparative Analysis**: High Tide vs AAVE performance comparison
4. **Visualizations**: Charts, graphs, LP curve evolution
5. **Statistical Analysis**: Monte Carlo results, confidence intervals
6. **Risk Assessment**: Stress test outcomes, protocol stability scores

This simulation system provides a complete framework for analyzing DeFi lending protocols, comparing liquidation mechanisms, and stress-testing protocol stability under various market conditions. The modular architecture allows for easy extension and modification while maintaining mathematical rigor and realistic agent behaviors.
```

I'll save this as a new markdown file at the root of the directory:

The system overview document has been created! This comprehensive document provides:

## 📋 **Complete System Breakdown**

### **🏗️ Architecture Overview**
- **5-layer modular architecture** from entry points to analysis frameworks
- **Clear separation of concerns** between engines, agents, protocols, and analysis

### **🔬 Core Protocol Mathematics**
- **`protocol.py`**: Kinked interest rate model, debt cap calculations, liquidation mechanics
- **`moet.py`**: Fee-less stablecoin with ±2% stability bands  
- **`yield_tokens.py`**: 10% APR yield system with continuous compounding
- **`uniswap_v3_math.py`**: Authentic tick-based concentrated liquidity with Q64.96 arithmetic

### **🤖 Agent System & Policies**
- **High Tide Agents**: Active rebalancing with yield tokens (3 risk profiles)
- **AAVE Agents**: Traditional liquidation (no rebalancing)
- **Supporting Agents**: Lenders, liquidators, traders with realistic behaviors

### **⚙️ Simulation Engines**
- **TidalProtocolEngine**: Base Tidal with Uniswap V3 integration
- **HighTideVaultEngine**: BTC decline scenario with rebalancing
- **AaveProtocolEngine**: Traditional liquidation comparison
- **BaseLendingEngine**: Common simulation framework

### **📊 Analysis Framework**
- **Metrics Calculator**: Protocol health scores, debt cap analysis
- **Visualization Suite**: LP curves, agent tracking, cost analysis
- **Stress Testing**: 10+ scenarios from flash crashes to black swan events

### **🔄 Complete Data Flow**
- Step-by-step walkthrough of High Tide vs AAVE simulations
- Mathematical relationships and formulas
- Usage examples and result structures

The document serves as a complete technical reference for understanding how the simulation system works, from the mathematical models underlying the protocols to the agent decision-making policies and comprehensive analysis frameworks. It's structured to help both developers working on the system and researchers analyzing the results.