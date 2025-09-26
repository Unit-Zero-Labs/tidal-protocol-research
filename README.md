# Tidal Protocol Simulation System

A comprehensive DeFi lending protocol simulation and analysis framework that models sophisticated lending strategies, liquidation mechanisms, and yield-bearing token systems with authentic Uniswap V3 mathematics and realistic agent behaviors.

## Overview

This repository contains a sophisticated multi-layer simulation system that models the complete Tidal Protocol lending ecosystem, featuring:

### Core Protocol Components
- **Tidal Protocol Engine**: Kinked interest rate models with Ebisu-style debt cap calculations
- **High Tide Vault Strategy**: Active rebalancing using yield tokens to prevent liquidations
- **AAVE Protocol Engine**: Traditional liquidation mechanisms for performance comparison
- **MOET Stablecoin**: Fee-less stablecoin with ±2% stability bands and peg maintenance
- **Yield Token System**: 10% APR rebasing tokens with continuous compound interest

### Advanced Mathematical Systems
- **Authentic Uniswap V3 Implementation**: Tick-based concentrated liquidity with Q64.96 fixed-point arithmetic
- **Cross-Tick Swap Mechanics**: Sophisticated multi-range trading with realistic slippage
- **Asymmetric Pool Configuration**: Configurable token ratios (10:90 to 90:10) with intelligent tick alignment
- **Monte Carlo Simulation Framework**: Statistical robustness with 10-50 agents per simulation

### Multi-Agent Ecosystem
- **High Tide Agents**: Active rebalancing with 3 risk profiles (Conservative, Moderate, Aggressive)
- **AAVE Agents**: Traditional liquidation behavior (passive until liquidation)
- **Pool Rebalancers**: ALM and algorithmic arbitrage agents for pool price accuracy
- **Liquidators & Traders**: Realistic market-making and liquidation execution

## Quick Start

### Prerequisites

- Python 3.8+ 
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/unit-zero-labs/tidal-protocol-research.git
cd tidal-protocol-research

# Create and activate virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r analysis_requirements.txt
```

### Dependencies

Core Python packages required:
- `matplotlib>=3.5.0` - Advanced visualization and charting
- `seaborn>=0.11.0` - Statistical plotting and analysis
- `pandas>=1.3.0` - Data manipulation and time series analysis
- `numpy>=1.21.0` - Numerical computing and mathematical operations

## 🚀 Usage Examples & Simulation Scenarios

### 1. High Tide vs AAVE Comparison (Recommended Start)

Compare High Tide active rebalancing against traditional AAVE liquidations:

```bash
python comprehensive_ht_vs_aave_analysis.py
```

**What this simulation does:**
- **Duration**: 60-minute BTC decline scenario
- **Market Stress**: 15-25% BTC price drop (from $100k to $75k-$85k)
- **High Tide Strategy**: Agents actively rebalance using yield tokens when health factors decline
- **AAVE Strategy**: Agents hold positions until liquidation (no rebalancing)
- **Analysis**: Side-by-side performance comparison with survival rates, costs, and efficiency metrics

### 2. Comprehensive Realistic Pool Analysis

Run detailed multi-configuration analysis:

```bash
python comprehensive_realistic_pool_analysis.py
```

**Advanced Features:**
- **Pool Configurations**: Multiple sizes ($250k, $500k, $2M) with varied concentrations
- **Risk Profiles**: Conservative (HF 2.1-2.4), Moderate (HF 1.5-1.8), Aggressive (HF 1.3-1.5)
- **LP Curve Evolution**: Real-time liquidity distribution tracking
- **Utilization Analysis**: Protocol sustainability under different stress conditions
- **Asymmetric Pool Testing**: 75/25 MOET:YT ratios with intelligent tick alignment

### 3. Monte Carlo Scenario Analysis

Statistical robustness testing with varied agent populations:

```bash
python balanced_scenario_monte_carlo.py
```

**Monte Carlo Features:**
- **Dynamic Agent Count**: 10-50 agents per simulation run
- **Risk Profile Randomization**: Varied initial and target health factors
- **Statistical Analysis**: Multiple runs for confidence intervals
- **Pool Arbitrage**: Optional ALM and algorithmic rebalancing agents

### 4. Target Health Factor Optimization

Analyze optimal health factor thresholds:

```bash
python target_health_factor_analysis.py
```

### 5. Long-Term Protocol Analysis

Extended 12-month simulation with realistic market dynamics:

```bash
python longterm_scenario_analysis.py
```

**Long-Term Features:**
- **Duration**: Up to 12 months with hourly price updates
- **Market Dynamics**: Geometric Brownian Motion for BTC price evolution
- **Pool Arbitrage**: ALM (12-hour intervals) and Algo (50 bps threshold) rebalancers
- **Yield Accrual**: Full 10% APR compound interest over extended periods
- **Flash Crash Events**: Optional extreme market stress testing

### 6. Yield Token Pool Capacity Testing

Examine pool liquidity limits and rebalancing capacity:

```bash
python yield_token_pool_capacity_analysis.py
```

**Capacity Analysis:**
- **Single Swap Limits**: Up to $350k trades within concentrated ranges
- **Consecutive Rebalancing**: Multi-agent competition for shared liquidity
- **Slippage Progression**: 0.01% (small trades) to 2%+ (range-crossing trades)
- **Pool Sustainability**: Liquidity exhaustion and recovery scenarios

### 7. Comprehensive Stress Testing Suite

Run full stress test library with multiple scenarios:

```bash
python tidal_protocol_sim/main.py --full-suite --monte-carlo 100
```

**Stress Test Categories:**
- **Single Asset Shocks**: ETH (-30%), BTC (-35%), FLOW (-50%)
- **Multi-Asset Crashes**: Crypto winter scenarios with correlated declines
- **Liquidity Crises**: MOET depeg events and pool liquidity drain
- **Parameter Sensitivity**: Collateral factors, liquidation thresholds, fee tiers
- **Extreme Events**: Black swan events, cascading liquidations

### 8. Individual Scenario Analysis

Test specific market conditions:

```bash
python tidal_protocol_sim/main.py --scenario ETH_Flash_Crash --detailed-analysis
```

## 🏗️ System Architecture

The simulation system follows a sophisticated 5-layer modular architecture designed for maximum flexibility and mathematical rigor:

```
┌─────────────────────────────────────────────────────────────┐
│                    Entry Points & CLI                       │
│  main.py, comprehensive_*.py, longterm_*.py, run_*.py      │
│  Monte Carlo Scripts, Stress Testing Suites               │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                  Simulation Engines                        │
│  HighTideVaultEngine, AaveProtocolEngine, TidalEngine     │
│  BaseLendingEngine, BTCPriceManager                        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 Agent System & Policies                    │
│  HighTideAgent, AaveAgent, PoolRebalancer (ALM/Algo)      │
│  TidalLender, Liquidator, Trader, BaseAgent               │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              Core Protocol Mathematics                      │
│  TidalProtocol, UniswapV3Math, YieldTokens,               │
│  MoetStablecoin, AssetPools, LiquidityPools                │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│          Analysis & Stress Testing Framework               │
│  Metrics Calculator, Visualization Suite, Stress Scenarios │
│  Results Management, Agent Tracking, LP Curve Analysis     │
└─────────────────────────────────────────────────────────────┘
```

### Orchestration vs Execution Layers

The system uses clear separation between orchestration and execution:

**🎯 Orchestration Layer (Engines):**
- Coordinates agent actions and decisions
- Tracks rebalancing events, slippage costs, and performance metrics  
- Records all trading activity for comprehensive analysis
- Manages simulation flow and agent lifecycle

**⚙️ Execution Layer (Agents + Pools):**
- Agents calculate portfolio needs and execute real swaps
- Pools execute authentic Uniswap V3 swaps with permanent state mutations
- Shared liquidity creates realistic competition between agents
- Real economic impact where each swap affects subsequent trades

## 🔬 Core Protocol Mathematics

### 1. Tidal Protocol Engine (`tidal_protocol_sim/core/protocol.py`)

**Purpose**: Core lending protocol with kinked interest rate model and Ebisu-style debt cap calculations

**Key Components:**
- **Asset Support**: ETH, BTC, FLOW, USDC, MOET with individual pool management
- **Kinked Interest Rate Model**: Base rate + multiplier below 80% kink, jump rate above
- **Collateral Factors**: ETH/BTC (75%), FLOW (50%), USDC (90%)
- **Liquidation Mechanics**: 8% penalty, 50% close factor, 1.2 target health factor

**Mathematical Models:**
```python
# Kinked Interest Rate Model
if utilization <= kink (0.80):
    rate = base_rate + (utilization × multiplier_per_block)
else:
    rate = base_rate + (kink × multiplier) + ((utilization - kink) × jump_rate)

# Ebisu-Style Debt Cap Formula
debt_cap = total_liquidation_capacity × dex_allocation × weighted_underwater_percentage
```

### 2. MOET Stablecoin System (`tidal_protocol_sim/core/moet.py`)

**Purpose**: Fee-less stablecoin with ±2% stability bands and peg maintenance

**Key Features:**
- **1:1 Minting/Burning**: No fees for optimal capital efficiency
- **Stability Bands**: $0.98 - $1.02 target range with pressure detection
- **Peg Maintenance**: Automatic stability pressure analysis and recommendations

### 3. Yield Token System (`tidal_protocol_sim/core/yield_tokens.py`)

**Purpose**: 10% APR rebasing tokens with continuous compound interest for High Tide strategy

**Advanced Features:**
- **Rebasing Mechanism**: Continuous value increase without quantity change
- **Flexible Creation**: Direct minting (minute 0) vs Uniswap V3 trading
- **Portfolio Management**: Yield-first sales with principal preservation
- **Real Swap Execution**: Engine-level coordination with pool state mutations

**Mathematical Model:**
```python
# Continuous yield accrual (per-minute precision)
minute_rate = APR * (minutes_elapsed / 525600)  # 525600 minutes/year
current_value = principal × (1 + minute_rate)
```

### 4. Authentic Uniswap V3 Implementation (`tidal_protocol_sim/core/uniswap_v3_math.py`)

**Purpose**: Production-grade tick-based concentrated liquidity with Q64.96 fixed-point arithmetic

**Advanced Capabilities:**
- **Tick System**: MIN_TICK (-887272) to MAX_TICK (887272) with proper spacing
- **Concentrated Liquidity**: 
  - MOET:BTC pools: 80% concentration around BTC price
  - MOET:Yield Token pools: 95% concentration around 1:1 peg
  - Asymmetric ratios: 10:90 to 90:10 with intelligent tick alignment
- **Cross-Tick Swaps**: Multi-step swaps with proper slippage calculation
- **Discrete Liquidity Ranges**: Three-tier system (concentrated core + wide ranges)

**Core Mathematical Functions:**
```python
# Tick to price conversion (Q64.96 format)
sqrt_price_x96 = int(1.0001^(tick/2) × 2^96)

# Liquidity delta calculations
amount0_delta = (liquidity × Q96 × (sqrt_price_b - sqrt_price_a)) / (sqrt_price_b × sqrt_price_a)
amount1_delta = (liquidity × (sqrt_price_b - sqrt_price_a)) / Q96

# Within-range price impact (whitepaper formula)
delta_sqrt_price = amount_in × Q96 / liquidity
```

## 🤖 Multi-Agent Ecosystem

### High Tide Agents (`tidal_protocol_sim/agents/high_tide_agent.py`)

**Strategy**: Active rebalancing using yield tokens to prevent liquidations

**Risk Profiles & Distribution:**
- **Conservative (30%)**: Initial HF 2.1-2.4, Target buffer 0.05-0.15
- **Moderate (40%)**: Initial HF 1.5-1.8, Target buffer 0.15-0.25  
- **Aggressive (30%)**: Initial HF 1.3-1.5, Target buffer 0.15-0.4

**Advanced Rebalancing Logic:**
1. **Initial Setup**: Deposit 1 BTC, borrow MOET based on initial HF, buy yield tokens
2. **Health Monitoring**: Continuous health factor tracking vs target thresholds
3. **Iterative Rebalancing**: Multi-cycle approach with slippage monitoring
4. **Yield-First Sales**: Sell accrued yield before touching principal
5. **Emergency Actions**: Full liquidation of remaining yield tokens if HF ≤ 1.0

### AAVE Agents (`tidal_protocol_sim/agents/aave_agent.py`)

**Strategy**: Traditional passive approach - hold positions until liquidation

**Key Differences from High Tide:**
- Same initial setup (1 BTC collateral, yield token purchase)
- **NO rebalancing** - positions held until liquidation
- Traditional AAVE liquidation: 50% collateral seizure + 5% bonus
- Passive yield accrual without portfolio management

### Pool Rebalancer Agents (`tidal_protocol_sim/agents/pool_rebalancer.py`)

**Purpose**: Maintain MOET:YT pool price accuracy through arbitrage

**ALM Rebalancer:**
- **Trigger**: Time-based (default: 12 hours)
- **Purpose**: Asset Liability Management - systematic pool maintenance

**Algo Rebalancer:**
- **Trigger**: Threshold-based (≥50 basis points deviation)
- **Purpose**: Algorithmic arbitrage - profit from price inefficiencies

**Arbitrage Mechanism:**
- Shared $500k liquidity pool (all MOET initially)
- External YT sales at true price to replenish reserves
- Profit = (Pool Price - True Price) × Amount Traded

## ⚙️ Simulation Engines

### 1. High Tide Vault Engine (`tidal_protocol_sim/engine/high_tide_vault_engine.py`)

**Purpose**: BTC decline scenario with sophisticated active rebalancing strategies

**Key Features:**
- **Market Stress Simulation**: 60-minute BTC decline (15-25% drop) with realistic volatility
- **Agent Orchestration**: Coordinates 10-50 High Tide agents with Monte Carlo variation
- **Real Swap Execution**: Engine-level coordination of yield token sales with pool state mutations
- **Iterative Rebalancing**: Multi-cycle approach with slippage monitoring and cost tracking

**Data Flow:**
```
Agent Needs MOET → Engine._execute_yield_token_sale() → Agent.execute_yield_token_sale() 
→ YieldTokenPool.execute_yield_token_sale() → Uniswap V3 Pool Swap (Real) → Pool State Updated
```

### 2. AAVE Protocol Engine (`tidal_protocol_sim/engine/aave_protocol_engine.py`)

**Purpose**: Traditional AAVE-style liquidation for direct comparison with High Tide

**Key Differences:**
- **No Rebalancing**: Agents hold positions until liquidation (passive strategy)
- **AAVE Liquidation**: 50% collateral + 5% bonus when HF ≤ 1.0
- **Same Market Conditions**: Identical BTC decline for fair comparison
- **Uniswap V3 Integration**: Real liquidation swaps through MOET:BTC pool

### 3. Base Lending Engine (`tidal_protocol_sim/engine/base_lending_engine.py`)

**Purpose**: Common simulation framework providing shared functionality

**Core Features:**
- Agent action processing loop with minute-by-minute updates
- Health factor monitoring and liquidation detection
- Comprehensive metrics recording throughout simulation lifecycle
- Common swap execution and slippage calculation infrastructure

### 4. BTC Price Manager (`tidal_protocol_sim/engine/btc_price_manager.py`)

**Purpose**: Realistic market stress simulation with authentic volatility patterns

**Price Dynamics:**
- **Historical Volatility**: Based on real BTC decline patterns
- **Convergence Logic**: Gradual convergence to target price in final 20% of simulation
- **Configurable Parameters**: Initial price, duration, final price range
- **Long-Term Support**: Geometric Brownian Motion for extended simulations

## 🔧 Configuration & Customization

### Core Configuration Files

**Simulation Parameters (`tidal_protocol_sim/engine/config.py`):**
- Agent populations and risk profile distributions
- Pool sizes and concentration levels
- BTC price decline parameters and volatility settings
- Monte Carlo variation controls

**Pool Configurations:**
```python
# MOET:BTC Pool (for liquidations)
moet_btc_pool_config = {
    "size": 500_000,         # $500k total liquidity
    "concentration": 0.80,   # 80% at current BTC price
    "fee_tier": 0.003        # 0.3% fee tier
}

# MOET:Yield Token Pool (for rebalancing)
moet_yt_pool_config = {
    "size": 500_000,         # $250k each side  
    "concentration": 0.95,   # 95% at 1:1 peg
    "token0_ratio": 0.75,    # 75% MOET, 25% YT
    "fee_tier": 0.0005       # 0.05% fee tier
}
```

**Agent Configuration:**
- **High Tide Agents**: 10-50 agents with Monte Carlo risk profile variation
- **AAVE Agents**: Identical setup to High Tide but no rebalancing capability
- **Pool Rebalancers**: ALM (12-hour intervals) and Algo (50 bps threshold) agents

### Advanced Customization Options

**Asymmetric Pool Ratios:**
- Configurable MOET:YT ratios from 10:90 to 90:10
- Intelligent tick alignment for precise target ratios
- Automatic bounds calculation with mathematical optimization

**Market Stress Parameters:**
- BTC decline duration (default: 60 minutes)
- Final price ranges (default: 15-25% decline)
- Volatility patterns based on historical data
- Long-term Geometric Brownian Motion for extended simulations

**Pool Arbitrage Settings:**
```python
# Pool Rebalancer Configuration
enable_pool_arbing = True                    # Enable arbitrage agents
alm_rebalance_interval_minutes = 720         # 12 hours
algo_deviation_threshold_bps = 50.0          # 50 basis points
total_rebalancer_liquidity = 500_000         # $500k shared liquidity
```

## 📚 Technical Documentation

### Comprehensive Documentation Library (`readmes/`)

**Core Mathematics:**
- [`UNISWAP_V3_MATH_README.md`](readmes/UNISWAP_V3_MATH_README.md): Authentic Uniswap V3 implementation with Q64.96 arithmetic
- [`YIELD_TOKENS_README.md`](readmes/YIELD_TOKENS_README.md): 10% APR rebasing system with portfolio management
- [`ASYMMETRIC_POOL_IMPLEMENTATION.md`](readmes/ASYMMETRIC_POOL_IMPLEMENTATION.md): Configurable token ratios with intelligent tick alignment

**Simulation Engines:**
- [`HIGH_TIDE_VAULT_ENGINE_README.md`](readmes/HIGH_TIDE_VAULT_ENGINE_README.md): Active rebalancing orchestration layer
- [`AAVE_PROTOCOL_ENGINE_README.md`](readmes/AAVE_PROTOCOL_ENGINE_README.md): Traditional liquidation comparison engine

**Specialized Systems:**
- [`POOL_REBALANCER_IMPLEMENTATION.md`](readmes/POOL_REBALANCER_IMPLEMENTATION.md): ALM and algorithmic arbitrage agents


## Workflow Examples

### 1. Research Workflow: High Tide Efficacy Analysis

```bash
# Step 1: Run baseline comparison
python comprehensive_ht_vs_aave_analysis.py

# Step 2: Analyze different pool configurations
python comprehensive_realistic_pool_analysis.py

# Step 3: Test statistical robustness
python balanced_scenario_monte_carlo.py

# Step 4: Examine long-term sustainability
python longterm_scenario_analysis.py
```

### 2. Parameter Sensitivity Analysis

```bash
# Test different health factor thresholds
python target_health_factor_analysis.py

# Analyze yield token pool capacity limits
python yield_token_pool_capacity_analysis.py

# Test pool rebalancing effectiveness
python rebalance_liquidity_test.py
```

### 3. Stress Testing Protocol Limits

```bash
# Run comprehensive stress test suite
python tidal_protocol_sim/main.py --full-suite --monte-carlo 50

# Test specific extreme scenarios
python tidal_protocol_sim/main.py --scenario BTC_Flash_Crash --detailed-analysis
python tidal_protocol_sim/main.py --scenario MOET_Depeg_Crisis --detailed-analysis
```