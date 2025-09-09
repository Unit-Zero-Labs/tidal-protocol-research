# Tidal Protocol Research & Simulation System

A comprehensive DeFi lending protocol simulation and analysis framework for comparing liquidation mechanisms, stress testing protocol stability, and analyzing High Tide active rebalancing strategies versus traditional AAVE-style liquidations.

##  Overview

This repository contains a sophisticated simulation system that models the Tidal Protocol lending ecosystem with authentic mathematical implementations of:

- **Tidal Protocol**: Kinked interest rate models with debt cap calculations
- **High Tide Strategy**: Active rebalancing using yield tokens to prevent liquidations
- **MOET Stablecoin**: Fee-less stablecoin with ±2% stability bands
- **Uniswap V3 Mathematics**: Tick-based concentrated liquidity with Q64.96 arithmetic
- **Comprehensive Agent System**: Multi-agent ecosystem with realistic behaviors

##  Quick Start

### Prerequisites

- Python 3.8+ 
- Git

### Clone & Setup

```bash
# Clone the repository
git clone https://github.com/your-org/tidal-protocol-research.git
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

The system requires the following Python packages:
- `matplotlib>=3.5.0` - Visualization and charting
- `seaborn>=0.11.0` - Statistical plotting
- `pandas>=1.3.0` - Data analysis and manipulation
- `numpy>=1.21.0` - Numerical computing

## 🏃‍♂️ Usage Examples

### 1. High Tide vs AAVE Comparison (Recommended Start)

Compare High Tide active rebalancing against traditional AAVE liquidations:

```bash
python run_high_tide_vs_aave_comparison.py
```

**What this does:**
- Simulates 60-minute BTC decline scenario (15-25% drop)
- High Tide agents actively rebalance using yield tokens
- AAVE agents hold positions until liquidation
- Generates comprehensive comparison analysis and charts

### 2. Comprehensive Realistic Analysis

Run detailed pool analysis with multiple configurations:

```bash
python comprehensive_realistic_pool_analysis.py
```

**Features:**
- Multiple pool sizes ($250k, $500k, $2M)
- Risk profile analysis (Conservative, Moderate, Aggressive)
- LP curve evolution tracking
- Utilization sustainability analysis

### 3. Target Health Factor Analysis

Analyze optimal health factor thresholds:

```bash
python target_health_factor_analysis.py
```

### 4. MOET Yield Token Borrow Cap Analysis

Examine borrow capacity constraints:

```bash
python moet_yt_borrow_cap_analysis.py
```

### 5. Full Stress Testing Suite

Run comprehensive stress tests with multiple scenarios:

```bash
python tidal_protocol_sim/main.py --full-suite --monte-carlo 100
```

### 6. Individual Scenario Analysis

Test specific market conditions:

```bash
python tidal_protocol_sim/main.py --scenario ETH_Flash_Crash --detailed-analysis
```

## 📊 Key Simulation Scenarios

### High Tide Strategy Simulation
- **Duration**: 60 minutes
- **BTC Decline**: 15-25% (from $100k to $75k-$85k)
- **Agent Behavior**: Active yield token rebalancing when health factors decline
- **Comparison**: Side-by-side with AAVE-style liquidations

### Stress Test Categories
- **Single Asset Shocks**: ETH (-30%), BTC (-35%), FLOW (-50%)
- **Multi-Asset Crashes**: Crypto winter scenarios
- **Liquidity Crises**: MOET depeg, pool liquidity drain
- **Parameter Sensitivity**: Collateral factors, liquidation thresholds
- **Extreme Events**: Black swan, cascading liquidations

## 🏗️ System Architecture

The simulation follows a modular 5-layer architecture:

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

## 🔬 Core Components

### Protocol Mathematics (`tidal_protocol_sim/core/`)

- **`protocol.py`**: Tidal Protocol with kinked interest rates and debt cap calculations
- **`moet.py`**: MOET stablecoin with fee-less 1:1 minting/burning
- **`yield_tokens.py`**: High Tide yield system with 10% APR continuous compounding
- **`uniswap_v3_math.py`**: Authentic Uniswap V3 tick-based concentrated liquidity

### Agent System (`tidal_protocol_sim/agents/`)

- **High Tide Agents**: Active rebalancing with 3 risk profiles (Conservative, Moderate, Aggressive)
- **AAVE Agents**: Traditional liquidation behavior (no rebalancing)
- **Supporting Agents**: Lenders, liquidators, traders with realistic market behaviors

### Simulation Engines (`tidal_protocol_sim/engine/`)

- **HighTideVaultEngine**: BTC decline scenario with active rebalancing
- **AaveProtocolEngine**: Traditional AAVE-style liquidation comparison
- **TidalEngine**: Base Tidal Protocol with Uniswap V3 integration
- **BaseLendingEngine**: Common simulation framework

### Analysis Framework (`tidal_protocol_sim/analysis/`)

- **Metrics Calculator**: Protocol health scores, debt cap analysis
- **Visualization Suite**: LP curves, agent tracking, cost analysis
- **Stress Testing**: Comprehensive scenario library with Monte Carlo support

## 📈 Results & Outputs

Simulations generate comprehensive results including:

### 1. Agent Performance Analysis
- Individual agent survival rates and costs
- Health factor evolution over time
- Rebalancing decision tracking
- Yield token portfolio management

### 2. Protocol Metrics
- Utilization rates across asset pools
- MOET stability and peg maintenance
- Liquidation efficiency and coverage
- Debt cap utilization and safety buffers

### 3. Comparative Analysis
- High Tide vs AAVE performance comparison
- Cost-benefit analysis of active rebalancing
- Risk-adjusted return calculations
- Statistical significance testing

### 4. Visualizations
- Interactive charts and dashboards
- LP curve evolution tracking
- Agent timeline analysis
- Protocol utilization heatmaps

All results are saved in the `tidal_protocol_sim/results/` directory with organized subdirectories for each analysis type.

## 🔧 Configuration

Key configuration files:
- `tidal_protocol_sim/engine/config.py`: Simulation parameters
- `analysis_requirements.txt`: Python dependencies
- `cleanup_simulation.sh`: Result cleanup utility

## 📚 Documentation

For detailed technical documentation:
- [`plans/system-overview.md`](plans/system-overview.md): Complete system architecture and mathematical models
- [`tidal_protocol_sim/core/UNISWAP_V3_MATH_README.md`](tidal_protocol_sim/core/UNISWAP_V3_MATH_README.md): Uniswap V3 implementation details
- [`tidal_protocol_sim/core/YIELD_TOKENS_README.md`](tidal_protocol_sim/core/YIELD_TOKENS_README.md): Yield token mechanics
- [`tidal_protocol_sim/engine/HIGH_TIDE_VAULT_ENGINE_README.md`](tidal_protocol_sim/engine/HIGH_TIDE_VAULT_ENGINE_README.md): High Tide strategy details

## 🧪 Example Workflow

1. **Start with High Tide vs AAVE comparison** to understand the core value proposition
2. **Run comprehensive analysis** to explore different configurations
3. **Examine specific scenarios** using the stress testing framework
4. **Analyze results** using the generated charts and CSV data
5. **Iterate parameters** to explore different market conditions

## Contributing

This is a research repository focused on DeFi protocol analysis. The modular architecture allows for easy extension of:
- New agent behaviors and strategies
- Additional stress test scenarios
- Enhanced analysis metrics
- Alternative protocol implementations

## 📄 License

[Add your license information here]
