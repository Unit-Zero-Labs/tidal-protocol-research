# Tidal Protocol Simulation: Modular Architecture

A **commercial-grade, modular DeFi simulation system** implementing the Agent-Action-Market pattern with **comprehensive Tidal Protocol integration** for sophisticated tokenomics modeling.

## 🏗️ Architecture Overview

This simulation follows the **Agent-Action-Market pattern** with clean separation of concerns:

```
Agents (State Only) → Policies (Decisions) → Actions (Universal Primitives) → Markets (Execution) → Events (Results)
```

### Key Features
- **Tidal Protocol Integration**: Comprehensive implementation of all Tidal-specific mechanisms
- **MOET Stablecoin System**: Full minting/burning mechanics with peg stability
- **Ebisu-Style Debt Caps**: Advanced risk management with liquidation capacity modeling
- **Universal Action System**: 40+ DeFi primitives + Tidal-specific actions
- **Plugin Architecture**: Generic framework + dedicated Tidal Protocol market
- **Zero Hardcoded Values**: All parameters from client configurations
- **Production-Ready**: Error handling, logging, comprehensive metrics

## 🌊 Tidal Protocol Features

### 1. Comprehensive Lending System
- **Multi-Asset Collateral**: ETH, BTC, FLOW, USDC with specific collateral factors
- **MOET Stablecoin**: Protocol-native stablecoin with $1.00 peg target
- **Kinked Interest Rates**: Sophisticated rate model with 80% kink point
- **Health Factor Management**: 1.5x target with liquidation at <1.0

### 2. MOET Stablecoin Mechanics
- **Minting Against Collateral**: Mint MOET by supplying collateral assets
- **Peg Stability System**: Automatic mechanisms to maintain $1.00 peg
- **Mint/Burn Fees**: 0.1% fees for stability and protocol revenue
- **Stability Bands**: ±2% bands with pressure mechanisms

### 3. Advanced Risk Management
- **Ebisu-Style Debt Caps**: A × B × C methodology for protocol-wide risk
- **Liquidation Cascades**: 8% penalty with 50% close factor
- **Concentrated Liquidity**: Uniswap V3-style pools for efficient liquidations
- **Real-Time Health Monitoring**: Continuous position health tracking

### 4. Integrated Liquidity System
- **MOET Trading Pairs**: MOET/USDC, MOET/ETH, MOET/BTC pools
- **Concentrated Liquidity**: 10% price range concentration for efficiency
- **Liquidation Capacity**: Dynamic calculation based on available liquidity
- **Fee Distribution**: 50% of protocol revenue to LP providers

### 5. Agent-Based Simulation
- **Tidal-Specific Policies**: Specialized lending behaviors for Tidal Protocol
- **Health Factor Strategies**: Emergency, conservative, and optimization modes
- **Collateral Diversification**: Intelligent asset allocation strategies
- **MOET Borrowing Logic**: Protocol-aware borrowing decisions

## Installation

1. Ensure you have Python 3.8+ installed
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

Run the modular simulation:
```bash
# Single simulation
python3 main.py --mode single --days 365 --verbose

# Monte Carlo analysis
python3 main.py --mode monte-carlo --runs 1000 --days 365

# Run examples
python3 main.py
```

### Tidal Protocol Configuration

Create Tidal-specific simulations:

```python
from src.config.schemas.tidal_config import create_default_config, PolicyConfig, PolicyType

# Create Tidal Protocol configuration
config = create_default_config()
config.simulation.agent_policies = [
    PolicyConfig(
        type=PolicyType.TIDAL_LENDER,  # Tidal-specific lender
        count=60,
        params={
            "target_health_factor": 2.0,
            "moet_borrowing_ratio": 0.7,
            "risk_tolerance": 0.6,
            "collateral_diversification": True
        }
    ),
    PolicyConfig(
        type=PolicyType.TRADER,
        count=40,
        params={"trading_frequency": 0.15, "risk_tolerance": 0.5}
    )
]

# Run Tidal simulation
from src.core.simulation.factory import SimulationFactory
simulation = SimulationFactory.create_simulation(config)
results = simulation.run_simulation(max_days=365)

# Analyze Tidal-specific metrics
tidal_data = results['history']['metrics'][-1]['tidal_protocol']
print(f"MOET Price: ${tidal_data['moet_stablecoin']['current_price']:.4f}")
print(f"Protocol Debt Cap: ${tidal_data['debt_cap']:,.0f}")
```

### Key Classes and Methods

#### TidalProtocolMarket
- `_handle_supply()`: Process collateral supply to earn interest
- `_handle_borrow()`: Handle MOET borrowing against collateral
- `_handle_liquidate()`: Execute liquidation of unhealthy positions
- `_handle_mint_moet()`: Direct MOET minting for stability
- `calculate_debt_cap()`: Ebisu-style debt cap calculation
- `_calculate_agent_health_factor()`: Real-time health monitoring

#### TidalLenderPolicy
- `_supply_strategy()`: Intelligent collateral supply decisions
- `_borrow_strategy()`: MOET borrowing with health factor management
- `_emergency_action()`: Crisis response for low health factors
- `_rebalance_strategy()`: Portfolio optimization across assets

#### TidalMath
- `calculate_kinked_interest_rate()`: Tidal's kinked rate model
- `calculate_debt_cap_ebisu_style()`: A × B × C debt cap formula
- `calculate_moet_mint_amount()`: MOET minting calculations
- `calculate_liquidation_amounts()`: Liquidation math with penalties

## 🏗️ Hybrid Architecture: Generic + Tidal-Specific

This simulation implements a **hybrid approach** that combines:

### Generic DeFi Framework
- **Universal Action System**: 40+ standardized DeFi primitives
- **Pluggable Markets**: AMM, lending, staking market interfaces
- **Configurable Policies**: Trader, lender, staker agent behaviors
- **Extensible Design**: Easy to add new protocols and mechanisms

### Tidal Protocol Specialization
- **TidalProtocolMarket**: Dedicated market with all Tidal-specific logic
- **MOET Stablecoin System**: Complete minting, burning, and peg management
- **Ebisu-Style Risk Management**: Sophisticated debt cap calculations
- **TidalLenderPolicy**: Protocol-aware agent behaviors
- **Tidal-Specific Math**: Kinked rates, liquidation formulas, stability mechanisms

### Benefits of Hybrid Approach
1. **Protocol Accuracy**: Deep Tidal-specific implementation captures all nuances
2. **Framework Flexibility**: Generic components enable other protocol simulations
3. **Modular Design**: Clean separation between generic and specific components
4. **Easy Extension**: Add new Tidal features or entirely new protocols

## Configuration Parameters

### Asset-Specific Parameters
```python
# Initial liquidity allocation
INITIAL_LIQUIDITY = {
    Asset.ETH: 7_000_000,    # $7M
    Asset.BTC: 3_500_000,    # $3.5M
    Asset.FLOW: 2_100_000,   # $2.1M
    Asset.USDC: 1_400_000    # $1.4M
}

# Collateral factors
COLLATERAL_FACTORS = {
    Asset.ETH: 0.75,    # 75% of value can be borrowed against
    Asset.BTC: 0.75,
    Asset.FLOW: 0.50,   # FLOW has lower collateral factor
    Asset.USDC: 0.90
}
```

### Interest Rate Model
```python
BASE_RATE_PER_BLOCK = 0
MULTIPLIER_PER_BLOCK = 11415525114
JUMP_PER_BLOCK = 253678335870
KINK = 8e17  # 80% utilization threshold
```

### Risk Parameters
```python
TARGET_HEALTH_FACTOR = 1.5
EXTREME_PRICE_DROPS = {
    Asset.ETH: 0.15,    # -15%
    Asset.BTC: 0.15,    # -15%
    Asset.FLOW: 0.35,   # -35%
    Asset.USDC: 0.15    # -15%
}
```

## Output and Analysis

### Protocol State Metrics
- Asset prices and utilization rates
- Borrow and supply APRs
- Health factors for each asset
- Protocol revenue and LP rewards
- Debt cap calculations

### Monte Carlo Results
- Statistical distributions of key metrics
- Risk analysis (e.g., percentage of scenarios with low health factors)
- Visualization plots for analysis

### Key Calculations

#### Health Factor
```
health_factor = (effective_collateral_value * collateral_factor) / borrowed_value
```

#### Utilization Rate
```
utilization = total_borrowed / total_supplied
```

#### Supply APR
```
supplyAPR = borrowAPR * utilization * (1 - reserve_factor)
```

#### Debt Cap (Ebisu-style)
```
debt_cap = A * B * C
```
Where:
- A: Liquidation capacity via DEX
- B: DEX liquidity allocation (35%)
- C: Underwater collateral percentage

## Extending the Simulation

### Adding New Assets
1. Add asset to `Asset` enum
2. Update `INITIAL_LIQUIDITY`, `INITIAL_PRICES`, `COLLATERAL_FACTORS`
3. Add to `EXTREME_PRICE_DROPS` if needed

### Modifying Interest Rate Model
1. Update constants in the configuration section
2. Modify `calculate_borrow_rate()` method in `TidalProtocol`

### Custom Risk Scenarios
1. Extend `MonteCarloSimulator.run_simulation()`
2. Add new shock types or correlation structures
3. Implement custom risk metrics

## Assumptions and Limitations

### Current Assumptions
- All liquidity is concentrated within 1 standard deviation of current price
- Simplified slippage calculation (not full Uniswap V3 math)
- Linear correlation between asset price changes
- Fixed collateral factors and health factor targets

### Limitations
- Does not model individual user positions
- Simplified liquidation mechanics
- No cross-asset correlation in extreme scenarios
- Assumes constant pool liquidity

## Contributing

To extend or modify the simulation:
1. Fork the repository
2. Create a feature branch
3. Implement changes with proper documentation
4. Add tests if applicable
5. Submit a pull request

## License

This simulation is provided for educational and research purposes. Please ensure compliance with any applicable licenses for the underlying protocols being simulated.

## Contact

For questions or suggestions about the simulation, please open an issue in the repository.
