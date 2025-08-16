# Tidal Protocol Lending Simulation

A comprehensive Python simulation of the Tidal lending protocol with Uniswap V3-style liquidity pools and Monte Carlo stress testing capabilities.

## Overview

This simulation models the Tidal lending protocol, which allows users to:
- Supply collateral assets (ETH, BTC, FLOW, USDC) to earn interest
- Borrow MOET (protocol stablecoin) against their collateral
- Participate in liquidity pools for protocol rewards

## Features

### 1. Protocol Configuration
- **Supported Assets**: ETH, BTC, FLOW, USDC, MOET (stablecoin)
- **Total Protocol Liquidity**: $14,000,000
- **Reserve Factor**: 15% of interest goes to protocol revenue
- **LP Rewards**: 50% of protocol revenue distributed to liquidity providers

### 2. Interest Rate Model
- **Kinked Interest Rate Model**: Linear rates below 80% utilization, jump rates above
- **Per-block calculations** with conversion to annual rates
- **Dynamic rates** based on asset utilization

### 3. Collateral and Risk Management
- **Collateral Factors**: ETH/BTC (75%), FLOW (50%), USDC (90%)
- **Target Health Factor**: 1.5x
- **Extreme Price Drops**: ETH/BTC/USDC (-15%), FLOW (-35%)

### 4. Uniswap V3-Style Liquidity Pools
- **Concentrated liquidity** modeling with normal distribution
- **Slippage calculations** for liquidation scenarios
- **Liquidation capacity** estimation for debt cap calculations

### 5. Monte Carlo Simulation
- **Stress testing** with random price shocks
- **Utilization changes** simulation
- **Risk analysis** across multiple scenarios
- **Statistical summaries** and visualizations

## Installation

1. Ensure you have Python 3.8+ installed
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the complete simulation:
```bash
python tidal_protocol_simulation.py
```

This will:
1. Initialize the protocol with default parameters
2. Run 1000 Monte Carlo simulations
3. Display summary statistics
4. Generate visualization plots

### Custom Simulations

You can modify the simulation parameters in the script:

```python
# Change number of simulations
simulator = MonteCarloSimulator(protocol, n_simulations=5000)

# Modify price shock parameters
shock = np.random.normal(-0.10, 0.05)  # Custom shock distribution

# Adjust utilization changes
utilization_change = np.random.normal(0, 0.15)  # Higher volatility
```

### Key Classes and Methods

#### TidalProtocol
- `calculate_borrow_rate(asset)`: Calculate borrow rate using kinked model
- `calculate_supply_rate(asset)`: Calculate supply rate based on utilization
- `calculate_debt_cap()`: Calculate debt cap using Ebisu methodology
- `apply_price_shock(asset, shock_percentage)`: Apply price shocks

#### UniswapV3Pool
- `calculate_slippage(amount_in, is_buy)`: Calculate trade slippage
- `get_liquidation_capacity(max_slippage)`: Estimate liquidation capacity

#### MonteCarloSimulator
- `run_simulation()`: Execute Monte Carlo simulations
- `generate_summary_statistics()`: Create statistical summaries
- `plot_results()`: Generate visualization plots

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
