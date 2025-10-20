# Advanced MOET System - Technical Documentation

## Overview

This document provides a comprehensive breakdown of the **Advanced MOET Stablecoin System**, which implements sophisticated reserve management, dynamic bond auctions, and economically-driven interest rate mechanisms. The system replaces simple utilization-based interest curves with a realistic economic model that ties MOET borrowing costs to the actual expense of maintaining the stablecoin's $1.00 peg through backing reserves.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Bonder System and Bond Auctions](#bonder-system-and-bond-auctions)
3. [Redeemer Contract and Reserve Management](#redeemer-contract-and-reserve-management)
4. [Interest Rate Calculation Formula](#interest-rate-calculation-formula)
5. [Integration with Tidal Protocol](#integration-with-tidal-protocol)
6. [Configuration and Governance Parameters](#configuration-and-governance-parameters)
7. [Simulation Integration](#simulation-integration)
8. [Performance Considerations](#performance-considerations)
9. [Usage Examples](#usage-examples)
10. [Comparison with Legacy System](#comparison-with-legacy-system)

## Architecture Overview

### Advanced MOET System Components

The Advanced MOET System consists of four interconnected components that work together to maintain the MOET stablecoin's peg and calculate realistic borrowing costs:

```
Advanced MOET System
├── MoetStablecoin (Core Stablecoin Logic)
│   ├── Basic mint/burn operations
│   ├── Peg stability monitoring
│   └── System state management
├── BonderSystem (Reserve Management)
│   ├── Dynamic bond auctions
│   ├── EMA cost tracking
│   └── Auction history management
├── RedeemerContract (Backing Reserves)
│   ├── USDC/USDF reserve pools
│   ├── Redemption processing
│   └── Reserve ratio monitoring
└── Interest Rate Engine
    ├── r_floor (governance profit margin)
    ├── r_bond_cost (EMA of auction costs)
    └── Peg stability adjustments
```

### Economic Model Philosophy

Unlike traditional DeFi protocols that use simple supply/demand curves, the Advanced MOET System implements an **economically-driven model** where:

- **Interest rates reflect real costs**: MOET borrowing rates are tied to the actual expense of maintaining backing reserves
- **Market-driven pricing**: Bond auction yields are determined by market conditions and reserve deficit urgency
- **Governance oversight**: Protocol parameters are controlled by governance while market forces determine operational costs
- **Peg maintenance**: The system automatically responds to reserve shortfalls through bond auctions

## Bonder System and Bond Auctions

### Bond Auction Mechanics

The `BonderSystem` class manages the core reserve maintenance mechanism through dynamic bond auctions:

#### Continuous Auction Logic

```python
def should_run_auction(self, reserve_state: ReserveState, total_moet_supply: float) -> bool:
    deficit = reserve_state.get_reserve_deficit(total_moet_supply)
    return deficit > 0.0  # Run auction whenever reserves < target
```

**Continuous Auction Conditions:**
- Auction runs whenever actual reserves < target reserves
- No trigger thresholds - immediate response to any deficit
- Auction stops automatically when reserves reach target (deficit = 0)

#### Pure Deficit-Based Pricing

Bond auctions use a pure deficit-based pricing model that directly reflects reserve pressure:

```python
# Pure deficit-based APR calculation
deficit_ratio = (target_reserves - actual_reserves) / target_reserves
bond_apr = max(0.0, deficit_ratio)

# No additional premiums or time-based increases
final_apr = bond_apr
```

**Pricing Parameters:**
- **Target Reserves**: 10% of total MOET supply
- **Initial Reserves**: 8% of total MOET supply (creates immediate 20% APR)
- **Continuous Auctions**: Always active when reserves < target
- **Real-time Updates**: APR recalculated every minute based on current deficit

#### Auction Fill Probability

The system simulates realistic market behavior with probability-based auction fills:

```python
def _calculate_fill_probability(self, auction: BondAuction, market_conditions: Dict) -> float:
    base_probability = 0.05  # 5% base chance per minute
    yield_factor = min(2.0, auction.final_apr / 0.05)  # Higher yield = higher probability
    stress_factor = market_conditions.get('stress_level', 1.0)
    return min(0.95, base_probability * yield_factor * stress_factor)
```

### EMA Cost Tracking

The Bonder System maintains an **Exponential Moving Average (EMA)** of bond costs with a 7-day half-life:

```python
def _update_bond_cost_ema(self, new_apr: float):
    half_life_minutes = 7 * 24 * 60  # 7 days = 10,080 minutes
    alpha = 1 - math.exp(-math.log(2) / half_life_minutes)
    self.current_bond_cost_ema = alpha * new_apr + (1 - alpha) * self.current_bond_cost_ema
```

**Benefits of EMA Tracking:**
- **Smooths volatility**: Prevents rapid interest rate swings from individual auctions
- **Historical context**: Incorporates recent market conditions into pricing
- **Predictable rates**: Provides stability for borrowers while reflecting real costs

## Redeemer Contract and Reserve Management

### Reserve State Management

The `RedeemerContract` maintains backing reserves in a 50/50 USDC/USDF split:

```python
@dataclass
class ReserveState:
    usdc_balance: float
    usdf_balance: float
    target_reserves_ratio: float  # 30% target
    
    def get_reserve_ratio(self, total_moet_supply: float) -> float:
        return self.total_reserves / total_moet_supply
    
    def get_reserve_deficit(self, total_moet_supply: float) -> float:
        target_amount = total_moet_supply * self.target_reserves_ratio
        return max(0, target_amount - self.total_reserves)
```

### Reserve Initialization

The system initializes reserves at **8% of initial MOET debt** to create immediate reserve pressure:

```python
def initialize_reserves(self, initial_moet_debt: float):
    initial_reserves = initial_moet_debt * 0.08  # 8% backing (creates immediate deficit)
    usdc_amount = initial_reserves * 0.50        # 4% USDC
    usdf_amount = initial_reserves * 0.50        # 4% USDF
```

**Strategic Benefits:**
- **Immediate 20% bond APR**: Creates significant reserve deficit from simulation start
- **Active auction system**: Forces continuous bond auction activity
- **Economic stress testing**: Tests system under realistic reserve pressure
- **Meaningful yield generation**: Creates substantial bond costs for MOET interest rates

### Redemption Processing

The Redeemer handles MOET redemptions at the $1.00 peg:

```python
def process_redemption(self, moet_amount: float) -> bool:
    if self.reserve_state.total_reserves >= moet_amount:
        # Redeem proportionally from USDC/USDF
        total_reserves = self.reserve_state.total_reserves
        usdc_ratio = self.reserve_state.usdc_balance / total_reserves
        usdf_ratio = self.reserve_state.usdf_balance / total_reserves
        
        self.reserve_state.usdc_balance -= moet_amount * usdc_ratio
        self.reserve_state.usdf_balance -= moet_amount * usdf_ratio
        return True
    return False
```

## Interest Rate Calculation Formula

### Core Formula: r_MOET = r_floor + r_bond_cost

The Advanced MOET System calculates interest rates using a **two-component formula**:

```python
def _calculate_moet_interest_rate(self) -> float:
    r_floor = self.governance_params['r_floor']          # 2% governance
    r_bond_cost = self.bonder_system.get_current_bond_cost()  # EMA of bond costs
    peg_adjustment = 0.0  # Future enhancement for ff-peg scenarios
    
    return r_floor + r_bond_cost + peg_adjustment
```

### Component Breakdown

#### r_floor (Governance Profit Margin)
- **Value**: 2% APR
- **Purpose**: Provides revenue to Tidal Treasury and insurance funds
- **Control**: Set by governance, remains stable
- **Allocation**: Funds protocol operations and risk management

#### r_bond_cost (Market-Driven Component)
- **Calculation**: EMA of actual bond auction APRs with 7-day half-life
- **Purpose**: Passes through the real cost of maintaining reserves
- **Variability**: Fluctuates based on market conditions and reserve stress
- **Economic Logic**: Borrowers pay the actual cost of peg maintenance

#### Peg Adjustment (Future Enhancement)
- **Current Value**: 0% (MOET maintained at $1.00 peg)
- **Future Use**: Dynamic adjustments when MOET trades off-peg
- **Mechanism**: 
  - MOET > $1.00 → Lower rates → Encourage minting → Increase supply
  - MOET < $1.00 → Higher rates → Encourage repayment → Decrease supply

### Interest Rate Update Logic

Interest rates update when the calculated rate changes by more than 1 basis point:

```python
new_rate = self._calculate_moet_interest_rate()
if abs(new_rate - self.current_moet_interest_rate) > 0.0001:  # 1 bps threshold
    self.current_moet_interest_rate = new_rate
    # Log rate change and components
```

## Integration with Tidal Protocol

### Protocol-Level Integration

The Advanced MOET System integrates seamlessly with the existing Tidal Protocol architecture:

```python
class TidalProtocol:
    def __init__(self, enable_advanced_moet: bool = False):
        self.moet_system = MoetStablecoin(enable_advanced_system=enable_advanced_moet)
        self.enable_advanced_moet = enable_advanced_moet
    
    def get_moet_borrow_rate(self) -> float:
        if self.enable_advanced_moet:
            return self.moet_system.get_current_interest_rate()
        else:
            # Fallback to legacy BTC pool utilization rate
            return self.asset_pools[Asset.BTC].calculate_borrow_rate()
```

### Agent Interest Updates

High Tide agents receive updated interest rates through the existing interface:

```python
def _update_agent_debt_interest(self, minute: int):
    # Uses new MOET system rate instead of BTC pool rate
    moet_borrow_rate = self.protocol.get_moet_borrow_rate()
    
    for agent in self.high_tide_agents:
        if agent.active:
            agent.update_debt_interest(minute, moet_borrow_rate)
```

### Simulation Loop Integration

The system processes MOET updates each simulation minute:

```python
# Process MOET system updates (bond auctions, interest rate calculations)
moet_update_results = self.protocol.process_moet_system_update(minute)

# Log significant events
if moet_update_results.get('bond_auction_triggered'):
    print(f"🔔 Bond auction triggered at minute {minute}")
if moet_update_results.get('interest_rate_updated'):
    print(f"📈 MOET rate updated: {moet_update_results['new_interest_rate']:.2%}")
```

## Configuration and Governance Parameters

### Governance Parameters

The system uses the following governance-controlled parameters:

```python
governance_params = {
    'r_floor': 0.02,                    # 2% governance profit margin
    'target_reserves_ratio': 0.10,      # 10% target reserve ratio (updated)
    'ema_half_life_days': 7,            # 7-day EMA smoothing
}
```

### Auction Parameters

Bond auction behavior is controlled by these parameters:

```python
# Auction timing (simplified)
base_auction_duration = 60          # 1 hour default duration
max_auction_duration = 360          # 6 hours maximum duration

# Pure deficit-based pricing (no premiums)
# APR = (Target Reserves - Actual Reserves) / Target Reserves

# Continuous operation (no triggers)
# Auction runs whenever deficit > 0
```

### Reserve Management

Reserve management follows these configured ratios:

```python
# Initial setup (creates immediate deficit)
initial_reserve_ratio = 0.08        # 8% of initial debt as reserves
usdc_usdf_split = 0.50             # 50/50 USDC/USDF split

# Target maintenance
target_reserves_ratio = 0.10        # 10% ongoing target
```

## Simulation Integration

### Full Year Simulation Integration

The Advanced MOET System integrates with `full_year_sim.py` through a simple toggle:

```python
class FullYearSimConfig:
    def __init__(self):
        # ... other config ...
        self.enable_advanced_moet_system = True  # Enable sophisticated system
```

### Engine Configuration

The system propagates through the engine hierarchy:

```python
# BaseLendingEngine initialization
enable_advanced_moet = getattr(config, 'enable_advanced_moet_system', False)
self.protocol = TidalProtocol(enable_advanced_moet=enable_advanced_moet)

# HighTideConfig setup
ht_config.enable_advanced_moet_system = self.config.enable_advanced_moet_system
```

### Reserve Initialization in Simulations

Reserves are automatically initialized based on total agent debt:

```python
if self.protocol.enable_advanced_moet:
    total_agent_debt = sum(agent.state.moet_debt for agent in self.high_tide_agents)
    self.protocol.initialize_moet_reserves(total_agent_debt)
    print(f"🏦 Initialized MOET reserves: ${total_agent_debt * 0.5:,.0f}")
```

## Performance Considerations

### Computational Efficiency

The Advanced MOET System is designed for efficient long-term simulations:

#### Memory Management
- **Bounded history**: Auction history limited to recent events
- **EMA calculations**: O(1) updates instead of full historical calculations
- **Efficient state tracking**: Minimal memory footprint per minute

#### Computational Complexity
- **Bond auction processing**: O(1) per minute when no auction active
- **Interest rate updates**: O(1) calculation with 1 bps threshold
- **Reserve management**: O(1) balance updates

### Optimization Features

The system includes several performance optimizations:

#### Auction Processing
- **Probability-based fills**: Avoids complex market simulation
- **Timeout handling**: Prevents infinite auction duration
- **Batch logging**: Reduces I/O overhead during simulations

#### Interest Rate Caching
- **1 bps threshold**: Prevents unnecessary updates for tiny changes
- **Component tracking**: Separate logging of r_floor and r_bond_cost
- **Efficient EMA**: Mathematical optimization for 7-day half-life

## Usage Examples

### Basic Integration Example

```python
# Enable Advanced MOET System
from tidal_protocol_sim.core.protocol import TidalProtocol

# Create protocol with advanced MOET
protocol = TidalProtocol(enable_advanced_moet=True)

# Initialize reserves (typically done automatically in simulations)
initial_debt = 1_000_000  # $1M total agent debt
protocol.initialize_moet_reserves(initial_debt)

# Process simulation minutes
for minute in range(1000):
    # Update MOET system (auctions, rates, etc.)
    results = protocol.process_moet_system_update(minute)
    
    # Get current borrowing rate for agents
    current_rate = protocol.get_moet_borrow_rate()
    
    # Check for significant events
    if results.get('bond_auction_completed'):
        auction = results['completed_auction']
        print(f"Auction filled: ${auction['amount_filled']:,.0f} at {auction['final_apr']:.2%}")
```

### Full Year Simulation Example

```python
# Configure full year simulation with Advanced MOET
from sim_tests.full_year_sim import FullYearSimConfig, FullYearSimulation

# Create configuration
config = FullYearSimConfig()
config.enable_advanced_moet_system = True  # Enable advanced system

# Run simulation
simulation = FullYearSimulation(config)
results = simulation.run_test()

# Analyze MOET system performance
moet_state = results['simulation_results']['moet_system_state']
print(f"Final interest rate: {moet_state['current_interest_rate']:.2%}")
print(f"Total auctions: {moet_state['bonder_system']['auction_history_count']}")
```

### Custom Configuration Example

```python
# Create MOET system with custom parameters
from tidal_protocol_sim.core.moet import MoetStablecoin

# Custom governance parameters
custom_params = {
    'r_floor': 0.015,           # 1.5% governance rate
    'target_reserves_ratio': 0.25,  # 25% target reserves
    'ema_half_life_days': 14,   # 14-day EMA smoothing
    'benchmark_rate': 0.04      # 4% benchmark rate
}

# Initialize with custom parameters (requires code modification)
moet_system = MoetStablecoin(enable_advanced_system=True)
moet_system.governance_params.update(custom_params)
```

## Comparison with Legacy System

### Legacy System Characteristics

The original MOET system used simple utilization-based rates:

```python
# Legacy approach
def get_legacy_rate():
    btc_pool = protocol.asset_pools[Asset.BTC]
    utilization = btc_pool.total_borrowed / btc_pool.total_supplied
    
    if utilization <= 0.80:  # Below kink
        rate = base_rate + (utilization * multiplier)
    else:  # Above kink
        rate = base_rate + kink_rate + ((utilization - 0.80) * jump_multiplier)
    
    return rate
```

**Legacy System Issues:**
- **Artificial rates**: Not tied to real economic costs
- **Pool tracking gaps**: MOET debt not properly tracked in pools
- **Zero utilization**: Effective 0% rates due to tracking bugs
- **No reserve backing**: No mechanism to maintain peg stability

### Advanced System Benefits

The new system addresses all legacy issues:

| Aspect | Legacy System | Advanced System |
|--------|---------------|-----------------|
| **Rate Basis** | Artificial utilization curve | Real economic costs (r_floor + r_bond_cost) |
| **Peg Maintenance** | No mechanism | Active reserve management + bond auctions |
| **Economic Realism** | Disconnected from costs | Directly tied to reserve maintenance costs |
| **Governance Control** | Limited parameters | Clear separation of governance vs market rates |
| **Market Response** | Static curves | Dynamic response to market conditions |
| **Reserve Backing** | None | 30% target with automatic replenishment |

## System State Monitoring and JSON Tracking

### Comprehensive State Reporting

The Advanced MOET System provides detailed state information and comprehensive tracking for analysis:

```python
# Get complete system state
state = protocol.get_moet_system_state()

# Key metrics available:
print(f"Interest Rate: {state['current_interest_rate']:.2%}")
print(f"Reserve Ratio: {state['reserve_state']['total_reserves'] / moet_supply:.1%}")
print(f"Bond Cost EMA: {state['bonder_system']['current_bond_cost_ema']:.2%}")
print(f"Recent Auctions: {len(state['bonder_system']['recent_auctions'])}")
```

### Auction Activity Summary

```python
# Get auction performance metrics
summary = protocol.moet_system.get_auction_summary()

print(f"Total Auctions: {summary['total_auctions']}")
print(f"Total Raised: ${summary['total_raised']:,.0f}")
print(f"Average APR: {summary['avg_apr']:.2%}")
print(f"Fill Rate: {summary['fill_rate']:.1%}")
print(f"Current EMA Cost: {summary['current_ema_cost']:.2%}")
```

### Real-Time Event Monitoring

The system provides detailed logging of all significant events:

```python
# Bond auction events
🔔 Bond auction triggered at minute 1440
   Target: $150,000
   Starting APR: 7.50%

✅ Bond auction completed at minute 1502
   Filled: $150,000
   Final APR: 8.25%
   Duration: 62 minutes

# Interest rate updates
📈 MOET rate updated: 4.15% (was 3.89%)
   r_floor: 2.00%
   r_bond_cost: 2.15% (EMA updated)
```

### Enhanced JSON Tracking

The system now provides minute-by-minute tracking of key metrics in simulation results:

```python
# JSON results include comprehensive MOET system data
"moet_system_state": {
    "tracking_data": {
        "moet_rate_history": [
            {
                "minute": 0,
                "moet_interest_rate": 0.02,
                "r_floor": 0.02,
                "r_bond_cost": 0.0
            },
            // ... minute-by-minute data
        ],
        "bond_apr_history": [
            {
                "minute": 0,
                "bond_apr": 0.20,
                "deficit_ratio": 0.20
            },
            // ... minute-by-minute bond APRs
        ],
        "reserve_history": [
            {
                "minute": 0,
                "target_reserves": 100000,
                "actual_reserves": 80000,
                "reserve_ratio": 0.08,
                "target_ratio": 0.10
            },
            // ... minute-by-minute reserve data
        ],
        "deficit_history": [
            {
                "minute": 0,
                "deficit": 20000,
                "deficit_ratio": 0.20
            },
            // ... minute-by-minute deficit tracking
        ]
    }
},
"moet_system_summary": {
    "initial_moet_rate": 0.02,
    "final_moet_rate": 0.045,
    "avg_moet_rate": 0.032,
    "max_bond_apr": 0.20,
    "avg_bond_apr": 0.15,
    "initial_reserve_ratio": 0.08,
    "final_reserve_ratio": 0.095,
    "max_deficit": 20000,
    "avg_deficit": 12500,
    "total_data_points": 525600
}
```

**Tracking Benefits:**
- **Complete audit trail**: Every minute of MOET system operation
- **Performance analysis**: Detailed metrics for system optimization
- **Economic validation**: Verify deficit-based pricing works as designed
- **Research insights**: Understand bond auction dynamics over time

This comprehensive monitoring enables detailed analysis of the MOET system's performance and provides insights into the economic dynamics driving interest rate changes.

## Enhanced Redeemer System with Dynamic Fee Structure

### Overview

The Enhanced Redeemer system introduces a sophisticated fee structure that incentivizes balanced pool interactions while generating revenue from convenience-seeking users. The system maintains a 50/50 USDC/USDF reserve pool and applies dynamic fees based on transaction impact.

### Fee Structure

#### Deposit Fees

**Balanced Deposits (50/50 USDC/USDF)**:
- Base fee: 0.01%
- Imbalance fee: 0% (no deviation from ideal weights)
- Total fee: 0.01%

**Imbalanced Deposits (Single asset or off-ratio)**:
- Base fee: 0.02%
- Imbalance fee: `K * max(0, Δw(post) - Δw(tol))^γ`
  - K = 50 bps (0.5%) scale factor
  - γ = 2.0 (quadratic scaling)
  - Δw(tol) = 2% tolerance band

#### Redemption Fees

**Proportional Redemptions**:
- Fee: 0% (maintains pool balance)
- User receives blend of USDC/USDF matching current ratios

**Single-Asset Redemptions**:
- Base fee: 0.02%
- Imbalance fee: Same formula as deposits
- Premium for convenience of receiving specific asset

### Implementation Examples

#### Example 1: Balanced Deposit
```python
# User deposits $10,000 (50/50 USDC/USDF)
deposit_result = protocol.mint_moet_from_deposit(5000, 5000)

# Results:
# - MOET minted: $9,999.00
# - Total fee: $1.00 (0.01%)
# - Base fee: $1.00
# - Imbalance fee: $0.00
# - Pool remains balanced
```

#### Example 2: Imbalanced Deposit
```python
# User deposits $10,000 (100% USDC)
deposit_result = protocol.mint_moet_from_deposit(10000, 0)

# Results:
# - MOET minted: $9,997.95
# - Total fee: $2.05 (0.021%)
# - Base fee: $2.00 (0.02%)
# - Imbalance fee: $0.05 (penalty for creating imbalance)
# - Pool becomes USDC-heavy
```

#### Example 3: Proportional Redemption
```python
# User redeems $5,000 MOET proportionally
redemption_result = protocol.redeem_moet_for_assets(5000, "proportional")

# Results:
# - USDC received: $2,750 (55% of current pool)
# - USDF received: $2,250 (45% of current pool)
# - Total fee: $0.00
# - Pool balance maintained
```

#### Example 4: Single-Asset Redemption
```python
# User redeems $3,000 MOET for USDC only
redemption_result = protocol.redeem_moet_for_assets(3000, "USDC")

# Results:
# - USDC received: $2,999.40
# - USDF received: $0.00
# - Total fee: $0.60 (0.02% base + imbalance fee)
# - Pool becomes more USDF-heavy
```

### Fee Calculation Formula

The imbalance fee uses the formula:
```
fee(imb) = K * max(0, Δw(post) - Δw(tol))^γ

Where:
- K = 0.005 (50 basis points scale factor)
- γ = 2.0 (quadratic convexity)
- Δw(post) = Post-transaction weight deviation from 50/50
- Δw(tol) = 0.02 (2% tolerance band)
```

### Integration with Bond Auctions

The Enhanced Redeemer works seamlessly with the existing bond auction system:

1. **Bond proceeds** are added as 50/50 USDC/USDF (maintains balance)
2. **Fee revenue** accumulates in reserves, reducing bond auction frequency
3. **Reserve targets** remain at 10% of total MOET supply
4. **Dual replenishment**: Both user fees and bond auctions maintain reserves

### API Methods

#### Protocol-Level Methods
```python
# Mint MOET from deposits
result = protocol.mint_moet_from_deposit(usdc_amount, usdf_amount)

# Redeem MOET for assets
result = protocol.redeem_moet_for_assets(moet_amount, desired_asset)

# Estimate fees before transactions
fee_estimate = protocol.estimate_deposit_fee(usdc_amount, usdf_amount)
fee_estimate = protocol.estimate_redemption_fee(moet_amount, desired_asset)

# Get pool information
weights = protocol.get_redeemer_pool_weights()
optimal = protocol.get_optimal_deposit_ratio(total_amount)
```

#### Direct Redeemer Methods
```python
# Access enhanced redeemer directly
redeemer = protocol.moet_system.redeemer

# Execute transactions
deposit_result = redeemer.deposit_assets_for_moet(usdc_amount, usdf_amount)
redemption_result = redeemer.redeem_moet_for_assets(moet_amount, desired_asset)

# Get state information
current_weights = redeemer.get_current_pool_weights()
optimal_ratio = redeemer.get_optimal_deposit_ratio(amount)
```

### Economic Incentives

The fee structure creates proper economic incentives:

1. **Balanced users pay minimal fees** (0.01% deposits, 0% redemptions)
2. **Imbalanced users pay for convenience** (higher fees + quadratic penalties)
3. **Pool maintains health** through economic disincentives for imbalance
4. **Protocol earns revenue** from imbalance fees and convenience premiums
5. **Reduced bond reliance** as fee revenue helps maintain target reserves

### Performance Characteristics

Based on integration testing:

- **Balanced deposits**: ~0.01% fee (e.g., $1 fee on $10k deposit)
- **Imbalanced deposits**: ~0.02-0.12% fee depending on deviation
- **Proportional redemptions**: 0% fee
- **Single-asset redemptions**: ~0.02-0.14% fee depending on impact
- **Pool rebalancing**: Automatic through economic incentives

This enhanced system provides a sustainable, economically-driven mechanism for maintaining MOET's peg while generating protocol revenue and reducing reliance on bond auctions.

---
