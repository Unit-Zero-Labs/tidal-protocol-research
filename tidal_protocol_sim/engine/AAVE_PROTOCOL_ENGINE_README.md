# AAVE Protocol Engine - Technical Documentation

## Overview

This document provides a comprehensive breakdown of the `aave_protocol_engine.py` script, which serves as the **orchestration layer** for the AAVE Protocol simulation. The engine coordinates traditional DeFi lending strategies with passive yield token management, implements AAVE-style liquidation mechanics with Uniswap V3 integration, and simulates realistic market stress conditions to compare against High Tide's active rebalancing approach.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components and Integration](#core-components-and-integration)
3. [Monte Carlo Simulation Framework](#monte-carlo-simulation-framework)
4. [Agent Orchestration and Management](#agent-orchestration-and-management)
5. [BTC Price Decline Simulation](#btc-price-decline-simulation)
6. [Yield Token Strategy Implementation](#yield-token-strategy-implementation)
7. [Liquidation and Risk Management](#liquidation-and-risk-management)
8. [Performance Analytics and Reporting](#performance-analytics-and-reporting)
9. [Configuration and Customization](#configuration-and-customization)
10. [Usage Examples](#usage-examples)

## Architecture Overview

### AAVE Protocol Engine as Orchestration Layer

The `AaveProtocolEngine` serves as the central coordination system that:

- **Inherits from `BaseLendingEngine`**: Gets core lending functionality without Tidal Protocol dependencies
- **Manages AAVE Agents**: Orchestrates 10-50 agents with varied risk profiles in Monte Carlo simulations
- **Coordinates Yield Token Strategies**: Integrates with the [Yield Token System](../core/YIELD_TOKENS_README.md) for passive portfolio management
- **Simulates Market Stress**: Implements BTC price decline scenarios using realistic volatility patterns
- **Implements AAVE Liquidations**: Traditional 50% collateral seizure + 5% bonus mechanics with Uniswap V3 swap integration
- **Tracks Performance**: Provides comprehensive analytics and reporting across all simulation components

### System Integration Architecture

```
AaveProtocolEngine (Orchestration Layer)
├── BaseLendingEngine (Core Lending)
│   ├── Protocol State Management
│   └── Basic Lending Operations
├── AAVE Agents (10-50 Monte Carlo Agents)
│   ├── Yield Token Portfolios (Passive)
│   ├── Risk Profile Management
│   └── No Rebalancing Logic
├── BTC Price Manager (Market Stress Simulation)
├── Uniswap V3 Integration (Liquidation Swaps)
└── Analytics & Reporting (Performance Tracking)
```

## Core Components and Integration

### 1. Base Lending Engine Integration

The engine inherits core lending functionality from the base `BaseLendingEngine`:

```python
class AaveProtocolEngine(BaseLendingEngine):
    """Pure AAVE Protocol implementation"""
    
    def __init__(self, config: AaveConfig):
        super().__init__(config)
        self.aave_config = config
        
        # AAVE liquidation parameters setup
        self._setup_aave_liquidation_pools()
```

**Key Inherited Capabilities:**
- **Core Lending Operations**: Basic borrow/lend functionality without Tidal Protocol complexity
- **Protocol State Management**: Interest accrual, debt tracking, and basic protocol operations
- **Agent Management**: Base agent coordination and action processing
- **Metrics Collection**: Basic performance tracking and result generation

### 2. Yield Token Pool Integration

The engine creates and manages the global yield token trading infrastructure:

```python
# Initialize AAVE specific components
self.yield_token_pool = YieldTokenPool(config.moet_btc_pool_size)
```

**Integration with Yield Token System:**
- **Pool Configuration**: Uses [Pool Configuration](../core/YIELD_TOKENS_README.md#pool-configuration) with standard concentration
- **Passive Trading Operations**: Leverages [Trading Operations](../core/YIELD_TOKENS_README.md#trading-operations) for initial purchases only
- **No Rebalancing**: AAVE agents don't actively trade yield tokens after initial purchase
- **Portfolio Management**: Coordinates with [Portfolio Management](../core/YIELD_TOKENS_README.md#portfolio-management) for passive tracking

### 3. Agent Creation and Management

The engine creates and manages a diverse set of AAVE agents with varied risk profiles:

```python
# Replace agents with AAVE agents
self.aave_agents = create_aave_agents(
    config.num_aave_agents,
    config.monte_carlo_agent_variation
)
```

**Agent Distribution (Monte Carlo):**
- **Conservative (30%)**: Initial HF = 2.1-2.4, Target HF = Initial - 0.05-0.15
- **Moderate (40%)**: Initial HF = 1.5-1.8, Target HF = Initial - 0.15-0.25  
- **Aggressive (30%)**: Initial HF = 1.3-1.5, Target HF = Initial - 0.15-0.4
- **Minimum Target HF**: 1.1 for all agents (safety threshold)
- **No Rebalancing**: `automatic_rebalancing = False` for all agents

## Monte Carlo Simulation Framework

### Simulation Configuration

The engine implements sophisticated Monte Carlo simulation capabilities:

```python
class AaveConfig(BaseLendingConfig):
    """Pure AAVE configuration - no Tidal dependencies"""
    
    def __init__(self):
        super().__init__()
        self.scenario_name = "AAVE_Protocol"
        
        # AAVE liquidation parameters
        self.liquidation_threshold = 0.85
        self.liquidation_bonus = 0.05  # 5%
        self.liquidation_percentage = 0.5  # 50%
        
        # Yield token parameters (for fair comparison)
        self.yield_apr = 0.10  # 10% APR
        self.moet_btc_pool_size = 500_000  # Same as High Tide for comparison
        
        # BTC price decline parameters (for fair comparison)
        self.btc_initial_price = 100_000.0
        self.btc_decline_duration = 60  # 60 minutes
        self.btc_final_price_range = (75_000.0, 85_000.0)  # 15-25% decline
        
        # Agent configuration matches High Tide
        self.num_aave_agents = 20
        self.monte_carlo_agent_variation = True
        
        # Disable rebalancing for AAVE scenario
        self.rebalancing_enabled = False
```

### Monte Carlo Agent Variation

The engine supports dynamic agent population and risk profile distribution:

```python
def create_aave_agents(num_agents: int, monte_carlo_variation: bool = True) -> list:
    """
    Create AAVE agents with SAME risk profile distribution as High Tide agents
    
    This ensures fair comparison between High Tide and AAVE strategies
    """
    if monte_carlo_variation:
        # Randomize agent count between 10-50 (same as High Tide)
        num_agents = random.randint(10, 50)
```

**Monte Carlo Features:**
- **Dynamic Agent Count**: 10-50 agents per simulation run
- **Risk Profile Randomization**: Varied initial and target health factors
- **BTC Price Variation**: Random final price within specified range
- **Statistical Analysis**: Multiple runs for robust performance metrics
- **Fair Comparison**: Identical agent distribution to High Tide for valid comparison

## Agent Orchestration and Management

### Agent Action Processing

The engine orchestrates agent decision-making and action execution:

```python
def _process_aave_agents(self, minute: int) -> Dict[str, Dict]:
    """Process AAVE agent actions for current minute"""
    swap_data = {}
    
    for agent in self.aave_agents:
        if not agent.active:
            continue
            
        # Get agent's decision
        protocol_state = self._get_protocol_state()
        protocol_state["current_step"] = minute
        
        action_type, params = agent.decide_action(protocol_state, self.state.current_prices)
        
        # Execute action and capture swap data
        success, agent_swap_data = self._execute_aave_action(agent, action_type, params, minute)
```

### AAVE Action Execution

The engine handles simplified action execution with no rebalancing:

```python
def _execute_aave_action(self, agent: AaveAgent, action_type: AgentAction, params: dict, minute: int) -> tuple:
    """Execute AAVE specific actions (only initial yield token purchase)"""
    if action_type == AgentAction.SWAP:
        swap_type = params.get("action_type", "")
        
        if swap_type == "buy_yield_tokens":
            success = self._execute_yield_token_purchase(agent, params, minute)
            return success, None
            
    # AAVE agents don't do any other swaps (no rebalancing)
    return False, None
```

**Supported Actions:**
- **Yield Token Purchase**: Initial investment only (no rebalancing)
- **No Leverage Management**: No dynamic borrowing based on health factors
- **No Emergency Actions**: No crisis management or liquidation prevention
- **Passive Strategy**: Buy and hold until liquidation

### Yield Token Purchase Orchestration

The engine coordinates yield token purchases with simple execution:

```python
def _execute_yield_token_purchase(self, agent: AaveAgent, params: dict, minute: int) -> bool:
    """Execute yield token purchase for agent (same as High Tide)"""
    moet_amount = params.get("moet_amount", 0.0)
    
    if moet_amount <= 0:
        return False
        
    success = agent.execute_yield_token_purchase(moet_amount, minute)
    
    if success:
        self.yield_token_pool.execute_yield_token_purchase(moet_amount)
        
        self.yield_token_trades.append({
            "minute": minute,
            "agent_id": agent.agent_id,
            "action": "purchase",
            "moet_amount": moet_amount,
            "agent_health_factor": agent.state.health_factor
        })
        
    return success
```

**Integration with Yield Token System:**
- **Simple Purchases**: Uses [Trading Operations](../core/YIELD_TOKENS_README.md#trading-operations) for initial purchases
- **No Sales**: AAVE agents never sell yield tokens (no rebalancing)
- **Pool State Management**: Coordinates with [Pool Integration](../core/YIELD_TOKENS_README.md#pool-integration) for state consistency
- **Passive Tracking**: Records purchases but no sales or rebalancing activity

## BTC Price Decline Simulation

### Market Stress Implementation

The engine implements realistic BTC price decline scenarios:

```python
# BTC Price Manager is used for BTC price decline
self.btc_price_manager = BTCPriceDeclineManager(
    initial_price=config.btc_initial_price,
    duration=config.btc_decline_duration,
    final_price_range=config.btc_final_price_range
)
```

**Price Decline Characteristics:**
- **Initial Price**: $100,000 (configurable)
- **Duration**: 60 minutes (configurable)
- **Final Range**: $75,000 - $85,000 (15-25% decline)
- **Volatility Pattern**: Historical decline rates with realistic variation
- **Convergence**: Gradual convergence to target price in final 20% of simulation

### Price Update Integration

The engine integrates BTC price updates throughout the simulation:

```python
for minute in range(steps):
    self.current_step = minute
    
    # Update BTC price
    new_btc_price = self.btc_price_manager.update_btc_price(minute)
    self.state.current_prices[Asset.BTC] = new_btc_price
    self.btc_price_history.append(new_btc_price)
    
    # Update protocol state
    self.protocol.current_block = minute
    self.protocol.accrue_interest()
    
    # Update agent debt interest
    self._update_agent_debt_interest(minute)
```

**Price Impact on System:**
- **Health Factor Updates**: All agents recalculate health factors with new BTC price
- **Collateral Value Changes**: BTC collateral value decreases, affecting borrowing capacity
- **No Rebalancing Triggers**: AAVE agents don't rebalance as health factors decline
- **Liquidation Risk**: Increased risk of liquidation as BTC price falls (passive response)

## Yield Token Strategy Implementation

### Initial Investment Strategy

The engine implements simple initial investment strategies:

```python
def _initial_yield_token_purchase(self, current_minute: int) -> tuple:
    """Purchase yield tokens with initially borrowed MOET"""
    moet_available = self.state.borrowed_balances.get(Asset.MOET, 0.0)
    
    if moet_available > 0:
        # Use all borrowed MOET to purchase yield tokens
        return (AgentAction.SWAP, {
            "action_type": "buy_yield_tokens",
            "moet_amount": moet_available,
            "current_minute": current_minute
        })
```

**Investment Logic:**
- **Full MOET Utilization**: All borrowed MOET converted to yield tokens
- **One-Time Purchase**: Only initial purchase, no subsequent trading
- **Passive Management**: Hold yield tokens until liquidation (no rebalancing)

### No Rebalancing Strategy

AAVE agents implement a **passive strategy** with no rebalancing:

```python
def decide_action(self, protocol_state: dict, asset_prices: Dict[Asset, float]) -> tuple:
    """
    AAVE-style decision logic:
    1. Initially purchase yield tokens with borrowed MOET (same as High Tide)
    2. NO rebalancing - hold position until liquidation
    3. Track health factor but take no action
    """
    current_minute = protocol_state.get("current_step", 0)
    
    # Update health factor
    self._update_health_factor(asset_prices)
    
    # Check if we need to purchase yield tokens initially (same as High Tide)
    if (self.state.moet_debt > 0 and 
        len(self.state.yield_token_manager.yield_tokens) == 0):
        return self._initial_yield_token_purchase(current_minute)
    
    # NO REBALANCING - this is the key difference from High Tide
    # AAVE agents hold their position until liquidation
    
    # Default action - hold position (no matter what the health factor is)
    return (AgentAction.HOLD, {})
```

**Passive Strategy Logic:**
- **No Health Factor Triggers**: No rebalancing when HF falls below target threshold
- **No Yield Sales**: Never sells yield tokens for debt repayment
- **No Leverage Management**: Never increases or decreases leverage
- **Hold Until Liquidation**: Maintains position until health factor ≤ 1.0

## Liquidation and Risk Management

### Health Factor Monitoring

The engine continuously monitors agent health factors:

```python
def _check_aave_liquidations(self, minute: int):
    """Check for AAVE-style liquidations (HF ≤ 1.0)"""
    for agent in self.aave_agents:
        if not agent.active:
            continue
            
        # Update health factor
        agent._update_health_factor(self.state.current_prices)
        
        # Check if liquidation is needed (HF ≤ 1.0)
        if agent.state.health_factor <= 1.0:
            # Liquidation is executed by the agent itself using Uniswap V3
            liquidation_event = agent.execute_aave_liquidation(minute, self.state.current_prices)
            
            if liquidation_event:
                self.liquidation_events.append(liquidation_event)
```

**Health Factor Calculation:**
- **Collateral Value**: BTC amount × BTC price × collateral factor (0.80)
- **Debt Value**: MOET debt × MOET price (1.0)
- **Health Factor**: Collateral Value / Debt Value
- **Liquidation Threshold**: HF ≤ 1.0 triggers liquidation

### AAVE-Style Liquidation with Uniswap V3 Integration

The engine handles AAVE-style liquidation mechanics but uses Uniswap V3 for the actual liquidation swaps:

```python
def execute_aave_liquidation(self, current_minute: int, asset_prices: Dict[Asset, float], 
                            pool_size_usd: float = 500_000) -> dict:
    """
    Execute AAVE-style liquidation with proper Uniswap V3 math:
    1. Seize 50% of collateral (BTC)
    2. Swap BTC -> MOET through Uniswap V3 pool
    3. Use MOET to pay down debt
    4. Liquidator receives 5% bonus on debt repaid (in BTC value)
    """
    if self.state.health_factor >= 1.0:
        return {}  # No liquidation needed
    
    # Calculate liquidation amounts
    btc_price = asset_prices.get(Asset.BTC, 100_000.0)
    current_btc_collateral = self.state.supplied_balances.get(Asset.BTC, 0.0)
    current_debt = self.state.moet_debt
    
    # AAVE liquidation mechanics
    # 1. Debt reduction: 50% of current debt
    debt_reduction = current_debt * 0.50
    
    # 2. Calculate how much BTC to seize to get enough MOET to repay debt
    # We need to account for the 5% liquidation bonus
    liquidation_bonus_rate = 0.05  # 5% bonus
    debt_repaid_value = debt_reduction  # MOET debt value (assuming 1:1 with USD)
    
    # 3. Calculate BTC needed: debt_repaid_value * (1 + bonus) / btc_price
    # This gives us the BTC value needed to get enough MOET after the bonus
    btc_value_needed = debt_repaid_value * (1 + liquidation_bonus_rate)
    btc_to_seize = btc_value_needed / btc_price
    
    # 4. Ensure we don't seize more than available collateral
    btc_to_seize = min(btc_to_seize, current_btc_collateral)
    
    if btc_to_seize <= 0:
        return {}  # No collateral to liquidate
    
    # 5. Use Uniswap V3 math to calculate actual MOET received from BTC swap
    from ..core.uniswap_v3_math import create_moet_btc_pool, UniswapV3SlippageCalculator
    
    # Create pool and calculator
    pool = create_moet_btc_pool(pool_size_usd, btc_price)
    calculator = UniswapV3SlippageCalculator(pool)
    
    # Calculate BTC -> MOET swap with slippage
    btc_value_to_swap = btc_to_seize * btc_price
    swap_result = calculator.calculate_swap_slippage(btc_value_to_swap, "BTC")
    
    # Actual MOET received from swap (after slippage and fees)
    actual_moet_received = swap_result["amount_out"]
    
    # 6. Calculate actual debt that can be repaid (limited by MOET received)
    actual_debt_repaid = min(debt_reduction, actual_moet_received)
    
    # 7. Calculate liquidation bonus (5% of debt repaid, in BTC value)
    liquidation_bonus_value = actual_debt_repaid * liquidation_bonus_rate
    liquidation_bonus_btc = liquidation_bonus_value / btc_price
    
    # 8. Execute liquidation
    self.state.supplied_balances[Asset.BTC] -= btc_to_seize
    self.state.moet_debt -= actual_debt_repaid
```

**Liquidation Mechanics:**
- **Target Debt Reduction**: 50% of current debt (traditional AAVE)
- **BTC Seizure Calculation**: Accounts for 5% liquidation bonus
- **Uniswap V3 Integration**: BTC->MOET swap through MOET:BTC pool with real slippage and fees
- **Slippage-Aware Repayment**: Uses actual MOET received (post-slippage) to repay debt
- **Liquidation Bonus**: 5% bonus calculated on actual debt repaid (post-slippage)
- **Real Market Conditions**: Liquidation flows through Uniswap V3 trading mechanism for realistic pricing

## Performance Analytics and Reporting

### Comprehensive Metrics Collection

The engine collects detailed performance metrics throughout the simulation:

```python
def _record_aave_metrics(self, minute: int):
    """Record AAVE specific metrics"""
    # Base metrics
    super()._record_metrics()
    
    # AAVE specific metrics
    agent_health_data = []
    for agent in self.aave_agents:
        portfolio = agent.get_detailed_portfolio_summary(self.state.current_prices, minute)
        agent_health_data.append({
            "agent_id": agent.agent_id,
            "health_factor": agent.state.health_factor,
            "risk_profile": agent.risk_profile,
            "target_hf": agent.state.target_health_factor,
            "initial_hf": agent.state.initial_health_factor,
            "cost_of_liquidation": portfolio["cost_of_liquidation"],
            "net_position_value": portfolio["net_position_value"],
            "yield_token_value": portfolio["yield_token_portfolio"]["total_current_value"],
            "liquidation_events": portfolio["liquidation_events_count"],
            "liquidation_penalties": portfolio["liquidation_penalties"],
            "remaining_collateral": portfolio["btc_amount"]
        })
```

**Metrics Tracked:**
- **Agent Health**: Health factors, risk profiles, target thresholds
- **Portfolio Performance**: Yield token values, liquidation costs, net positions
- **Liquidation Activity**: Liquidation events, penalties, collateral seized
- **Risk Metrics**: Survival rates, liquidation frequency, penalty collection

### Results Generation

The engine generates comprehensive simulation results:

```python
def _generate_aave_results(self) -> dict:
    """Generate comprehensive AAVE simulation results"""
    base_results = super()._generate_results()
    
    # Calculate AAVE specific metrics
    final_minute = self.aave_config.btc_decline_duration - 1
    
    # Agent outcomes
    agent_outcomes = []
    total_cost_of_liquidation = 0.0
    total_liquidation_penalties = 0.0
    survival_by_risk_profile = {"conservative": 0, "moderate": 0, "aggressive": 0}
    
    for agent in self.aave_agents:
        agent._update_health_factor(self.state.current_prices)
        
        portfolio = agent.get_detailed_portfolio_summary(
            self.state.current_prices, 
            final_minute
        )
        
        outcome = {
            "agent_id": agent.agent_id,
            "risk_profile": agent.risk_profile,
            "target_health_factor": agent.state.target_health_factor,
            "initial_health_factor": agent.state.initial_health_factor,
            "final_health_factor": agent.state.health_factor,
            "cost_of_liquidation": portfolio["cost_of_liquidation"],
            "net_position_value": portfolio["net_position_value"],
            "total_yield_earned": portfolio["yield_token_portfolio"]["total_accrued_yield"],
            "total_yield_sold": 0.0,  # AAVE agents don't sell yield tokens
            "liquidation_events": len(agent.get_liquidation_history()),
            "survived": agent.active and agent.state.health_factor > 1.0,
            "liquidation_penalties": portfolio["liquidation_penalties"],
            "yield_token_value": portfolio["yield_token_portfolio"]["total_current_value"],
            "remaining_collateral": portfolio["btc_amount"]
        }
```

**Result Categories:**
- **Survival Statistics**: Overall and risk-profile-specific survival rates
- **Liquidation Analysis**: Total and average liquidation costs and penalties
- **Yield Token Activity**: Purchase volumes (no sales in AAVE)
- **Agent Health History**: Minute-by-minute health factor tracking
- **BTC Price History**: Complete price decline trajectory

## Configuration and Customization

### AAVE Configuration

The engine supports extensive configuration options:

```python
class AaveConfig(BaseLendingConfig):
    """Pure AAVE configuration - no Tidal dependencies"""
    
    def __init__(self):
        super().__init__()
        self.scenario_name = "AAVE_Protocol"
        
        # AAVE liquidation parameters
        self.liquidation_threshold = 0.85
        self.liquidation_bonus = 0.05  # 5%
        self.liquidation_percentage = 0.5  # 50%
        
        # Yield token parameters (for fair comparison)
        self.yield_apr = 0.10  # 10% APR
        self.moet_btc_pool_size = 500_000  # Same as Tidal for comparison
        
        # BTC price decline parameters (for fair comparison)
        self.btc_initial_price = 100_000.0
        self.btc_decline_duration = 60  # 60 minutes
        self.btc_final_price_range = (75_000.0, 85_000.0)  # 15-25% decline
        
        # Agent configuration
        self.num_aave_agents = 20
        self.monte_carlo_agent_variation = True
        
        # Disable rebalancing for AAVE scenario
        self.rebalancing_enabled = False
```

**Configuration Options:**
- **Liquidation Parameters**: Threshold, bonus rate, and percentage for AAVE-style liquidations
- **Pool Sizes**: MOET:BTC pool configuration for Uniswap V3 integration
- **Simulation Duration**: BTC decline duration and final price ranges
- **Agent Parameters**: Count, risk profiles, Monte Carlo variation
- **Yield Token Strategy**: Passive management (no rebalancing)
- **Fair Comparison**: Identical parameters to High Tide for valid comparison

### Monte Carlo Customization

The engine supports sophisticated Monte Carlo customization:

```python
# Agent configuration
self.num_aave_agents = 20
self.monte_carlo_agent_variation = True

# BTC price decline parameters
self.btc_final_price_range = (75_000.0, 85_000.0)  # 15-25% decline
```

**Monte Carlo Features:**
- **Dynamic Agent Count**: 10-50 agents per simulation
- **Risk Profile Distribution**: Configurable conservative/moderate/aggressive ratios
- **Price Variation**: Random final BTC prices within specified ranges
- **Statistical Analysis**: Multiple runs for robust performance metrics
- **Fair Comparison**: Identical agent distribution to High Tide for valid comparison

## Usage Examples

### Basic Simulation Setup

```python
# Create AAVE configuration
config = AaveConfig()
config.num_aave_agents = 25
config.btc_decline_duration = 60
config.btc_final_price_range = (70_000.0, 80_000.0)

# Initialize AAVE Protocol Engine
engine = AaveProtocolEngine(config)

# Run simulation
results = engine.run_simulation()

# Analyze results
print(f"Survival Rate: {results['survival_statistics']['survival_rate']:.2%}")
print(f"Total Liquidation Cost: ${results['cost_analysis']['total_cost_of_liquidation']:,.2f}")
```

### Advanced Configuration

```python
# Custom configuration for stress testing
config = AaveConfig()
config.scenario_name = "AAVE_Stress_Test"
config.moet_btc_pool_size = 500_000  # Larger pool
config.btc_decline_duration = 120  # Longer decline
config.btc_final_price_range = (60_000.0, 70_000.0)  # Deeper decline
config.liquidation_bonus = 0.08  # Higher liquidation bonus
config.num_aave_agents = 50  # More agents
config.monte_carlo_agent_variation = True

# Run stress test
engine = AaveProtocolEngine(config)
results = engine.run_simulation()
```

### Monte Carlo Analysis

```python
# Run multiple Monte Carlo simulations
monte_carlo_results = []

for run in range(10):  # 10 simulation runs
    config = AaveConfig()
    config.monte_carlo_agent_variation = True  # Enable variation
    
    engine = AaveProtocolEngine(config)
    results = engine.run_simulation()
    
    monte_carlo_results.append({
        'run': run,
        'survival_rate': results['survival_statistics']['survival_rate'],
        'total_liquidation_cost': results['cost_analysis']['total_cost_of_liquidation'],
        'total_penalties': results['cost_analysis']['total_liquidation_penalties'],
        'agent_count': len(results['agent_outcomes'])
    })

# Analyze Monte Carlo results
avg_survival = sum(r['survival_rate'] for r in monte_carlo_results) / len(monte_carlo_results)
avg_cost = sum(r['total_liquidation_cost'] for r in monte_carlo_results) / len(monte_carlo_results)

print(f"Average Survival Rate: {avg_survival:.2%}")
print(f"Average Liquidation Cost: ${avg_cost:,.2f}")
```

### Integration with Analysis Tools

```python
# Run simulation and generate charts
engine = AaveProtocolEngine(AaveConfig())
results = engine.run_simulation()

# Access detailed data for analysis
agent_health_history = results['agent_health_history']
btc_price_history = results['btc_price_history']
liquidation_events = results['liquidation_activity']['liquidation_events']

# Generate performance charts
from tidal_protocol_sim.analysis.aave_charts import generate_aave_dashboard
generate_aave_dashboard(results, save_path="aave_analysis.png")
```

### Comparison with High Tide

```python
# Run both AAVE and High Tide simulations for comparison
aave_config = AaveConfig()
aave_engine = AaveProtocolEngine(aave_config)
aave_results = aave_engine.run_simulation()

high_tide_config = HighTideConfig()
high_tide_engine = HighTideVaultEngine(high_tide_config)
high_tide_results = high_tide_engine.run_simulation()

# Compare results
print(f"AAVE Survival Rate: {aave_results['survival_statistics']['survival_rate']:.2%}")
print(f"High Tide Survival Rate: {high_tide_results['survival_statistics']['survival_rate']:.2%}")
print(f"AAVE Liquidation Cost: ${aave_results['cost_analysis']['total_cost_of_liquidation']:,.2f}")
print(f"High Tide Rebalancing Cost: ${high_tide_results['cost_analysis']['total_cost_of_rebalancing']:,.2f}")
```

## Key Features Summary

### 1. Orchestration Layer Architecture
- **Central Coordination**: Manages all simulation components and agent interactions
- **Base Lending Integration**: Leverages core lending functionality without Tidal Protocol complexity
- **Modular Integration**: Seamlessly integrates with yield token and price management systems

### 2. Monte Carlo Simulation Framework
- **Dynamic Agent Population**: 10-50 agents with varied risk profiles per simulation
- **Statistical Robustness**: Multiple runs for comprehensive performance analysis
- **Configurable Parameters**: Extensive customization options for different scenarios
- **Fair Comparison**: Identical agent distribution to High Tide for valid comparison

### 3. Passive Yield Token Strategy
- **One-Time Purchase**: Initial yield token purchase only, no subsequent trading
- **No Rebalancing**: Agents hold positions until liquidation (passive strategy)
- **Simple Management**: Basic portfolio tracking without active management

### 4. Market Stress Simulation
- **Realistic BTC Decline**: Historical volatility patterns with configurable parameters
- **Health Factor Monitoring**: Continuous risk assessment and liquidation management
- **Passive Response**: No crisis management or liquidation prevention strategies

### 5. AAVE-Style Liquidation with Uniswap V3
- **Traditional Mechanics**: 50% collateral seizure + 5% bonus (classic AAVE liquidation thresholds)
- **Uniswap V3 Integration**: BTC->MOET swap through MOET:BTC pool with real slippage and fees
- **Real Market Conditions**: Liquidation flows through Uniswap V3 trading mechanism for realistic pricing
- **Slippage-Aware Calculations**: Accounts for actual slippage and fees in liquidation calculations

### 6. Comprehensive Analytics
- **Performance Tracking**: Detailed metrics collection throughout simulation lifecycle
- **Liquidation Analysis**: Survival rates, liquidation costs, and penalty collection
- **Result Generation**: Comprehensive reporting with statistical analysis
- **Comparison Ready**: Structured results for direct comparison with High Tide

### 7. Integration with Core Systems
- **Yield Token System**: Complete integration with [Portfolio Management](../core/YIELD_TOKENS_README.md#portfolio-management)
- **Uniswap V3 Math**: Full access to [Concentrated Liquidity Mathematics](../core/UNISWAP_V3_MATH_README.md#concentrated-liquidity) for liquidation swaps
- **Protocol State**: Seamless coordination with base lending infrastructure
