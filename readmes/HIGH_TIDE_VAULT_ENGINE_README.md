# High Tide Vault Engine - Technical Documentation

## Overview

This document provides a comprehensive breakdown of the `high_tide_vault_engine.py` script, which serves as the **orchestration layer** for the High Tide Protocol simulation. The engine coordinates sophisticated yield-bearing token strategies, manages Monte Carlo simulations, and integrates advanced Uniswap V3 concentrated liquidity mathematics to simulate realistic DeFi lending scenarios under market stress conditions.

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

### High Tide Vault Engine as Orchestration Layer

The `HighTideVaultEngine` serves as the central coordination system that:

- **Inherits from `TidalProtocolEngine`**: Gets all Uniswap V3 functionality and sophisticated concentrated liquidity mathematics
- **Manages High Tide Agents**: Orchestrates 10-50 agents with varied risk profiles in Monte Carlo simulations
- **Coordinates Yield Token Strategies**: Integrates with the [Yield Token System](../core/YIELD_TOKENS_README.md) for sophisticated portfolio management
- **Simulates Market Stress**: Implements BTC price decline scenarios using realistic volatility patterns
- **Tracks Performance**: Provides comprehensive analytics and reporting across all simulation components

### System Integration Architecture

```
HighTideVaultEngine (Orchestration Layer)
├── TidalProtocolEngine (Base Protocol)
│   ├── Uniswap V3 Math (Concentrated Liquidity)
│   └── Protocol State Management
├── High Tide Agents (10-50 Monte Carlo Agents)
│   ├── Yield Token Portfolios
│   ├── Risk Profile Management
│   └── Automatic Rebalancing Logic
├── BTC Price Manager (Market Stress Simulation)
└── Analytics & Reporting (Performance Tracking)
```

### Orchestration vs Execution Layers

The High Tide system uses a clear separation between orchestration and execution:

**🎯 Orchestration Layer (HighTideVaultEngine):**
- **Coordinates** agent actions and decisions
- **Tracks** rebalancing events, slippage costs, and performance metrics
- **Records** all trading activity for comprehensive analysis
- **Manages** simulation flow and agent lifecycle

**⚙️ Execution Layer (Agents + Pools):**
- **Agents** calculate portfolio needs and call pool methods directly
- **YieldTokenPool** executes real Uniswap V3 swaps with permanent pool state mutations
- **Pool state** is shared across all agents, creating realistic competition for liquidity
- **Real economic impact** where each swap affects subsequent trades

**🔄 Data Flow for Rebalancing:**
```
1. Agent calculates MOET needed for target health factor
2. Agent requests engine._execute_yield_token_sale()
3. Engine calls agent.execute_yield_token_sale()
4. Agent calls YieldTokenPool.execute_yield_token_sale() (REAL SWAP)
5. Pool state mutates permanently
6. Engine records slippage, costs, and event data
7. Agent portfolio updates to reflect actual tokens sold
```

## Core Components and Integration

### 1. Tidal Protocol Integration

The engine inherits sophisticated Uniswap V3 functionality from the base `TidalProtocolEngine`:

```python
class HighTideVaultEngine(TidalProtocolEngine):
    """High Tide Yield Vaults built on Tidal Protocol"""
    
    def __init__(self, config: HighTideConfig):
        # Initialize with High Tide config - gets all Uniswap V3 functionality from Tidal
        super().__init__(config)
        self.high_tide_config = config
```

**Key Inherited Capabilities:**
- **Uniswap V3 Concentrated Liquidity**: Full access to [Uniswap V3 Math](../core/UNISWAP_V3_MATH_README.md#core-math-functions) for sophisticated trading operations
- **Cross-Tick Swap Support**: Advanced [Cross-Tick Swap Mechanics](../core/UNISWAP_V3_MATH_README.md#cross-tick-swap-mechanics) for complex trading scenarios
- **Slippage Calculation**: Real-time [Slippage Calculation](../core/UNISWAP_V3_MATH_README.md#slippage-calculation) using authentic Uniswap V3 mathematics
- **Pool State Management**: Complete [Pool Implementation](../core/UNISWAP_V3_MATH_README.md#pool-implementation) with tick-based pricing

### 2. Yield Token Pool Integration

The engine creates and manages the global yield token trading infrastructure:

```python
# Initialize High Tide specific components
self.yield_token_pool = YieldTokenPool(
    initial_moet_reserve=config.moet_yield_pool_size,
    concentration=config.yield_token_concentration
)
```

**Integration with Yield Token System:**
- **Pool Configuration**: Uses [Pool Configuration](../core/YIELD_TOKENS_README.md#pool-configuration) with 95% concentration at 1:1 peg
- **Trading Operations**: Leverages [Trading Operations](../core/YIELD_TOKENS_README.md#trading-operations) for MOET ↔ Yield Token conversions
- **Slippage Management**: Implements [Slippage and Fee Management](../core/YIELD_TOKENS_README.md#slippage-and-fee-management) using real Uniswap V3 math
- **Portfolio Management**: Coordinates with [Portfolio Management](../core/YIELD_TOKENS_README.md#portfolio-management) for individual agent portfolios

### 3. Agent Creation and Management

The engine creates and manages a diverse set of High Tide agents with varied risk profiles:

```python
# Replace agents with High Tide agents
self.high_tide_agents = create_high_tide_agents(
    config.num_high_tide_agents,
    config.monte_carlo_agent_variation,
    self.yield_token_pool
)
```

**Agent Distribution (Monte Carlo):**
- **Conservative (30%)**: Initial HF = 2.1-2.4, Target HF = Initial - 0.05-0.15
- **Moderate (40%)**: Initial HF = 1.5-1.8, Target HF = Initial - 0.15-0.25  
- **Aggressive (30%)**: Initial HF = 1.3-1.5, Target HF = Initial - 0.15-0.4
- **Minimum Target HF**: 1.1 for all agents (safety threshold)

## Monte Carlo Simulation Framework

### Simulation Configuration

The engine implements sophisticated Monte Carlo simulation capabilities:

```python
class HighTideConfig(TidalConfig):
    """Configuration for High Tide scenario"""
    
    def __init__(self):
        super().__init__()
        self.scenario_name = "High_Tide_Vault"
        
        # High Tide specific parameters
        self.moet_yield_pool_size = 250_000  # $250K each side
        self.btc_decline_duration = 60  # 60 minutes
        
        # Yield Token Pool Configuration
        self.yield_token_concentration = 0.95  # 95% concentration
        
        # Agent configuration for High Tide
        self.num_high_tide_agents = 20
        self.monte_carlo_agent_variation = True
        
        # BTC price decline parameters
        self.btc_final_price_range = (75_000.0, 85_000.0)  # 15-25% decline
```

### Monte Carlo Agent Variation

The engine supports dynamic agent population and risk profile distribution:

```python
def create_high_tide_agents(num_agents: int, monte_carlo_variation: bool = True, yield_token_pool = None) -> list:
    """
    Create High Tide agents with varied risk profiles
    
    Risk Profile Distribution:
    - Conservative (30%): Initial HF = 2.1-2.4, Target HF = Initial - 0.05-0.15
    - Moderate (40%): Initial HF = 1.5-1.8, Target HF = Initial - 0.15-0.25
    - Aggressive (30%): Initial HF = 1.3-1.5, Target HF = Initial - 0.15-0.4
    
    Minimum Target HF = 1.1 for all agents
    """
    if monte_carlo_variation:
        # Randomize agent count between 10-50
        num_agents = random.randint(10, 50)
```

**Monte Carlo Features:**
- **Dynamic Agent Count**: 10-50 agents per simulation run
- **Risk Profile Randomization**: Varied initial and target health factors
- **BTC Price Variation**: Random final price within specified range
- **Statistical Analysis**: Multiple runs for robust performance metrics

## Agent Orchestration and Management

### Agent Action Processing

The engine orchestrates agent decision-making and action execution:

```python
def _process_high_tide_agents(self, minute: int) -> Dict[str, Dict]:
    """Process High Tide agent actions for current minute"""
    swap_data = {}
    
    for agent in self.high_tide_agents:
        if not agent.active:
            continue
            
        # Get agent's decision
        protocol_state = self._get_protocol_state()
        protocol_state["current_step"] = minute
        
        action_type, params = agent.decide_action(protocol_state, self.state.current_prices)
        
        # Execute action and capture swap data
        success, agent_swap_data = self._execute_high_tide_action(agent, action_type, params, minute)
```

### High Tide Action Execution

The engine handles sophisticated action execution with Uniswap V3 integration:

```python
def _execute_high_tide_action(self, agent: HighTideAgent, action_type: AgentAction, params: dict, minute: int) -> tuple:
    """Execute High Tide specific actions"""
    if action_type == AgentAction.SWAP:
        swap_type = params.get("action_type", "")
        
        if swap_type == "buy_yield_tokens":
            success = self._execute_yield_token_purchase(agent, params, minute)
            return success, None
        elif swap_type in ["sell_yield_tokens", "emergency_sell_all_yield"]:
            success, swap_data = self._execute_yield_token_sale(agent, params, minute)
            return success, swap_data
    
    elif action_type == AgentAction.BORROW:
        if params.get("leverage_increase", False):
            success = self._execute_leverage_increase_borrow(agent, params, minute)
            return success, None
        else:
            return super()._execute_agent_action(agent, action_type, params)
            
    elif action_type == AgentAction.LIQUIDATE:
        success = self._execute_liquidation(agent, params)
        return success, None
        
    return False, None
```

**Supported Actions:**
- **Yield Token Purchase**: Initial investment and leverage increases
- **Yield Token Sale**: Rebalancing via yield token sales
- **Emergency Sell All Yield**: Full liquidation of remaining yield tokens
- **Borrow (Leverage Increase)**: Dynamic borrowing based on health factors
- **Liquidate**: Emergency liquidation when HF ≤ 1.0

### Yield Token Purchase Orchestration

The engine coordinates yield token purchases with flexible creation methods:

```python
def _execute_yield_token_purchase(self, agent: HighTideAgent, params: dict, minute: int) -> bool:
    """Execute yield token purchase for agent"""
    moet_amount = params.get("moet_amount", 0.0)
    
    if moet_amount <= 0:
        return False
    
    # Determine if we should use direct minting (minute 0 + config enabled)
    use_direct_minting = (minute == 0 and self.high_tide_config.use_direct_minting_for_initial)
    
    success = agent.execute_yield_token_purchase(moet_amount, minute, use_direct_minting)
    
    if success:
        # Always update pool state to maintain synchronization
        # For direct minting, we need to update the pool's internal reserves
        if use_direct_minting:
            # For direct minting, update pool reserves to reflect the 1:1 minting
            # This ensures pool state stays synchronized with agent state
            self.yield_token_pool.moet_reserve += moet_amount
            self.yield_token_pool.yield_token_reserve += moet_amount
        else:
            # For regular purchases, use the pool's execute method
            self.yield_token_pool.execute_yield_token_purchase(moet_amount)
```

**Integration with Yield Token System:**
- **Direct Minting**: Uses [Flexible Yield Token Creation](../core/YIELD_TOKENS_README.md#flexible-yield-token-creation) for minute 0
- **Uniswap V3 Trading**: Leverages [Trading Operations](../core/YIELD_TOKENS_README.md#trading-operations) for subsequent purchases
- **Pool State Management**: Coordinates with [Pool Integration](../core/YIELD_TOKENS_README.md#pool-integration) for state consistency

### Yield Token Sale Orchestration

The engine manages sophisticated yield token sales for rebalancing:

```python
def _execute_yield_token_sale(self, agent: HighTideAgent, params: dict, minute: int) -> tuple:
    """Execute yield token sale for rebalancing via engine orchestration"""
    moet_amount_needed = params.get("moet_needed", 0.0)
    swap_type = params.get("swap_type", "rebalancing")
    
    if moet_amount_needed <= 0:
        return False, None
        
    # ORCHESTRATION LAYER: Engine coordinates the real swap execution
    # 1. Agent calculates yield tokens to sell and updates portfolio
    # 2. Agent calls YieldTokenPool.execute_yield_token_sale() for real swap
    # 3. Engine records the event and tracks slippage/costs
    
    success, swap_data = agent.execute_yield_token_sale()
    
    if success and swap_data:
        moet_received = swap_data.get("moet_received", 0.0)
        yt_swapped = swap_data.get("yt_swapped", 0.0)
        
        # Calculate slippage for engine tracking
        slippage_cost = moet_amount_needed - moet_received
        slippage_percentage = (slippage_cost / moet_amount_needed) * 100 if moet_amount_needed > 0 else 0
        
        # Record rebalancing event at engine level
        rebalancing_event = {
            "agent_id": agent.agent_id,
            "minute": minute,
            "moet_needed": moet_amount_needed,
            "moet_raised": moet_received,
            "swap_type": swap_type,
            "slippage_cost": slippage_cost,
            "slippage_percentage": slippage_percentage,
            "health_factor_before": agent.state.health_factor,
            "health_factor_after": agent.state.health_factor
        }
        self.rebalancing_events.append(rebalancing_event)
        
        return True, {
            "moet_received": moet_received,
            "yt_swapped": yt_swapped,
            "slippage_cost": slippage_cost,
            "slippage_percentage": slippage_percentage
        }
    
    return False, None
```

**Sale Strategies:**
- **Rebalancing Sales**: Engine orchestrates yield token sales to reduce MOET debt with real pool state mutations
- **Emergency Sales**: Sell all remaining yield tokens when health factor is critical
- **Engine Tracking**: All sales are recorded at the engine level with slippage costs and performance metrics

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
- **Rebalancing Triggers**: Agents may need to rebalance as health factors decline
- **Liquidation Risk**: Increased risk of liquidation as BTC price falls

## Yield Token Strategy Implementation

### Initial Investment Strategy

The engine implements sophisticated initial investment strategies:

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
- **1:1 Initial Rate**: Direct minting at minute 0 for balanced pool establishment
- **Uniswap V3 Trading**: Subsequent purchases use [Trading Operations](../core/YIELD_TOKENS_README.md#trading-operations)

### Rebalancing Strategy

The engine coordinates sophisticated iterative rebalancing strategies:

```python
def _execute_iterative_rebalancing(self, initial_moet_needed: float, current_minute: int, asset_prices: Dict[Asset, float]) -> tuple:
    """Execute iterative rebalancing with slippage monitoring"""
    moet_needed = initial_moet_needed
    total_moet_raised = 0.0
    total_yield_tokens_sold = 0.0
    rebalance_cycle = 0
    
    print(f"        🔄 {self.agent_id}: Starting iterative rebalancing - need ${moet_needed:,.2f} MOET")
    
    while (self.state.health_factor < self.state.target_health_factor and 
           self.state.yield_token_manager.yield_tokens and
           rebalance_cycle < 10):  # Max 10 cycles to prevent infinite loops
        
        rebalance_cycle += 1
        print(f"        🔄 Rebalance Cycle {rebalance_cycle}: Need ${moet_needed:,.2f} MOET")
        
        # Calculate yield tokens to sell (1:1 assumption)
        yield_tokens_to_sell = moet_needed
        
        # CRITICAL FIX: Use engine's real swap execution instead of YieldTokenManager bookkeeping
        if self.engine:
            # Let the engine execute the REAL swap with pool state mutations
            success, swap_data = self.engine._execute_yield_token_sale(
                self, 
                {"moet_needed": moet_needed, "swap_type": "rebalancing"}, 
                current_minute
            )
            
            if success and swap_data:
                moet_received = swap_data.get("moet_received", 0.0)
                actual_yield_tokens_sold_value = swap_data.get("yt_swapped", 0.0)
            else:
                moet_received = 0.0
                actual_yield_tokens_sold_value = 0.0
        else:
            # WARNING: This fallback should not happen in production! Engine reference missing.
            print(f"⚠️  WARNING: Agent {self.agent_id} using YieldTokenManager fallback - engine reference missing!")
            moet_received, actual_yield_tokens_sold_value = self.state.yield_token_manager.sell_yield_tokens(yield_tokens_to_sell, current_minute)
        
        if moet_received <= 0:
            print(f"        ❌ No MOET received from yield token sale - liquidity exhausted")
            break
        
        # Check slippage threshold (>5% slippage)
        if moet_received < 0.95 * actual_yield_tokens_sold_value:
            slippage_percent = (1 - moet_received / actual_yield_tokens_sold_value) * 100
            print(f"        ⚠️  HIGH SLIPPAGE: {actual_yield_tokens_sold_value:,.2f} yield tokens → ${moet_received:,.2f} MOET ({slippage_percent:.1f}% slippage)")
        
        # Pay down debt
        debt_repayment = min(moet_received, self.state.moet_debt)
        self.state.moet_debt -= debt_repayment
        total_moet_raised += moet_received
        total_yield_tokens_sold += actual_yield_tokens_sold_value
        
        # Update health factor with actual prices
        self._update_health_factor(asset_prices)
        
        print(f"        📊 Cycle {rebalance_cycle}: Received ${moet_received:,.2f} MOET, repaid ${debt_repayment:,.2f} debt, new HF: {self.state.health_factor:.3f}")
        
        # Check if we've reached target
        if self.state.health_factor >= self.state.target_health_factor:
            print(f"        ✅ Target HF reached: {self.state.health_factor:.3f} >= {self.state.target_health_factor:.3f}")
            break
        
        # Calculate remaining MOET needed for next cycle
        collateral_value = self._calculate_effective_collateral_value(asset_prices)
        target_debt = collateral_value / self.state.initial_health_factor
        moet_needed = self.state.moet_debt - target_debt
        
        if moet_needed <= 0:
            break
```

**Iterative Rebalancing Logic:**
- **Multi-Cycle Approach**: Continues rebalancing until target health factor is reached
- **Slippage Monitoring**: Tracks and reports high slippage (>5%) during rebalancing
- **Progressive Debt Reduction**: Each cycle reduces debt and recalculates remaining needs
- **Safety Limits**: Maximum 10 cycles to prevent infinite loops
- **Real-Time Health Factor Updates**: Updates health factor after each rebalancing cycle

### Leverage Management

The engine implements dynamic leverage management:

```python
def _check_leverage_opportunity(self, asset_prices: Dict[Asset, float]) -> bool:
    """Check if agent can increase leverage when HF > initial HF"""
    if self.state.health_factor > self.state.initial_health_factor:
        return True
    return False

def _execute_leverage_increase(self, asset_prices: Dict[Asset, float], current_minute: int) -> tuple:
    """Increase leverage by borrowing more MOET to restore initial HF"""
    collateral_value = self._calculate_effective_collateral_value(asset_prices)
    current_debt = self.state.moet_debt
    
    # Calculate target debt for initial HF
    target_debt = collateral_value / self.state.initial_health_factor
    additional_moet_needed = target_debt - current_debt
    
    if additional_moet_needed <= 0:
        return (AgentAction.HOLD, {})
    
    return (AgentAction.BORROW, {
        "amount": additional_moet_needed,
        "current_minute": current_minute,
        "leverage_increase": True
    })
```

**Leverage Strategy:**
- **Opportunistic Borrowing**: Increases leverage when health factor improves
- **Target Restoration**: Aims to restore initial health factor through additional borrowing
- **Yield Token Investment**: New MOET immediately invested in yield tokens

## Liquidation and Risk Management

### Health Factor Monitoring

The engine continuously monitors agent health factors:

```python
def _check_high_tide_liquidations(self, minute: int):
    """Check for High Tide liquidations (HF ≤ 1.0)"""
    for agent in self.high_tide_agents:
        if not agent.active:
            continue
            
        # Update health factor
        agent._update_health_factor(self.state.current_prices)
        
        # Check if liquidation is needed (HF ≤ 1.0)
        if agent.state.health_factor <= 1.0:
            liquidation_event = agent.execute_high_tide_liquidation(minute, self.state.current_prices, self)
            
            if liquidation_event:
                self.liquidation_events.append(liquidation_event)
```

**Health Factor Calculation:**
- **Collateral Value**: BTC amount × BTC price × collateral factor (0.80)
- **Debt Value**: MOET debt × MOET price (1.0)
- **Health Factor**: Collateral Value / Debt Value
- **Liquidation Threshold**: HF ≤ 1.0 triggers liquidation

### Emergency Liquidation

The engine handles emergency liquidation scenarios with full Uniswap V3 integration:

```python
def execute_high_tide_liquidation(self, current_minute: int, asset_prices: Dict[Asset, float], simulation_engine) -> Optional[Dict]:
    """Execute High Tide liquidation with Uniswap V3 BTC->MOET swap"""
    
    # Ensure we have BTC price from simulation engine
    btc_price = asset_prices.get(Asset.BTC)
    if btc_price is None:
        raise ValueError(f"BTC price not provided in asset_prices for liquidation at minute {current_minute}")
    
    # Calculate how much debt to repay to bring HF back to 1.1
    collateral_value = self._calculate_effective_collateral_value(asset_prices)
    target_debt = collateral_value / 1.1  # Target HF of 1.1
    current_debt = self.state.moet_debt
    debt_to_repay = current_debt - target_debt
    
    if debt_to_repay <= 0:
        return None
    
    # Step 1: Calculate BTC needed for debt repayment
    btc_to_repay_debt = debt_to_repay / btc_price
    available_btc = self.state.supplied_balances.get(Asset.BTC, 0.0)
    
    if btc_to_repay_debt > available_btc:
        btc_to_repay_debt = available_btc
    
    # Step 2: Execute BTC->MOET swap through Uniswap V3 pool
    swap_result = simulation_engine.slippage_calculator.calculate_swap_slippage(
        btc_to_repay_debt, "BTC"
    )
    
    actual_moet_received = swap_result["amount_out"]
    slippage_amount = swap_result["slippage_amount"]
    slippage_percent = swap_result["slippage_percent"]
    
    # Step 3: Repay debt with actual MOET received
    actual_debt_repaid = min(actual_moet_received, self.state.moet_debt)
    self.state.moet_debt -= actual_debt_repaid
    
    # Step 4: Calculate and seize liquidation bonus (5% of debt repaid)
    liquidation_bonus = actual_debt_repaid * 0.05
    btc_bonus = liquidation_bonus / btc_price
    total_btc_seized = btc_to_repay_debt + btc_bonus
    
    # Update agent state
    self.state.supplied_balances[Asset.BTC] -= total_btc_seized
```

**Liquidation Mechanics:**
- **Target Health Factor**: 1.1 (safety buffer above liquidation threshold)
- **Two-Step BTC Seizure**: First for debt repayment, second for 5% liquidator bonus
- **Uniswap V3 Integration**: BTC->MOET swap through MOET:BTC pool with real slippage
- **Slippage-Aware Bonus**: 5% bonus calculated on actual debt repaid (post-slippage)
- **Real Market Conditions**: Liquidation flows through trading mechanism for realistic pricing

## Performance Analytics and Reporting

### Comprehensive Metrics Collection

The engine collects detailed performance metrics throughout the simulation:

```python
def _record_high_tide_metrics(self, minute: int):
    """Record High Tide specific metrics"""
    # Base metrics
    super()._record_metrics()
    
    # High Tide specific metrics
    agent_health_data = []
    for agent in self.high_tide_agents:
        portfolio = agent.get_detailed_portfolio_summary(self.state.current_prices, minute)
        agent_health_data.append({
            "agent_id": agent.agent_id,
            "health_factor": agent.state.health_factor,
            "risk_profile": agent.risk_profile,
            "target_hf": agent.state.target_health_factor,
            "initial_hf": agent.state.initial_health_factor,
            "cost_of_rebalancing": portfolio["cost_of_rebalancing"],
            "net_position_value": portfolio["net_position_value"],
            "yield_token_value": portfolio["yield_token_portfolio"]["total_current_value"],
            "total_yield_sold": portfolio["total_yield_sold"],
            "rebalancing_events": portfolio["rebalancing_events_count"]
        })
```

**Metrics Tracked:**
- **Agent Health**: Health factors, risk profiles, target thresholds
- **Portfolio Performance**: Yield token values, rebalancing costs, net positions
- **Trading Activity**: Yield token purchases, sales, rebalancing events
- **Risk Metrics**: Liquidation events, emergency actions, survival rates

### Results Generation

The engine generates comprehensive simulation results:

```python
def _generate_high_tide_results(self) -> dict:
    """Generate comprehensive High Tide simulation results"""
    base_results = super()._generate_results()
    
    # Calculate High Tide specific metrics
    final_minute = self.high_tide_config.btc_decline_duration - 1
    
    # Agent outcomes
    agent_outcomes = []
    total_cost_of_rebalancing = 0.0
    survival_by_risk_profile = {"conservative": 0, "moderate": 0, "aggressive": 0}
    
    for agent in self.high_tide_agents:
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
            "cost_of_rebalancing": portfolio["cost_of_rebalancing"],
            "net_position_value": portfolio["net_position_value"],
            "total_yield_earned": portfolio["yield_token_portfolio"]["total_accrued_yield"],
            "total_yield_sold": portfolio["total_yield_sold"],
            "rebalancing_events": len(agent.get_rebalancing_history()),
            "survived": agent.state.health_factor > 1.0,
            "yield_token_value": portfolio["yield_token_portfolio"]["total_current_value"]
        }
```

**Result Categories:**
- **Survival Statistics**: Overall and risk-profile-specific survival rates
- **Cost Analysis**: Total and average rebalancing costs
- **Yield Token Activity**: Purchase/sale volumes, trading frequency
- **Agent Health History**: Minute-by-minute health factor tracking
- **BTC Price History**: Complete price decline trajectory

## Configuration and Customization

### High Tide Configuration

The engine supports extensive configuration options:

```python
class HighTideConfig(TidalConfig):
    """Configuration for High Tide scenario"""
    
    def __init__(self):
        super().__init__()
        self.scenario_name = "High_Tide_Vault"
        
        # High Tide specific parameters
        self.moet_yield_pool_size = 250_000  # $250K each side
        self.btc_decline_duration = 60  # 60 minutes
        
        # Yield Token Pool Configuration
        self.yield_token_concentration = 0.95  # 95% concentration
        
        # Agent configuration for High Tide
        self.num_high_tide_agents = 20
        self.monte_carlo_agent_variation = True
        
        # BTC price decline parameters
        self.btc_final_price_range = (75_000.0, 85_000.0)  # 15-25% decline
        
        # Yield Token Creation Method
        self.use_direct_minting_for_initial = True  # True = 1:1 minting at minute 0
```

**Configuration Options:**
- **Pool Sizes**: MOET:BTC and MOET:Yield Token pool configurations
- **Simulation Duration**: BTC decline duration and final price ranges
- **Agent Parameters**: Count, risk profiles, Monte Carlo variation
- **Yield Token Strategy**: Direct minting vs. Uniswap V3 trading
- **Concentration Levels**: Liquidity concentration for different pool types

### Monte Carlo Customization

The engine supports Monte Carlo customization:

```python
# Agent configuration for High Tide
self.num_high_tide_agents = 20
self.monte_carlo_agent_variation = True

# BTC price decline parameters
self.btc_final_price_range = (75_000.0, 85_000.0)  # 15-25% decline
```

**Monte Carlo Features:**
- **Dynamic Agent Count**: 10-50 agents per simulation
- **Risk Profile Distribution**: Configurable conservative/moderate/aggressive ratios
- **Price Variation**: Random final BTC prices within specified ranges
- **Statistical Analysis**: Multiple runs for robust performance metrics

## Usage Examples

### Basic Simulation Setup

```python
# Create High Tide configuration
config = HighTideConfig()
config.num_high_tide_agents = 25
config.btc_decline_duration = 60
config.btc_final_price_range = (70_000.0, 80_000.0)

# Initialize High Tide Vault Engine
engine = HighTideVaultEngine(config)

# Run simulation
results = engine.run_simulation()

# Analyze results
print(f"Survival Rate: {results['survival_statistics']['survival_rate']:.2%}")
print(f"Total Rebalancing Cost: ${results['cost_analysis']['total_cost_of_rebalancing']:,.2f}")
```

### Advanced Configuration

```python
# Custom configuration for stress testing
config = HighTideConfig()
config.scenario_name = "High_Tide_Stress_Test"
config.moet_yield_pool_size = 500_000  # Larger pool
config.btc_decline_duration = 120  # Longer decline
config.btc_final_price_range = (60_000.0, 70_000.0)  # Deeper decline
config.yield_token_concentration = 0.98  # Tighter concentration
config.num_high_tide_agents = 50  # More agents
config.monte_carlo_agent_variation = True

# Run stress test
engine = HighTideVaultEngine(config)
results = engine.run_simulation()
```

### Monte Carlo Analysis

```python
# Run multiple Monte Carlo simulations
monte_carlo_results = []

for run in range(10):  # 10 simulation runs
    config = HighTideConfig()
    config.monte_carlo_agent_variation = True  # Enable variation
    
    engine = HighTideVaultEngine(config)
    results = engine.run_simulation()
    
    monte_carlo_results.append({
        'run': run,
        'survival_rate': results['survival_statistics']['survival_rate'],
        'total_cost': results['cost_analysis']['total_cost_of_rebalancing'],
        'agent_count': len(results['agent_outcomes'])
    })

# Analyze Monte Carlo results
avg_survival = sum(r['survival_rate'] for r in monte_carlo_results) / len(monte_carlo_results)
avg_cost = sum(r['total_cost'] for r in monte_carlo_results) / len(monte_carlo_results)

print(f"Average Survival Rate: {avg_survival:.2%}")
print(f"Average Total Cost: ${avg_cost:,.2f}")
```

### Integration with Analysis Tools

```python
# Run simulation and generate charts
engine = HighTideVaultEngine(HighTideConfig())
results = engine.run_simulation()

# Access detailed data for analysis
agent_health_history = results['agent_health_history']
btc_price_history = results['btc_price_history']
rebalancing_events = results['rebalancing_events']

# Generate performance charts
from tidal_protocol_sim.analysis.high_tide_charts import generate_high_tide_dashboard
generate_high_tide_dashboard(results, save_path="high_tide_analysis.png")
```

## Key Features Summary

### 1. Orchestration Layer Architecture
- **Central Coordination**: Engine orchestrates all agent actions while agents execute real swaps
- **Separation of Concerns**: Clear distinction between orchestration (engine) and execution (agents/pools)
- **Real Pool State Mutations**: All swaps permanently affect shared liquidity pools
- **Comprehensive Tracking**: Engine records all trading activity, slippage costs, and performance metrics
- **Inheritance-Based Design**: Leverages Tidal Protocol's Uniswap V3 functionality

### 2. Monte Carlo Simulation Framework
- **Dynamic Agent Population**: 10-50 agents with varied risk profiles per simulation
- **Statistical Robustness**: Multiple runs for comprehensive performance analysis
- **Configurable Parameters**: Extensive customization options for different scenarios

### 3. Advanced Yield Token Strategy
- **Flexible Creation Methods**: Direct minting vs. Uniswap V3 trading based on simulation needs
- **Sophisticated Rebalancing**: Yield-first sales with principal preservation strategies
- **Real-Time Slippage**: Integration with [Uniswap V3 Slippage Calculation](../core/UNISWAP_V3_MATH_README.md#slippage-calculation)

### 4. Market Stress Simulation
- **Realistic BTC Decline**: Historical volatility patterns with configurable parameters
- **Health Factor Monitoring**: Continuous risk assessment and liquidation management
- **Emergency Response**: Crisis management and liquidation prevention strategies

### 5. Comprehensive Analytics
- **Performance Tracking**: Detailed metrics collection throughout simulation lifecycle
- **Risk Analysis**: Survival rates, rebalancing costs, and yield token activity
- **Result Generation**: Comprehensive reporting with statistical analysis

### 6. Integration with Core Systems
- **Uniswap V3 Math**: Full access to [Concentrated Liquidity Mathematics](../core/UNISWAP_V3_MATH_README.md#concentrated-liquidity)
- **Yield Token System**: Complete integration with [Portfolio Management](../core/YIELD_TOKENS_README.md#portfolio-management)
- **Protocol State**: Seamless coordination with Tidal Protocol's lending infrastructure

