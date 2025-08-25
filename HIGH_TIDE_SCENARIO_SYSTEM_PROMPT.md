# **System Prompt: Implement High Tide Scenario for Tidal Protocol Simulation**

## **Overview**
Implement a comprehensive "High Tide" scenario that simulates an actively managed lending protocol where users deposit BTC collateral, borrow MOET stablecoin, and automatically purchase yield-bearing tokens. The system actively rebalances positions during BTC price declines to prevent liquidations, providing a comparison against traditional lending protocols (Aave-style).

## **Core High Tide Mechanics**

### **1. Protocol Flow**
1. **Initial Setup**: Users deposit BTC collateral and borrow MOET stablecoin
2. **Automatic Yield Token Purchase**: Borrowed MOET is automatically used to purchase yield tokens at $1.00 each
3. **Continuous Yield Accrual**: Yield tokens earn 10% APR, accruing continuously each minute
4. **Active Monitoring**: Protocol monitors health factors via scheduled cronjobs every minute
5. **Automatic Rebalancing**: When health factor falls below maintenance threshold, protocol sells yield tokens to repay debt
6. **Liquidation Fallback**: If all yield tokens are sold and HF still falls to 1.0, traditional liquidation occurs

### **2. Agent Configuration**
- **Minimum 10 agents** per simulation
- **Monte Carlo Variation**: 10-50 agents randomly distributed per run
- **Risk Profile Distribution**:
  - **Conservative Agents** (30%): Initial HF = 2.1-2.4, Maintenance HF = 2.0-2.2
  - **Moderate Agents** (40%): Initial HF = 1.5-1.8, Maintenance HF = 1.3-1.75  
  - **Aggressive Agents** (30%): Initial HF = 1.3-1.5, Maintenance HF = 1.25-1.3
- **Collateral Distribution**: Each agent deposits exactly 1 BTC ($100,000 initial value)
- **Effective Collateral**: 80% of BTC value can be borrowed against
- **Color Coding**: Chart lines colored by target health factor ranges for visualization

### **3. BTC Price Decline Mechanics**

#### **Historical Volatility Implementation**
```python
# Base decline rates from 2025 historical data
DECLINE_RATES = [-0.54%, -0.53%, -0.46%, -0.43%, -0.40%]
MAX_DECLINE = -0.95%  # Absolute maximum from 3-year historical data

# Per-minute price calculation
def calculate_btc_price_change():
    base_decline = random.choice(DECLINE_RATES)
    # Add small random variation (±0.05%)
    variation = random.uniform(-0.0005, 0.0005)
    # Occasionally use maximum decline (5% probability)
    if random.random() < 0.05:
        return MAX_DECLINE
    return base_decline + variation
```

#### **Price Evolution Pattern**
- **Starting Price**: $100,000 BTC
- **Duration**: 60+ minutes (configurable)
- **Pattern**: Gradual decline using historical volatility
- **Target Range**: $75,000-$85,000 final price (25-15% decline)

### **4. Yield Token Implementation**

#### **Continuous Yield Accrual**
```python
class YieldToken:
    def __init__(self, principal_amount: float, apr: float = 0.10):
        self.principal = principal_amount
        self.apr = apr
        self.creation_time = current_minute
        
    def get_current_value(self, current_minute: int) -> float:
        minutes_elapsed = current_minute - self.creation_time
        # Convert APR to per-minute rate: APR / (365 * 24 * 60)
        minute_rate = self.apr / 525600
        return self.principal * (1 + minute_rate * minutes_elapsed)
        
    def get_accrued_yield(self, current_minute: int) -> float:
        return self.get_current_value(current_minute) - self.principal
```

#### **Yield Token Trading**
- **Initial Purchase**: MOET borrowed → Yield Tokens at $1.00 each
- **Price Appreciation**: $1.00 + accrued yield based on 10% APR
- **Trading Pool**: Yield Token ↔ MOET conversion with minimal slippage

### **5. Active Rebalancing System**

#### **Health Factor Monitoring**
```python
def check_rebalancing_needed(agent: HighTideAgent) -> bool:
    current_hf = calculate_health_factor(agent)
    return current_hf < agent.maintenance_health_factor

def calculate_health_factor(agent: HighTideAgent) -> float:
    collateral_value = agent.btc_amount * current_btc_price * 0.8  # 80% effective
    total_debt = agent.moet_debt
    return collateral_value / total_debt if total_debt > 0 else float('inf')
```

#### **Rebalancing Logic**
1. **Trigger**: Health factor falls below maintenance threshold
2. **Target**: Return health factor to initial target level
3. **Selling Priority**:
   - First: Sell accrued yield above principal amount
   - Second: Sell principal yield tokens if needed
4. **Calculation**: 
   ```python
   target_debt = collateral_value / target_health_factor
   debt_reduction_needed = current_debt - target_debt
   ```

#### **Uniswap V3 Integration**
- **Pool Configuration**: $250K MOET : $250K BTC (50/50)
- **Fee Tier**: 0.01% (1 basis point)
- **Slippage Calculation**: Use constant product formula
- **Swap Execution**: Include price impact and slippage in rebalancing calculations

### **6. Liquidation Mechanics**

#### **High Tide Liquidation**
- **Trigger**: Health factor reaches 1.0 after all yield tokens sold
- **Process**: Standard liquidation with 5% bonus to liquidator
- **Target**: Liquidate sufficient debt to bring HF to 1.1

#### **Aave Comparison Strategy**
- **No Active Management**: Positions held until HF reaches 1.0
- **Liquidation**: 50% of debt liquidated when HF < 1.0
- **Recovery**: Position recovers to HF = 1.1 after liquidation

## **Implementation Requirements**

### **7. New Classes to Implement**

#### **HighTideAgent**
```python
class HighTideAgent(BaseAgent):
    def __init__(self, agent_id: str, initial_hf: float, maintenance_hf: float):
        self.target_health_factor = initial_hf
        self.maintenance_health_factor = maintenance_hf
        self.yield_tokens = []
        self.automatic_rebalancing = True
        
    def execute_rebalancing(self):
        # Implement yield token selling logic
        pass
        
    def calculate_cost_of_liquidation(self, final_btc_price: float) -> float:
        # Net Position Value = Collateral Value + (Debt - Yield Token Value)
        collateral_value = self.btc_amount * final_btc_price
        net_position = collateral_value + (self.moet_debt - self.total_yield_value)
        cost_of_liquidation = final_btc_price - net_position
        return cost_of_liquidation
```

#### **YieldTokenManager**
```python
class YieldTokenManager:
    def mint_yield_tokens(self, moet_amount: float) -> List[YieldToken]:
        # Convert MOET to yield tokens
        pass
        
    def sell_yield_tokens(self, amount_needed: float) -> float:
        # Sell yield tokens starting with highest yield first
        pass
        
    def calculate_total_value(self) -> float:
        # Sum all yield token values including accrued yield
        pass
```

#### **HighTideScenario**
```python
class HighTideScenario(StressTestScenario):
    def __init__(self):
        super().__init__(
            "High_Tide_BTC_Decline",
            "BTC gradual decline with active yield token rebalancing",
            self._setup_high_tide_scenario,
            duration=60
        )
    
    def _setup_high_tide_scenario(self, engine: TidalSimulationEngine):
        # Configure BTC price decline
        # Initialize High Tide agents
        # Setup MOET:BTC liquidity pool
        pass
```

### **8. Integration Points**

#### **Simulation Engine Updates**
- **Price Evolution**: Add gradual BTC price decline mechanism
- **Rebalancing Checks**: Execute health factor checks each minute
- **Yield Accrual**: Update yield token values continuously
- **Pool Updates**: Track MOET:BTC pool state and utilization

#### **Configuration Extensions**
```python
class HighTideConfig(SimulationConfig):
    def __init__(self):
        super().__init__()
        # High Tide specific parameters
        self.yield_apr = 0.10
        self.moet_btc_pool_size = 250_000  # $250K each side
        self.btc_decline_duration = 60  # minutes
        self.rebalancing_enabled = True
        self.comparison_mode = True  # Include Aave strategy comparison
```

### **9. Comprehensive Visualization Suite**

#### **Required Charts**
1. **Net Position Value Over Time**
   - Multi-agent lines colored by target health factor
   - Comparison: High Tide vs Aave strategy
   - Shows cost of liquidation evolution

2. **Yield Token Activity Timeline**
   - Individual agent selling events
   - Volume of yield tokens sold vs time
   - Remaining yield token balances

3. **Health Factor Distribution**
   - Real-time HF distribution across agents
   - Trigger points (maintenance thresholds)
   - Recovery patterns post-rebalancing

4. **Protocol Utilization Dashboard**
   - MOET:BTC pool liquidity over time
   - Total yield tokens outstanding
   - Protocol treasury accumulation

5. **Strategy Comparison Summary**
   - Final cost of liquidation: High Tide vs Aave
   - Agent survival rates by strategy
   - Total protocol revenue by strategy

#### **Chart Styling Requirements**
- **Color Coding**: Lines colored by agent target health factor
- **Legend**: Clear identification of agent risk profiles
- **Annotations**: Mark key events (rebalancing, liquidations)
- **Dual Y-Axes**: Price evolution vs position metrics

### **10. Monte Carlo Analysis Framework**

#### **Parameter Variations**
```python
def create_monte_carlo_variations():
    variations = []
    for run in range(num_runs):
        config = HighTideConfig()
        # Randomize agent count (10-50)
        config.num_high_tide_agents = random.randint(10, 50)
        
        # Randomize risk profile distribution
        config.risk_profile_mix = generate_random_risk_mix()
        
        # Randomize BTC decline pattern
        config.decline_severity = random.uniform(0.15, 0.30)  # 15-30% decline
        
        # Randomize initial pool state
        config.pool_utilization = random.uniform(0.6, 0.9)
        
        variations.append(config)
    return variations
```

#### **Statistical Analysis**
- **Distribution Analysis**: Cost of liquidation distributions
- **Survival Rates**: Agent success rates by risk profile
- **Efficiency Metrics**: Protocol revenue and user outcomes
- **Comparative Analysis**: High Tide vs Aave across all runs

### **11. Success Criteria**

#### **Functional Requirements**
✅ **Gradual BTC Price Decline**: Historical volatility-based decline pattern  
✅ **Active Rebalancing**: Automatic yield token selling when HF < maintenance  
✅ **Continuous Yield Accrual**: 10% APR compounding per minute  
✅ **Monte Carlo Capability**: 10-50 agent variations across runs  
✅ **Strategy Comparison**: High Tide vs Aave liquidation outcomes  

#### **Visualization Requirements**
✅ **Multi-Agent Charts**: Color-coded by risk profile  
✅ **Timeline Charts**: Yield token activity and rebalancing events  
✅ **Comparison Charts**: Strategy performance side-by-side  
✅ **Distribution Analysis**: Health factor and outcome distributions  

#### **Performance Metrics**
✅ **Cost of Liquidation**: Accurate calculation and tracking  
✅ **Protocol Revenue**: Treasury accumulation from various sources  
✅ **User Outcomes**: Net position value preservation  
✅ **System Efficiency**: Rebalancing effectiveness and gas costs  

### **12. Implementation Priority**

1. **Core Mechanics** (Week 1):
   - YieldToken class with continuous accrual
   - HighTideAgent with rebalancing logic
   - BTC price decline with historical patterns

2. **Protocol Integration** (Week 2):
   - MOET:BTC pool setup and management
   - Health factor monitoring system
   - Automatic rebalancing execution

3. **Scenario Framework** (Week 3):
   - HighTideScenario implementation
   - Monte Carlo parameter variations
   - Aave comparison strategy

4. **Visualization Suite** (Week 4):
   - Multi-agent timeline charts
   - Strategy comparison dashboard
   - Statistical analysis and reporting

This High Tide scenario will provide comprehensive insights into the effectiveness of active position management versus traditional liquidation mechanisms, with robust visualization and statistical analysis capabilities.
