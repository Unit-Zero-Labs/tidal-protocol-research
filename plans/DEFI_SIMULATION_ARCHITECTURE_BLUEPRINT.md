# DeFi Simulation Architecture Blueprint: Time Chamber Standards

> **A comprehensive guide for building commercial-grade, modular DeFi simulation systems**
> 
> Based on the Time Chamber architecture by Unit Zero Labs - a production-ready framework that demonstrates exceptional modularity through Agent-Action-Market patterns, comprehensive configuration management, and plugin-based extensibility.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Architectural Principles](#core-architectural-principles)
3. [System Architecture Overview](#system-architecture-overview)
4. [Agent-Action-Market Pattern](#agent-action-market-pattern)
5. [Configuration System Excellence](#configuration-system-excellence)
6. [Market Plugin System](#market-plugin-system)
7. [Universal Action Framework](#universal-action-framework)
8. [Data Flow Architecture](#data-flow-architecture)
9. [Analysis and Metrics](#analysis-and-metrics)
10. [Extension Patterns](#extension-patterns)
11. [Implementation Roadmap](#implementation-roadmap)
12. [Best Practices](#best-practices)

---

## Executive Summary

Time Chamber represents a **commercial-grade, modular agent-based simulation system** for DeFi protocol tokenomics. The architecture demonstrates exceptional modularity through its **Agent-Action-Market pattern**, comprehensive configuration management, and plugin-based extensibility. 

**Key Achievements:**
- **Zero Hardcoded Values**: All 140+ parameters sourced from client configurations
- **Universal Action System**: 70+ DeFi primitives cover all major mechanisms  
- **Plugin Architecture**: Markets and policies are composable, independent components
- **Google Sheets Integration**: Live configuration updates from non-technical stakeholders
- **Mathematical Accuracy**: Proper implementation of DeFi protocol mathematics
- **Kubernetes-Ready**: Horizontal scaling for large Monte Carlo simulations

This document serves as a blueprint for developing similarly sophisticated DeFi simulation systems.

---

## Core Architectural Principles

### 1. Agent-Action-Market Pattern

The system implements **clean separation of concerns**:

```
Agents (State Only) → Policies (Decisions) → Actions (Universal Primitives) → Markets (Execution) → Events (Results)
```

- **Agents**: Hold only state (balances, history) - no business logic
- **Policies**: Pure decision functions that emit universal Action primitives  
- **Actions**: Universal DeFi primitives (SWAP_BUY, STAKE, SUPPLY, etc.)
- **Markets**: Independent plugins that execute Actions and manage state
- **Engine**: Coordinates the Agent → Action → Market flow

### 2. Configuration-Driven Architecture

- **Single Source of Truth**: Google Sheets with 140+ comprehensive parameters
- **No Hardcoded Values**: All simulation parameters come from configuration
- **Multi-Client Support**: Different clients produce genuinely different simulations
- **Backward Compatibility**: Legacy configurations work during transitions

### 3. Plugin-Based Extensibility

- **Market Registry**: Central routing system for action execution
- **Policy System**: Composable agent behaviors without code duplication
- **Universal Actions**: Standardized DeFi action types cover all mechanisms

---

## System Architecture Overview

### Directory Structure

```
src/
├── core/
│   ├── engine/
│   │   ├── engine.py          # Main orchestration loop
│   │   ├── primitives.py      # Action/Event/MarketSnapshot definitions  
│   │   ├── factory.py         # Routes configs to appropriate engines
│   │   └── monte_carlo.py     # Statistical simulation wrapper
│   ├── agents/
│   │   ├── base.py           # BaseAgent + BasePolicy interfaces
│   │   ├── population.py     # Agent population management
│   │   └── policies/         # Pure decision logic implementations
│   │       ├── trader.py     # Price momentum trading
│   │       ├── staker.py     # APY-driven staking
│   │       ├── lender.py     # Lending protocol interaction
│   │       └── velodrome_lp.py # Vote-escrow governance
│   ├── markets/              # Independent DeFi mechanism plugins
│   │   ├── base.py          # BaseMarket + MarketRegistry
│   │   ├── uniswap_v2.py    # Constant product AMM
│   │   ├── velodrome.py     # Vote-escrow AMM with governance
│   │   ├── compound.py      # Pooled lending market
│   │   ├── euler.py         # Isolated lending market
│   │   └── staking.py       # Token staking rewards
│   └── math/                # Pure mathematical calculations
│       ├── uniswap_v2.py    # AMM pricing formulas
│       ├── velodrome.py     # ve-token and governance math
│       ├── compound.py      # Interest rate models
│       └── euler.py         # Risk-isolated lending math
├── config/
│   ├── schemas/
│   │   └── radcad.py         # 140+ parameter Pydantic schema
│   ├── csv_converter.py      # RadCAD → YAML conversion
│   └── client_manager.py     # Runtime configuration loading
├── integrations/
│   └── google_sheets/
│       ├── connector.py      # Google Sheets API wrapper
│       └── radcad_parser.py  # RadCAD-specific parsing
└── analysis/
    ├── metrics/
    │   └── calculator.py     # Comprehensive metrics
    └── visualization/        # Chart generation
```

---

## Agent-Action-Market Pattern

### Agent Architecture

**Agents hold only state - no business logic:**

```python
@dataclass
class AgentState:
    """Clean state separation"""
    token_balance: float
    staked_balance: float
    lp_balance: float
    ve_balance: float
    cash_balance: float
    last_action_timestamp: int = 0

class BaseAgent:
    """Universal agent container"""
    def __init__(self, agent_id: str, initial_state: AgentState, policy: BasePolicy):
        self.agent_id = agent_id
        self.state = initial_state
        self.policy = policy
    
    def decide_actions(self, snapshot: MarketSnapshot) -> List[Action]:
        """Delegate decisions to policy"""
        return self.policy.decide(snapshot)
```

### Policy System

**Pure decision functions with no state mutations:**

```python
class BasePolicy(ABC):
    @abstractmethod
    def decide(self, snapshot: MarketSnapshot) -> List[Action]:
        """Pure decision logic - no state mutations"""
        pass

# Example Implementation
class TraderPolicy(BasePolicy):
    def decide(self, snapshot: MarketSnapshot) -> List[Action]:
        # Price momentum analysis
        if self.detect_upward_trend(snapshot):
            return [Action(ActionKind.SWAP_BUY, params={...})]
        elif self.detect_downward_trend(snapshot):
            return [Action(ActionKind.SWAP_SELL, params={...})]
        return [Action(ActionKind.HOLD, params={})]
```

**Available Policy Types:**
- `TraderPolicy`: Price momentum-based trading
- `StakerPolicy`: APY-driven staking behavior  
- `LPPolicy`: Risk-adjusted liquidity provision
- `LenderPolicy`: Lending protocol interaction
- `VelodromeLPPolicy`: Vote-escrow governance participation
- `HardwareSellerPolicy`: DePIN device sales
- `NodeOperatorPolicy`: Node operation and rewards

---

## Configuration System Excellence

### Comprehensive Parameter Schema

The system manages **140+ parameters** across 10 categories:

```python
class RadcadClientConfig(BaseModel):
    """Complete RadCAD client configuration schema"""
    
    # 1. Token Supply & Fundraising (25 params)
    initial_total_supply: Optional[int]
    public_sale_valuation: Optional[int]
    angel_raised: Optional[int]
    # ... more fundraising parameters
    
    # 2. Token Allocations (11 params)
    team_allocation: Optional[float]
    advisor_allocation: Optional[float]
    community_allocation: Optional[float]
    # ... more allocation parameters
    
    # 3. Vesting Schedules (45 params)
    team_initial_vesting: Optional[int]
    team_cliff: Optional[int]
    team_vesting_duration: Optional[int]
    # ... vesting for all allocations
    
    # 4. Airdrop Parameters (6 params)
    airdrop_date1: Optional[str]
    airdrop_amount1: Optional[float]
    # ... multiple airdrops
    
    # 5. Adoption & Revenue (20 params)
    initial_token_holders: Optional[int]
    product_income_per_month: Optional[float]
    # ... growth and revenue parameters
    
    # 6. Expenses & Cash Flow (15 params)
    salaries_per_month: Optional[float]
    buyback_fixed_per_month: Optional[float]
    # ... operational parameters
    
    # 7. Utility Mechanisms (12 params)
    staking_share: Optional[float]
    liquidity_mining_APR: Optional[float]
    # ... mechanism parameters
    
    # 8. Point Systems (8 params)
    off_chain_point_system_flag: Optional[str]
    point_to_token_conversion_rate: Optional[float]
    # ... point system parameters
    
    # 9. Advanced Simulation (15 params)
    speculation_factor: Optional[float]
    agent_behavior: Optional[str]
    # ... simulation tuning parameters
    
    # 10. Mechanism Parameters (8 params)
    staker_rev_share: Optional[float]
    mint_burn_ratio: Optional[float]
    # ... protocol mechanism parameters
```

### Google Sheets Integration

**Live configuration from spreadsheets:**

```python
# Command-line import from Google Sheets
python -m cli.sheets_import import \
  --sheet-url "https://docs.google.com/spreadsheets/d/[SHEET_ID]/edit" \
  --client-id "client-name" \
  --client-name "Client Display Name" \
  --output "configs/clients/client-name.yaml"

# Programmatic usage
from src.integrations.google_sheets.radcad_parser import convert_radcad_sheet_to_yaml

config_path = convert_radcad_sheet_to_yaml(
    sheet_url=sheet_url,
    client_id="unique-client-id", 
    client_name="Client Name",
    output_path="configs/clients/client.yaml"
)
```

**Benefits:**
- **Non-Technical Editing**: Clients modify parameters without code changes
- **Version Control**: Sheet history provides parameter change tracking
- **Collaborative**: Multiple stakeholders can review/edit parameters
- **Live Updates**: Configurations update directly from client spreadsheets

---

## Market Plugin System

### Market Architecture

```python
class BaseMarket(ABC):
    """Abstract base for all DeFi markets"""
    
    @abstractmethod
    def route(self, action: Action, simulation_state: Dict[str, Any]) -> List[Event]:
        """Execute action and return events"""
        pass
    
    @abstractmethod  
    def end_of_block(self, simulation_state: Dict[str, Any]) -> None:
        """Handle periodic updates (interest, rewards)"""
        pass
    
    @abstractmethod
    def get_market_data(self) -> Dict[str, Any]:
        """Provide data for MarketSnapshot"""
        pass

class MarketRegistry:
    """Central action routing system"""
    
    def register_market(self, market: BaseMarket, action_kinds: List[ActionKind]):
        """Register market for specific actions"""
        
    def route_action(self, action: Action, simulation_state: Dict) -> List[Event]:
        """Route action to appropriate market"""
```

### Implemented Markets

**AMM Markets:**
```python
class UniswapV2Market(BaseMarket):
    """Constant product AMM (x*y=k)"""
    def route(self, action: Action, simulation_state: Dict) -> List[Event]:
        if action.kind == ActionKind.SWAP_BUY:
            return self._handle_swap_buy(action, agent)
        elif action.kind == ActionKind.ADD_LIQUIDITY:
            return self._handle_add_liquidity(action, agent)

class VelodromeMarket(BaseMarket):
    """Vote-escrow AMM with governance"""
    def route(self, action: Action, simulation_state: Dict) -> List[Event]:
        if action.kind == ActionKind.LOCK_TOKENS:
            return self._handle_lock_tokens(action, agent)
        elif action.kind == ActionKind.VOTE_FOR_POOL:
            return self._handle_pool_voting(action, agent)
```

**Lending Markets:**
```python
class CompoundLendingMarket(BaseMarket):
    """Pooled lending with cToken mechanics"""
    def route(self, action: Action, simulation_state: Dict) -> List[Event]:
        if action.kind == ActionKind.SUPPLY:
            return self._handle_supply(action, agent)
        elif action.kind == ActionKind.BORROW:
            return self._handle_borrow(action, agent)

class EulerLendingMarket(BaseMarket):
    """Isolated lending markets with risk isolation"""
    def route(self, action: Action, simulation_state: Dict) -> List[Event]:
        if action.kind == ActionKind.SUPPLY:
            return self._handle_isolated_supply(action, agent)
        elif action.kind == ActionKind.LIQUIDATE:
            return self._handle_liquidation(action, agent)
```

**Staking Markets:**
```python
class StakingMarket(BaseMarket):
    """Standard token staking with configurable rewards"""
    def route(self, action: Action, simulation_state: Dict) -> List[Event]:
        if action.kind == ActionKind.STAKE:
            return self._handle_stake(action, agent)
        elif action.kind == ActionKind.CLAIM_REWARD:
            return self._handle_claim_rewards(action, agent)
```

---

## Universal Action Framework

### Action System Design

**70+ universal DeFi actions** that cover all major mechanisms:

```python
class ActionKind(Enum):
    """Enumeration of all possible DeFi actions"""
    
    # ── Trading
    SWAP_BUY            = auto()
    SWAP_SELL           = auto()
    LIMIT_ORDER         = auto()
    CANCEL_ORDER        = auto()

    # ── Liquidity & AMMs
    ADD_LIQUIDITY       = auto()
    REMOVE_LIQUIDITY    = auto()
    COLLECT_FEES        = auto()

    # ── Lending / Borrowing
    SUPPLY              = auto()  # Supply assets to lending protocol
    WITHDRAW            = auto()  # Withdraw supplied assets
    DEPOSIT_COLLATERAL  = auto()
    WITHDRAW_COLLATERAL = auto()
    BORROW              = auto()
    REPAY               = auto()
    LIQUIDATE           = auto()

    # ── Staking / Locking
    STAKE               = auto()
    UNSTAKE             = auto()
    LOCK_TOKENS         = auto()   # Lock tokens for vote-escrow
    EXTEND_LOCK         = auto()
    UNLOCK              = auto()

    # ── Yield / Rewards
    CLAIM_REWARD        = auto()
    COMPOUND_REWARD     = auto()
    DELEGATE_VOTE       = auto()
    VOTE_GAUGE_WEIGHT   = auto()

    # ── Treasury / Protocol Ops
    BUYBACK_BURN        = auto()
    MINT                = auto()
    BURN                = auto()
    ALLOCATE_INCENTIVE  = auto()

    # ── Governance
    CREATE_PROPOSAL     = auto()
    VOTE_PROPOSAL       = auto()
    VOTE_FOR_POOL       = auto()  # Velodrome-style pool voting

    # ── DePIN & Device Mining
    ONBOARD_NODE        = auto()
    REPORT_UPTIME       = auto()
    OFFBOARD_NODE       = auto()

    # ── Hold (no action)
    HOLD                = auto()
```

### Action-Event Flow

```python
@dataclass
class Action:
    """Agent intention"""
    kind: ActionKind
    agent_id: str  
    params: Dict[str, Any]  # market_id, amount, etc.
    ts: int

@dataclass  
class Event:
    """Market execution result"""
    action_kind: ActionKind
    agent_id: str
    market_id: str
    result: Dict[str, Any]  # tokens_received, fees_paid, etc.
    ts: int

@dataclass
class MarketSnapshot:
    """Read-only market state for agent decisions"""
    timestamp: int
    token_price: float
    market_cap: float
    staking_apy: float
    pool_apy: float
    daily_volume: float
    protocol_treasury: float
    markets: Dict[str, Dict[str, Any]]  # market-specific data
```

---

## Data Flow Architecture

### End-to-End Pipeline

```
Google Sheets (140+ params) 
    ↓ [Google Sheets API]
RadCAD DataFrame
    ↓ [RadCADCsvConverter]  
RadcadClientConfig (Pydantic validation)
    ↓ [to_legacy_format()]
Legacy YAML Configuration
    ↓ [SimulationFactory]
TokenomicsSimulation Engine
    ↓ [Agent-Action-Market Loop]
Daily Events & State Updates  
    ↓ [MetricsCalculator]
Comprehensive Analysis Results
```

### Daily Simulation Loop

```python
# Main simulation loop
for day in range(max_days):
    # 1. Create read-only market snapshot
    snapshot = MarketSnapshot(
        timestamp=day,
        token_price=current_price,
        market_data=market_registry.get_all_market_data()
    )
    
    # 2. Agents emit actions via policies
    all_actions = []
    for agent in agent_population.agents:
        actions = agent.decide_actions(snapshot)
        all_actions.extend(actions)
    
    # 3. Shuffle actions (simulate block ordering)
    random.shuffle(all_actions)
    
    # 4. Markets execute actions
    all_events = []
    for action in all_actions:
        events = market_registry.route_action(action, simulation_state)
        all_events.extend(events)
    
    # 5. End-of-block operations
    market_registry.end_of_block_all(simulation_state)
    
    # 6. Update metrics
    metrics_history.append(collect_comprehensive_metrics())
```

---

## Analysis and Metrics

### Comprehensive Metrics System

```python
class TokenomicsMetricsCalculator:
    """Calculate comprehensive tokenomics metrics"""
    
    def calculate_price_metrics(self) -> Dict[str, Any]:
        """Price evolution, volatility, appreciation scenarios"""
        return {
            'final_price_stats': {
                'mean': np.mean(final_prices),
                'percentiles': {p: np.percentile(final_prices, p) for p in [5, 25, 75, 95]}
            },
            'price_volatility': {
                'mean': np.mean(volatilities),
                'median': np.median(volatilities)
            },
            'price_appreciation': {
                'mean_change': np.mean(price_changes),
                'appreciation_scenarios': success_rate
            }
        }
        
    def calculate_protocol_metrics(self) -> Dict[str, Any]:  
        """Treasury health, market cap analysis"""
        return {
            'treasury_health': {
                'final_balance_mean': np.mean(final_treasuries),
                'positive_balance_probability': success_rate,
                'bankruptcy_risk': failure_rate
            },
            'market_cap_analysis': {
                'final_market_cap_mean': np.mean(market_caps),
                'market_cap_percentiles': percentile_analysis
            }
        }
        
    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """VaR, price movement scenarios, stability ratios"""
        return {
            'value_at_risk': {
                'var_95': np.percentile(returns, 5),
                'var_99': np.percentile(returns, 1)
            },
            'price_movement_scenarios': {
                'high_appreciation': scenario_probabilities,
                'stable_price': stability_analysis,
                'extreme_movements': tail_risk_analysis
            }
        }
```

### Key Metrics Categories

**Price Analytics:**
- Final price distributions and percentiles
- Price volatility analysis over time
- Price appreciation/depreciation scenarios
- Return distributions and tail risks

**Protocol Health:**
- Treasury balance evolution and sustainability
- Revenue vs expenses analysis
- Market capitalization growth patterns
- Cash flow and runway analysis

**Risk Metrics:**
- Value at Risk (95%/99% confidence intervals)
- Maximum drawdown analysis
- Scenario probability distributions
- Price stability ratios and volatility measures

**Agent Behavior:**
- Staking participation rates over time
- Trading volume patterns and liquidity
- LP provision dynamics and yields
- Governance participation metrics

---

## Extension Patterns

### Adding New Client Behaviors

**For 99% of cases: Create new Policy classes**

```python
class DAOVoterPolicy(BasePolicy):
    """Policy for DAO governance participation"""
    
    def __init__(self, voting_frequency: float = 0.1, governance_weight: float = 0.3):
        self.voting_frequency = voting_frequency
        self.governance_weight = governance_weight
        self.last_vote_time = 0
    
    def decide(self, snapshot: MarketSnapshot) -> List[Action]:
        actions = []
        
        # Vote on proposals monthly  
        if (snapshot.timestamp - self.last_vote_time) >= 30:
            actions.append(Action(
                kind=ActionKind.VOTE_PROPOSAL,
                params={
                    "market_id": "governance",
                    "proposal_id": "current_proposal", 
                    "vote": "yes"
                }
            ))
            self.last_vote_time = snapshot.timestamp
        
        # Stake for voting power quarterly
        if snapshot.timestamp % 90 == 0:
            actions.append(Action(
                kind=ActionKind.STAKE,
                params={
                    "market_id": "staking",
                    "amount_pct": self.governance_weight
                }
            ))
            
        return actions or [Action(ActionKind.HOLD, params={})]

# Usage in configuration
config = {
    'agent_policies': [
        {'type': 'DAOVoterPolicy', 'count': 50, 'params': {'voting_frequency': 0.2}},
        {'type': 'TraderPolicy', 'count': 100, 'params': {'trading_frequency': 0.1}},
    ]
}
```

### Adding New DeFi Mechanisms

**Create new Market classes for novel protocols:**

```python
class OptionsMarket(BaseMarket):
    """Options trading market"""
    
    def __init__(self, underlying_asset: str):
        super().__init__(f"options_{underlying_asset}")
        self.underlying_asset = underlying_asset
        self.open_positions = {}
        self.option_prices = {}
        
    def route(self, action: Action, simulation_state: Dict) -> List[Event]:
        if action.kind == ActionKind.BUY_CALL_OPTION:
            return self._handle_buy_call(action, simulation_state)
        elif action.kind == ActionKind.EXERCISE_OPTION:
            return self._handle_exercise(action, simulation_state)
        return []
    
    def _handle_buy_call(self, action: Action, simulation_state: Dict) -> List[Event]:
        agent = simulation_state['agents'][action.agent_id]
        
        strike_price = action.params['strike_price']
        expiry = action.params['expiry']
        premium = self._calculate_option_premium(strike_price, expiry)
        
        if agent.state.cash_balance >= premium:
            agent.state.cash_balance -= premium
            
            option_id = f"call_{action.agent_id}_{strike_price}_{expiry}"
            self.open_positions[option_id] = {
                'type': 'call',
                'strike': strike_price,
                'expiry': expiry,
                'owner': action.agent_id
            }
            
            return [Event(
                action_kind=action.kind,
                agent_id=action.agent_id,
                market_id=self.id,
                result={'option_id': option_id, 'premium_paid': premium}
            )]
        
        return []
    
    def end_of_block(self, simulation_state: Dict) -> None:
        """Handle option expiries and settlement"""
        current_day = simulation_state['current_day']
        expired_options = [
            opt_id for opt_id, opt in self.open_positions.items() 
            if opt['expiry'] <= current_day
        ]
        
        for opt_id in expired_options:
            self._settle_expired_option(opt_id, simulation_state)
    
    def get_market_data(self) -> Dict[str, Any]:
        return {
            'open_interest': len(self.open_positions),
            'total_premium_volume': sum(self.option_prices.values()),
            'underlying_asset': self.underlying_asset
        }

# Register the new market
def register_options_market(market_registry: MarketRegistry):
    options_market = OptionsMarket("TOKEN")
    market_registry.register_market(options_market, [
        ActionKind.BUY_CALL_OPTION,
        ActionKind.SELL_PUT_OPTION,
        ActionKind.EXERCISE_OPTION
    ])
```

### Adding New Actions

**For truly novel DeFi mechanisms, extend the ActionKind enum:**

```python
class ActionKind(Enum):
    # ... existing actions ...
    
    # ── Options Trading (New Category)
    BUY_CALL_OPTION     = auto()
    SELL_PUT_OPTION     = auto()
    EXERCISE_OPTION     = auto()
    
    # ── Insurance (New Category)  
    BUY_COVERAGE        = auto()
    CLAIM_INSURANCE     = auto()
    
    # ── Prediction Markets (New Category)
    CREATE_MARKET       = auto()
    BUY_SHARES          = auto()
    RESOLVE_MARKET      = auto()
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

**Core Architecture Setup**
- [ ] Implement Agent-Action-Market pattern
- [ ] Create universal Action system with base ActionKind enum
- [ ] Build MarketRegistry for action routing
- [ ] Establish BaseAgent and BasePolicy interfaces
- [ ] Create MarketSnapshot for read-only state access

```python
# Milestone: Basic simulation loop working
for day in range(simulation_days):
    snapshot = create_market_snapshot()
    actions = collect_agent_actions(snapshot)
    events = execute_actions_via_markets(actions)
    update_state_from_events(events)
```

### Phase 2: Core Markets (Weeks 5-8)

**Essential DeFi Markets**
- [ ] Implement UniswapV2Market (constant product AMM)
- [ ] Create StakingMarket (basic token staking)
- [ ] Build CompoundLendingMarket (pooled lending)
- [ ] Add basic TraderPolicy and StakerPolicy
- [ ] Integrate mathematical accuracy for pricing

```python
# Milestone: Multi-market simulation
markets = [
    UniswapV2Market(token_reserve=100000, usdc_reserve=50000),
    StakingMarket(base_apy=0.08),
    CompoundLendingMarket(base_rate=0.02)
]
```

### Phase 3: Configuration System (Weeks 9-12)

**Comprehensive Parameter Management**
- [ ] Design parameter schema (start with 50+ core parameters)
- [ ] Build Google Sheets integration with API
- [ ] Create configuration validation system
- [ ] Implement legacy format conversion
- [ ] Add configuration-driven agent creation

```python
# Milestone: Configuration-driven simulations
config = load_from_google_sheets(sheet_url)
simulation = SimulationFactory.create_simulation(config)
results = simulation.run_monte_carlo(runs=100)
```

### Phase 4: Advanced Features (Weeks 13-16)

**Production-Ready Features**
- [ ] Add comprehensive metrics calculator
- [ ] Create visualization system
- [ ] Implement Kubernetes orchestration
- [ ] Build Monte Carlo statistical analysis
- [ ] Add advanced markets (Velodrome, Euler)

```python
# Milestone: Production deployment
python main.py --config client.yaml --k8s --parallel-jobs 4 --runs 1000
```

---

## Best Practices

### Code Organization Standards

**File Structure Principles:**
- **One canonical file per concept**: No `_v2`, `_enhanced`, or `_new` suffixes
- **Pure functions**: Mathematical calculations separated from state management  
- **Interface compliance**: All components implement abstract base classes
- **Type safety**: Strong typing with Pydantic schemas and type hints

**Example Structure:**
```python
# ✅ Good: Clear, single-purpose files
src/core/markets/uniswap_v2.py      # Uniswap V2 market implementation
src/core/math/uniswap_v2.py         # Pure AMM mathematics
src/core/agents/policies/trader.py  # Trading decision logic

# ❌ Bad: Versioned or unclear files  
src/core/markets/uniswap_v2_enhanced.py
src/core/markets/trading_market_new.py
src/core/agents/trader_agent_v3.py
```

### Configuration Management

**Parameter Handling:**
- **No hardcoded defaults**: All values must come from client configurations
- **Comprehensive validation**: Pydantic schemas prevent invalid configurations
- **Legacy compatibility**: Smooth transitions during architecture upgrades
- **Version control**: Configuration changes tracked through Git and sheet history

```python
# ✅ Good: Configuration-driven
class RadcadClientConfig(BaseModel):
    staking_apy: Optional[float] = None  # Must be provided
    initial_supply: Optional[int] = None # Must be provided
    
    def validate_required_fields(self):
        if self.staking_apy is None:
            raise ValueError("staking_apy must be provided in configuration")

# ❌ Bad: Hardcoded defaults
class SimulationConfig:
    staking_apy: float = 0.08  # Hidden assumption
    initial_supply: int = 1000000  # Hardcoded value
```

### Testing Strategy

**Component Testing:**
```python
# Test mathematical accuracy
def test_uniswap_pricing():
    pool = LiquidityPool(token_a=100000, token_b=50000)
    tokens_out, new_price = pool.swap_exact_tokens_for_tokens(1000, 'b')
    assert abs(tokens_out - expected_value) < 0.01

# Test policy decisions  
def test_trader_policy():
    policy = TraderPolicy(trading_frequency=0.2)
    snapshot = MarketSnapshot(token_price=1.5, price_volatility=0.1)
    actions = policy.decide(snapshot)
    assert len(actions) > 0
    assert all(isinstance(a, Action) for a in actions)

# Test market routing
def test_market_execution():
    market = UniswapV2Market(100000, 50000)
    action = Action(ActionKind.SWAP_BUY, "agent1", {"amount_in": 1000})
    events = market.route(action, simulation_state)
    assert len(events) > 0
    assert events[0].result['tokens_received'] > 0
```

### Performance Optimization

**Scalability Guidelines:**
- **Agent Scale**: Design for 100-1000+ agents efficiently
- **Simulation Duration**: Support 1-10 years (365-3650 days)
- **Monte Carlo**: Enable 50-1000 runs for statistical significance  
- **Memory Usage**: Target ~100MB for typical configurations
- **Parallel Execution**: Design for Kubernetes horizontal scaling

```python
# ✅ Good: Efficient agent processing
def process_agents_batch(agents: List[BaseAgent], snapshot: MarketSnapshot) -> List[Action]:
    """Process agents in batches for memory efficiency"""
    all_actions = []
    batch_size = 100
    
    for i in range(0, len(agents), batch_size):
        batch = agents[i:i + batch_size]
        batch_actions = []
        
        for agent in batch:
            actions = agent.decide_actions(snapshot)
            batch_actions.extend(actions)
        
        all_actions.extend(batch_actions)
    
    return all_actions

# ❌ Bad: Memory-intensive processing
def process_agents_inefficient(agents: List[BaseAgent]) -> List[Action]:
    """Loads all agent histories into memory"""
    full_histories = [agent.load_complete_history() for agent in agents]  # Memory intensive
    return process_with_full_context(full_histories)
```

### Error Handling and Observability

**Comprehensive Logging:**
```python
import logging

# Structured logging for production
logging.info(f"Day {day} - Events executed: {event_counts}")
logging.info(f"Market state: Price=${price:.4f}, Volume=${volume:,.0f}")

# Error handling with context
try:
    events = market.route(action, simulation_state)
except InsufficientBalanceError as e:
    logging.warning(f"Agent {action.agent_id} insufficient balance: {e}")
except MarketError as e:
    logging.error(f"Market {market.id} execution failed: {e}")
    # Continue simulation with other agents
```

**Metrics Tracking:**
```python
# Comprehensive metrics collection
metrics = {
    'day': current_day,
    'market_state': {
        'token_price': snapshot.token_price,
        'market_cap': snapshot.market_cap,
        'daily_volume': snapshot.daily_volume
    },
    'agent_metrics': {
        'total_agents': len(agents),
        'active_traders': count_active_traders(),
        'staking_participation': calculate_staking_rate()
    },
    'protocol_metrics': {
        'treasury_balance': protocol_treasury,
        'total_revenue': total_protocol_revenue,
        'circulating_supply': current_circulating_supply
    }
}
```

---

## Conclusion

This architecture blueprint demonstrates how to build a **commercial-grade DeFi simulation system** that combines modularity, accuracy, and scalability. The Time Chamber framework sets a new standard for tokenomics simulation through its Agent-Action-Market pattern, comprehensive configuration management, and plugin-based extensibility.

**Key Success Factors:**

1. **Separation of Concerns**: Agents handle state, policies make decisions, markets execute actions
2. **Configuration-Driven**: Zero hardcoded values, all parameters from client data
3. **Mathematical Accuracy**: Proper implementation of DeFi protocol mathematics
4. **Plugin Architecture**: Markets and policies are composable, independent components
5. **Production-Ready**: Kubernetes scaling, comprehensive metrics, error handling

**Implementation Priority:**
1. Start with the Agent-Action-Market pattern as your foundation
2. Build 2-3 core markets (AMM, Staking, Lending) to prove the concept
3. Add comprehensive configuration management with Google Sheets integration  
4. Implement metrics and visualization for client deliverables
5. Scale with Kubernetes orchestration for large Monte Carlo simulations

This blueprint provides the architectural foundation for any serious DeFi simulation system, ensuring both the modularity needed for diverse client requirements and the mathematical rigor required for accurate tokenomics modeling.

---

*Based on the Time Chamber architecture by Unit Zero Labs - a production-ready framework serving multiple DeFi protocols with sophisticated tokenomics modeling.*
