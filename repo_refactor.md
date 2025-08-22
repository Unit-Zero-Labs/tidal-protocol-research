# System Prompt: Tidal Protocol Codebase Refactoring

## **Objective**
Refactor the existing codebase to create a streamlined, Tidal Protocol-focused simulation system that maintains essential modularity for agent policies and stress testing while eliminating unnecessary generic framework complexity. The primary goal is to enable comprehensive stress testing of the lending mechanism, liquidity parameters, liquidation methodology, and protocol stability.

## **Core Requirements**

### **1. Preserve Essential Tidal Protocol Mechanics**
- **Kinked Interest Rate Model**: Exact implementation with Tidal's specific parameters (base_rate_per_block, multiplier_per_block, jump_per_block, kink at 80%)
- **MOET Stablecoin System**: Minting/burning mechanics, peg stability bands (±2%), **no mint/burn fees**
- **Ebisu-Style Debt Cap**: A × B × C formula (liquidation capacity × DEX allocation × underwater percentage)
- **Multi-Asset Collateral**: ETH, BTC, FLOW, USDC with specific collateral factors and liquidation thresholds
- **Liquidation Engine**: 8% penalty, 50% close factor, health factor calculations
- **Concentrated Liquidity Pools**: UniswapV3-style math for MOET pairs

### **2. Maintain Strategic Modularity**

#### **Agent Policy System** (Simplified but Flexible)
```python
# Keep these policy types only:
class PolicyType(Enum):
    TIDAL_LENDER = "tidal_lender"    # Core lending behavior
    BASIC_TRADER = "basic_trader"     # Simple trading for liquidity
    LIQUIDATOR = "liquidator"        # Liquidation bot behavior

# Agent behaviors should focus on:
- Supply/withdraw decisions based on APY thresholds
- Borrow MOET against collateral with health factor management
- Emergency actions when health factor < 1.1
- Basic liquidation opportunities
```

#### **Stress Testing Framework** (Essential Scenarios)
```python
# Focus on these stress test categories:
1. Single Asset Price Shocks (ETH -30%, BTC -35%, FLOW -50%, USDC depeg)
2. Multi-Asset Crashes (crypto winter scenarios)
3. Liquidity Crises (concentrated liquidity depletion)
4. Parameter Sensitivity (collateral factors, liquidation thresholds)
5. Cascading Liquidations (health factor deterioration chains)
```

## **3. Elimination Targets**

### **Remove Completely:**
- Generic DeFi framework abstractions (BaseMarket, BasePolicy)
- Multiple market support (UniswapV2, Compound, Staking)
- Complex configuration schemas with Pydantic validation
- Event-driven architecture with action/event queues
- Google Sheets integration and external connectors
- Advanced agent strategies (rebalancing, optimization, position history)
- Factory patterns and dependency injection systems
- **MOET mint/burn fees** (simplified to fee-less minting/burning)

### **Simplify Drastically:**
- Configuration management → Simple parameter dictionaries
- Agent decision-making → Rule-based logic with clear thresholds
- Simulation engine → Direct function calls instead of event system
- Metrics calculation → Focus on protocol stability metrics only
- Visualization → Essential charts for stress testing results

## **4. Target Architecture**

```
tidal_protocol_sim/
├── core/
│   ├── protocol.py           # Main TidalProtocol class (lending logic)
│   ├── math.py              # Pure math functions (interest rates, liquidations)
│   ├── liquidity_pools.py   # MOET pair pools with concentrated liquidity
│   └── moet.py              # MOET stablecoin mechanics (no fees)
├── agents/
│   ├── base_agent.py        # Minimal agent interface
│   ├── tidal_lender.py      # Core lending behavior
│   ├── trader.py            # Basic trading for market activity
│   └── liquidator.py        # Liquidation bot logic
├── simulation/
│   ├── engine.py            # Streamlined simulation runner
│   ├── state.py             # Protocol and agent state management
│   └── config.py            # Parameter definitions and scenarios
├── stress_testing/
│   ├── scenarios.py         # Stress test scenario definitions
│   ├── runner.py            # Stress test execution engine
│   └── analyzer.py          # Results analysis and metrics
├── analysis/
│   ├── metrics.py           # Protocol stability metrics
│   ├── liquidation_analysis.py # Liquidation efficiency analysis
│   └── visualization.py     # Essential charts and dashboards
└── main.py                  # Entry point for stress testing
```

## **5. Specific Implementation Guidelines**

### **Protocol Core (protocol.py)**
```python
class TidalProtocol:
    """Streamlined Tidal Protocol implementation focused on lending mechanics"""
    
    # Essential state only:
    - asset_pools: Dict[Asset, AssetPool]  # ETH, BTC, FLOW, USDC pools
    - moet_system: MoetStablecoin
    - liquidity_pools: Dict[str, LiquidityPool]  # MOET pairs
    - protocol_treasury: float
    
    # Core methods only:
    - supply(agent_id, asset, amount) -> bool
    - borrow(agent_id, amount_moet) -> bool  # Only MOET borrowing
    - repay(agent_id, amount_moet) -> bool
    - liquidate(liquidator_id, target_id, collateral_asset, repay_amount) -> bool
    - calculate_health_factor(agent_id) -> float
    - calculate_debt_cap() -> float
    - accrue_interest() -> None
```

### **MOET System (moet.py)**
```python
class MoetStablecoin:
    """Simplified MOET stablecoin without mint/burn fees"""
    
    # Key parameters:
    - total_supply: float
    - target_price: float = 1.0
    - stability_bands: tuple = (0.98, 1.02)  # ±2% bands
    
    # Core methods (no fees):
    - mint(amount) -> amount  # 1:1 minting, no fees
    - burn(amount) -> amount  # 1:1 burning, no fees
    - is_peg_stable() -> bool
    - calculate_stability_action() -> Optional[str]
```

### **Agent Behavior (tidal_lender.py)**
```python
class TidalLender:
    """Simplified but effective lending agent"""
    
    # Key parameters only:
    - target_health_factor: float = 1.5
    - min_supply_apy: float = 0.02
    - moet_borrowing_ratio: float = 0.6
    - risk_tolerance: float = 0.5
    
    # Decision logic:
    def decide_action(self, protocol_state) -> Action:
        hf = self.calculate_health_factor()
        if hf < 1.1: return emergency_repay()
        elif hf < 1.5: return conservative_action()
        elif high_apy_opportunity(): return supply_action()
        elif can_borrow_safely(): return borrow_action()
        else: return hold()
```

### **Stress Testing Focus**
```python
# Priority stress scenarios for Tidal Protocol:
STRESS_SCENARIOS = [
    # Liquidation efficiency tests
    {"name": "ETH_Flash_Crash", "eth_drop": -0.40, "duration": 1},
    {"name": "Cascading_Liquidations", "multi_asset_drop": {"ETH": -0.30, "BTC": -0.25}},
    
    # Liquidity crisis tests  
    {"name": "MOET_Depeg", "moet_price": 0.95, "liquidity_drain": 0.5},
    {"name": "Pool_Liquidity_Crisis", "liquidity_reduction": 0.8},
    
    # Parameter sensitivity
    {"name": "Collateral_Factor_Stress", "cf_reduction": 0.1},
    {"name": "Liquidation_Threshold_Test", "lt_reduction": 0.05},
]
```

## **6. Success Metrics**

### **Code Reduction Targets:**
- Total lines of code: < 2,000 (vs current ~4,000+)
- Number of files: < 15 (vs current 20+)
- Configuration complexity: Simple dictionaries (vs Pydantic schemas)
- Agent complexity: < 200 lines per agent type

### **Functional Requirements:**
- ✅ All essential Tidal Protocol mechanics preserved
- ✅ Accurate debt cap calculations under stress
- ✅ Proper liquidation cascade modeling  
- ✅ MOET peg stability analysis (without fee complications)
- ✅ Monte Carlo stress testing (1000+ runs)
- ✅ Liquidity parameter sensitivity analysis
- ✅ Protocol revenue and treasury tracking

### **Performance Targets:**
- Simulation speed: 30-day simulation in < 10 seconds
- Memory usage: < 500MB for 1000 Monte Carlo runs
- Stress test suite: Complete analysis in < 5 minutes

## **7. Migration Strategy**

1. **Phase 1**: Extract core Tidal mechanics from both codebases into streamlined `protocol.py`
2. **Phase 2**: Implement simplified agent system with focus on lending behaviors
3. **Phase 3**: Build targeted stress testing framework for liquidation analysis
4. **Phase 4**: Create focused visualization for protocol stability metrics
5. **Phase 5**: Validate results against original `tidal_sim_v1` baseline

## **8. Validation Criteria**

The refactored system must produce equivalent results to `tidal_sim_v1` for:
- Debt cap calculations under various market conditions
- Interest rate calculations across utilization ranges  
- Health factor calculations and liquidation triggers
- MOET price stability mechanisms (simplified without fees)
- Protocol revenue calculations

**Success Definition**: A streamlined codebase that maintains the accuracy of `tidal_sim_v1` while adding the modularity needed for comprehensive stress testing and liquidity analysis, with 50% fewer lines of code and 3x faster execution.

## **9. Key Simplifications from Original Analysis**

### **MOET System Simplification**
- **Removed**: mint_fee and burn_fee parameters and calculations
- **Simplified**: Direct 1:1 minting/burning against collateral
- **Benefit**: Cleaner liquidation calculations without fee adjustments
- **Impact**: More predictable MOET supply dynamics in stress scenarios

### **Focus Areas for Stress Testing**
1. **Liquidation Methodology**: Efficiency of 8% penalty system under various market conditions
2. **Liquidity Parameters**: How concentrated liquidity affects liquidation capacity
3. **Protocol Stability**: MOET peg maintenance without fee-based stabilization mechanisms
4. **Debt Cap Accuracy**: Validation of A × B × C formula under extreme scenarios
5. **Cascading Effects**: Multi-agent liquidation scenarios and protocol resilience

This refactoring prioritizes **clarity and accuracy** over generic extensibility, ensuring the simulation can effectively model Tidal Protocol's specific risk parameters and liquidation mechanisms under stress conditions.
