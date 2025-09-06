[1 tool called]

# 🔧 Engine Architecture Refactoring Plan

## **Problem Summary**
The current engine architecture is inconsistent and confusing:
- AAVE engine inherits from Tidal (wrong conceptually)
- Sophisticated Uniswap V3 math only used in High Tide, not base Tidal
- Comparison scripts compare incompatible architectures


## **Target Architecture**

```
BaseLendingEngine (Abstract)
├── Common lending mechanics
├── Health factor calculations  
├── Basic liquidation framework
└── Agent management

TidalProtocolEngine (BaseLendingEngine)
├── Uses sophisticated Uniswap V3 math
├── MOET:BTC pools with proper slippage
├── Tidal-specific lending mechanics
└── Foundation for all Tidal variants

HighTideVaultEngine (TidalProtocolEngine)  
├── Inherits all Tidal Uniswap V3 functionality
├── Adds MOET:YieldToken pools
├── Adds rebalancing mechanisms
└── Yield vault strategies

AaveProtocolEngine (BaseLendingEngine)
├── Pure AAVE implementation
├── Traditional liquidation (50% + 5% bonus)  
├── AAVE's actual DEX mechanics
└── No Tidal dependencies
```

## **Refactoring Tasks**

### **Phase 1: Create Base Architecture**

#### **Task 1.1: Create BaseLendingEngine**
```python
# File: tidal_protocol_sim/simulation/base_lending_engine.py

class BaseLendingEngine:
    """Abstract base class for all lending protocol simulations"""
    
    def __init__(self, config):
        self.config = config
        self.agents = {}
        self.current_step = 0
        self.liquidation_events = []
        
    def run_simulation(self, steps: int) -> Dict:
        """Abstract method - must be implemented by subclasses"""
        raise NotImplementedError
        
    def _process_agent_actions(self):
        """Common agent processing logic"""
        pass
        
    def _check_liquidations(self):
        """Common liquidation checking logic"""
        pass
        
    def _record_metrics(self):
        """Common metrics recording"""
        pass
```

#### **Task 1.2: Refactor TidalProtocolEngine**
```python
# File: tidal_protocol_sim/simulation/tidal_engine.py (rename from engine.py)

from .base_lending_engine import BaseLendingEngine
from ..core.uniswap_v3_math import UniswapV3Pool, UniswapV3SlippageCalculator

class TidalProtocolEngine(BaseLendingEngine):
    """Tidal Protocol with sophisticated Uniswap V3 mathematics"""
    
    def __init__(self, config: TidalConfig):
        super().__init__(config)
        
        # Initialize Uniswap V3 pools for ALL Tidal simulations
        self._setup_uniswap_v3_pools()
        
    def _setup_uniswap_v3_pools(self):
        """Setup Uniswap V3 pools with proper math"""
        from ..core.uniswap_v3_math import create_moet_btc_pool
        
        self.moet_btc_pool = create_moet_btc_pool(
            pool_size=self.config.moet_btc_pool_size,
            btc_price=self.config.btc_initial_price,
            concentration=self.config.moet_btc_concentration
        )
        
        self.slippage_calculator = UniswapV3SlippageCalculator(self.moet_btc_pool)
```

### **Phase 2: Fix High Tide Engine**

#### **Task 2.1: Refactor HighTideVaultEngine**
```python
# File: tidal_protocol_sim/simulation/high_tide_engine.py

from .tidal_engine import TidalProtocolEngine  # Changed inheritance

class HighTideVaultEngine(TidalProtocolEngine):  # Now inherits from Tidal
    """High Tide Yield Vaults built on Tidal Protocol"""
    
    def __init__(self, config: HighTideConfig):
        super().__init__(config)  # Gets all Uniswap V3 functionality
        
        # Add yield token pools ON TOP of existing Tidal functionality
        self._setup_yield_token_pools()
        
    def _setup_yield_token_pools(self):
        """Add yield token functionality to existing Tidal base"""
        from ..core.uniswap_v3_math import create_yield_token_pool
        
        self.yield_token_pool = create_yield_token_pool(
            pool_size=self.config.moet_yield_pool_size,
            btc_price=self.config.btc_initial_price,
            concentration=self.config.yield_token_concentration
        )
        
        self.yield_token_slippage_calculator = UniswapV3SlippageCalculator(
            self.yield_token_pool
        )
```

### **Phase 3: Create Pure AAVE Engine**

#### **Task 3.1: Create Independent AaveProtocolEngine**
```python
# File: tidal_protocol_sim/simulation/aave_engine.py

from .base_lending_engine import BaseLendingEngine  # Changed inheritance

class AaveConfig:
    """Pure AAVE configuration - no Tidal dependencies"""
    
    def __init__(self):
        self.scenario_name = "AAVE_Protocol"
        self.liquidation_threshold = 0.85
        self.liquidation_bonus = 0.05  # 5%
        self.liquidation_percentage = 0.5  # 50%
        # No Uniswap V3 parameters

class AaveProtocolEngine(BaseLendingEngine):  # No Tidal inheritance
    """Pure AAVE Protocol implementation"""
    
    def __init__(self, config: AaveConfig):
        super().__init__(config)
        
        # AAVE uses traditional AMM pools, not Uniswap V3
        self._setup_aave_liquidation_pools()
        
    def _setup_aave_liquidation_pools(self):
        """Setup AAVE's actual liquidation mechanisms"""
        # Traditional constant product AMM
        # 50% liquidation + 5% bonus
        pass
        
    def _execute_aave_liquidation(self, agent, collateral_asset, debt_amount):
        """Traditional AAVE liquidation: 50% collateral + 5% bonus"""
        liquidation_amount = debt_amount * 0.5  # 50% max
        bonus = liquidation_amount * 0.05  # 5% bonus
        return liquidation_amount + bonus
```

### **Phase 4: Fix Comparison Scripts**

#### **Task 4.1: Update High Tide vs AAVE Comparison**
```python
# File: run_high_tide_vs_aave_comparison.py

from tidal_protocol_sim.simulation.high_tide_engine import HighTideVaultEngine, HighTideConfig
from tidal_protocol_sim.simulation.aave_engine import AaveProtocolEngine, AaveConfig

def run_comparison():
    """Now comparing apples to apples: both use same scenarios but different protocols"""
    
    # High Tide: Tidal + Yield Vaults
    ht_config = HighTideConfig()
    ht_engine = HighTideVaultEngine(ht_config)  # Gets Uniswap V3 from Tidal base
    
    # AAVE: Pure AAVE protocol
    aave_config = AaveConfig()  # No Tidal dependencies
    aave_engine = AaveProtocolEngine(aave_config)  # Pure AAVE
    
    # Fair comparison: same market conditions, different protocols
```

#### **Task 4.2: Update Borrow Cap Analysis**
```python
# File: moet_yt_borrow_cap_analysis.py

from tidal_protocol_sim.simulation.tidal_engine import TidalProtocolEngine, TidalConfig

def run_borrow_cap_analysis():
    """Test Tidal protocol's borrow capacity using proper Uniswap V3 math"""
    
    # Use base Tidal engine with Uniswap V3 math
    tidal_config = TidalConfig()
    tidal_engine = TidalProtocolEngine(tidal_config)  # Now has Uniswap V3
    
    # Test liquidation capacity with proper slippage calculations
```

## **File Structure Changes**

### **Rename/Move Files:**
```
OLD → NEW
tidal_protocol_sim/simulation/engine.py → tidal_protocol_sim/simulation/tidal_engine.py
tidal_protocol_sim/simulation/high_tide_engine.py → tidal_protocol_sim/simulation/high_tide_vault_engine.py
```

### **New Files to Create:**
```
tidal_protocol_sim/simulation/base_lending_engine.py
tidal_protocol_sim/simulation/configs/
├── base_config.py
├── tidal_config.py  
├── high_tide_config.py
└── aave_config.py
```

## **Migration Steps**

### **Step 1: Create Base Classes**
1. Create `BaseLendingEngine` with common functionality
2. Create `BaseLendingConfig` with common parameters
3. Test base classes work independently

### **Step 2: Migrate Tidal Engine**
1. Rename `engine.py` → `tidal_engine.py`
2. Add Uniswap V3 math to base Tidal engine
3. Update `TidalSimulationEngine` → `TidalProtocolEngine`
4. Test Tidal engine with Uniswap V3 math

### **Step 3: Fix High Tide Engine**
1. Change inheritance: `TidalSimulationEngine` → `TidalProtocolEngine`
2. Remove duplicate Uniswap V3 setup (inherit from Tidal)
3. Focus on yield token additions only
4. Test High Tide inherits Tidal functionality

### **Step 4: Rebuild AAVE Engine**
1. Remove Tidal inheritance
2. Inherit from `BaseLendingEngine` only
3. Implement pure AAVE liquidation mechanics
4. Remove all Uniswap V3 references

### **Step 5: Update Analysis Scripts**
1. Update imports to use new engine names
2. Fix comparison logic to use proper engines
3. Test all analysis scripts work with new architecture

## **Testing Plan**

### **Unit Tests:**
```python
def test_tidal_engine_has_uniswap_v3():
    """Verify base Tidal engine uses Uniswap V3 math"""
    
def test_high_tide_inherits_tidal():
    """Verify High Tide gets Uniswap V3 from Tidal base"""
    
def test_aave_independent():
    """Verify AAVE has no Tidal dependencies"""
    
def test_comparison_scripts():
    """Verify comparison scripts use correct engines"""
```

### **Integration Tests:**
```python
def test_high_tide_vs_aave_comparison():
    """Test full comparison with proper architectures"""
    
def test_borrow_cap_analysis():
    """Test borrow cap analysis uses Tidal Uniswap V3"""
```

## **Benefits of This Refactor**

1. **Architectural Clarity**: Each engine has a clear, logical purpose
2. **Uniswap V3 Utilization**: Your week of work becomes the foundation for all Tidal simulations
3. **Proper Comparisons**: High Tide vs AAVE compares different protocols, not different variants of the same protocol
4. **Maintainability**: Clear inheritance hierarchy makes future changes easier
5. **Extensibility**: Easy to add new protocols or Tidal variants

