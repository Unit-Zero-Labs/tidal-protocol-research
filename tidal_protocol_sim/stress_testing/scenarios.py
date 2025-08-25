#!/usr/bin/env python3
"""
Stress Test Scenario Definitions

Priority scenarios for testing liquidation methodology, liquidity parameters,
and protocol stability as specified in the refactoring requirements.
"""

from typing import Dict, List, Callable
from ..core.protocol import Asset
from ..simulation.config import StressTestScenarios
from ..simulation.engine import TidalSimulationEngine
from ..simulation.state import SimulationState


class StressTestScenario:
    """Individual stress test scenario"""
    
    def __init__(self, name: str, description: str, setup_func: Callable, duration: int = 100):
        self.name = name
        self.description = description
        self.setup_func = setup_func
        self.duration = duration
        self.results = None
    
    def apply_to_engine(self, engine: TidalSimulationEngine):
        """Apply scenario to simulation engine"""
        self.setup_func(engine)
    
    def run(self, engine: TidalSimulationEngine) -> dict:
        """Run the stress test scenario"""
        print(f"Running stress test: {self.name}")
        print(f"Description: {self.description}")
        
        # Apply scenario setup
        self.apply_to_engine(engine)
        
        # Run simulation
        results = engine.run_simulation(self.duration)
        self.results = results
        
        return results


class TidalStressTestSuite:
    """Complete stress test suite for Tidal Protocol"""
    
    def __init__(self):
        self.scenarios = self._create_scenarios()
        self.results = {}
    
    def _create_scenarios(self) -> List[StressTestScenario]:
        """Create all stress test scenarios"""
        
        return [
            # Liquidation efficiency tests
            StressTestScenario(
                "ETH_Flash_Crash",
                "ETH drops 40% to test liquidation efficiency",
                lambda engine: self._apply_price_shock(engine, {Asset.ETH: -0.40}),
                duration=50
            ),
            
            StressTestScenario(
                "Cascading_Liquidations", 
                "Multi-asset drop to trigger cascading liquidations",
                lambda engine: self._apply_price_shock(engine, {Asset.ETH: -0.30, Asset.BTC: -0.25}),
                duration=100
            ),
            
            # Liquidity crisis tests
            StressTestScenario(
                "MOET_Depeg",
                "MOET depegs to $0.95 with 50% liquidity drain",
                lambda engine: self._apply_moet_depeg_scenario(engine),
                duration=200
            ),
            
            StressTestScenario(
                "Pool_Liquidity_Crisis",
                "80% reduction in pool liquidity",
                lambda engine: self._reduce_pool_liquidity(engine, 0.8),
                duration=150
            ),
            
            # Parameter sensitivity
            StressTestScenario(
                "Collateral_Factor_Stress",
                "Reduce collateral factors by 10%",
                lambda engine: self._adjust_collateral_factors(engine, -0.1),
                duration=100
            ),
            
            StressTestScenario(
                "Liquidation_Threshold_Test",
                "Reduce liquidation thresholds by 5%",
                lambda engine: self._adjust_liquidation_thresholds(engine, -0.05),
                duration=100
            ),
            
            # Protocol stability tests
            StressTestScenario(
                "High_Utilization_Stress",
                "Push utilization rates to 95% across all assets",
                lambda engine: self._force_high_utilization(engine),
                duration=200
            ),
            
            StressTestScenario(
                "Interest_Rate_Spike",
                "Trigger interest rate spike above kink point",
                lambda engine: self._trigger_interest_spike(engine),
                duration=150
            ),
            
            # Extreme scenarios
            StressTestScenario(
                "Black_Swan_Event",
                "Multiple simultaneous shocks",
                lambda engine: self._black_swan_scenario(engine),
                duration=300
            ),
            
            StressTestScenario(
                "Debt_Cap_Stress",
                "Test debt cap under extreme liquidation conditions",
                lambda engine: self._debt_cap_stress_test(engine),
                duration=100
            ),
            
            # High Tide scenario
            StressTestScenario(
                "High_Tide_BTC_Decline",
                "BTC gradual decline with active yield token rebalancing",
                lambda engine: self._setup_high_tide_scenario(engine),
                duration=60
            )
        ]
    
    def _apply_price_shock(self, engine: TidalSimulationEngine, shocks: Dict[Asset, float]):
        """Apply immediate price shocks"""
        engine.state.apply_price_shock(shocks)
    
    def _apply_moet_depeg_scenario(self, engine: TidalSimulationEngine):
        """Apply MOET depeg with liquidity drain"""
        # Depeg MOET
        engine.state.current_prices[Asset.MOET] = 0.95
        
        # Reduce liquidity in MOET pools by 50%
        for pool_key, pool in engine.protocol.liquidity_pools.items():
            if "MOET" in pool_key:
                for asset in pool.reserves:
                    pool.reserves[asset] *= 0.5
                pool.lp_token_supply *= 0.5
    
    def _reduce_pool_liquidity(self, engine: TidalSimulationEngine, reduction: float):
        """Reduce liquidity in all pools"""
        for pool in engine.protocol.liquidity_pools.values():
            for asset in pool.reserves:
                pool.reserves[asset] *= (1 - reduction)
            pool.lp_token_supply *= (1 - reduction)
    
    def _adjust_collateral_factors(self, engine: TidalSimulationEngine, adjustment: float):
        """Adjust collateral factors"""
        for asset, pool in engine.protocol.asset_pools.items():
            pool.collateral_factor *= (1 + adjustment)
            pool.collateral_factor = max(0.1, min(1.0, pool.collateral_factor))
    
    def _adjust_liquidation_thresholds(self, engine: TidalSimulationEngine, adjustment: float):
        """Adjust liquidation thresholds"""
        for asset, pool in engine.protocol.asset_pools.items():
            pool.liquidation_threshold *= (1 + adjustment)
            pool.liquidation_threshold = max(0.1, min(1.0, pool.liquidation_threshold))
    
    def _force_high_utilization(self, engine: TidalSimulationEngine):
        """Force high utilization rates"""
        for asset, pool in engine.protocol.asset_pools.items():
            # Set borrowed amount to 95% of supplied
            target_utilization = 0.95
            pool.total_borrowed = pool.total_supplied * target_utilization
    
    def _trigger_interest_spike(self, engine: TidalSimulationEngine):
        """Trigger interest rate spike by pushing above kink"""
        for asset, pool in engine.protocol.asset_pools.items():
            # Push utilization above kink (80%) to trigger jump rate
            pool.total_borrowed = pool.total_supplied * 0.90  # 90% utilization
    
    def _black_swan_scenario(self, engine: TidalSimulationEngine):
        """Apply multiple simultaneous stress factors"""
        # Multiple price shocks
        shocks = {
            Asset.ETH: -0.45,
            Asset.BTC: -0.40,
            Asset.FLOW: -0.60,
            Asset.USDC: -0.08  # Severe depeg
        }
        engine.state.apply_price_shock(shocks)
        
        # Reduce liquidity
        self._reduce_pool_liquidity(engine, 0.7)  # 70% reduction
        
        # Adjust risk parameters
        self._adjust_collateral_factors(engine, -0.15)  # 15% reduction
    
    def _debt_cap_stress_test(self, engine: TidalSimulationEngine):
        """Test debt cap under extreme conditions"""
        # Maximize debt near cap
        current_debt_cap = engine.protocol.calculate_debt_cap()
        
        # Simulate high borrowing demand
        for asset, pool in engine.protocol.asset_pools.items():
            pool.total_borrowed = pool.total_supplied * 0.85  # High utilization
        
        # Apply moderate price shock to test cap effectiveness
        shocks = {Asset.ETH: -0.20, Asset.BTC: -0.20, Asset.FLOW: -0.30}
        engine.state.apply_price_shock(shocks)
        
    def _setup_high_tide_scenario(self, engine: TidalSimulationEngine):
        """Setup High Tide scenario - this is a placeholder for integration with HighTideSimulationEngine"""
        # Note: This scenario requires special handling in the stress test runner
        # to use HighTideSimulationEngine instead of the regular TidalSimulationEngine
        
        # For now, apply a gradual BTC decline pattern
        # The actual High Tide logic will be handled by HighTideSimulationEngine
        print("Setting up High Tide scenario - BTC decline with active rebalancing")
        
        # Initialize BTC decline (this will be overridden by HighTideSimulationEngine)
        initial_btc_price = engine.state.current_prices.get(Asset.BTC, 100_000.0)
        target_decline = -0.20  # 20% decline as baseline
        
        # This is a placeholder - actual scenario execution happens in HighTideSimulationEngine
    
    def run_all_scenarios(self, base_config) -> Dict[str, dict]:
        """Run all stress test scenarios"""
        print("Starting Tidal Protocol Stress Test Suite")
        print("=" * 50)
        
        for scenario in self.scenarios:
            try:
                # Create fresh engine for each scenario
                from ..simulation.config import SimulationConfig
                config = SimulationConfig()
                engine = TidalSimulationEngine(config)
                
                # Run scenario
                results = scenario.run(engine)
                self.results[scenario.name] = results
                
                print(f"✓ Completed: {scenario.name}")
                
            except Exception as e:
                print(f"✗ Failed: {scenario.name} - {str(e)}")
                self.results[scenario.name] = {"error": str(e)}
        
        print("\nAll stress tests completed!")
        return self.results
    
    def run_scenario(self, scenario_name: str, base_config) -> dict:
        """Run a specific scenario"""
        scenario = next((s for s in self.scenarios if s.name == scenario_name), None)
        
        if not scenario:
            raise ValueError(f"Scenario '{scenario_name}' not found")
        
        from ..simulation.config import SimulationConfig
        config = SimulationConfig()
        engine = TidalSimulationEngine(config)
        
        return scenario.run(engine)
    
    def get_scenario_names(self) -> List[str]:
        """Get list of all scenario names"""
        return [scenario.name for scenario in self.scenarios]