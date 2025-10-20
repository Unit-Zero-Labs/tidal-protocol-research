#!/usr/bin/env python3
"""
Parameter definitions and scenarios

Simple parameter dictionaries instead of complex Pydantic schemas.
"""

from typing import Dict, List
from ..core.protocol import Asset


class SimulationConfig:
    """Simple simulation configuration"""
    
    def __init__(self):
        # Agent configuration
        self.num_lenders = 5
        self.num_traders = 3
        self.num_liquidators = 2
        
        # Initial balances
        self.lender_initial_balance = 100_000.0
        self.trader_initial_balance = 50_000.0
        self.liquidator_initial_balance = 200_000.0
        
        # Simulation parameters
        self.simulation_steps = 1000
        self.price_update_frequency = 10  # Update prices every 10 steps
        self.metrics_recording_frequency = 1
        
        # Market parameters
        self.base_volatility = 0.02  # 2% daily volatility
        self.flow_volatility = 0.05  # Higher volatility for FLOW
        self.usdc_volatility = 0.001  # Low volatility for stablecoin
        
        # Enhanced MOET System Configuration
        self.enable_advanced_moet_system = False  # Enable 4-pool structure and Enhanced Redeemer
        self.num_arbitrage_agents = 0  # Number of MOET arbitrage agents for peg maintenance
        self.arbitrage_agent_balance = 100_000.0  # Initial balance per arbitrage agent


class StressTestScenarios:
    """Priority stress test scenarios as specified in refactor requirements"""
    
    # Single Asset Price Shocks
    ETH_FLASH_CRASH = {
        "name": "ETH_Flash_Crash",
        "description": "ETH drops 30% instantly",
        "price_shocks": {Asset.ETH: -0.30},
        "duration": 1
    }
    
    BTC_CRASH = {
        "name": "BTC_Crash", 
        "description": "BTC drops 35% instantly",
        "price_shocks": {Asset.BTC: -0.35},
        "duration": 1
    }
    
    FLOW_CRASH = {
        "name": "FLOW_Crash",
        "description": "FLOW drops 50% instantly", 
        "price_shocks": {Asset.FLOW: -0.50},
        "duration": 1
    }
    
    USDC_DEPEG = {
        "name": "USDC_Depeg",
        "description": "USDC depegs to $0.95",
        "price_shocks": {Asset.USDC: -0.05},
        "duration": 1
    }
    
    # Multi-Asset Crashes (crypto winter scenarios)
    CRYPTO_WINTER = {
        "name": "Crypto_Winter",
        "description": "Multi-asset crash",
        "price_shocks": {
            Asset.ETH: -0.30,
            Asset.BTC: -0.25,
            Asset.FLOW: -0.45
        },
        "duration": 1
    }
    
    CASCADING_LIQUIDATIONS = {
        "name": "Cascading_Liquidations",
        "description": "Cascading liquidation scenario",
        "price_shocks": {
            Asset.ETH: -0.30,
            Asset.BTC: -0.25
        },
        "duration": 1
    }
    
    # Liquidity Crisis Tests
    MOET_DEPEG = {
        "name": "MOET_Depeg",
        "description": "MOET depegs with liquidity drain",
        "price_shocks": {Asset.MOET: -0.05},
        "liquidity_drain": 0.5,
        "duration": 5
    }
    
    POOL_LIQUIDITY_CRISIS = {
        "name": "Pool_Liquidity_Crisis", 
        "description": "80% liquidity reduction in pools",
        "liquidity_reduction": 0.8,
        "duration": 10
    }
    
    # Parameter sensitivity
    COLLATERAL_FACTOR_STRESS = {
        "name": "Collateral_Factor_Stress",
        "description": "Reduce collateral factors by 10%",
        "cf_reduction": 0.1,
        "duration": 1
    }
    
    LIQUIDATION_THRESHOLD_TEST = {
        "name": "Liquidation_Threshold_Test",
        "description": "Reduce liquidation thresholds by 5%",
        "lt_reduction": 0.05,
        "duration": 1
    }
    
    @classmethod
    def get_all_scenarios(cls) -> List[dict]:
        """Get all stress test scenarios"""
        return [
            cls.ETH_FLASH_CRASH,
            cls.BTC_CRASH,
            cls.FLOW_CRASH,
            cls.USDC_DEPEG,
            cls.CRYPTO_WINTER,
            cls.CASCADING_LIQUIDATIONS,
            cls.MOET_DEPEG,
            cls.POOL_LIQUIDITY_CRISIS,
            cls.COLLATERAL_FACTOR_STRESS,
            cls.LIQUIDATION_THRESHOLD_TEST
        ]
    
    @classmethod
    def get_scenario_by_name(cls, name: str) -> dict:
        """Get specific scenario by name"""
        scenarios = cls.get_all_scenarios()
        for scenario in scenarios:
            if scenario["name"] == name:
                return scenario
        return None


class ProtocolParameters:
    """Tidal Protocol parameters"""
    
    # Interest rate model (kinked rates)
    BASE_RATE_PER_BLOCK = 0
    MULTIPLIER_PER_BLOCK = 11415525114
    JUMP_PER_BLOCK = 253678335870
    KINK = 0.80  # 80% utilization
    
    # Collateral factors
    COLLATERAL_FACTORS = {
        Asset.ETH: 0.75,
        Asset.BTC: 0.75,
        Asset.FLOW: 0.50,
        Asset.USDC: 0.90
    }
    
    # Liquidation parameters
    LIQUIDATION_PENALTY = 0.08  # 8% penalty
    CLOSE_FACTOR = 0.5  # Max 50% of debt can be liquidated
    LIQUIDATION_THRESHOLDS = {
        Asset.ETH: 0.80,
        Asset.BTC: 0.80,
        Asset.FLOW: 0.60,
        Asset.USDC: 0.92
    }
    
    # Protocol parameters
    TARGET_HEALTH_FACTOR = 1.5
    DEX_LIQUIDITY_ALLOCATION = 0.35  # 35%
    RESERVE_FACTOR = 0.15  # 15%
    
    # MOET parameters
    MOET_PEG_TARGET = 1.0
    MOET_STABILITY_BANDS = (0.98, 1.02)  # ±2%
    
    # Extreme price drop scenarios for debt cap
    EXTREME_PRICE_DROPS = {
        Asset.ETH: 0.15,
        Asset.BTC: 0.15,
        Asset.FLOW: 0.35,
        Asset.USDC: 0.15
    }