#!/usr/bin/env python3
"""
Configuration schemas for the Tidal Protocol simulation.

This module defines comprehensive Pydantic schemas for all simulation parameters,
following the configuration-driven architecture pattern.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from enum import Enum


class AssetType(str, Enum):
    """Supported asset types"""
    ETH = "ETH"
    BTC = "BTC"
    FLOW = "FLOW"
    USDC = "USDC"
    MOET = "MOET"


class PolicyType(str, Enum):
    """Available agent policy types"""
    HOLD = "hold"
    TRADER = "trader"
    LENDER = "lender"
    TIDAL_LENDER = "tidal_lender"
    STAKER = "staker"
    LP_PROVIDER = "lp_provider"


class MarketType(str, Enum):
    """Available market types"""
    UNISWAP_V2 = "uniswap_v2"
    COMPOUND_LENDING = "compound_lending"
    STAKING = "staking"
    TIDAL_PROTOCOL = "tidal_protocol"


class AssetConfig(BaseModel):
    """Configuration for a single asset"""
    symbol: AssetType
    initial_price: float = Field(gt=0, description="Initial price in USD")
    initial_supply: Optional[float] = Field(None, gt=0, description="Initial circulating supply")
    collateral_factor: float = Field(ge=0, le=1, description="Collateral factor for lending")
    extreme_drop_percentage: float = Field(ge=-1, le=0, description="Extreme price drop scenario")
    volatility_std: float = Field(ge=0, le=1, default=0.1, description="Price volatility standard deviation")


class InterestRateModelConfig(BaseModel):
    """Interest rate model configuration"""
    base_rate_per_block: float = Field(ge=0, default=0, description="Base interest rate per block")
    multiplier_per_block: float = Field(ge=0, description="Linear rate multiplier per block")
    jump_per_block: float = Field(ge=0, description="Jump rate multiplier per block")
    kink: float = Field(ge=0, le=1, description="Utilization threshold for jump rate")
    
    # Block timing
    blocks_per_minute: int = Field(gt=0, default=30, description="Blocks per minute")
    blocks_per_day: int = Field(gt=0, default=43200, description="Blocks per day")
    blocks_per_year: int = Field(gt=0, default=15768000, description="Blocks per year")


class PolicyConfig(BaseModel):
    """Configuration for an agent policy"""
    type: PolicyType
    count: int = Field(gt=0, description="Number of agents with this policy")
    params: Dict[str, Any] = Field(default_factory=dict, description="Policy-specific parameters")
    initial_balance_usd: float = Field(gt=0, default=10000, description="Initial balance per agent in USD")
    
    @validator('params')
    def validate_policy_params(cls, v, values):
        """Validate policy-specific parameters"""
        policy_type = values.get('type')
        
        if policy_type == PolicyType.TRADER:
            # Validate trader parameters
            trading_frequency = v.get('trading_frequency', 0.1)
            if not 0 <= trading_frequency <= 1:
                raise ValueError("trading_frequency must be between 0 and 1")
        
        elif policy_type == PolicyType.LENDER:
            # Validate lender parameters
            min_supply_apy = v.get('min_supply_apy', 0.02)
            if not 0 <= min_supply_apy <= 1:
                raise ValueError("min_supply_apy must be between 0 and 1")
        
        return v


class MarketConfig(BaseModel):
    """Configuration for a market"""
    type: MarketType
    market_id: str = Field(description="Unique market identifier")
    enabled: bool = Field(default=True, description="Whether market is enabled")
    params: Dict[str, Any] = Field(default_factory=dict, description="Market-specific parameters")


class UniswapV2Config(MarketConfig):
    """Uniswap V2 market configuration"""
    type: MarketType = MarketType.UNISWAP_V2
    initial_reserves: Dict[AssetType, float] = Field(description="Initial token reserves")
    fee_rate: float = Field(ge=0, le=1, default=0.003, description="Trading fee rate")
    
    @validator('initial_reserves')
    def validate_reserves(cls, v):
        """Validate initial reserves"""
        if not v:
            raise ValueError("initial_reserves cannot be empty")
        
        for asset, amount in v.items():
            if amount <= 0:
                raise ValueError(f"Reserve for {asset} must be positive")
        
        return v


class CompoundLendingConfig(MarketConfig):
    """Compound lending market configuration"""
    type: MarketType = MarketType.COMPOUND_LENDING
    reserve_factor: float = Field(ge=0, le=1, default=0.15, description="Reserve factor")
    interest_rate_model: InterestRateModelConfig = Field(description="Interest rate model")
    supported_assets: List[AssetType] = Field(description="Assets supported for lending")


class StakingConfig(MarketConfig):
    """Staking market configuration"""
    type: MarketType = MarketType.STAKING
    base_apy: float = Field(ge=0, le=1, description="Base staking APY")
    staking_token: AssetType = Field(default=AssetType.MOET, description="Token to stake")
    reward_token: AssetType = Field(default=AssetType.MOET, description="Reward token")
    min_stake_amount: float = Field(gt=0, default=100, description="Minimum stake amount")


class TidalProtocolMarketConfig(MarketConfig):
    """Tidal Protocol specific market configuration"""
    type: MarketType = MarketType.TIDAL_PROTOCOL
    
    # Asset pool configurations
    initial_liquidity: Dict[AssetType, float] = Field(description="Initial liquidity per asset")
    collateral_factors: Dict[AssetType, float] = Field(description="Collateral factors per asset")
    liquidation_thresholds: Dict[AssetType, float] = Field(description="Liquidation thresholds per asset")
    
    # Interest rate model parameters
    base_rate_per_block: float = Field(ge=0, default=0, description="Base interest rate per block")
    multiplier_per_block: float = Field(ge=0, default=11415525114, description="Linear rate multiplier")
    jump_per_block: float = Field(ge=0, default=253678335870, description="Jump rate multiplier")
    kink: float = Field(ge=0, le=1, default=0.8, description="Kink point utilization")
    
    # Protocol parameters
    reserve_factor: float = Field(ge=0, le=1, default=0.15, description="Reserve factor")
    lp_rewards_factor: float = Field(ge=0, le=1, default=0.50, description="LP rewards factor")
    target_health_factor: float = Field(gt=1, default=1.2, description="Target health factor")
    liquidation_penalty: float = Field(ge=0, le=1, default=0.08, description="Liquidation penalty")
    
    # MOET stablecoin parameters
    moet_initial_supply: float = Field(gt=0, default=1000000, description="Initial MOET supply")
    moet_mint_fee: float = Field(ge=0, le=1, default=0.001, description="MOET mint fee")
    moet_burn_fee: float = Field(ge=0, le=1, default=0.001, description="MOET burn fee")
    moet_stability_bands: Dict[str, float] = Field(
        default_factory=lambda: {"upper": 1.02, "lower": 0.98},
        description="MOET price stability bands"
    )
    
    # Debt cap parameters (Ebisu-style)
    dex_liquidity_allocation: float = Field(ge=0, le=1, default=0.35, description="DEX liquidity allocation")
    extreme_price_drops: Dict[AssetType, float] = Field(
        description="Extreme price drop scenarios per asset"
    )
    
    # Liquidity pool parameters
    liquidity_pools: Dict[str, Dict[AssetType, float]] = Field(
        description="Initial reserves for MOET trading pairs"
    )
    
    @validator('collateral_factors')
    def validate_collateral_factors(cls, v):
        """Validate collateral factors are reasonable"""
        for asset, factor in v.items():
            if not 0 <= factor <= 1:
                raise ValueError(f"Collateral factor for {asset} must be between 0 and 1")
        return v
    
    @validator('extreme_price_drops')
    def validate_price_drops(cls, v):
        """Validate price drops are negative"""
        for asset, drop in v.items():
            if drop >= 0:
                raise ValueError(f"Extreme price drop for {asset} must be negative")
        return v


class SimulationConfig(BaseModel):
    """Main simulation configuration"""
    # Basic parameters
    name: str = Field(description="Simulation name")
    description: Optional[str] = Field(None, description="Simulation description")
    max_days: int = Field(gt=0, default=365, description="Maximum simulation days")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    
    # Assets
    assets: Dict[AssetType, AssetConfig] = Field(description="Asset configurations")
    
    # Agent policies
    agent_policies: List[PolicyConfig] = Field(description="Agent policy configurations")
    
    # Markets
    markets: List[MarketConfig] = Field(description="Market configurations")
    
    # Protocol parameters
    protocol_treasury_initial: float = Field(ge=0, default=0, description="Initial protocol treasury")
    lp_rewards_factor: float = Field(ge=0, le=1, default=0.5, description="LP rewards factor")
    target_health_factor: float = Field(gt=1, default=1.5, description="Target health factor")
    
    # Monte Carlo parameters
    monte_carlo_runs: int = Field(gt=0, default=100, description="Number of Monte Carlo runs")
    price_shock_enabled: bool = Field(default=True, description="Enable price shocks")
    
    @validator('agent_policies')
    def validate_agent_policies(cls, v):
        """Validate agent policies"""
        if not v:
            raise ValueError("At least one agent policy must be specified")
        
        total_agents = sum(policy.count for policy in v)
        if total_agents <= 0:
            raise ValueError("Total agent count must be positive")
        
        return v
    
    @validator('markets')
    def validate_markets(cls, v):
        """Validate market configurations"""
        if not v:
            raise ValueError("At least one market must be specified")
        
        market_ids = [market.market_id for market in v]
        if len(market_ids) != len(set(market_ids)):
            raise ValueError("Market IDs must be unique")
        
        return v


class TidalProtocolConfig(BaseModel):
    """Complete Tidal Protocol configuration"""
    client_id: str = Field(description="Unique client identifier")
    client_name: str = Field(description="Client display name")
    version: str = Field(default="1.0.0", description="Configuration version")
    
    # Core simulation configuration
    simulation: SimulationConfig = Field(description="Simulation configuration")
    
    # Risk parameters
    risk_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "low_health_factor": 1.5,
            "medium_health_factor": 1.2,
            "high_health_factor": 1.1,
            "max_utilization": 0.95
        },
        description="Risk threshold parameters"
    )
    
    # Reporting parameters
    reporting: Dict[str, Any] = Field(
        default_factory=lambda: {
            "summary_percentiles": [0.05, 0.25, 0.75, 0.95],
            "histogram_bins": 50,
            "plot_figure_size": [18, 12]
        },
        description="Reporting and visualization parameters"
    )
    
    def validate_config(self) -> bool:
        """Validate the entire configuration"""
        errors = []
        
        # Check asset consistency
        required_assets = {AssetType.MOET, AssetType.USDC}  # Minimum required assets
        available_assets = set(self.simulation.assets.keys())
        
        if not required_assets.issubset(available_assets):
            missing = required_assets - available_assets
            errors.append(f"Missing required assets: {missing}")
        
        # Check market-asset consistency
        for market in self.simulation.markets:
            if hasattr(market, 'supported_assets'):
                for asset in market.supported_assets:
                    if asset not in available_assets:
                        errors.append(f"Market {market.market_id} references unavailable asset: {asset}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
        
        return True
    
    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy configuration format for backward compatibility"""
        # Extract key parameters for legacy format
        assets = self.simulation.assets
        
        return {
            "INITIAL_PRICES": {
                asset.value: config.initial_price 
                for asset, config in assets.items()
            },
            "COLLATERAL_FACTORS": {
                asset.value: config.collateral_factor 
                for asset, config in assets.items()
            },
            "EXTREME_PRICE_DROPS": {
                asset.value: config.extreme_drop_percentage 
                for asset, config in assets.items()
            },
            "RESERVE_FACTOR": 0.15,  # Default from compound lending
            "LP_REWARDS_FACTOR": self.simulation.lp_rewards_factor,
            "TARGET_HEALTH_FACTOR": self.simulation.target_health_factor,
            "DEFAULT_N_SIMULATIONS": self.simulation.monte_carlo_runs
        }


def create_default_config() -> TidalProtocolConfig:
    """Create a default configuration for testing"""
    assets = {
        AssetType.ETH: AssetConfig(
            symbol=AssetType.ETH,
            initial_price=4400.0,
            collateral_factor=0.75,
            extreme_drop_percentage=-0.15
        ),
        AssetType.BTC: AssetConfig(
            symbol=AssetType.BTC,
            initial_price=118000.0,
            collateral_factor=0.75,
            extreme_drop_percentage=-0.15
        ),
        AssetType.FLOW: AssetConfig(
            symbol=AssetType.FLOW,
            initial_price=0.40,
            collateral_factor=0.50,
            extreme_drop_percentage=-0.35
        ),
        AssetType.USDC: AssetConfig(
            symbol=AssetType.USDC,
            initial_price=1.0,
            collateral_factor=0.90,
            extreme_drop_percentage=-0.15
        ),
        AssetType.MOET: AssetConfig(
            symbol=AssetType.MOET,
            initial_price=1.0,
            initial_supply=1000000.0,
            collateral_factor=0.0,
            extreme_drop_percentage=0.0
        )
    }
    
    agent_policies = [
        PolicyConfig(
            type=PolicyType.TIDAL_LENDER,
            count=80,
            params={
                "min_supply_apy": 0.02, 
                "target_health_factor": 1.2,
                "moet_borrowing_ratio": 0.7,
                "risk_tolerance": 0.6,
                "collateral_diversification": True
            },
            initial_balance_usd=25000
        ),
        PolicyConfig(
            type=PolicyType.LENDER,
            count=20,
            params={
                "min_supply_apy": 0.015,
                "target_health_factor": 1.2,
                "supply_ratio": 0.9
            },
            initial_balance_usd=15000
        )
    ]
    
    markets = [
        TidalProtocolMarketConfig(
            market_id="tidal_protocol",
            initial_liquidity={
                AssetType.ETH: 7_000_000,
                AssetType.BTC: 3_500_000,
                AssetType.FLOW: 2_100_000,
                AssetType.USDC: 1_400_000
            },
            collateral_factors={
                AssetType.ETH: 0.9,
                AssetType.BTC: 0.9,
                AssetType.FLOW: 0.6,
                AssetType.USDC: 1.0
            },
            liquidation_thresholds={
                AssetType.ETH: 0.92,
                AssetType.BTC: 0.92,
                AssetType.FLOW: 0.65,
                AssetType.USDC: 1.0
            },
            extreme_price_drops={
                AssetType.ETH: -0.15,
                AssetType.BTC: -0.15,
                AssetType.FLOW: -0.35,
                AssetType.USDC: -0.15
            },
            liquidity_pools={
                "MOET_USDC": {AssetType.MOET: 1250000, AssetType.USDC: 1250000},
                "MOET_ETH": {AssetType.MOET: 1250000, AssetType.ETH: 284.09},
                "MOET_BTC": {AssetType.MOET: 1250000, AssetType.BTC: 10.59},
                "MOET_FLOW": {AssetType.MOET: 1250000, AssetType.FLOW: 3125000}
            }
        ),
        UniswapV2Config(
            market_id="uniswap_v2",
            initial_reserves={
                AssetType.MOET: 500000.0,
                AssetType.USDC: 500000.0,
                AssetType.ETH: 166.67,
                AssetType.BTC: 11.11
            }
        )
    ]
    
    simulation = SimulationConfig(
        name="Default Tidal Protocol Simulation",
        description="Default configuration for testing",
        assets=assets,
        agent_policies=agent_policies,
        markets=markets
    )
    
    return TidalProtocolConfig(
        client_id="default",
        client_name="Default Configuration",
        simulation=simulation
    )
