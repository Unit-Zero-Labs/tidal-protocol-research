#!/usr/bin/env python3
"""
Tidal Protocol Implementation

This module contains the core TidalProtocol class focused on lending mechanics,
following the refactoring guidelines to simplify and eliminate generic framework complexity.
"""

from typing import Dict, Optional, Tuple
from enum import Enum
import math


class Asset(Enum):
    """Supported assets"""
    ETH = "ETH"
    BTC = "BTC" 
    FLOW = "FLOW"
    USDC = "USDC"
    MOET = "MOET"


class AssetPool:
    """Individual asset pool within Tidal Protocol"""
    
    def __init__(self, asset: Asset, total_supplied: float = 0.0):
        self.asset = asset
        self.total_supplied = total_supplied
        self.total_borrowed = 0.0
        self.reserve_balance = 0.0
        
        # Interest accrual tracking
        self.supply_index = 1.0
        self.borrow_index = 1.0
        self.last_update_block = 0
        
        # Set asset-specific parameters
        self._set_asset_parameters()
        
        # Kinked interest rate model parameters (Tidal-specific)
        self.base_rate_per_block = 0
        self.multiplier_per_block = 11415525114
        self.jump_per_block = 253678335870
        self.kink = 0.80
    
    def _set_asset_parameters(self):
        """Set asset-specific collateral factors and thresholds"""
        if self.asset == Asset.ETH:
            self.collateral_factor = 0.80
            self.liquidation_threshold = 0.80
        elif self.asset == Asset.BTC:
            self.collateral_factor = 0.80
            self.liquidation_threshold = 0.85
        elif self.asset == Asset.FLOW:
            self.collateral_factor = 0.50
            self.liquidation_threshold = 0.60
        elif self.asset == Asset.USDC:
            self.collateral_factor = 0.90
            self.liquidation_threshold = 0.92
        
        self.liquidation_penalty = 0.05  # 5% liquidation penalty
        self.reserve_factor = 0.15  # 15% of interest to protocol
    
    @property
    def utilization_rate(self) -> float:
        """Calculate current utilization rate"""
        if self.total_supplied <= 0:
            return 0.0
        return self.total_borrowed / self.total_supplied
    
    def calculate_borrow_rate(self) -> float:
        """Calculate borrow rate using Tidal's kinked interest model"""
        utilization = self.utilization_rate
        
        if utilization <= self.kink:
            rate = self.base_rate_per_block + (utilization * self.multiplier_per_block / 1e18)
        else:
            base_rate = self.base_rate_per_block + (self.kink * self.multiplier_per_block / 1e18)
            jump_rate = (utilization - self.kink) * self.jump_per_block / 1e18
            rate = base_rate + jump_rate
        
        # Convert to annual rate (Tidal-specific block timing)
        blocks_per_year = 15768000
        return rate * blocks_per_year / 1e18
    
    def calculate_supply_rate(self) -> float:
        """Calculate supply rate based on borrow rate and utilization"""
        borrow_rate = self.calculate_borrow_rate()
        return borrow_rate * self.utilization_rate * (1 - self.reserve_factor)




class TidalProtocol:
    """Streamlined Tidal Protocol implementation focused on lending mechanics"""
    
    def __init__(self):
        # Essential state only
        self.asset_pools = self._initialize_asset_pools()
        
        # Import MoetStablecoin here to avoid circular imports
        from .moet import MoetStablecoin
        self.moet_system = MoetStablecoin()
        
        # Uniswap V3 pools are handled by simulation engines, not the core protocol
        self.protocol_treasury = 0.0
        
        # Protocol parameters
        self.target_health_factor = 1.2
        self.liquidation_close_factor = 0.5  # Max 50% of debt can be liquidated
        self.dex_liquidity_allocation = 0.35  # 35% for debt cap calculations
        self.current_block = 0
        
        # Extreme scenario parameters for debt cap
        self.extreme_price_drops = {
            Asset.ETH: 0.15,
            Asset.BTC: 0.15,
            Asset.FLOW: 0.35,
            Asset.USDC: 0.15
        }
    
    def _initialize_asset_pools(self) -> Dict[Asset, AssetPool]:
        """Initialize asset pools with Tidal-specific parameters"""
        return {
            Asset.ETH: AssetPool(Asset.ETH, 7_000_000),   # $7M initial
            Asset.BTC: AssetPool(Asset.BTC, 3_500_000),   # $3.5M initial
            Asset.FLOW: AssetPool(Asset.FLOW, 2_100_000), # $2.1M initial
            Asset.USDC: AssetPool(Asset.USDC, 1_400_000), # $1.4M initial
        }
    
    
    def supply(self, agent_id: str, asset: Asset, amount: float) -> bool:
        """Supply asset to protocol"""
        if asset not in self.asset_pools:
            return False
        
        pool = self.asset_pools[asset]
        self._accrue_interest(asset)
        
        # Update pool state
        pool.total_supplied += amount
        return True
    
    def borrow(self, agent_id: str, amount_moet: float) -> bool:
        """Borrow MOET against collateral"""
        # Check debt cap
        current_debt_cap = self.calculate_debt_cap()
        total_moet_debt = sum(pool.total_borrowed for pool in self.asset_pools.values())
        
        if total_moet_debt + amount_moet > current_debt_cap:
            return False
        
        # Mint MOET (fee-less as per refactor spec)
        self.moet_system.mint(amount_moet)
        return True
    
    def repay(self, agent_id: str, amount_moet: float) -> bool:
        """Repay MOET debt"""
        self.moet_system.burn(amount_moet)
        return True
    
    
    def calculate_health_factor(self, agent_id: str) -> float:
        """Calculate agent's health factor"""
        # Simplified - would need agent state tracking in full implementation
        return 1.5  # Placeholder
    
    def calculate_debt_cap(self, liquidation_capacity: float = 0.0) -> float:
        """Calculate debt cap using Ebisu-style methodology: A × B × C"""
        # A: Liquidation capacity (provided by simulation engine)
        total_liquidation_capacity = liquidation_capacity
        
        # B: DEX liquidity allocation factor
        dex_allocation = self.dex_liquidity_allocation
        
        # C: Weighted underwater collateral percentage
        total_collateral_value = sum(pool.total_supplied for pool in self.asset_pools.values())
        weighted_underwater = 0.0
        
        if total_collateral_value > 0:
            for asset, pool in self.asset_pools.items():
                weight = pool.total_supplied / total_collateral_value
                drop_percentage = self.extreme_price_drops.get(asset, 0.15)
                weighted_underwater += weight * drop_percentage
        
        debt_cap = total_liquidation_capacity * dex_allocation * weighted_underwater
        return max(debt_cap, 100000.0)  # Minimum $100k debt cap
    
    def accrue_interest(self) -> None:
        """Accrue interest for all asset pools"""
        for asset in self.asset_pools:
            self._accrue_interest(asset)
    
    def _accrue_interest(self, asset: Asset):
        """Accrue interest for an asset pool"""
        if asset not in self.asset_pools:
            return
        
        pool = self.asset_pools[asset]
        blocks_elapsed = self.current_block - pool.last_update_block
        
        if blocks_elapsed <= 0:
            return
        
        # Calculate interest
        borrow_rate_per_block = pool.calculate_borrow_rate() / 15768000
        interest_factor = (1 + borrow_rate_per_block) ** blocks_elapsed
        
        # Update indices
        pool.borrow_index *= interest_factor
        
        if pool.total_supplied > 0:
            total_interest = pool.total_borrowed * (interest_factor - 1)
            reserve_amount = total_interest * pool.reserve_factor
            supply_interest = total_interest - reserve_amount
            
            pool.supply_index *= (1 + supply_interest / pool.total_supplied)
            self.protocol_treasury += reserve_amount
        
        pool.last_update_block = self.current_block
    


