#!/usr/bin/env python3
"""
Streamlined Tidal Protocol Implementation

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
    
    def __init__(self, enable_advanced_moet: bool = False):
        # Essential state only
        self.asset_pools = self._initialize_asset_pools()
        
        # Import MoetStablecoin here to avoid circular imports
        from .moet import MoetStablecoin
        self.moet_system = MoetStablecoin(enable_advanced_system=enable_advanced_moet)
        self.enable_advanced_moet = enable_advanced_moet
        
        # Uniswap V3 pools are handled by simulation engines, not the core protocol
        self.protocol_treasury = 0.0
        
        # Protocol parameters
        self.target_health_factor = 1.2
        self.liquidation_close_factor = 0.5  # Max 50% of debt can be liquidated
        self.dex_liquidity_allocation = 0.65  # 65% for debt cap calculations
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
            Asset.ETH: AssetPool(Asset.ETH, 0.0),   
            Asset.BTC: AssetPool(Asset.BTC, 0.0),   
            Asset.FLOW: AssetPool(Asset.FLOW, 0.0), 
            Asset.USDC: AssetPool(Asset.USDC, 0.0), 
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
        """Calculate debt cap using methodology: A × B × C"""
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
    
    def process_moet_system_update(self, current_minute: int, market_conditions: dict = None) -> dict:
        """Process MOET system updates including bond auctions and interest rate calculations"""
        if self.enable_advanced_moet:
            return self.moet_system.process_minute_update(current_minute, market_conditions)
        return {'advanced_system_enabled': False}
    
    def initialize_moet_reserves(self, total_agent_debt: float):
        """Initialize MOET reserves based on total agent debt"""
        if self.enable_advanced_moet:
            self.moet_system.initialize_reserves(total_agent_debt)
    
    def get_moet_borrow_rate(self) -> float:
        """Get current MOET borrowing rate"""
        if self.enable_advanced_moet:
            return self.moet_system.get_current_interest_rate()
        else:
            # Fallback to simple rate calculation for legacy system
            # Use BTC pool as proxy since that's what agents borrow against
            btc_pool = self.asset_pools.get(Asset.BTC)
            if btc_pool:
                return btc_pool.calculate_borrow_rate()
            return 0.05  # 5% default rate
    
    def get_moet_system_state(self) -> dict:
        """Get comprehensive MOET system state"""
        return self.moet_system.get_state()
    
    def mint_moet_from_deposit(self, usdc_amount: float, usdf_amount: float) -> dict:
        """Mint MOET from USDC/USDF deposit through enhanced Redeemer"""
        if self.enable_advanced_moet:
            from .moet import DepositResult
            result = self.moet_system.mint_from_deposit(usdc_amount, usdf_amount)
            return {
                'success': True,
                'moet_minted': result.moet_minted,
                'total_fee': result.total_fee,
                'base_fee': result.base_fee,
                'imbalance_fee': result.imbalance_fee,
                'fee_percentage': result.fee_percentage,
                'is_balanced': result.is_balanced,
                'post_deviation': result.post_deviation
            }
        else:
            # Legacy 1:1 minting
            total_deposit = usdc_amount + usdf_amount
            self.moet_system.mint(total_deposit)
            return {
                'success': True,
                'moet_minted': total_deposit,
                'total_fee': 0.0,
                'base_fee': 0.0,
                'imbalance_fee': 0.0,
                'fee_percentage': 0.0,
                'is_balanced': True,
                'post_deviation': 0.0
            }
    
    def redeem_moet_for_assets(self, moet_amount: float, desired_asset: str = "proportional") -> dict:
        """Redeem MOET for underlying assets through enhanced Redeemer"""
        if self.enable_advanced_moet:
            from .moet import RedemptionResult
            result = self.moet_system.redeem_for_assets(moet_amount, desired_asset)
            return {
                'success': result.usdc_received > 0 or result.usdf_received > 0,
                'usdc_received': result.usdc_received,
                'usdf_received': result.usdf_received,
                'total_fee': result.total_fee,
                'base_fee': result.base_fee,
                'imbalance_fee': result.imbalance_fee,
                'fee_percentage': result.fee_percentage,
                'is_proportional': result.is_proportional,
                'post_deviation': result.post_deviation
            }
        else:
            # Legacy proportional redemption
            result = self.moet_system.burn(moet_amount)
            return {
                'success': result > 0,
                'usdc_received': result * 0.5,
                'usdf_received': result * 0.5,
                'total_fee': 0.0,
                'base_fee': 0.0,
                'imbalance_fee': 0.0,
                'fee_percentage': 0.0,
                'is_proportional': True,
                'post_deviation': 0.0
            }
    
    def estimate_deposit_fee(self, usdc_amount: float, usdf_amount: float) -> dict:
        """Estimate fees for a potential MOET deposit"""
        if self.enable_advanced_moet and hasattr(self.moet_system, 'redeemer') and self.moet_system.redeemer:
            return self.moet_system.redeemer.estimate_deposit_fee(usdc_amount, usdf_amount)
        else:
            return {
                'total_fee': 0.0,
                'base_fee': 0.0,
                'imbalance_fee': 0.0,
                'fee_percentage': 0.0,
                'is_balanced': True,
                'post_deviation': 0.0
            }
    
    def estimate_redemption_fee(self, moet_amount: float, desired_asset: str = "proportional") -> dict:
        """Estimate fees for a potential MOET redemption"""
        if self.enable_advanced_moet and hasattr(self.moet_system, 'redeemer') and self.moet_system.redeemer:
            return self.moet_system.redeemer.estimate_redemption_fee(moet_amount, desired_asset)
        else:
            return {
                'total_fee': 0.0,
                'base_fee': 0.0,
                'imbalance_fee': 0.0,
                'fee_percentage': 0.0,
                'is_proportional': True,
                'post_deviation': 0.0
            }
    
    def get_redeemer_pool_weights(self) -> dict:
        """Get current USDC/USDF pool composition from Redeemer"""
        if self.enable_advanced_moet and hasattr(self.moet_system, 'redeemer') and self.moet_system.redeemer:
            return self.moet_system.redeemer.get_current_pool_weights()
        else:
            return {
                'usdc_ratio': 0.50,
                'usdf_ratio': 0.50,
                'ideal_usdc_ratio': 0.50,
                'ideal_usdf_ratio': 0.50,
                'weight_deviation': 0.0
            }
    
    def get_optimal_deposit_ratio(self, total_amount: float) -> dict:
        """Get optimal deposit composition to minimize fees"""
        if self.enable_advanced_moet and hasattr(self.moet_system, 'redeemer') and self.moet_system.redeemer:
            return self.moet_system.redeemer.get_optimal_deposit_ratio(total_amount)
        else:
            return {
                'usdc_amount': total_amount * 0.50,
                'usdf_amount': total_amount * 0.50,
                'estimated_fee': 0.0,
                'fee_percentage': 0.0
            }
    
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
    


