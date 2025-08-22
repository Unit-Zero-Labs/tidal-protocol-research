#!/usr/bin/env python3
"""
MOET Pair Liquidity Pools with Concentrated Liquidity

This module handles MOET trading pairs with UniswapV3-style concentrated liquidity.
"""

from typing import Dict, Tuple
from .math import TidalMath
from .protocol import Asset


class ConcentratedLiquidityPool:
    """MOET pair pool with concentrated liquidity"""
    
    def __init__(self, asset: Asset, moet_reserve: float, asset_reserve: float):
        self.asset = asset
        self.reserves = {Asset.MOET: moet_reserve, asset: asset_reserve}
        self.fee_rate = 0.003  # 0.3% trading fee
        self.lp_token_supply = moet_reserve
        
        # Concentration parameters
        self.concentration_multiplier = 3.0  # 3x liquidity concentration
        self.price_range_factor = 0.1  # 10% price range
    
    def get_liquidation_capacity(self, max_slippage: float = 0.04) -> float:
        """Calculate liquidation capacity before hitting max slippage"""
        moet_reserve = self.reserves.get(Asset.MOET, 0.0)
        if moet_reserve <= 0:
            return 0.0
        
        # Using constant product formula: slippage = amount_in / (reserve_in + amount_in)
        max_liquidation_amount = (max_slippage * moet_reserve) / (1 - max_slippage)
        
        # Apply safety factor
        safety_factor = 0.8
        return max_liquidation_amount * safety_factor
    
    def calculate_swap_output(self, amount_in: float, asset_in: Asset, asset_out: Asset) -> Tuple[float, float, float]:
        """Calculate swap output with concentrated liquidity"""
        reserve_in = self.reserves.get(asset_in, 0.0)
        reserve_out = self.reserves.get(asset_out, 0.0)
        
        if reserve_in <= 0 or reserve_out <= 0:
            return 0.0, 0.0, 0.0
        
        return TidalMath.calculate_concentrated_liquidity_swap(
            amount_in, reserve_in, reserve_out, self.concentration_multiplier, self.fee_rate
        )
    
    def update_reserves(self, asset_in: Asset, amount_in: float, asset_out: Asset, amount_out: float):
        """Update reserves after a swap"""
        if asset_in in self.reserves and asset_out in self.reserves:
            self.reserves[asset_in] += amount_in
            self.reserves[asset_out] -= amount_out
    
    def get_price(self, base_asset: Asset = None) -> float:
        """Get current price (MOET per unit of asset)"""
        if base_asset is None:
            base_asset = self.asset
        
        moet_reserve = self.reserves.get(Asset.MOET, 0.0)
        asset_reserve = self.reserves.get(base_asset, 0.0)
        
        if asset_reserve <= 0:
            return 0.0
        
        return moet_reserve / asset_reserve


class LiquidityPoolManager:
    """Manages all MOET trading pairs"""
    
    def __init__(self):
        self.pools = self._initialize_pools()
    
    def _initialize_pools(self) -> Dict[str, ConcentratedLiquidityPool]:
        """Initialize all MOET trading pairs with $2.5M total liquidity each"""
        return {
            "MOET_USDC": ConcentratedLiquidityPool(Asset.USDC, 1_250_000, 1_250_000),
            "MOET_ETH": ConcentratedLiquidityPool(Asset.ETH, 1_250_000, 284.09),    # ~$1.25M at $4400/ETH
            "MOET_BTC": ConcentratedLiquidityPool(Asset.BTC, 1_250_000, 10.59),     # ~$1.25M at $118k/BTC
            "MOET_FLOW": ConcentratedLiquidityPool(Asset.FLOW, 1_250_000, 3_125_000), # ~$1.25M at $0.40/FLOW
        }
    
    def get_pool(self, asset: Asset) -> ConcentratedLiquidityPool:
        """Get pool for MOET/Asset pair"""
        pool_key = f"MOET_{asset.value}"
        return self.pools.get(pool_key)
    
    def get_total_liquidation_capacity(self) -> float:
        """Get total liquidation capacity across all pools"""
        return sum(pool.get_liquidation_capacity() for pool in self.pools.values())
    
    def execute_swap(self, pool_key: str, amount_in: float, asset_in: Asset, asset_out: Asset) -> Tuple[float, float, float]:
        """Execute swap through specified pool"""
        if pool_key not in self.pools:
            return 0.0, 0.0, 0.0
        
        pool = self.pools[pool_key]
        amount_out, fee_amount, slippage = pool.calculate_swap_output(amount_in, asset_in, asset_out)
        
        # Update pool reserves
        pool.update_reserves(asset_in, amount_in, asset_out, amount_out)
        
        return amount_out, fee_amount, slippage
    
    def get_moet_price(self, asset: Asset) -> float:
        """Get MOET price in terms of specified asset"""
        pool = self.get_pool(asset)
        if pool is None:
            return 1.0  # Default to $1
        
        return pool.get_price(asset)