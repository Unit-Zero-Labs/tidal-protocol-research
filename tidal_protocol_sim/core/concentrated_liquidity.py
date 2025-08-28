#!/usr/bin/env python3
"""
Advanced Uniswap V3 Concentrated Liquidity System

Implements discrete liquidity bins instead of continuous distributions:
- MOET:BTC: 80% at peg ±0.99%, 100k liquidity at ±1% in both directions
- MOET:Yield Token: 95% at peg, remaining distributed in 1 basis point increments
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class LiquidityBin:
    """Represents a discrete liquidity bin at a specific price point"""
    price: float  # Price at this bin
    liquidity: float  # Amount of liquidity in this bin
    bin_index: int  # Sequential bin index
    is_active: bool = True  # Whether this bin has active liquidity


class ConcentratedLiquidityPool:
    """
    Advanced Uniswap V3-style pool with discrete liquidity bins
    
    Implements the specific concentration requirements:
    - MOET:BTC: 80% at peg ±0.99%, 100k at ±1%
    - MOET:Yield Token: 95% at peg, remaining in 1bp increments
    """
    
    def __init__(self, 
                 pool_name: str,
                 total_liquidity: float,
                 btc_price: float = 100_000.0,
                 num_bins: int = 100):
        self.pool_name = pool_name
        self.total_liquidity = total_liquidity
        self.btc_price = btc_price
        self.num_bins = num_bins
        
        # Determine correct peg price based on pool type
        if "MOET:BTC" in pool_name:
            self.peg_price = 0.00001  # 1 BTC = 100,000 MOET
            self.concentration_type = "moet_btc"
        else:
            self.peg_price = 1.0  # 1:1 for yield tokens
            self.concentration_type = "yield_token"
        
        # Initialize liquidity distribution
        self.bins: List[LiquidityBin] = []
        self._initialize_liquidity_distribution()
    
    def _initialize_liquidity_distribution(self):
        """Initialize liquidity distribution across discrete bins"""
        
        if self.concentration_type == "moet_btc":
            self._initialize_moet_btc_distribution()
        else:
            self._initialize_yield_token_distribution()
    
    def _initialize_moet_btc_distribution(self):
        """Initialize MOET:BTC liquidity with 80% at peg ±0.99%, 100k at ±1%"""
        
        # Calculate price range for bins
        # We want to cover from 0.00001 * 0.99 to 0.00001 * 1.01 (roughly)
        min_price = self.peg_price * 0.99  # 0.0000099
        max_price = self.peg_price * 1.01  # 0.0000101
        
        # Create price points for bins
        prices = np.linspace(min_price, max_price, self.num_bins)
        
        # Calculate liquidity distribution
        for i, price in enumerate(prices):
            bin_liquidity = 0.0
            
            # Calculate distance from peg (normalized)
            distance_from_peg = abs(price - self.peg_price) / self.peg_price
            
            if distance_from_peg <= 0.0099:  # Within ±0.99%
                # 80% of total liquidity in this range
                # Use bell curve distribution centered at peg
                peak_factor = math.exp(-(distance_from_peg / 0.005) ** 2)  # Gaussian falloff
                bin_liquidity = (self.total_liquidity * 0.8 * peak_factor) / self.num_bins
                
            elif distance_from_peg <= 0.01:  # At ±1%
                # 100k liquidity at exactly ±1% from peg
                bin_liquidity = 100_000 / 2  # Split between two ±1% bins
                
            else:
                # Minimal liquidity outside concentrated range
                bin_liquidity = (self.total_liquidity * 0.05) / self.num_bins
            
            # Ensure minimum liquidity
            bin_liquidity = max(bin_liquidity, 1000)  # Minimum $1k per bin
            
            self.bins.append(LiquidityBin(
                price=price,
                liquidity=bin_liquidity,
                bin_index=i,
                is_active=bin_liquidity > 1000
            ))
    
    def _initialize_yield_token_distribution(self):
        """Initialize MOET:Yield Token liquidity with 95% at peg, remaining in 1bp increments"""
        
        # Calculate price range for bins (1 basis point = 0.01%)
        # We want to cover from 0.9999 to 1.0001 (roughly)
        min_price = self.peg_price * 0.9999  # 0.9999
        max_price = self.peg_price * 1.0001  # 1.0001
        
        # Create price points for bins
        prices = np.linspace(min_price, max_price, self.num_bins)
        
        # Calculate liquidity distribution
        for i, price in enumerate(prices):
            bin_liquidity = 0.0
            
            # Calculate distance from peg (normalized)
            distance_from_peg = abs(price - self.peg_price) / self.peg_price
            
            if distance_from_peg <= 0.0001:  # Within ±1 basis point
                # 95% of total liquidity in this range
                # Use bell curve distribution centered at peg
                peak_factor = math.exp(-(distance_from_peg / 0.00005) ** 2)  # Very tight Gaussian
                bin_liquidity = (self.total_liquidity * 0.95 * peak_factor) / self.num_bins
                
            else:
                # Remaining 5% distributed in 1bp increments
                # Each bin gets equal share of remaining liquidity
                bin_liquidity = (self.total_liquidity * 0.05) / self.num_bins
            
            # Ensure minimum liquidity
            bin_liquidity = max(bin_liquidity, 1000)  # Minimum $1k per bin
            
            self.bins.append(LiquidityBin(
                price=price,
                liquidity=bin_liquidity,
                bin_index=i,
                is_active=bin_liquidity > 1000
            ))
    
    def get_liquidity_at_price(self, target_price: float) -> float:
        """Get liquidity available at a specific price point"""
        
        # Find the closest bin to the target price
        closest_bin = min(self.bins, key=lambda b: abs(b.price - target_price))
        
        if closest_bin.is_active:
            return closest_bin.liquidity
        else:
            return 0.0
    
    def get_total_active_liquidity(self) -> float:
        """Get total liquidity across all active bins"""
        return sum(bin.liquidity for bin in self.bins if bin.is_active)
    
    def get_liquidity_distribution(self) -> Tuple[List[float], List[float]]:
        """Get price and liquidity arrays for charting"""
        active_bins = [bin for bin in self.bins if bin.is_active]
        prices = [bin.price for bin in active_bins]
        liquidity = [bin.liquidity for bin in active_bins]
        return prices, liquidity
    
    def get_bin_data_for_charts(self) -> List[Dict]:
        """Get bin data formatted for bar chart visualization"""
        return [
            {
                "bin_index": bin.bin_index,
                "price": bin.price,
                "liquidity": bin.liquidity,
                "is_active": bin.is_active,
                "price_label": self._format_price_label(bin.price)
            }
            for bin in self.bins
        ]
    
    def _format_price_label(self, price: float) -> str:
        """Format price for display labels"""
        if self.concentration_type == "moet_btc":
            # For MOET:BTC, show as BTC per MOET
            return f"{price:.6f}"
        else:
            # For yield tokens, show as MOET per yield token
            return f"{price:.4f}"
    
    def simulate_price_impact(self, trade_amount: float, trade_direction: str) -> Dict:
        """
        Simulate the impact of a trade on the liquidity distribution
        
        Args:
            trade_amount: Amount being traded
            trade_direction: "buy" or "sell"
            
        Returns:
            Dict with price impact and liquidity changes
        """
        
        # Find the bin closest to current peg
        peg_bin = min(self.bins, key=lambda b: abs(b.price - self.peg_price))
        
        # Calculate how much liquidity this trade consumes
        liquidity_consumed = min(trade_amount, peg_bin.liquidity)
        
        # Calculate price impact based on liquidity depth
        if liquidity_consumed > 0:
            # Price impact increases as we consume more liquidity
            price_impact = (liquidity_consumed / peg_bin.liquidity) * 0.001  # 0.1% max impact
            
            # Determine new price based on trade direction
            if trade_direction == "buy":
                new_price = self.peg_price * (1 + price_impact)
            else:
                new_price = self.peg_price * (1 - price_impact)
        else:
            new_price = self.peg_price
            price_impact = 0.0
        
        return {
            "original_price": self.peg_price,
            "new_price": new_price,
            "price_impact": price_impact,
            "liquidity_consumed": liquidity_consumed,
            "remaining_liquidity": peg_bin.liquidity - liquidity_consumed,
            "trade_amount": trade_amount
        }
    
    def update_liquidity_distribution(self, price_change: float):
        """
        Update liquidity distribution after a price change
        
        Args:
            price_change: Percentage change in price (e.g., 0.05 for 5% increase)
        """
        
        # Shift the peg price
        self.peg_price *= (1 + price_change)
        
        # Update bin prices proportionally
        for bin in self.bins:
            bin.price *= (1 + price_change)
            
            # Deactivate bins that are too far from the new peg
            distance_from_peg = abs(bin.price - self.peg_price) / self.peg_price
            
            if self.concentration_type == "moet_btc":
                # Deactivate if more than 2% from peg
                bin.is_active = distance_from_peg <= 0.02
            else:
                # Deactivate if more than 0.5% from peg
                bin.is_active = distance_from_peg <= 0.005


def create_moet_btc_concentrated_pool(pool_size_usd: float, btc_price: float = 100_000.0) -> ConcentratedLiquidityPool:
    """Create a MOET:BTC concentrated liquidity pool"""
    return ConcentratedLiquidityPool(
        pool_name="MOET:BTC",
        total_liquidity=pool_size_usd,
        btc_price=btc_price,
        num_bins=100
    )


def create_yield_token_concentrated_pool(pool_size_usd: float, btc_price: float = 100_000.0) -> ConcentratedLiquidityPool:
    """Create a MOET:Yield Token concentrated liquidity pool"""
    return ConcentratedLiquidityPool(
        pool_name="MOET:Yield_Token",
        total_liquidity=pool_size_usd,
        btc_price=btc_price,
        num_bins=100
    )
