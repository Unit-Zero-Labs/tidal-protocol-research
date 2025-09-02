#!/usr/bin/env python3
"""
Advanced Uniswap V3 Concentrated Liquidity System

Implements discrete liquidity bins with accurate slippage calculations:
- MOET:BTC: 80% at peg ±0.99%, 100k liquidity at ±1% in both directions
- MOET:Yield Token: 95% at peg, remaining distributed in 1 basis point increments

Provides both trading functionality and visualization data for bar charts.
"""

import math
import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass


@dataclass
class LiquidityBin:
    """Represents a discrete liquidity bin at a specific price point"""
    price: float  # Price at this bin
    liquidity: float  # Amount of liquidity in this bin
    bin_index: int  # Sequential bin index
    is_active: bool = True  # Whether this bin has active liquidity


@dataclass
class UniswapV3Pool:
    """Advanced Uniswap v3 pool with discrete liquidity bins"""
    pool_name: str  # "MOET:BTC" or "MOET:Yield_Token"
    total_liquidity: float  # Total pool size in USD
    btc_price: float = 100_000.0  # BTC price in USD
    num_bins: int = 100  # Number of discrete liquidity bins
    fee_tier: float = 0.003  # 0.3% fee tier
    concentration: float = 0.80  # Concentration level (0.80 = 80% at peg)
    
    # Legacy fields for backward compatibility
    token0_reserve: Optional[float] = None  # MOET reserve (calculated from bins)
    token1_reserve: Optional[float] = None  # BTC reserve (calculated from bins)
    sqrt_price_x96: Optional[float] = None
    liquidity: Optional[float] = None
    tick_current: Optional[int] = None
    
    def __post_init__(self):
        """Initialize discrete liquidity bins and derived values"""
        # Determine pool type and peg price
        if "MOET:BTC" in self.pool_name:
            self.peg_price = 0.00001  # 1 BTC = 100,000 MOET
            self.concentration_type = "moet_btc"
        else:
            self.peg_price = 1.0  # 1:1 for yield tokens
            self.concentration_type = "yield_token"
        
        # Initialize discrete liquidity bins
        self.bins: List[LiquidityBin] = []
        self._initialize_liquidity_distribution()
        
        # Sort bins by price for proper visualization
        self.bins.sort(key=lambda b: b.price)
        
        # Re-assign bin indices after sorting to maintain order
        for i, bin in enumerate(self.bins):
            bin.bin_index = i
        
        # Calculate legacy fields from bins for backward compatibility
        self._update_legacy_fields()
    
    def _initialize_liquidity_distribution(self):
        """Initialize liquidity distribution across discrete bins"""
        if self.concentration_type == "moet_btc":
            self._initialize_moet_btc_distribution()
        else:
            self._initialize_yield_token_distribution()
    
    def _initialize_moet_btc_distribution(self):
        """Initialize MOET:BTC liquidity distribution with proper Uniswap V3 concentrated liquidity:
        - 80% of total liquidity in the SINGLE bin at the exact peg price (0.00001 BTC per MOET)
        - $100k liquidity distributed in 100 basis point increments on each side of the peg
        - Remaining liquidity in outer bins
        """
        
        peg_price = self.peg_price  # 0.00001 BTC per MOET
        
        # 1. Create the SINGLE peg bin with 80% of total liquidity
        peg_liquidity = self.total_liquidity * self.concentration  # 80% of total
        self.bins.append(LiquidityBin(
            price=peg_price,
            liquidity=peg_liquidity,
            bin_index=0,
            is_active=True
        ))
        
        # 2. Create bins in 100 basis point increments on each side of the peg
        # 100 basis points = 1% = 0.01
        bp_increment = 0.01  # 1%
        bins_per_side = 10   # 10 bins on each side (10% range each side)
        
        # Calculate liquidity per bin for the $100k total
        total_side_liquidity = 100_000  # $100k as specified
        liquidity_per_side_bin = total_side_liquidity / (bins_per_side * 2)  # Split between both sides
        
        # Lower side bins (below peg)
        for i in range(1, bins_per_side + 1):
            price = peg_price * (1 - bp_increment * i)  # Each bin is 1% lower
            self.bins.append(LiquidityBin(
                price=price,
                liquidity=liquidity_per_side_bin,
                bin_index=i,
                is_active=True
            ))
        
        # Upper side bins (above peg)
        for i in range(1, bins_per_side + 1):
            price = peg_price * (1 + bp_increment * i)  # Each bin is 1% higher
            self.bins.append(LiquidityBin(
                price=price,
                liquidity=liquidity_per_side_bin,
                bin_index=bins_per_side + i,
                is_active=True
            ))
        
        # 3. Create outer bins with remaining liquidity
        remaining_liquidity = self.total_liquidity - peg_liquidity - total_side_liquidity
        outer_bins_count = self.num_bins - (1 + bins_per_side * 2)  # Remaining bins
        
        if outer_bins_count > 0 and remaining_liquidity > 0:
            outer_liquidity_per_bin = remaining_liquidity / outer_bins_count
            
            # Lower outer range (beyond 10% from peg)
            lower_outer_count = outer_bins_count // 2
            for i in range(lower_outer_count):
                price = peg_price * (1 - bp_increment * (bins_per_side + 1 + i))
                self.bins.append(LiquidityBin(
                    price=price,
                    liquidity=outer_liquidity_per_bin,
                    bin_index=1 + bins_per_side * 2 + i,
                    is_active=outer_liquidity_per_bin > 1000
                ))
            
            # Upper outer range (beyond 10% from peg)
            upper_outer_count = outer_bins_count - lower_outer_count
            for i in range(upper_outer_count):
                price = peg_price * (1 + bp_increment * (bins_per_side + 1 + i))
                self.bins.append(LiquidityBin(
                    price=price,
                    liquidity=outer_liquidity_per_bin,
                    bin_index=1 + bins_per_side * 2 + lower_outer_count + i,
                    is_active=outer_liquidity_per_bin > 1000
                ))
    
    def _initialize_yield_token_distribution(self):
        """Initialize MOET:Yield Token liquidity distribution with single peg bin model:
        - 95% of total liquidity in the SINGLE bin at the exact peg price (1.0)
        - Remaining 5% distributed in 1 basis point increments on each side of the peg
        """
        
        peg_price = self.peg_price  # 1.0 MOET per Yield Token
        
        # 1. Create the SINGLE peg bin with 95% of total liquidity
        peg_liquidity = self.total_liquidity * self.concentration  # 95% of total
        self.bins.append(LiquidityBin(
            price=peg_price,
            liquidity=peg_liquidity,
            bin_index=0,
            is_active=True
        ))
        
        # 2. Create bins in 1 basis point increments on each side of the peg
        # 1 basis point = 0.01% = 0.0001
        bp_increment = 0.0001  # 1 basis point
        bins_per_side = 50     # 50 bins on each side (5% range each side)
        
        # Calculate liquidity per bin for the remaining 5%
        remaining_liquidity = self.total_liquidity * (1 - self.concentration)  # 5%
        liquidity_per_side_bin = remaining_liquidity / (bins_per_side * 2)  # Split between both sides
        
        # Lower side bins (below peg)
        for i in range(1, bins_per_side + 1):
            price = peg_price * (1 - bp_increment * i)  # Each bin is 1bp lower
            self.bins.append(LiquidityBin(
                price=price,
                liquidity=liquidity_per_side_bin,
                bin_index=i,
                is_active=True
            ))
        
        # Upper side bins (above peg)
        for i in range(1, bins_per_side + 1):
            price = peg_price * (1 + bp_increment * i)  # Each bin is 1bp higher
            self.bins.append(LiquidityBin(
                price=price,
                liquidity=liquidity_per_side_bin,
                bin_index=bins_per_side + i,
                is_active=True
            ))
        
        # 3. Create outer bins with any remaining liquidity
        outer_bins_count = self.num_bins - (1 + bins_per_side * 2)  # Remaining bins
        
        if outer_bins_count > 0:
            # Any remaining liquidity goes to outer bins
            outer_liquidity_per_bin = 1000  # Small amount for outer bins
            
            # Lower outer range (beyond 5% from peg)
            lower_outer_count = outer_bins_count // 2
            for i in range(lower_outer_count):
                price = peg_price * (1 - bp_increment * (bins_per_side + 1 + i))
                self.bins.append(LiquidityBin(
                    price=price,
                    liquidity=outer_liquidity_per_bin,
                    bin_index=1 + bins_per_side * 2 + i,
                    is_active=True
                ))
            
            # Upper outer range (beyond 5% from peg)
            upper_outer_count = outer_bins_count - lower_outer_count
            for i in range(upper_outer_count):
                price = peg_price * (1 + bp_increment * (bins_per_side + 1 + i))
                self.bins.append(LiquidityBin(
                    price=price,
                    liquidity=outer_liquidity_per_bin,
                    bin_index=1 + bins_per_side * 2 + lower_outer_count + i,
                    is_active=True
                ))
    
    def _update_legacy_fields(self):
        """Update legacy fields for backward compatibility"""
        # Calculate total reserves from bins
        total_active_liquidity = sum(bin.liquidity for bin in self.bins if bin.is_active)
        
        # Split reserves 50/50 for compatibility
        self.token0_reserve = total_active_liquidity / 2  # MOET reserve in USD
        self.token1_reserve = total_active_liquidity / 2  # BTC reserve in USD
        
        # Calculate price and derived values
        current_price = self.get_price()
        self.sqrt_price_x96 = math.sqrt(current_price) * (2 ** 96)
        self.liquidity = math.sqrt(self.token0_reserve * self.token1_reserve)
    
    def get_price(self) -> float:
        """Get current effective price from active bins"""
        # Calculate weighted average price from active bins
        if not self.bins or not any(b.is_active for b in self.bins):
            return self.peg_price
            
        total_liquidity = 0
        weighted_price = 0
        
        for bin in self.bins:
            if bin.is_active and bin.liquidity > 0:
                total_liquidity += bin.liquidity
                weighted_price += bin.price * bin.liquidity
        
        if total_liquidity > 0:
            return weighted_price / total_liquidity
        else:
            return self.peg_price


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
        """Simulate the impact of a trade on the liquidity distribution"""
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
        """Update liquidity distribution after a price change"""
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
        
        # Update legacy fields
        self._update_legacy_fields()


class UniswapV3SlippageCalculator:
    """
    Advanced slippage calculator using discrete liquidity bins
    """
    
    def __init__(self, pool_state: UniswapV3Pool):
        self.pool = pool_state
        
    def calculate_swap_slippage(
        self, 
        amount_in: float, 
        token_in: str,  # "MOET" or "BTC"
        concentrated_range: float = 0.2  # Legacy parameter, now uses actual bins
    ) -> Dict[str, float]:
        """
        Calculate slippage for a swap using discrete liquidity bins
        
        Args:
            amount_in: Amount of input token to swap
            token_in: Which token is being swapped in ("MOET" or "BTC")
            concentrated_range: Legacy parameter for backward compatibility
            
        Returns:
            Dict with swap details including slippage
        """
        
        if token_in == "MOET":
            return self._calculate_moet_to_btc_swap_with_bins(amount_in)
        elif token_in == "BTC":
            return self._calculate_btc_to_moet_swap_with_bins(amount_in)
        else:
            raise ValueError("token_in must be 'MOET' or 'BTC'")
    
    def _calculate_moet_to_btc_swap_with_bins(self, moet_amount: float) -> Dict[str, float]:
        """Calculate MOET -> BTC swap using discrete liquidity bins"""
        
        # Current price (BTC per MOET) - should be 0.00001
        current_price = self.pool.get_price()
        
        # moet_amount is in USD terms, apply fees
        moet_after_fees = moet_amount * (1 - self.pool.fee_tier)
        trading_fees = moet_amount * self.pool.fee_tier
        
        # Calculate swap using discrete bins
        remaining_moet_value = moet_after_fees  # USD value of MOET to swap
        total_btc_out = 0.0
        bins_consumed = []
        
        # Sort bins by price (closest to peg first)
        sorted_bins = sorted([b for b in self.pool.bins if b.is_active], 
                           key=lambda b: abs(b.price - self.pool.peg_price))
        
        for bin in sorted_bins:
            if remaining_moet_value <= 0:
                break
                
            # Each bin has liquidity split 50/50 between MOET and BTC sides
            bin_moet_liquidity_usd = bin.liquidity / 2
            moet_to_consume = min(remaining_moet_value, bin_moet_liquidity_usd)
            
            # Convert to actual BTC using bin price
            # bin.price is BTC per MOET, so MOET_USD * (BTC/MOET) / BTC_PRICE = BTC tokens
            btc_tokens_from_bin = (moet_to_consume / 1.0) * bin.price  # MOET tokens * BTC per MOET
            
            total_btc_out += btc_tokens_from_bin
            remaining_moet_value -= moet_to_consume
            
            bins_consumed.append({
                "bin_index": bin.bin_index,
                "price": bin.price,
                "moet_consumed": moet_to_consume,
                "btc_output": btc_tokens_from_bin
            })
        
        # Calculate expected output without slippage
        moet_tokens_in = moet_amount / 1.0  # USD to MOET tokens (1 MOET = $1)
        expected_btc_out = moet_tokens_in * current_price
        
        # Calculate slippage
        slippage_amount = max(0, expected_btc_out - total_btc_out)
        slippage_percentage = (slippage_amount / expected_btc_out) * 100 if expected_btc_out > 0 else 0
        
        # Calculate effective price (BTC per MOET)
        effective_price = total_btc_out / moet_tokens_in if moet_tokens_in > 0 else current_price
        price_impact = ((current_price - effective_price) / current_price) * 100 if current_price > 0 else 0
        
        return {
            "amount_in": moet_amount,
            "token_in": "MOET",
            "amount_out": total_btc_out,
            "token_out": "BTC",
            "expected_amount_out": expected_btc_out,
            "slippage_amount": slippage_amount,
            "slippage_percentage": slippage_percentage,
            "price_impact_percentage": abs(price_impact),
            "trading_fees": trading_fees,
            "current_price": current_price,
            "new_price": effective_price,
            "effective_liquidity": self.pool.get_total_active_liquidity(),
            "bins_consumed": bins_consumed
        }
    
    def _calculate_btc_to_moet_swap_with_bins(self, btc_amount: float) -> Dict[str, float]:
        """Calculate BTC -> MOET swap using discrete liquidity bins"""
        
        # Current price (BTC per MOET)
        current_price_btc_per_moet = self.pool.get_price()
        # For BTC -> MOET swap, we need MOET per BTC (inverse)
        current_price_moet_per_btc = 1.0 / current_price_btc_per_moet if current_price_btc_per_moet > 0 else 100000.0
        
        # Amount of BTC going in (after fees) - btc_amount is in USD since we're tracking USD values
        btc_after_fees = btc_amount * (1 - self.pool.fee_tier)
        trading_fees = btc_amount * self.pool.fee_tier
        
        # Calculate swap using discrete bins
        remaining_btc_usd = btc_after_fees  # This is in USD
        total_moet_out = 0.0
        bins_consumed = []
        
        # Sort bins by price (closest to peg first for better execution)
        sorted_bins = sorted([b for b in self.pool.bins if b.is_active], 
                           key=lambda b: abs(b.price - self.pool.peg_price))
        
        for bin in sorted_bins:
            if remaining_btc_usd <= 0:
                break
                
            # Calculate how much BTC this bin can handle
            # bin.liquidity is total USD liquidity, half is BTC side
            bin_capacity_btc_usd = bin.liquidity / 2  # BTC side capacity in USD
            btc_to_consume_usd = min(remaining_btc_usd, bin_capacity_btc_usd)
            
            # Convert USD to BTC tokens (BTC is at $100,000)
            btc_tokens = btc_to_consume_usd / self.pool.btc_price  # USD to BTC tokens
            
            # Calculate MOET output from this bin
            # bin.price is BTC per MOET, so BTC / (BTC/MOET) = MOET tokens
            moet_tokens = btc_tokens / bin.price  # BTC / (BTC/MOET) = MOET tokens
            
            # Convert MOET tokens to USD (since 1 MOET = $1)
            moet_from_bin_usd = moet_tokens * 1.0  # MOET tokens to USD
            
            total_moet_out += moet_from_bin_usd
            remaining_btc_usd -= btc_to_consume_usd
            
            bins_consumed.append({
                "bin_index": bin.bin_index,
                "price": bin.price,
                "btc_consumed": btc_to_consume_usd,  # Store in USD for consistency
                "moet_output": moet_from_bin_usd
            })
        
        # Calculate slippage metrics
        # btc_amount is in USD, convert to BTC tokens then to MOET
        btc_tokens_in = btc_amount / self.pool.btc_price  # USD to BTC tokens
        moet_tokens_out = btc_tokens_in / current_price_btc_per_moet  # BTC tokens / (BTC/MOET) = MOET tokens
        expected_moet_out = moet_tokens_out * 1.0  # MOET tokens to USD (1 MOET = $1)
        slippage_amount = max(0, expected_moet_out - total_moet_out)
        slippage_percentage = (slippage_amount / expected_moet_out) * 100 if expected_moet_out > 0 else 0
        
        # Calculate effective price and price impact
        # effective_price should be MOET per BTC in USD terms
        effective_price_moet_per_btc = total_moet_out / btc_amount if btc_amount > 0 else current_price_moet_per_btc
        price_impact = ((current_price_moet_per_btc - effective_price_moet_per_btc) / current_price_moet_per_btc) * 100 if current_price_moet_per_btc > 0 else 0
        
        # Effective liquidity is the sum of active bin liquidity
        effective_liquidity = self.pool.get_total_active_liquidity()
        
        return {
            "amount_in": btc_amount,
            "token_in": "BTC", 
            "amount_out": total_moet_out,
            "token_out": "MOET",
            "expected_amount_out": expected_moet_out,
            "slippage_amount": slippage_amount,
            "slippage_percentage": slippage_percentage,
            "price_impact_percentage": abs(price_impact),
            "trading_fees": trading_fees,
            "current_price": current_price_moet_per_btc,
            "new_price": effective_price_moet_per_btc,
            "effective_liquidity": effective_liquidity,
            "bins_consumed": bins_consumed
        }
    

    
    def update_pool_state(self, swap_result: Dict[str, float]):
        """Update pool state after a swap by consuming liquidity from bins"""
        
        # Update bin liquidity based on consumption
        if "bins_consumed" in swap_result:
            for bin_consumption in swap_result["bins_consumed"]:
                bin_index = bin_consumption["bin_index"]
                if bin_index < len(self.pool.bins):
                    bin = self.pool.bins[bin_index]
                    
                    # Reduce bin liquidity based on actual consumption
                    # Since we consumed from one side, reduce total liquidity by 2x the consumed amount
                    if swap_result["token_in"] == "MOET":
                        liquidity_consumed = bin_consumption["moet_consumed"] * 2
                    else:
                        liquidity_consumed = bin_consumption.get("btc_consumed", 0) * 2
                    
                    # Update bin liquidity
                    bin.liquidity = max(0, bin.liquidity - liquidity_consumed)
                    
                    # Deactivate bin if liquidity too low
                    if bin.liquidity < 1000:  # Minimum $1k threshold
                        bin.is_active = False
        
        # Update pool's current price based on the swap impact
        if "new_price" in swap_result and swap_result["new_price"] > 0:
            # Small price adjustment based on trade impact
            price_impact = swap_result.get("price_impact_percentage", 0) / 100.0
            if swap_result["token_in"] == "MOET":
                # MOET -> BTC swap should increase MOET price slightly (less MOET in pool)
                self.pool.peg_price *= (1 + price_impact * 0.1)  # Small adjustment
            else:
                # BTC -> MOET swap should decrease MOET price slightly (more MOET in pool)
                self.pool.peg_price *= (1 - price_impact * 0.1)  # Small adjustment
        
        # Update legacy fields for backward compatibility
        self.pool._update_legacy_fields()


def create_moet_btc_pool(pool_size_usd: float, btc_price: float = 100_000.0, concentration: float = 0.80) -> UniswapV3Pool:
    """
    Create a MOET:BTC Uniswap v3 pool with discrete liquidity bins
    
    Args:
        pool_size_usd: Total pool size in USD
        btc_price: Current BTC price in USD (default: $100,000)
        concentration: Liquidity concentration level (0.80 = 80% at peg)
        
    Returns:
        UniswapV3Pool instance with discrete bins
    """
    
    return UniswapV3Pool(
        pool_name="MOET:BTC",
        total_liquidity=pool_size_usd,
        btc_price=btc_price,
        num_bins=100,
        fee_tier=0.003,  # 0.3% fee tier
        concentration=concentration
    )


def create_yield_token_pool(pool_size_usd: float, btc_price: float = 100_000.0, concentration: float = 0.95) -> UniswapV3Pool:
    """
    Create a MOET:Yield Token Uniswap v3 pool with discrete liquidity bins
    
    Args:
        pool_size_usd: Total pool size in USD
        btc_price: Current BTC price in USD (for consistency)
        concentration: Liquidity concentration level (0.95 = 95% at peg)
        
    Returns:
        UniswapV3Pool instance with discrete bins
    """
    
    return UniswapV3Pool(
        pool_name="MOET:Yield_Token",
        total_liquidity=pool_size_usd,
        btc_price=btc_price,
        num_bins=100,
        fee_tier=0.003,  # 0.3% fee tier
        concentration=concentration
    )


# Legacy factory function for backward compatibility
def create_moet_btc_concentrated_pool(pool_size_usd: float, btc_price: float = 100_000.0) -> UniswapV3Pool:
    """Legacy function - redirects to create_moet_btc_pool"""
    return create_moet_btc_pool(pool_size_usd, btc_price)


def create_yield_token_concentrated_pool(pool_size_usd: float, btc_price: float = 100_000.0) -> UniswapV3Pool:
    """Legacy function - redirects to create_yield_token_pool"""
    return create_yield_token_pool(pool_size_usd, btc_price)


def calculate_rebalancing_cost_with_slippage(
    moet_amount: float,
    pool_size_usd: float = 500_000,
    concentrated_range: float = 0.2,
    btc_price: float = 100_000.0
) -> Dict[str, float]:
    """
    Calculate the total cost of rebalancing including Uniswap v3 slippage
    
    Args:
        moet_amount: Amount of MOET to swap for debt repayment
        pool_size_usd: Total MOET:BTC pool size in USD
        concentrated_range: Liquidity concentration range (0.2 = 20%)
        btc_price: Current BTC price in USD (default: $100,000)
        
    Returns:
        Dict with cost breakdown including slippage
    """
    
    # Create pool state with correct MOET:BTC ratio
    pool = create_moet_btc_pool(pool_size_usd, btc_price)
    calculator = UniswapV3SlippageCalculator(pool)
    
    # Calculate swap (MOET -> BTC to repay debt)
    swap_result = calculator.calculate_swap_slippage(moet_amount, "MOET", concentrated_range)
    
    # Total cost includes slippage and fees
    total_cost = swap_result["slippage_amount"] + swap_result["trading_fees"]
    
    return {
        "moet_amount_swapped": moet_amount,
        "btc_received": swap_result["amount_out"],
        "expected_btc_without_slippage": swap_result["expected_amount_out"],
        "slippage_cost": swap_result["slippage_amount"],
        "trading_fees": swap_result["trading_fees"],
        "total_swap_cost": total_cost,
        "slippage_percentage": swap_result["slippage_percentage"],
        "price_impact_percentage": swap_result["price_impact_percentage"],
        "effective_liquidity": swap_result["effective_liquidity"]
    }


def calculate_liquidation_cost_with_slippage(
    collateral_btc_amount: float,
    btc_price: float,
    liquidation_percentage: float = 0.5,
    liquidation_bonus: float = 0.05,
    pool_size_usd: float = 500_000,
    concentrated_range: float = 0.2
) -> Dict[str, float]:
    """
    Calculate the total cost of AAVE-style liquidation including Uniswap v3 slippage
    
    Args:
        collateral_btc_amount: Amount of BTC collateral to liquidate
        btc_price: Current BTC price in USD
        liquidation_percentage: Percentage of collateral to liquidate (0.5 = 50%)
        liquidation_bonus: Liquidation bonus rate (0.05 = 5%)
        pool_size_usd: Total MOET:BTC pool size in USD
        concentrated_range: Liquidity concentration range
        
    Returns:
        Dict with liquidation cost breakdown including slippage
    """
    
    # Amount of BTC to liquidate
    btc_to_liquidate = collateral_btc_amount * liquidation_percentage
    btc_value_to_liquidate = btc_to_liquidate * btc_price
    
    # Create pool state
    pool = create_moet_btc_pool(pool_size_usd, btc_price)
    calculator = UniswapV3SlippageCalculator(pool)
    
    # Calculate swap (BTC -> MOET for debt repayment)
    swap_result = calculator.calculate_swap_slippage(btc_value_to_liquidate, "BTC", concentrated_range)
    
    # Liquidation bonus cost
    bonus_cost = btc_value_to_liquidate * liquidation_bonus
    
    # Total liquidation cost includes slippage, fees, and bonus
    total_cost = swap_result["slippage_amount"] + swap_result["trading_fees"] + bonus_cost
    
    return {
        "btc_liquidated": btc_to_liquidate,
        "btc_value_liquidated": btc_value_to_liquidate,
        "moet_received": swap_result["amount_out"],
        "expected_moet_without_slippage": swap_result["expected_amount_out"],
        "slippage_cost": swap_result["slippage_amount"],
        "trading_fees": swap_result["trading_fees"],
        "liquidation_bonus_cost": bonus_cost,
        "total_liquidation_cost": total_cost,
        "slippage_percentage": swap_result["slippage_percentage"],
        "price_impact_percentage": swap_result["price_impact_percentage"],
        "effective_liquidity": swap_result["effective_liquidity"]
    }
