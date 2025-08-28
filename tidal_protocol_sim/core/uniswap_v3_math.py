#!/usr/bin/env python3
"""
Uniswap v3 Concentrated Liquidity Math

Implements slippage calculations for swaps through MOET:BTC Uniswap v3 pool
with concentrated liquidity mechanics.
"""

import math
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class UniswapV3Pool:
    """Represents a Uniswap v3 pool state"""
    token0_reserve: float  # MOET reserve (in USD value)
    token1_reserve: float  # BTC reserve (in USD value)
    fee_tier: float = 0.003  # 0.3% fee tier
    sqrt_price_x96: Optional[float] = None
    liquidity: Optional[float] = None
    tick_current: Optional[int] = None
    btc_price: float = 100_000.0  # BTC price in USD for correct price calculation
    
    def __post_init__(self):
        """Initialize derived values"""
        if self.sqrt_price_x96 is None:
            # Calculate sqrt price from reserves (price = BTC per MOET)
            # Both reserves are stored in USD
            if self.token0_reserve > 0:  # MOET reserve > 0
                btc_tokens = self.token1_reserve / self.btc_price  # Convert USD to BTC tokens
                moet_tokens = self.token0_reserve  # MOET tokens (1:1 with USD)
                price = btc_tokens / moet_tokens  # BTC per MOET
            else:
                price = 0.00001  # Default: 1 BTC = 100,000 MOET
            self.sqrt_price_x96 = math.sqrt(price) * (2 ** 96)
        
        if self.liquidity is None:
            # Calculate liquidity from reserves
            # L = sqrt(x * y) for uniform distribution
            self.liquidity = math.sqrt(self.token0_reserve * self.token1_reserve)
    
    def get_price(self) -> float:
        """Get current price as BTC per MOET"""
        # Both reserves are stored in USD
        # MOET is 1:1 with USD, so $250k = 250,000 MOET
        # BTC reserve is $250k, which equals 2.5 BTC at $100k/BTC
        # Price = BTC per MOET = (BTC reserve USD / BTC price) / (MOET reserve USD)
        # Price = ($250k / $100k) / $250k = 2.5 / 250,000 = 0.00001 BTC per MOET
        
        if self.token0_reserve > 0:  # MOET reserve > 0
            btc_tokens = self.token1_reserve / self.btc_price  # Convert USD to BTC tokens
            moet_tokens = self.token0_reserve  # MOET tokens (1:1 with USD)
            return btc_tokens / moet_tokens  # BTC per MOET
        else:
            return 0.00001  # Default: 1 BTC = 100,000 MOET


class UniswapV3SlippageCalculator:
    """
    Calculates slippage for swaps through Uniswap v3 concentrated liquidity pools
    """
    
    def __init__(self, pool_state: UniswapV3Pool):
        self.pool = pool_state
        
    def calculate_swap_slippage(
        self, 
        amount_in: float, 
        token_in: str,  # "MOET" or "BTC"
        concentrated_range: float = 0.2  # 20% price range concentration
    ) -> Dict[str, float]:
        """
        Calculate slippage for a swap through the Uniswap v3 pool
        
        Args:
            amount_in: Amount of input token to swap
            token_in: Which token is being swapped in ("MOET" or "BTC")
            concentrated_range: Range around current price where liquidity is concentrated
            
        Returns:
            Dict with swap details including slippage
        """
        
        if token_in == "MOET":
            return self._calculate_moet_to_btc_swap(amount_in, concentrated_range)
        elif token_in == "BTC":
            return self._calculate_btc_to_moet_swap(amount_in, concentrated_range)
        else:
            raise ValueError("token_in must be 'MOET' or 'BTC'")
    
    def _calculate_moet_to_btc_swap(self, moet_amount: float, concentrated_range: float) -> Dict[str, float]:
        """Calculate MOET -> BTC swap with concentrated liquidity"""
        
        # Current price (BTC per MOET) - use the pool's correct price calculation
        current_price = self.pool.get_price()
        
        # Amount of MOET going in (after fees)
        moet_after_fees = moet_amount * (1 - self.pool.fee_tier)
        
        # Simple constant product formula: x * y = k
        # When we add moet_after_fees to x, we need to solve for new y
        k = self.pool.token0_reserve * self.pool.token1_reserve
        new_moet_reserve = self.pool.token0_reserve + moet_after_fees
        new_btc_reserve = k / new_moet_reserve
        
        # BTC amount out
        btc_amount_out = self.pool.token1_reserve - new_btc_reserve
        
        # Ensure we don't exceed available liquidity and amount is positive
        max_btc_out = self.pool.token1_reserve * 0.95  # Leave 5% buffer
        btc_amount_out = max(0, min(btc_amount_out, max_btc_out))
        
        # Calculate actual new price using the pool's method
        # Temporarily update reserves to calculate new price
        old_moet = self.pool.token0_reserve
        old_btc = self.pool.token1_reserve
        self.pool.token0_reserve = new_moet_reserve
        self.pool.token1_reserve = new_btc_reserve
        new_price = self.pool.get_price()
        # Restore original reserves (will be updated properly by update_pool_state)
        self.pool.token0_reserve = old_moet
        self.pool.token1_reserve = old_btc
        
        # Calculate slippage metrics
        expected_btc_out = moet_amount * current_price  # Without slippage
        slippage_amount = max(0, expected_btc_out - btc_amount_out)
        slippage_percentage = (slippage_amount / expected_btc_out) * 100 if expected_btc_out > 0 else 0
        
        # Price impact
        price_impact = ((current_price - new_price) / current_price) * 100 if current_price > 0 else 0
        
        # Trading fees
        trading_fees = moet_amount * self.pool.fee_tier
        
        # Apply concentration factor to slippage (concentrated liquidity increases slippage)
        # For 95% concentration (0.05 range), this creates realistic stablecoin-like behavior
        concentration_multiplier = 1.0 + (1.0 / concentrated_range - 1.0) * 0.05  # 5% impact per concentration level
        adjusted_slippage = slippage_amount * concentration_multiplier
        adjusted_slippage_percentage = (adjusted_slippage / expected_btc_out) * 100 if expected_btc_out > 0 else 0
        
        return {
            "amount_in": moet_amount,
            "token_in": "MOET",
            "amount_out": btc_amount_out,
            "token_out": "BTC",
            "expected_amount_out": expected_btc_out,
            "slippage_amount": adjusted_slippage,
            "slippage_percentage": adjusted_slippage_percentage,
            "price_impact_percentage": abs(price_impact),
            "trading_fees": trading_fees,
            "current_price": current_price,
            "new_price": new_price,
            "effective_liquidity": self.pool.liquidity
        }
    
    def _calculate_btc_to_moet_swap(self, btc_amount: float, concentrated_range: float) -> Dict[str, float]:
        """Calculate BTC -> MOET swap with concentrated liquidity"""
        
        # Current price (BTC per MOET) - use the pool's correct price calculation
        current_price_btc_per_moet = self.pool.get_price()
        # For BTC -> MOET swap, we need MOET per BTC (inverse)
        current_price_moet_per_btc = 1.0 / current_price_btc_per_moet if current_price_btc_per_moet > 0 else 100000.0
        
        # Amount of BTC going in (after fees)
        btc_after_fees = btc_amount * (1 - self.pool.fee_tier)
        
        # Simple constant product formula: x * y = k
        # When we add btc_after_fees to y, we need to solve for new x
        k = self.pool.token0_reserve * self.pool.token1_reserve
        new_btc_reserve = self.pool.token1_reserve + btc_after_fees
        new_moet_reserve = k / new_btc_reserve
        
        # MOET amount out
        moet_amount_out = self.pool.token0_reserve - new_moet_reserve
        
        # Ensure we don't exceed available liquidity and amount is positive
        max_moet_out = self.pool.token0_reserve * 0.95  # Leave 5% buffer
        moet_amount_out = max(0, min(moet_amount_out, max_moet_out))
        
        # Calculate actual new price using the pool's method
        # Temporarily update reserves to calculate new price
        old_moet = self.pool.token0_reserve
        old_btc = self.pool.token1_reserve
        self.pool.token0_reserve = new_moet_reserve
        self.pool.token1_reserve = new_btc_reserve
        new_price_btc_per_moet = self.pool.get_price()
        new_price_moet_per_btc = 1.0 / new_price_btc_per_moet if new_price_btc_per_moet > 0 else 100000.0
        # Restore original reserves (will be updated properly by update_pool_state)
        self.pool.token0_reserve = old_moet
        self.pool.token1_reserve = old_btc
        
        # Calculate slippage metrics
        expected_moet_out = btc_amount * current_price_moet_per_btc  # Without slippage
        slippage_amount = max(0, expected_moet_out - moet_amount_out)
        slippage_percentage = (slippage_amount / expected_moet_out) * 100 if expected_moet_out > 0 else 0
        
        # Price impact
        price_impact = ((current_price_moet_per_btc - new_price_moet_per_btc) / current_price_moet_per_btc) * 100 if current_price_moet_per_btc > 0 else 0
        
        # Trading fees
        trading_fees = btc_amount * self.pool.fee_tier
        
        # Apply concentration factor to slippage (concentrated liquidity increases slippage)
        # For 95% concentration (0.05 range), this creates realistic stablecoin-like behavior
        concentration_multiplier = 1.0 + (1.0 / concentrated_range - 1.0) * 0.05  # 5% impact per concentration level
        adjusted_slippage = slippage_amount * concentration_multiplier
        adjusted_slippage_percentage = (adjusted_slippage / expected_moet_out) * 100 if expected_moet_out > 0 else 0
        
        return {
            "amount_in": btc_amount,
            "token_in": "BTC", 
            "amount_out": moet_amount_out,
            "token_out": "MOET",
            "expected_amount_out": expected_moet_out,
            "slippage_amount": adjusted_slippage,
            "slippage_percentage": adjusted_slippage_percentage,
            "price_impact_percentage": abs(price_impact),
            "trading_fees": trading_fees,
            "current_price": current_price_moet_per_btc,
            "new_price": new_price_moet_per_btc,
            "effective_liquidity": self.pool.liquidity
        }
    
    def _calculate_effective_liquidity(self, concentrated_range: float) -> float:
        """
        Calculate effective liquidity based on concentration range
        
        In Uniswap v3, liquidity is concentrated around the current price.
        This increases effective liquidity but also increases slippage for large trades.
        """
        
        # Base liquidity from reserves
        base_liquidity = math.sqrt(self.pool.token0_reserve * self.pool.token1_reserve)
        
        # Concentration factor - smaller range = higher concentration
        # Concentration increases effective liquidity but also price impact
        concentration_factor = 1.0 / concentrated_range
        
        # Effective liquidity is higher due to concentration but with diminishing returns
        effective_liquidity = base_liquidity * (1 + math.log(1 + concentration_factor))
        
        return effective_liquidity
    
    def update_pool_state(self, swap_result: Dict[str, float]):
        """Update pool state after a swap"""
        
        if swap_result["token_in"] == "MOET":
            # MOET in, BTC out
            self.pool.token0_reserve += swap_result["amount_in"]
            self.pool.token1_reserve -= swap_result["amount_out"]
        else:
            # BTC in, MOET out
            self.pool.token1_reserve += swap_result["amount_in"]
            self.pool.token0_reserve -= swap_result["amount_out"]
        
        # Recalculate derived values using the pool's correct price calculation
        if self.pool.token0_reserve > 0 and self.pool.token1_reserve > 0:
            price = self.pool.get_price()
            self.pool.sqrt_price_x96 = math.sqrt(price) * (2 ** 96)
            self.pool.liquidity = math.sqrt(self.pool.token0_reserve * self.pool.token1_reserve)


def create_moet_btc_pool(pool_size_usd: float, btc_price: float = 100_000.0) -> UniswapV3Pool:
    """
    Create a MOET:BTC Uniswap v3 pool with specified total size
    
    Args:
        pool_size_usd: Total pool size in USD (will be split 50/50)
        btc_price: Current BTC price in USD (default: $100,000)
        
    Returns:
        UniswapV3Pool instance
    """
    
    # Split pool 50/50 between MOET and BTC (in USD value)
    usd_per_side = pool_size_usd / 2
    
    # Store both reserves in USD for consistency
    # MOET is 1:1 with USD, so $250k = 250,000 MOET
    # BTC reserve is $250k (equivalent to 2.5 BTC at $100k/BTC)
    moet_reserve_usd = usd_per_side
    btc_reserve_usd = usd_per_side
    
    return UniswapV3Pool(
        token0_reserve=moet_reserve_usd,  # MOET reserve in USD
        token1_reserve=btc_reserve_usd,   # BTC reserve in USD
        fee_tier=0.003,  # 0.3% fee tier
        btc_price=btc_price  # Pass BTC price for correct price calculation
    )


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
