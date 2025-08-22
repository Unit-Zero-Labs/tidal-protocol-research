#!/usr/bin/env python3
"""
Pure mathematical calculations for Uniswap V2-style AMM.

This module contains only mathematical functions with no state mutations,
following the clean separation of concerns principle.
"""

import math
from typing import Tuple, Optional


def calculate_swap_output(
    amount_in: float,
    reserve_in: float,
    reserve_out: float,
    fee_rate: float = 0.003
) -> Tuple[float, float]:
    """
    Calculate output amount for a constant product AMM swap
    
    Uses the formula: amount_out = (amount_in_with_fee * reserve_out) / (reserve_in + amount_in_with_fee)
    Where: amount_in_with_fee = amount_in * (1 - fee_rate)
    
    Args:
        amount_in: Amount of input token
        reserve_in: Reserve of input token
        reserve_out: Reserve of output token
        fee_rate: Trading fee rate (default 0.3%)
        
    Returns:
        Tuple of (amount_out, fee_amount)
    """
    if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
        return 0.0, 0.0
    
    # Calculate fee
    fee_amount = amount_in * fee_rate
    amount_in_with_fee = amount_in - fee_amount
    
    # Constant product formula
    amount_out = (amount_in_with_fee * reserve_out) / (reserve_in + amount_in_with_fee)
    
    # Ensure we don't drain the pool
    amount_out = min(amount_out, reserve_out * 0.99)  # Leave 1% minimum liquidity
    
    return amount_out, fee_amount


def calculate_price_impact(
    amount_in: float,
    reserve_in: float,
    reserve_out: float,
    fee_rate: float = 0.003
) -> float:
    """
    Calculate price impact of a swap
    
    Args:
        amount_in: Amount of input token
        reserve_in: Reserve of input token
        reserve_out: Reserve of output token
        fee_rate: Trading fee rate
        
    Returns:
        Price impact as a percentage (0-1)
    """
    if reserve_in <= 0 or amount_in <= 0:
        return 0.0
    
    # Current price (before swap)
    current_price = reserve_out / reserve_in
    
    # Calculate new reserves after swap
    amount_out, _ = calculate_swap_output(amount_in, reserve_in, reserve_out, fee_rate)
    new_reserve_in = reserve_in + amount_in
    new_reserve_out = reserve_out - amount_out
    
    if new_reserve_in <= 0 or new_reserve_out <= 0:
        return 1.0  # Maximum impact
    
    # New price (after swap)
    new_price = new_reserve_out / new_reserve_in
    
    # Calculate price impact
    if current_price > 0:
        price_impact = abs(new_price - current_price) / current_price
    else:
        price_impact = 0.0
    
    return min(price_impact, 1.0)  # Cap at 100%


def calculate_slippage(
    amount_in: float,
    expected_amount_out: float,
    actual_amount_out: float
) -> float:
    """
    Calculate slippage between expected and actual output
    
    Args:
        amount_in: Input amount
        expected_amount_out: Expected output without price impact
        actual_amount_out: Actual output with price impact
        
    Returns:
        Slippage percentage (0-1)
    """
    if expected_amount_out <= 0:
        return 0.0
    
    slippage = (expected_amount_out - actual_amount_out) / expected_amount_out
    return max(0.0, slippage)  # Slippage is always positive


def calculate_liquidity_tokens(
    amount_a: float,
    amount_b: float,
    reserve_a: float,
    reserve_b: float,
    total_supply: float
) -> float:
    """
    Calculate LP tokens to mint when adding liquidity
    
    Args:
        amount_a: Amount of token A to add
        amount_b: Amount of token B to add
        reserve_a: Current reserve of token A
        reserve_b: Current reserve of token B
        total_supply: Current total LP token supply
        
    Returns:
        LP tokens to mint
    """
    if amount_a <= 0 or amount_b <= 0:
        return 0.0
    
    if reserve_a <= 0 or reserve_b <= 0:
        # First liquidity provision - use geometric mean
        return math.sqrt(amount_a * amount_b)
    
    # Existing pool - maintain proportionality
    # LP tokens = min(amount_a/reserve_a, amount_b/reserve_b) * total_supply
    ratio_a = amount_a / reserve_a
    ratio_b = amount_b / reserve_b
    
    # Use the smaller ratio to maintain pool balance
    mint_ratio = min(ratio_a, ratio_b)
    
    return mint_ratio * total_supply


def calculate_liquidity_removal(
    lp_tokens: float,
    total_supply: float,
    reserve_a: float,
    reserve_b: float
) -> Tuple[float, float]:
    """
    Calculate token amounts when removing liquidity
    
    Args:
        lp_tokens: LP tokens to burn
        total_supply: Total LP token supply
        reserve_a: Current reserve of token A
        reserve_b: Current reserve of token B
        
    Returns:
        Tuple of (amount_a, amount_b) to return
    """
    if lp_tokens <= 0 or total_supply <= 0:
        return 0.0, 0.0
    
    # Calculate share of pool
    share = lp_tokens / total_supply
    
    # Calculate amounts to return
    amount_a = reserve_a * share
    amount_b = reserve_b * share
    
    return amount_a, amount_b


def calculate_optimal_swap_amount(
    balance_a: float,
    balance_b: float,
    reserve_a: float,
    reserve_b: float,
    fee_rate: float = 0.003
) -> Optional[float]:
    """
    Calculate optimal amount to swap to add liquidity with single-sided deposit
    
    This function calculates how much of token A to swap for token B
    to achieve the optimal ratio for liquidity provision.
    
    Args:
        balance_a: Available balance of token A
        balance_b: Available balance of token B
        reserve_a: Pool reserve of token A
        reserve_b: Pool reserve of token B
        fee_rate: Trading fee rate
        
    Returns:
        Optimal amount of token A to swap, or None if not possible
    """
    if balance_a <= 0 or reserve_a <= 0 or reserve_b <= 0:
        return None
    
    # Target ratio for optimal liquidity provision
    target_ratio = reserve_b / reserve_a
    current_ratio = balance_b / balance_a if balance_a > 0 else 0
    
    # If we already have the right ratio, no swap needed
    if abs(current_ratio - target_ratio) < 0.01:  # 1% tolerance
        return 0.0
    
    # If we have too much token B relative to A, we need to swap A for B
    if current_ratio < target_ratio:
        # Calculate optimal swap amount using quadratic formula
        # This is derived from the AMM math and optimal liquidity conditions
        
        fee_factor = 1 - fee_rate
        
        # Quadratic coefficients for optimal swap calculation
        a_coeff = 1
        b_coeff = 2 * reserve_a * fee_factor
        c_coeff = reserve_a * reserve_a * fee_factor * fee_factor - balance_a * reserve_a * fee_factor * target_ratio
        
        # Solve quadratic equation
        discriminant = b_coeff * b_coeff - 4 * a_coeff * c_coeff
        
        if discriminant < 0:
            return None  # No real solution
        
        # Take the positive root
        swap_amount = (-b_coeff + math.sqrt(discriminant)) / (2 * a_coeff)
        
        # Ensure swap amount is within bounds
        if 0 < swap_amount < balance_a:
            return swap_amount
    
    return None


def calculate_impermanent_loss(
    price_ratio_initial: float,
    price_ratio_current: float
) -> float:
    """
    Calculate impermanent loss for a liquidity position
    
    Args:
        price_ratio_initial: Initial price ratio (price_b / price_a)
        price_ratio_current: Current price ratio (price_b / price_a)
        
    Returns:
        Impermanent loss as a percentage (0-1)
    """
    if price_ratio_initial <= 0 or price_ratio_current <= 0:
        return 0.0
    
    # Calculate price change ratio
    price_change = price_ratio_current / price_ratio_initial
    
    # Impermanent loss formula
    # IL = 2 * sqrt(price_change) / (1 + price_change) - 1
    numerator = 2 * math.sqrt(price_change)
    denominator = 1 + price_change
    
    lp_value_ratio = numerator / denominator
    
    # Impermanent loss is the difference from holding
    impermanent_loss = 1 - lp_value_ratio
    
    return max(0.0, impermanent_loss)


def calculate_apy_from_fees(
    daily_fees: float,
    total_liquidity: float,
    days: int = 365
) -> float:
    """
    Calculate APY from trading fees
    
    Args:
        daily_fees: Average daily fees collected
        total_liquidity: Total liquidity in the pool
        days: Days in year (default 365)
        
    Returns:
        APY as a percentage (0-1)
    """
    if total_liquidity <= 0 or daily_fees < 0:
        return 0.0
    
    # Annual fees
    annual_fees = daily_fees * days
    
    # APY = annual_fees / total_liquidity
    apy = annual_fees / total_liquidity
    
    return apy
