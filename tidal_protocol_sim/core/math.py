#!/usr/bin/env python3
"""
Tidal Protocol Mathematical Functions

Pure mathematical functions for Tidal Protocol calculations including
kinked interest rates, liquidations, and debt cap formulas.
"""

import math
from typing import Tuple, Dict


class TidalMath:
    """Pure mathematical functions for Tidal Protocol calculations"""
    
    # Tidal-specific constants
    BLOCKS_PER_YEAR = 15768000
    SCALE_FACTOR = 1e18
    
    @staticmethod
    def calculate_kinked_interest_rate(
        utilization: float,
        base_rate_per_block: float,
        multiplier_per_block: float,
        jump_per_block: float,
        kink: float
    ) -> float:
        """Calculate interest rate using Tidal's kinked interest rate model"""
        if utilization <= kink:
            rate_per_block = base_rate_per_block + (utilization * multiplier_per_block / TidalMath.SCALE_FACTOR)
        else:
            base_rate = base_rate_per_block + (kink * multiplier_per_block / TidalMath.SCALE_FACTOR)
            jump_rate = (utilization - kink) * jump_per_block / TidalMath.SCALE_FACTOR
            rate_per_block = base_rate + jump_rate
        
        return rate_per_block * TidalMath.BLOCKS_PER_YEAR / TidalMath.SCALE_FACTOR
    
    @staticmethod
    def calculate_supply_rate(borrow_rate: float, utilization: float, reserve_factor: float) -> float:
        """Calculate supply rate from borrow rate"""
        return borrow_rate * utilization * (1 - reserve_factor)
    
    @staticmethod
    def calculate_health_factor(collateral_value: float, debt_value: float, collateral_factor: float) -> float:
        """Calculate health factor for a position"""
        if debt_value <= 0:
            return float('inf')
        
        effective_collateral = collateral_value * collateral_factor
        return effective_collateral / debt_value
    
    @staticmethod
    def calculate_liquidation_amounts(
        debt_amount: float,
        collateral_amount: float,
        collateral_price: float,
        liquidation_penalty: float,
        close_factor: float = 0.5
    ) -> Tuple[float, float, float]:
        """Calculate liquidation amounts"""
        max_repay = debt_amount * close_factor
        
        collateral_value_needed = max_repay * (1 + liquidation_penalty)
        collateral_to_seize = collateral_value_needed / collateral_price
        
        if collateral_to_seize > collateral_amount:
            collateral_to_seize = collateral_amount
            max_repay = (collateral_to_seize * collateral_price) / (1 + liquidation_penalty)
        
        liquidation_bonus = max_repay * liquidation_penalty
        
        return max_repay, collateral_to_seize, liquidation_bonus
    
    @staticmethod
    def calculate_debt_cap_ebisu_style(
        liquidation_capacity: float,
        dex_allocation_factor: float,
        underwater_collateral_percentage: float
    ) -> float:
        """Calculate debt cap using Ebisu-style methodology: A × B × C"""
        return liquidation_capacity * dex_allocation_factor * underwater_collateral_percentage
    
    @staticmethod
    def calculate_constant_product_swap(
        amount_in: float,
        reserve_in: float,
        reserve_out: float,
        fee_rate: float = 0.003
    ) -> Tuple[float, float, float]:
        """Calculate swap using constant product formula"""
        if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
            return 0.0, 0.0, 0.0
        
        # Apply trading fee
        fee_amount = amount_in * fee_rate
        amount_in_after_fee = amount_in - fee_amount
        
        # Constant product formula
        k = reserve_in * reserve_out
        new_reserve_in = reserve_in + amount_in_after_fee
        new_reserve_out = k / new_reserve_in
        amount_out = reserve_out - new_reserve_out
        
        # Calculate slippage
        expected_price = reserve_out / reserve_in
        actual_price = amount_out / amount_in if amount_in > 0 else 0.0
        slippage_percent = ((expected_price - actual_price) / expected_price) * 100.0 if expected_price > 0 else 0.0
        
        return amount_out, fee_amount, slippage_percent
    
    @staticmethod
    def calculate_concentrated_liquidity_swap(
        amount_in: float,
        reserve_in: float,
        reserve_out: float,
        concentration_multiplier: float = 3.0,
        fee_rate: float = 0.003
    ) -> Tuple[float, float, float]:
        """Calculate swap with concentrated liquidity"""
        effective_reserve_in = reserve_in * concentration_multiplier
        effective_reserve_out = reserve_out * concentration_multiplier
        
        return TidalMath.calculate_constant_product_swap(
            amount_in, effective_reserve_in, effective_reserve_out, fee_rate
        )