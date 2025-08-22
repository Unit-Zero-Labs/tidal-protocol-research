#!/usr/bin/env python3
"""
Tidal Protocol Mathematical Calculations

This module contains all the pure mathematical functions specific to Tidal Protocol,
including kinked interest rate models, MOET stability calculations, and Ebisu-style
debt cap formulas.
"""

import math
from typing import Tuple, Dict, Any, Optional


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
        """
        Calculate interest rate using Tidal's kinked interest rate model
        
        Args:
            utilization: Current utilization rate (0-1)
            base_rate_per_block: Base rate per block
            multiplier_per_block: Linear multiplier per block
            jump_per_block: Jump multiplier per block (above kink)
            kink: Kink point utilization (0-1)
            
        Returns:
            Annual interest rate
        """
        if utilization <= kink:
            # Below kink: linear rate
            rate_per_block = base_rate_per_block + (utilization * multiplier_per_block / TidalMath.SCALE_FACTOR)
        else:
            # Above kink: jump rate
            base_rate = base_rate_per_block + (kink * multiplier_per_block / TidalMath.SCALE_FACTOR)
            jump_rate = (utilization - kink) * jump_per_block / TidalMath.SCALE_FACTOR
            rate_per_block = base_rate + jump_rate
        
        # Convert to annual rate
        annual_rate = rate_per_block * TidalMath.BLOCKS_PER_YEAR / TidalMath.SCALE_FACTOR
        return annual_rate
    
    @staticmethod
    def calculate_supply_rate(
        borrow_rate: float,
        utilization: float,
        reserve_factor: float
    ) -> float:
        """
        Calculate supply rate from borrow rate
        
        Formula: supply_rate = borrow_rate * utilization * (1 - reserve_factor)
        
        Args:
            borrow_rate: Current borrow rate
            utilization: Current utilization rate
            reserve_factor: Reserve factor (0-1)
            
        Returns:
            Supply rate
        """
        return borrow_rate * utilization * (1 - reserve_factor)
    
    @staticmethod
    def calculate_compound_interest(
        principal: float,
        rate_per_block: float,
        blocks_elapsed: int
    ) -> float:
        """
        Calculate compound interest over blocks
        
        Args:
            principal: Initial amount
            rate_per_block: Interest rate per block
            blocks_elapsed: Number of blocks
            
        Returns:
            Final amount after compound interest
        """
        if rate_per_block <= 0 or blocks_elapsed <= 0:
            return principal
        
        return principal * ((1 + rate_per_block) ** blocks_elapsed)
    
    @staticmethod
    def calculate_health_factor(
        collateral_value: float,
        debt_value: float,
        collateral_factor: float
    ) -> float:
        """
        Calculate health factor for a position
        
        Args:
            collateral_value: Total collateral value in USD
            debt_value: Total debt value in USD
            collateral_factor: Weighted average collateral factor
            
        Returns:
            Health factor (>1 is healthy, <1 is liquidatable)
        """
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
        """
        Calculate liquidation amounts
        
        Args:
            debt_amount: Total debt amount
            collateral_amount: Available collateral amount
            collateral_price: Price of collateral asset
            liquidation_penalty: Liquidation penalty (e.g., 0.08 for 8%)
            close_factor: Maximum portion of debt that can be liquidated
            
        Returns:
            Tuple of (max_repay_amount, collateral_to_seize, liquidation_bonus)
        """
        # Maximum debt that can be repaid
        max_repay = debt_amount * close_factor
        
        # Collateral value needed (including penalty)
        collateral_value_needed = max_repay * (1 + liquidation_penalty)
        collateral_to_seize = collateral_value_needed / collateral_price
        
        # Ensure we don't seize more collateral than available
        if collateral_to_seize > collateral_amount:
            collateral_to_seize = collateral_amount
            max_repay = (collateral_to_seize * collateral_price) / (1 + liquidation_penalty)
        
        liquidation_bonus = max_repay * liquidation_penalty
        
        return max_repay, collateral_to_seize, liquidation_bonus
    
    @staticmethod
    def calculate_moet_mint_amount(
        collateral_value_usd: float,
        collateral_factor: float,
        mint_fee: float,
        existing_debt: float = 0.0,
        target_health_factor: float = 1.5
    ) -> Tuple[float, float, float]:
        """
        Calculate how much MOET can be minted against collateral
        
        Args:
            collateral_value_usd: Collateral value in USD
            collateral_factor: Collateral factor for the asset
            mint_fee: Minting fee rate
            existing_debt: Existing MOET debt
            target_health_factor: Target health factor to maintain
            
        Returns:
            Tuple of (max_mint_amount, mint_fee_amount, resulting_health_factor)
        """
        effective_collateral = collateral_value_usd * collateral_factor
        
        # Calculate max total debt based on target health factor
        max_total_debt = effective_collateral / target_health_factor
        
        # Subtract existing debt
        max_new_debt = max(0, max_total_debt - existing_debt)
        
        # Apply mint fee
        mint_fee_amount = max_new_debt * mint_fee
        net_mint_amount = max_new_debt - mint_fee_amount
        
        # Calculate resulting health factor
        new_total_debt = existing_debt + max_new_debt
        resulting_health_factor = effective_collateral / new_total_debt if new_total_debt > 0 else float('inf')
        
        return net_mint_amount, mint_fee_amount, resulting_health_factor
    
    @staticmethod
    def calculate_debt_cap_ebisu_style(
        liquidation_capacity: float,
        dex_allocation_factor: float,
        underwater_collateral_percentage: float
    ) -> float:
        """
        Calculate debt cap using Ebisu-style methodology
        
        Formula: Debt Cap = A × B × C
        Where:
        A: Amount able to be profitably liquidated via DEX
        B: Allocation of DEX liquidity to other lending markets
        C: Percentage of collateral that is underwater in extreme drop
        
        Args:
            liquidation_capacity: Total liquidation capacity across DEX
            dex_allocation_factor: Percentage of DEX liquidity allocated (e.g., 0.35)
            underwater_collateral_percentage: Weighted average of underwater collateral
            
        Returns:
            Maximum safe debt cap
        """
        return liquidation_capacity * dex_allocation_factor * underwater_collateral_percentage
    
    @staticmethod
    def calculate_underwater_collateral_percentage(
        collateral_values: Dict[str, float],
        extreme_price_drops: Dict[str, float]
    ) -> float:
        """
        Calculate weighted average of underwater collateral in extreme scenarios
        
        Args:
            collateral_values: Dict of asset -> collateral value
            extreme_price_drops: Dict of asset -> extreme drop percentage
            
        Returns:
            Weighted average underwater percentage
        """
        total_collateral_value = sum(collateral_values.values())
        
        if total_collateral_value <= 0:
            return 0.0
        
        weighted_underwater = 0.0
        
        for asset, collateral_value in collateral_values.items():
            if asset in extreme_price_drops:
                weight = collateral_value / total_collateral_value
                underwater_percentage = extreme_price_drops[asset]
                weighted_underwater += weight * underwater_percentage
        
        return weighted_underwater
    
    @staticmethod
    def calculate_constant_product_swap(
        amount_in: float,
        reserve_in: float,
        reserve_out: float,
        fee_rate: float = 0.003
    ) -> Tuple[float, float, float, float]:
        """
        Calculate swap using constant product formula with proper slippage calculation
        
        Formula: x * y = k
        Where x = reserve_in, y = reserve_out, k = constant
        
        Args:
            amount_in: Input amount (Δx)
            reserve_in: Input token reserves (x)
            reserve_out: Output token reserves (y)
            fee_rate: Trading fee rate
            
        Returns:
            Tuple of (amount_out, fee_amount, slippage_percent, actual_price)
        """
        if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # Step 1: Calculate expected price before trade
        # P(expected) = y/x
        expected_price = reserve_out / reserve_in
        
        # Step 2: Apply trading fee
        fee_amount = amount_in * fee_rate
        amount_in_after_fee = amount_in - fee_amount
        
        # Step 3: Calculate constant k
        k = reserve_in * reserve_out
        
        # Step 4: Calculate new reserves after swap
        # x(new) = x + Δx
        new_reserve_in = reserve_in + amount_in_after_fee
        
        # y(new) = k / x(new)
        new_reserve_out = k / new_reserve_in
        
        # Step 5: Calculate amount out
        # Δy = y - y(new)
        amount_out = reserve_out - new_reserve_out
        
        # Ensure we don't drain the pool (safety check)
        amount_out = min(amount_out, reserve_out * 0.99)
        
        # Step 6: Calculate actual price paid per unit of output token
        # P(actual) = Δx / Δy (how much input per unit of output)
        # But we want price in terms of output per input for comparison
        actual_price_per_input = amount_out / amount_in if amount_in > 0 else 0.0
        
        # Step 7: Calculate slippage
        # Compare actual output per input vs expected output per input
        # expected_output_per_input = reserve_out / reserve_in
        expected_output_per_input = expected_price
        
        if expected_output_per_input > 0:
            # Slippage = (expected - actual) / expected * 100%
            # Positive slippage means you get less than expected
            slippage_percent = ((expected_output_per_input - actual_price_per_input) / expected_output_per_input) * 100.0
        else:
            slippage_percent = 0.0
        
        return amount_out, fee_amount, slippage_percent, actual_price_per_input
    
    @staticmethod
    def calculate_uniswap_v3_concentrated_liquidity(
        amount_in: float,
        reserve_in: float,
        reserve_out: float,
        current_price: float,
        price_std_dev: float,
        fee_rate: float = 0.003
    ) -> Tuple[float, float, float, float]:
        """
        Calculate swap with Uniswap V3 concentrated liquidity (normally distributed)
        
        Liquidity is concentrated within 1 standard deviation of current price
        
        Args:
            amount_in: Input amount
            reserve_in: Total reserves of input token
            reserve_out: Total reserves of output token  
            current_price: Current price (reserve_out/reserve_in)
            price_std_dev: Price standard deviation for liquidity distribution
            fee_rate: Trading fee rate
            
        Returns:
            Tuple of (amount_out, fee_amount, slippage_percent, actual_price)
        """
        if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # Calculate liquidity concentration factor
        # In Uniswap V3, liquidity is concentrated around current price
        # We model this by increasing effective liquidity within 1 std dev
        
        # Price range for concentrated liquidity (±1 std dev)
        price_lower = current_price * (1 - price_std_dev)
        price_upper = current_price * (1 + price_std_dev)
        
        # Calculate concentration factor based on normal distribution
        # More liquidity is available within the price range
        concentration_multiplier = 3.0  # 3x more liquidity in concentrated range
        
        # Effective reserves for the concentrated range
        effective_reserve_in = reserve_in * concentration_multiplier
        effective_reserve_out = reserve_out * concentration_multiplier
        
        # Use constant product formula with concentrated liquidity
        return TidalMath.calculate_constant_product_swap(
            amount_in, effective_reserve_in, effective_reserve_out, fee_rate
        )
    
    @staticmethod
    def calculate_lp_token_mint(
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
        
        if reserve_a <= 0 or reserve_b <= 0 or total_supply <= 0:
            # First liquidity provision
            return math.sqrt(amount_a * amount_b)
        
        # Maintain proportionality
        ratio_a = amount_a / reserve_a
        ratio_b = amount_b / reserve_b
        
        # Use the smaller ratio to maintain pool balance
        mint_ratio = min(ratio_a, ratio_b)
        
        return mint_ratio * total_supply
    
    @staticmethod
    def calculate_slippage_with_concentration(
        trade_amount: float,
        available_liquidity: float,
        concentration_factor: float = 0.1
    ) -> float:
        """
        Calculate slippage for concentrated liquidity
        
        Args:
            trade_amount: Size of the trade
            available_liquidity: Available liquidity in price range
            concentration_factor: Liquidity concentration factor
            
        Returns:
            Slippage percentage
        """
        if available_liquidity <= 0:
            return 1.0  # 100% slippage (no liquidity)
        
        effective_liquidity = available_liquidity * (1 + concentration_factor)
        
        # Simplified slippage calculation
        slippage = trade_amount / (effective_liquidity * 0.1)
        
        # Cap slippage at 100%
        return min(slippage, 1.0)
    
    @staticmethod
    def calculate_stability_mechanism_pressure(
        current_price: float,
        target_price: float,
        upper_band: float,
        lower_band: float
    ) -> Tuple[str, float]:
        """
        Calculate stability mechanism pressure for MOET
        
        Args:
            current_price: Current MOET price
            target_price: Target price (usually 1.0)
            upper_band: Upper stability band (e.g., 1.02)
            lower_band: Lower stability band (e.g., 0.98)
            
        Returns:
            Tuple of (action_type, pressure_magnitude)
        """
        if current_price > upper_band:
            pressure = (current_price - upper_band) / target_price
            return "mint_pressure", pressure
        elif current_price < lower_band:
            pressure = (lower_band - current_price) / target_price
            return "burn_pressure", pressure
        else:
            return "stable", 0.0
    
    @staticmethod
    def calculate_protocol_revenue_distribution(
        total_revenue: float,
        lp_rewards_factor: float,
        treasury_factor: float = None
    ) -> Tuple[float, float, float]:
        """
        Calculate protocol revenue distribution
        
        Args:
            total_revenue: Total protocol revenue
            lp_rewards_factor: Percentage to LP rewards
            treasury_factor: Percentage to treasury (if None, remainder goes to treasury)
            
        Returns:
            Tuple of (lp_rewards, treasury_amount, other_allocations)
        """
        lp_rewards = total_revenue * lp_rewards_factor
        
        if treasury_factor is not None:
            treasury_amount = total_revenue * treasury_factor
            other_allocations = total_revenue - lp_rewards - treasury_amount
        else:
            treasury_amount = total_revenue - lp_rewards
            other_allocations = 0.0
        
        return lp_rewards, treasury_amount, other_allocations
    
    @staticmethod
    def calculate_optimal_liquidation_size(
        debt_amount: float,
        collateral_amount: float,
        collateral_price: float,
        target_health_factor: float,
        liquidation_penalty: float,
        close_factor: float = 0.5
    ) -> float:
        """
        Calculate optimal liquidation size to restore target health factor
        
        Args:
            debt_amount: Current debt amount
            collateral_amount: Available collateral
            collateral_price: Collateral asset price
            target_health_factor: Desired health factor after liquidation
            liquidation_penalty: Liquidation penalty rate
            close_factor: Maximum liquidation percentage
            
        Returns:
            Optimal liquidation amount
        """
        # Calculate collateral value
        collateral_value = collateral_amount * collateral_price
        
        # Calculate debt to repay to achieve target health factor
        # HF = collateral_value / remaining_debt = target_health_factor
        # remaining_debt = collateral_value / target_health_factor
        target_remaining_debt = collateral_value / target_health_factor
        
        # Amount to repay
        optimal_repay = debt_amount - target_remaining_debt
        
        # Apply constraints
        max_repay = debt_amount * close_factor
        optimal_repay = min(optimal_repay, max_repay)
        optimal_repay = max(0, optimal_repay)
        
        return optimal_repay
