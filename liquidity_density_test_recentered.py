#!/usr/bin/env python3
"""
Test script to verify liquidity density calculations AFTER LiquidityRangeManager 
has recentered positions around the true YT price.

This simulates what the pool should look like after range updates.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tidal_protocol_sim.core.uniswap_v3_math import create_yield_token_pool, UniswapV3SlippageCalculator, sqrt_price_x96_to_tick
from tidal_protocol_sim.core.yield_tokens import calculate_true_yield_token_price

def test_recentered_pool_at_hour(hour: int, swap_amount_yt: float = 750.0):
    """
    Test liquidity density with pool recentered around true YT price
    (simulating what LiquidityRangeManager should have done)
    
    Args:
        hour: Simulation hour (0-8760 for full year)
        swap_amount_yt: Amount of YT to swap for MOET
    """
    print(f"\n🔬 TESTING RECENTERED POOL AT HOUR {hour}")
    print("=" * 60)
    
    # Convert hour to minute
    minute = hour * 60
    
    # Calculate true YT price at this time (10% APR continuous)
    true_yt_price = calculate_true_yield_token_price(minute, 0.10, 1.0)
    print(f"📊 True YT Price at hour {hour}: ${true_yt_price:.6f}")
    
    # Create pool but we'll manually recenter it around the true price
    print(f"🏊 Creating pool with recentered positions:")
    print(f"   - Total liquidity: $500,000")
    print(f"   - MOET:YT ratio: 75:25")
    print(f"   - Concentration: 95%")
    print(f"   - CENTER PRICE: ${true_yt_price:.6f} (not $1.00)")
    
    # Create base pool
    pool = create_yield_token_pool(
        pool_size_usd=500_000,
        concentration=0.95,
        token0_ratio=0.75  # 75% MOET, 25% YT
    )
    
    # Manually recenter the pool around the true YT price
    # This simulates what the LiquidityRangeManager should do
    
    # Update pool price to match true YT price
    target_sqrt_price_x96 = int((true_yt_price ** 0.5) * (2 ** 96))
    pool.sqrt_price_x96 = target_sqrt_price_x96
    pool.tick_current = sqrt_price_x96_to_tick(target_sqrt_price_x96)
    
    # Clear existing positions (simulate LiquidityRangeManager clearing old positions)
    pool.positions.clear()
    pool.ticks.clear()
    pool.liquidity = 0
    
    # Recreate positions centered around the new price
    # This is what _initialize_asymmetric_yield_token_positions should do after recentering
    
    import math
    
    # Calculate new tick range around true price (±1% range)
    # Using the same asymmetric calculation but centered on true_yt_price
    
    # Step 1: Fix upper bound at +1% from TRUE price
    P_upper = true_yt_price * 1.01
    b = math.sqrt(P_upper)
    
    # Step 2: Calculate lower bound for 75/25 ratio
    R = 0.75 / 0.25  # 3
    x = math.sqrt(true_yt_price)  # Current sqrt price at TRUE price
    a = 1 - (b - x) / (R * b)
    
    # Step 3: Convert to price bounds
    P_lower = a ** 2
    
    print(f"   - Recentered range: [${P_lower:.6f}, ${P_upper:.6f}]")
    print(f"   - Range width: {((P_upper/P_lower - 1) * 100):.2f}%")
    
    # Convert to ticks
    tick_lower_exact = math.log(P_lower) / math.log(1.0001)
    tick_upper_exact = math.log(P_upper) / math.log(1.0001)
    
    # Round to valid tick spacing (assuming spacing = 10)
    tick_spacing = 10
    tick_lower = int(tick_lower_exact // tick_spacing) * tick_spacing
    tick_upper = int(tick_upper_exact // tick_spacing) * tick_spacing
    if tick_upper <= tick_lower:
        tick_upper = tick_lower + tick_spacing
    
    print(f"   - Tick range: [{tick_lower}, {tick_upper}]")
    
    # Calculate concentrated liquidity (95% of total)
    concentrated_liquidity_usd = 500_000 * 0.95
    
    # Calculate proper liquidity amount using Uniswap V3 math
    # For simplicity, use the same approach as the original initialization
    coeff_0 = (b - x) / b  # MOET coefficient
    coeff_1 = x - a        # YT coefficient
    coeff_sum = coeff_0 + coeff_1
    
    if coeff_sum > 0:
        L = concentrated_liquidity_usd / coeff_sum
        proper_liquidity = int(L * 1e6)  # Scale for precision
    else:
        proper_liquidity = int(concentrated_liquidity_usd * 1e6)
    
    # Add the main concentrated position
    from tidal_protocol_sim.core.uniswap_v3_math import Position, TickInfo
    main_position = Position(tick_lower, tick_upper, proper_liquidity)
    pool.positions.append(main_position)
    
    # Update tick data
    pool.ticks[tick_lower] = TickInfo()
    pool.ticks[tick_upper] = TickInfo()
    pool.ticks[tick_lower].liquidity_net = proper_liquidity
    pool.ticks[tick_lower].liquidity_gross = proper_liquidity
    pool.ticks[tick_lower].initialized = True
    pool.ticks[tick_upper].liquidity_net = -proper_liquidity
    pool.ticks[tick_upper].liquidity_gross = proper_liquidity
    pool.ticks[tick_upper].initialized = True
    
    # Add backup positions (5% split into two ranges)
    backup_liquidity = int((500_000 * 0.05 / 2) * 1e6)
    
    # Lower backup position
    tick_lower_backup = tick_lower - 1000
    backup_position_lower = Position(tick_lower_backup, tick_lower, backup_liquidity)
    pool.positions.append(backup_position_lower)
    
    # Upper backup position  
    tick_upper_backup = tick_upper + 1000
    backup_position_upper = Position(tick_upper, tick_upper_backup, backup_liquidity)
    pool.positions.append(backup_position_upper)
    
    # Update current liquidity if current tick is in range
    if tick_lower <= pool.tick_current < tick_upper:
        pool.liquidity = proper_liquidity
    else:
        # Check backup positions
        if tick_lower_backup <= pool.tick_current < tick_lower:
            pool.liquidity = backup_liquidity
        elif tick_upper <= pool.tick_current < tick_upper_backup:
            pool.liquidity = backup_liquidity
        else:
            pool.liquidity = 0
    
    print(f"   - Pool sqrt_price_x96: {pool.sqrt_price_x96:,}")
    print(f"   - Pool tick_current: {pool.tick_current}")
    print(f"   - Pool price: ${pool.get_price():.6f}")
    
    # Check active liquidity
    active_liquidity = pool._calculate_active_liquidity_from_ticks(pool.tick_current)
    print(f"   - Active liquidity: {active_liquidity:,}")
    
    # Display pool positions
    print(f"\n📋 Recentered Pool Positions:")
    for i, pos in enumerate(pool.positions):
        tick_lower_price = 1.0001 ** pos.tick_lower
        tick_upper_price = 1.0001 ** pos.tick_upper
        is_active = pos.tick_lower <= pool.tick_current < pos.tick_upper
        print(f"   Position {i}: ticks [{pos.tick_lower}, {pos.tick_upper})")
        print(f"     Price range: [${tick_lower_price:.6f}, ${tick_upper_price:.6f})")
        print(f"     Liquidity: {pos.liquidity:,}")
        print(f"     Active: {'✅' if is_active else '❌'}")
    
    # Create slippage calculator
    calculator = UniswapV3SlippageCalculator(pool)
    
    # Test the swap: YT -> MOET
    print(f"\n💱 TESTING SWAP: ${swap_amount_yt:.2f} YT → MOET")
    print("-" * 40)
    
    try:
        swap_result = calculator.calculate_swap_slippage(
            amount_in=swap_amount_yt,
            token_in="Yield_Token",
            concentrated_range=0.05  # 5% range (legacy parameter)
        )
        
        print(f"✅ Swap Successful!")
        print(f"   YT Input: ${swap_result['amount_in']:.2f}")
        print(f"   MOET Output: ${swap_result['amount_out']:.2f}")
        print(f"   Expected Output: ${swap_amount_yt * true_yt_price:.2f}")
        print(f"   Slippage Amount: ${swap_result['slippage_amount']:.2f}")
        print(f"   Slippage Percentage: {swap_result['slippage_percent']:.2f}%")
        print(f"   Trading Fees: ${swap_result['trading_fees']:.2f}")
        print(f"   Price Impact: {swap_result['price_impact']:.2f}%")
        
        # Calculate effective exchange rate
        effective_rate = swap_result['amount_out'] / swap_result['amount_in']
        print(f"   Effective Rate: ${effective_rate:.6f} MOET per YT")
        print(f"   True Rate: ${true_yt_price:.6f} MOET per YT")
        print(f"   Rate Difference: ${true_yt_price - effective_rate:.6f}")
        
        return swap_result
        
    except Exception as e:
        print(f"❌ Swap Failed: {e}")
        return None

if __name__ == "__main__":
    print("🧪 RECENTERED LIQUIDITY DENSITY TEST")
    print("Testing pool behavior AFTER LiquidityRangeManager recentering")
    
    # Test hour 4000 with properly recentered positions
    test_recentered_pool_at_hour(4000, 750.0)
    
    print("\n✅ Test completed!")
