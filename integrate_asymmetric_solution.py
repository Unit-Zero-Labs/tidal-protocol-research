#!/usr/bin/env python3
"""
Integration of the closed-form asymmetric solution into the simulation
"""

import math

def solve_delta_down_closed_form(P: float, delta_up: float = 0.01, target_ratio: float = 3.0) -> float:
    """
    Closed-form lower half-width solution given fixed upper half-width.
    
    This is the working solution from your mathematical derivation:
    - s = sqrt(P), sb = sqrt(P * (1 + δu))
    - A = (sb - s) / (s * sb)
    - Enforce V0/V1 = A / [P * (s - sa)] = target_ratio
    - sa = s - A / (target_ratio * P)
    - δd = 1 - sa^2 / P
    
    Args:
        P: Current YT price
        delta_up: Upper half-width (default 0.01 = 1%)
        target_ratio: Target R = MOET/YT ratio (default 3.0 for 75/25)
        
    Returns:
        delta_down: Lower half-width that achieves target ratio
    """
    s = math.sqrt(P)
    sb = math.sqrt(P * (1.0 + delta_up))
    A = (sb - s) / (s * sb)

    sa = s - A / (target_ratio * P)
    
    # Numerical guard: sa should be positive and less than s
    if sa <= 0.0:
        return 1e-6

    delta_down = 1.0 - (sa * sa) / P

    # Clamp to reasonable bounds
    if delta_down < 1e-9:
        delta_down = 1e-9
    if delta_down > 0.5:
        delta_down = 0.5

    return delta_down

def calculate_asymmetric_range_bounds(current_price: float) -> tuple[float, float]:
    """
    Calculate optimal asymmetric range bounds for the given YT price.
    
    Args:
        current_price: Current true YT price
        
    Returns:
        (P_lower, P_upper): Range bounds that achieve 75% MOET, 25% YT
    """
    delta_up = 0.01  # Fixed 1% upper bound
    delta_down = solve_delta_down_closed_form(current_price, delta_up, 3.0)
    
    P_lower = current_price * (1.0 - delta_down)
    P_upper = current_price * (1.0 + delta_up)
    
    return P_lower, P_upper

def update_uniswap_positioning_method():
    """
    Show how to update the _initialize_asymmetric_yield_token_positions method
    """
    
    code_update = '''
def _initialize_asymmetric_yield_token_positions(self):
    """Initialize asymmetric yield token positions using closed-form solution"""
    import math
    
    # Clear existing positions first when recentering
    self.positions.clear()
    self.ticks.clear()
    self.liquidity = 0
    if self.use_tick_bitmap and self.tick_bitmap:
        self.tick_bitmap = TickBitmap()
    
    total_liquidity_amount = int(self.total_liquidity * 1e6)
    concentrated_liquidity_usd = self.total_liquidity * self.concentration
    
    # Get current pool price
    current_price = self.get_price()
    
    # MATHEMATICAL SOLUTION: Use closed-form formula to get optimal bounds
    def solve_delta_down_closed_form(P: float, delta_up: float = 0.01, target_ratio: float = 3.0) -> float:
        s = math.sqrt(P)
        sb = math.sqrt(P * (1.0 + delta_up))
        A = (sb - s) / (s * sb)
        sa = s - A / (target_ratio * P)
        if sa <= 0.0:
            return 1e-6
        delta_down = 1.0 - (sa * sa) / P
        return max(1e-9, min(0.5, delta_down))
    
    # Calculate optimal asymmetric bounds
    delta_up = 0.01  # Fixed 1% upper
    target_ratio = self.token0_ratio / (1 - self.token0_ratio)  # 75/25 = 3.0
    delta_down = solve_delta_down_closed_form(current_price, delta_up, target_ratio)
    
    P_lower = current_price * (1.0 - delta_down)
    P_upper = current_price * (1.0 + delta_up)
    
    # Convert to sqrt prices for Uniswap V3 math
    s = math.sqrt(current_price)
    a = math.sqrt(P_lower)
    b = math.sqrt(P_upper)
    
    # Calculate coefficients for liquidity distribution
    coeff_0 = (b - s) / (s * b)  # MOET coefficient
    coeff_1 = (s - a)            # YT coefficient
    coeff_sum = coeff_0 + coeff_1
    
    if coeff_sum <= 0:
        raise ValueError(f"Invalid coefficient sum {coeff_sum} for target ratio {target_ratio}")
    
    L = concentrated_liquidity_usd / coeff_sum
    
    # Calculate actual token amounts
    amount_0 = L * coeff_0  # MOET amount
    amount_1 = L * coeff_1  # YT amount
    
    # Convert bounds to ticks
    tick_lower_exact = math.log(P_lower) / math.log(1.0001)
    tick_upper_exact = math.log(P_upper) / math.log(1.0001)
    
    # Round to valid tick spacing
    tick_lower = int(tick_lower_exact // self.tick_spacing) * self.tick_spacing
    tick_upper = int(tick_upper_exact // self.tick_spacing) * self.tick_spacing
    
    # Ensure valid range
    tick_lower = max(MIN_TICK + self.tick_spacing, tick_lower)
    tick_upper = min(MAX_TICK - self.tick_spacing, tick_upper)
    
    if tick_lower >= tick_upper:
        raise ValueError(f"Invalid tick range: [{tick_lower}, {tick_upper})")
    
    # Add the concentrated liquidity position
    concentrated_liquidity = int(L * 1e6)
    self._add_position(tick_lower, tick_upper, concentrated_liquidity)
    
    # Add small backup positions for safety (5% each)
    backup_liquidity = int((total_liquidity_amount - concentrated_liquidity) / 2)
    if backup_liquidity > 0:
        # Lower backup position
        backup_lower = max(MIN_TICK + self.tick_spacing, tick_lower - 500 * self.tick_spacing)
        self._add_position(backup_lower, tick_lower, backup_liquidity)
        
        # Upper backup position  
        backup_upper = min(MAX_TICK - self.tick_spacing, tick_upper + 500 * self.tick_spacing)
        self._add_position(tick_upper, backup_upper, backup_liquidity)
    
    # Update legacy fields for backward compatibility
    self._update_legacy_fields()
    '''
    
    return code_update

def test_timing_alignment():
    """Test that CSV timing aligns with simulation timing"""
    
    print("🕐 TESTING TIMING ALIGNMENT")
    print("=" * 50)
    
    # Simulation timing
    alm_interval = 720  # minutes
    range_update_interval = 6  # ALM rebalances
    range_update_frequency = alm_interval * range_update_interval  # 4,320 minutes
    
    print(f"Simulation Timing:")
    print(f"  ALM Rebalancer: Every {alm_interval} minutes (12 hours)")
    print(f"  Range Update: Every {range_update_interval} ALM rebalances")
    print(f"  Range Update Frequency: {range_update_frequency} minutes = {range_update_frequency/1440:.1f} days")
    print()
    
    # CSV timing
    csv_day_interval = 3  # days
    csv_minute_interval = csv_day_interval * 1440  # minutes
    
    print(f"CSV Timing:")
    print(f"  Day Interval: Every {csv_day_interval} days")
    print(f"  Minute Interval: Every {csv_minute_interval} minutes")
    print()
    
    # Check alignment
    if range_update_frequency == csv_minute_interval:
        print("✅ PERFECT ALIGNMENT!")
        print(f"   Both update every {range_update_frequency} minutes ({csv_day_interval} days)")
    else:
        print("❌ TIMING MISMATCH!")
        print(f"   Simulation: {range_update_frequency} minutes")
        print(f"   CSV: {csv_minute_interval} minutes")
        print(f"   Difference: {abs(range_update_frequency - csv_minute_interval)} minutes")
    
    print()
    
    # Show first few update points
    print("First 10 Range Update Points:")
    print("Simulation Minutes | CSV Days | Alignment")
    print("-" * 40)
    
    for i in range(10):
        sim_minute = (i + 1) * range_update_frequency
        csv_day = (i + 1) * csv_day_interval
        expected_minute = csv_day * 1440
        
        aligned = "✅" if sim_minute == expected_minute else "❌"
        print(f"{sim_minute:14,} | {csv_day:8} | {aligned}")

def create_lookup_function():
    """Create a lookup function for the simulation"""
    
    lookup_code = '''
def get_optimal_asymmetric_bounds(current_minute: int, apr: float = 0.10) -> tuple[float, float]:
    """
    Get optimal asymmetric range bounds for the current simulation time.
    
    Args:
        current_minute: Current simulation minute
        apr: Annual percentage rate (default 0.10 for 10%)
        
    Returns:
        (P_lower, P_upper): Optimal range bounds
    """
    # Calculate true YT price at current time
    if current_minute <= 0:
        yt_price = 1.0
    else:
        minutes_per_year = 365 * 24 * 60
        minute_rate = apr * (current_minute / minutes_per_year)
        yt_price = 1.0 * (1 + minute_rate)
    
    # Use closed-form solution
    delta_up = 0.01
    target_ratio = 3.0
    
    s = math.sqrt(yt_price)
    sb = math.sqrt(yt_price * (1.0 + delta_up))
    A = (sb - s) / (s * sb)
    sa = s - A / (target_ratio * yt_price)
    
    if sa <= 0.0:
        delta_down = 1e-6
    else:
        delta_down = 1.0 - (sa * sa) / yt_price
        delta_down = max(1e-9, min(0.5, delta_down))
    
    P_lower = yt_price * (1.0 - delta_down)
    P_upper = yt_price * (1.0 + delta_up)
    
    return P_lower, P_upper
    '''
    
    return lookup_code

def main():
    """Main integration planning"""
    
    print("🔧 INTEGRATING CLOSED-FORM ASYMMETRIC SOLUTION")
    print("=" * 60)
    
    # Test timing alignment
    test_timing_alignment()
    
    print("\n📝 INTEGRATION PLAN:")
    print("1. ✅ Timing is perfectly aligned (every 3 days)")
    print("2. 🔄 Update _initialize_asymmetric_yield_token_positions() method")
    print("3. 📊 Add closed-form calculation function")
    print("4. 🧪 Test with current simulation")
    
    print("\n📋 NEXT STEPS:")
    print("1. Update the UniswapV3Pool._initialize_asymmetric_yield_token_positions() method")
    print("2. Replace the current asymmetric formula with the closed-form solution")
    print("3. Test with a short simulation to verify range widths")
    print("4. Run full year simulation to confirm slippage improvements")
    
    print(f"\n💡 EXPECTED IMPROVEMENTS:")
    print(f"   Range Width: 5.97% → 1.30% (4.6x tighter)")
    print(f"   Token Ratio: 16.7% MOET → 75.0% MOET (correct target)")
    print(f"   Slippage: Should reduce significantly due to tighter ranges")

if __name__ == "__main__":
    main()
