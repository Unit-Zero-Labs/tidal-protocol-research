#!/usr/bin/env python3
"""
Regenerate the optimal range CSV using the EXACT same YT price calculation as the simulation
"""

import sys
sys.path.append('tidal_protocol_sim')

import math
from datetime import datetime, timedelta
from tidal_protocol_sim.core.yield_tokens import calculate_true_yield_token_price

def solve_delta_down_closed_form(P: float, delta_up: float = 0.01, target_ratio: float = 3.0) -> float:
    """
    Closed-form lower half-width solution given fixed upper half-width.
    This is the working mathematical solution from the user.
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

def generate_corrected_csv():
    """Generate CSV using the EXACT same YT price calculation as the simulation"""
    
    print("🔄 REGENERATING OPTIMAL RANGE CSV")
    print("Using simulation's YT price calculation method")
    print("=" * 60)
    
    # Use the same parameters as simulation
    start_date = datetime(2025, 1, 1)
    delta_up = 0.01
    target_ratio = 3.0
    
    print("day,date_iso,P,delta_down,delta_up,lower,upper")
    
    # Generate every 3 days for a full year (same as original)
    for day in range(0, 366, 3):
        # Convert day to minute (same as simulation timing)
        minute = day * 1440  # 1440 minutes per day
        
        # Use EXACT same YT price calculation as simulation
        yt_price = calculate_true_yield_token_price(minute, 0.10, 1.0)
        
        # Calculate optimal bounds using closed-form solution
        delta_down = solve_delta_down_closed_form(yt_price, delta_up, target_ratio)
        
        # Calculate range bounds
        P_lower = yt_price * (1.0 - delta_down)
        P_upper = yt_price * (1.0 + delta_up)
        
        # Format date
        date_iso = (start_date + timedelta(days=day)).date().isoformat()
        
        # Output CSV row
        print(f"{day},{date_iso},{yt_price:.6f},{delta_down:.6f},{delta_up:.6f},{P_lower:.6f},{P_upper:.6f}")

def validate_specific_points():
    """Validate specific points against simulation output"""
    
    print("\n🔍 VALIDATION AGAINST SIMULATION")
    print("=" * 40)
    
    # Test points we know from simulation
    test_cases = [
        (0, "Simulation start"),
        (4320, "First range update (Day 3)"),  
        (414720, "Failure point (Day 288)")
    ]
    
    for minute, description in test_cases:
        day = minute / 1440
        yt_price = calculate_true_yield_token_price(minute, 0.10, 1.0)
        delta_down = solve_delta_down_closed_form(yt_price, 0.01, 3.0)
        P_lower = yt_price * (1.0 - delta_down)
        P_upper = yt_price * (1.0 + 0.01)
        range_width = (P_upper - P_lower) / yt_price * 100
        
        print(f"{description}:")
        print(f"  Minute: {minute:,} (Day {day:.1f})")
        print(f"  YT Price: ${yt_price:.6f}")
        print(f"  Range: [{P_lower:.6f}, {P_upper:.6f}]")
        print(f"  Width: {range_width:.2f}%")
        print(f"  δ_down: {delta_down:.6f}, δ_up: 0.010000")
        print()

if __name__ == "__main__":
    # First validate our calculations
    validate_specific_points()
    
    print("🎯 Generating corrected CSV...")
    generate_corrected_csv()
    
    print(f"\n✅ CSV regeneration complete!")
    print(f"   Copy the output above to replace optimal_range_lookup.csv")
    print(f"   This CSV now uses the EXACT same YT price calculation as the simulation")
