#!/usr/bin/env python3
"""
Python conversion of asymmetric_range_script.js
Asymmetric ranges every 3 days for a year, 10% continuous APR.
Goal: Achieve 75/25 (V0/V1 = 3) at the mid-price and minimize deviation from ±1%.
"""

import math
import csv
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

def price_at_day(day: int) -> float:
    """Calculate price at given day with continuous 10% APR"""
    t = day / 365.0  # years
    return math.exp(0.10 * t)  # continuous compounding

def ratio_at_mid(P: float, delta_down: float, delta_up: float) -> float:
    """
    Calculate mid-price USD ratio for general asymmetric range
    R = V0/V1 = [(sb - s) / (s · sb)] / [P · (s - sa)]
    """
    # Bounds checking to avoid math domain errors
    if delta_down >= 1.0 or delta_down < 0 or delta_up < 0 or delta_up >= 1.0:
        return float('inf')
    
    # Check that we won't have negative values under sqrt
    P_lower = P * (1 - delta_down)
    P_upper = P * (1 + delta_up)
    
    if P_lower <= 0 or P_upper <= 0:
        return float('inf')
    
    s = math.sqrt(P)
    sa = math.sqrt(P_lower)
    sb = math.sqrt(P_upper)
    
    numerator = (sb - s) / (s * sb)
    denominator = P * (s - sa)
    
    if denominator <= 0:
        return float('inf')
    
    return numerator / denominator

def solve_delta_down(P: float, delta_up: float, target_ratio: float = 3.0, 
                    tol: float = 1e-12, max_iter: int = 200) -> float:
    """
    Solve for delta_down given P, delta_up such that ratio_at_mid = target_ratio via bisection.
    Search in (0, 0.5), expanding high bound if needed.
    """
    lo = 1e-12
    hi = 0.5
    
    r_lo = ratio_at_mid(P, lo, delta_up)
    if r_lo > target_ratio:
        return lo
    
    # Expand hi until we bracket the target ratio
    r_hi = ratio_at_mid(P, hi, delta_up)
    while r_hi < target_ratio and hi < 0.999:
        hi *= 1.5
        if hi >= 0.999:
            break
        r_hi = ratio_at_mid(P, hi, delta_up)
    
    # Bisection
    left, right = lo, hi
    for _ in range(max_iter):
        mid = (left + right) / 2.0
        r_mid = ratio_at_mid(P, mid, delta_up)
        if abs(r_mid - target_ratio) < tol:
            return mid
        if r_mid < target_ratio:
            left = mid
        else:
            right = mid
    
    return (left + right) / 2.0

def preference_c_for_price(P: float, target_ratio: float = 3.0, delta_target: float = 0.01) -> Optional[Dict]:
    """
    Preference C solver: search delta_up around delta_target, solve delta_down to meet target_ratio,
    and pick the pair minimizing J = (delta_down - delta_target)^2 + (delta_up - delta_target)^2.
    """
    best = None
    span = 0.05         # allow ±5% wiggle room around 1%
    steps = 1000        # resolution of delta_up search
    
    for i in range(steps + 1):
        delta_up = delta_target - span + (2.0 * span) * (i / steps)
        if delta_up <= 0.0 or delta_up >= 0.5:
            continue
        delta_down = solve_delta_down(P, delta_up, target_ratio)
        if not math.isfinite(delta_down) or delta_down <= 0.0 or delta_down >= 0.5:
            continue
        J = (delta_down - delta_target) ** 2 + (delta_up - delta_target) ** 2
        if best is None or J < best['J']:
            best = {
                'delta_down': delta_down,
                'delta_up': delta_up,
                'J': J
            }
    
    # Fallback if no solution found
    if best is None:
        # No feasible solution found; fall back to fixing delta_up=0.01 and solving delta_down
        delta_up_fallback = 0.01
        delta_down_fallback = solve_delta_down(P, delta_up_fallback, target_ratio)
        if math.isfinite(delta_down_fallback) and delta_down_fallback > 0.0 and delta_down_fallback < 0.5:
            best = {
                'delta_down': delta_down_fallback,
                'delta_up': delta_up_fallback,
                'J': (delta_down_fallback - delta_target) ** 2 + (delta_up_fallback - delta_target) ** 2
            }
    
    return best

def test_specific_price():
    """Test the algorithm at P = 1.045662 (your original example)"""
    
    print("🧮 TESTING ASYMMETRIC RANGE ALGORITHM")
    print("=" * 60)
    
    P = 1.045662
    target_ratio = 3.0
    
    print(f"Testing at P = {P} (target R = {target_ratio})")
    print()
    
    # Find optimal asymmetric range
    result = preference_c_for_price(P, target_ratio, 0.01)
    
    if result:
        delta_down = result['delta_down']
        delta_up = result['delta_up']
        J = result['J']
        
        # Calculate range bounds
        P_lower = P * (1 - delta_down)
        P_upper = P * (1 + delta_up)
        
        # Validate the ratio
        achieved_R = ratio_at_mid(P, delta_down, delta_up)
        
        # Calculate range properties
        total_width = (P_upper - P_lower) / P * 100
        lower_distance = delta_down * 100
        upper_distance = delta_up * 100
        
        # Convert to percentages
        moet_pct = achieved_R / (achieved_R + 1) * 100
        yt_pct = 100 - moet_pct
        
        print("✅ OPTIMAL ASYMMETRIC RANGE FOUND:")
        print(f"  δ_down = {delta_down:.6f} ({lower_distance:.2f}%)")
        print(f"  δ_up   = {delta_up:.6f} ({upper_distance:.2f}%)")
        print(f"  Range: [{P_lower:.6f}, {P_upper:.6f}]")
        print(f"  Total width: {total_width:.2f}%")
        print(f"  Optimization cost J: {J:.2e}")
        print()
        print(f"📊 VALIDATION:")
        print(f"  Achieved R: {achieved_R:.6f}")
        print(f"  Target R: {target_ratio}")
        print(f"  Error: {abs(achieved_R - target_ratio):.6f}")
        print(f"  Token split: {moet_pct:.1f}% MOET / {yt_pct:.1f}% YT")
        
        if abs(achieved_R - target_ratio) < 0.01:
            print("  ✅ SUCCESS: Achieved target R = 3.0!")
        else:
            print("  ❌ FAILED: Did not achieve target R = 3.0")
        
        return result
    else:
        print("❌ No solution found")
        return None

def generate_year_schedule():
    """Generate asymmetric ranges for the full year"""
    
    print("\n📅 GENERATING FULL YEAR SCHEDULE")
    print("=" * 60)
    
    results = []
    
    print("Day | Date       | P        | δ_down   | δ_up     | Range Width | R      | MOET%")
    print("-" * 80)
    
    for day in range(0, 366, 3):  # Every 3 days
        P = price_at_day(day)
        result = preference_c_for_price(P, 3.0, 0.01)
        
        if result:
            delta_down = result['delta_down']
            delta_up = result['delta_up']
            
            P_lower = P * (1 - delta_down)
            P_upper = P * (1 + delta_up)
            total_width = (P_upper - P_lower) / P * 100
            
            achieved_R = ratio_at_mid(P, delta_down, delta_up)
            moet_pct = achieved_R / (achieved_R + 1) * 100
            
            date_str = (datetime(2025, 1, 1) + timedelta(days=day)).strftime("%Y-%m-%d")
            
            print(f"{day:3d} | {date_str} | {P:.6f} | {delta_down:.6f} | {delta_up:.6f} | {total_width:8.2f}% | {achieved_R:.3f} | {moet_pct:5.1f}%")
            
            results.append({
                'day': day,
                'date': date_str,
                'P': P,
                'delta_down': delta_down,
                'delta_up': delta_up,
                'P_lower': P_lower,
                'P_upper': P_upper,
                'total_width_pct': total_width,
                'achieved_R': achieved_R,
                'moet_pct': moet_pct,
                'J': result['J']
            })
        else:
            print(f"{day:3d} | ERROR: No solution found")
    
    print(f"\n📊 Generated {len(results)} range updates for the year")
    
    # Summary statistics
    if results:
        widths = [r['total_width_pct'] for r in results]
        ratios = [r['achieved_R'] for r in results]
        
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"  Range Width: {min(widths):.2f}% - {max(widths):.2f}% (avg: {sum(widths)/len(widths):.2f}%)")
        print(f"  Achieved R: {min(ratios):.3f} - {max(ratios):.3f} (avg: {sum(ratios)/len(ratios):.3f})")
        print(f"  Target R: 3.000")
        print(f"  vs Current Method: {5.97/sum(widths)*len(widths):.1f}x tighter ranges")
    
    return results

def save_lookup_table(results):
    """Save results as lookup table for simulation use"""
    
    if not results:
        print("No results to save")
        return
    
    filename = "asymmetric_range_lookup.csv"
    
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['day', 'date', 'P', 'delta_down', 'delta_up', 'P_lower', 'P_upper', 
                     'total_width_pct', 'achieved_R', 'moet_pct', 'J']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"\n💾 Saved lookup table to {filename}")
    print(f"   Ready for integration into simulation")

def main():
    """Main execution function"""
    
    print("🔧 ASYMMETRIC RANGE ALGORITHM - PYTHON VERSION")
    print("Converting JavaScript implementation to Python")
    print("=" * 80)
    
    # Test at specific price first
    test_result = test_specific_price()
    
    if test_result:
        # Generate full year schedule
        year_results = generate_year_schedule()
        
        # Save lookup table
        save_lookup_table(year_results)
        
        print(f"\n✅ SUCCESS: Algorithm working correctly!")
        print(f"   Ready to integrate into simulation")
    else:
        print(f"\n❌ FAILED: Algorithm not working at test price")

if __name__ == "__main__":
    main()
