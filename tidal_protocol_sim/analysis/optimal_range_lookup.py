#!/usr/bin/env python3
"""
Optimal Range Lookup Table for Asymmetric Liquidity Positioning

Pre-computed optimal range bounds using closed-form mathematical solution.
Achieves 75% MOET / 25% YT ratio (R=3.0) with minimal range width.

Usage:
    lookup = OptimalRangeLookup()
    P_lower, P_upper = lookup.get_optimal_bounds(current_minute)
"""

import csv
import math
from typing import Tuple, Optional
import os

class OptimalRangeLookup:
    """
    Pre-computed lookup table for optimal asymmetric range bounds.
    
    Data generated using closed-form mathematical solution:
    - Target ratio: R = MOET/YT = 3.0 (75% MOET, 25% YT)  
    - Upper bound: Fixed at +1% from current price
    - Lower bound: Calculated to achieve exact R=3.0
    - Range width: ~1.3% (vs current 5.97%)
    - Update frequency: Every 3 days (4,320 minutes)
    """
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize the lookup table.
        
        Args:
            csv_path: Path to CSV file. If None, uses default location.
        """
        if csv_path is None:
            # Default to same directory as this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(os.path.dirname(current_dir), "optimal_range_lookup_corrected.csv")
        
        self.csv_path = csv_path
        self.lookup_data = {}
        self._load_data()
    
    def _load_data(self):
        """Load the pre-computed lookup data from CSV"""
        try:
            with open(self.csv_path, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    day = int(row['day'])
                    minute = day * 1440  # Convert days to minutes
                    
                    self.lookup_data[minute] = {
                        'day': day,
                        'date_iso': row['date_iso'],
                        'P': float(row['P']),
                        'delta_down': float(row['delta_down']),
                        'delta_up': float(row['delta_up']),
                        'lower': float(row['lower']),
                        'upper': float(row['upper'])
                    }
            
            print(f"✅ Loaded {len(self.lookup_data)} optimal range entries from {self.csv_path}")
            
        except FileNotFoundError:
            print(f"❌ Could not find lookup table at {self.csv_path}")
            print("   Please run the range generation script first")
            self.lookup_data = {}
        except Exception as e:
            print(f"❌ Error loading lookup table: {str(e)}")
            self.lookup_data = {}
    
    def get_optimal_bounds(self, current_minute: int) -> Tuple[float, float]:
        """
        Get optimal asymmetric range bounds for the current simulation time.
        
        Args:
            current_minute: Current simulation minute
            
        Returns:
            (P_lower, P_upper): Optimal range bounds that achieve R=3.0
        """
        # Range updates happen every 4,320 minutes (3 days)
        # Find the most recent range update point
        range_update_minute = (current_minute // 4320) * 4320
        
        # Handle edge cases
        if range_update_minute <= 0:
            range_update_minute = 0
        
        # Look up the optimal bounds
        if range_update_minute in self.lookup_data:
            data = self.lookup_data[range_update_minute]
            return data['lower'], data['upper']
        
        # Fallback: calculate dynamically if not in lookup table
        print(f"⚠️  Range update minute {range_update_minute} not in lookup table, calculating dynamically")
        return self._calculate_fallback_bounds(current_minute)
    
    def _calculate_fallback_bounds(self, current_minute: int) -> Tuple[float, float]:
        """
        Fallback calculation if lookup table doesn't have the data.
        Uses the same closed-form solution as the pre-computed table.
        """
        # Calculate YT price at current time (10% APR continuous)
        if current_minute <= 0:
            yt_price = 1.0
        else:
            minutes_per_year = 365 * 24 * 60
            minute_rate = 0.10 * (current_minute / minutes_per_year)
            yt_price = 1.0 * (1 + minute_rate)
        
        # Closed-form solution
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
    
    def get_range_info(self, current_minute: int) -> dict:
        """
        Get detailed information about the optimal range for debugging.
        
        Args:
            current_minute: Current simulation minute
            
        Returns:
            Dictionary with range details including width, ratio, etc.
        """
        range_update_minute = (current_minute // 4320) * 4320
        P_lower, P_upper = self.get_optimal_bounds(current_minute)
        
        # Calculate range properties
        if range_update_minute in self.lookup_data:
            data = self.lookup_data[range_update_minute]
            yt_price = data['P']
            delta_down = data['delta_down']
            delta_up = data['delta_up']
        else:
            # Calculate for fallback
            minutes_per_year = 365 * 24 * 60
            minute_rate = 0.10 * (current_minute / minutes_per_year)
            yt_price = 1.0 * (1 + minute_rate)
            delta_down = (yt_price - P_lower) / yt_price
            delta_up = (P_upper - yt_price) / yt_price
        
        range_width = (P_upper - P_lower) / yt_price * 100
        
        return {
            'current_minute': current_minute,
            'range_update_minute': range_update_minute,
            'yt_price': yt_price,
            'P_lower': P_lower,
            'P_upper': P_upper,
            'delta_down': delta_down,
            'delta_up': delta_up,
            'range_width_pct': range_width,
            'target_ratio': 3.0,
            'expected_moet_pct': 75.0,
            'expected_yt_pct': 25.0,
            'data_source': 'lookup_table' if range_update_minute in self.lookup_data else 'fallback_calculation'
        }
    
    def validate_lookup_table(self) -> bool:
        """
        Validate that the lookup table covers the expected simulation timeframe.
        
        Returns:
            True if lookup table is valid for full year simulation
        """
        if not self.lookup_data:
            print("❌ No lookup data loaded")
            return False
        
        # Check coverage for full year (365 days = 525,600 minutes)
        max_minute = max(self.lookup_data.keys())
        min_minute = min(self.lookup_data.keys())
        
        expected_max = 363 * 1440  # Day 363 in minutes
        
        print(f"📊 Lookup Table Coverage:")
        print(f"   Range: {min_minute:,} - {max_minute:,} minutes")
        print(f"   Expected: 0 - {expected_max:,} minutes")
        print(f"   Entries: {len(self.lookup_data)}")
        print(f"   Update frequency: Every 4,320 minutes (3 days)")
        
        if max_minute >= expected_max:
            print("✅ Lookup table covers full year simulation")
            return True
        else:
            print("⚠️  Lookup table may not cover full year simulation")
            return False

def test_lookup_integration():
    """Test the lookup table integration"""
    
    print("🧪 TESTING OPTIMAL RANGE LOOKUP INTEGRATION")
    print("=" * 60)
    
    # Initialize lookup
    lookup = OptimalRangeLookup()
    
    # Validate coverage
    is_valid = lookup.validate_lookup_table()
    
    if not is_valid:
        return False
    
    print("\n🔍 Testing specific time points:")
    
    # Test key simulation points
    test_minutes = [0, 4320, 8640, 17280, 100000, 200000, 300000, 400000, 500000]
    
    for minute in test_minutes:
        if minute > 525600:  # Skip if beyond 1 year
            continue
            
        info = lookup.get_range_info(minute)
        P_lower, P_upper = lookup.get_optimal_bounds(minute)
        
        print(f"Minute {minute:6,} (Day {minute/1440:6.1f}): "
              f"YT=${info['yt_price']:.6f} → "
              f"Range=[{P_lower:.6f}, {P_upper:.6f}] "
              f"Width={info['range_width_pct']:.2f}% "
              f"({info['data_source']})")
    
    print(f"\n✅ Lookup table integration test completed!")
    return True

if __name__ == "__main__":
    test_lookup_integration()
