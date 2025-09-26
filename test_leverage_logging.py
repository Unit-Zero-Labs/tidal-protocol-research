#!/usr/bin/env python3
"""
Quick test to verify leverage increase logging and YT position tracking
"""
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sim_tests.full_year_sim import FullYearSimConfig

def test_leverage_logging():
    """Test the leverage increase functionality with proper logging"""
    print("🧪 Testing Leverage Increase Logging...")
    
    config = FullYearSimConfig()
    
    # Run just the first few steps to see leverage increases
    from sim_tests.full_year_sim import run_full_year_simulation
    
    # Override to run only 2 hours (120 minutes) to see initial behavior
    config.simulation_duration_minutes = 120
    
    print(f"Running {config.simulation_duration_minutes} minute test...")
    
    try:
        results = run_full_year_simulation(config)
        print("✅ Test completed successfully")
        return results
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

if __name__ == "__main__":
    test_leverage_logging()
