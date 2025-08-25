#!/usr/bin/env python3
"""
Verify Results Path Configuration

Quick script to verify that results are being saved in the correct location:
tidal_protocol_sim/results/ instead of the repository root.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_results_path():
    """Verify that the results path is configured correctly"""
    
    print("=" * 60)
    print("VERIFYING RESULTS PATH CONFIGURATION")
    print("=" * 60)
    
    try:
        from tidal_protocol_sim.stress_testing.runner import StressTestRunner
        from tidal_protocol_sim.analysis.results_manager import ResultsManager
        
        print("✅ Successfully imported required modules")
        
        # Test StressTestRunner configuration
        runner = StressTestRunner(auto_save=True)
        
        if runner.results_manager:
            results_path = runner.results_manager.base_results_dir
            print(f"📁 StressTestRunner results path: {results_path}")
            
            # Check if path is correct
            expected_path = Path(__file__).parent / "tidal_protocol_sim" / "results"
            if str(results_path) == str(expected_path):
                print("✅ Results path is correctly configured!")
                print(f"   Results will be saved in: {results_path}")
            else:
                print("❌ Results path configuration issue")
                print(f"   Expected: {expected_path}")
                print(f"   Actual: {results_path}")
        else:
            print("❌ Results manager not initialized")
            
        # Test direct ResultsManager configuration
        print("\n" + "-" * 40)
        print("Testing direct ResultsManager configuration:")
        
        # Test default (should be repo root)
        default_manager = ResultsManager()
        print(f"📁 Default ResultsManager path: {default_manager.base_results_dir}")
        
        # Test configured path
        correct_path = Path(__file__).parent / "tidal_protocol_sim" / "results"
        configured_manager = ResultsManager(str(correct_path))
        print(f"📁 Configured ResultsManager path: {configured_manager.base_results_dir}")
        
        print("\n" + "=" * 60)
        print("VERIFICATION COMPLETE")
        print("=" * 60)
        
        print("\nTo test the High Tide scenario with correct results path:")
        print("python tidal_protocol_sim/main.py --scenario High_Tide_BTC_Decline")
        print("\nResults will be saved in:")
        print("tidal_protocol_sim/results/High_Tide_BTC_Decline/run_XXX_YYYYMMDD_HHMMSS/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_results_path()
    sys.exit(0 if success else 1)
