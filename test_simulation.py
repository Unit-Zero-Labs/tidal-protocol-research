#!/usr/bin/env python3
"""
Simple test script for Tidal Protocol Simulation
"""

def test_imports():
    """Test that all required modules can be imported"""
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy import stats
        print("✓ All scientific libraries imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import scientific libraries: {e}")
        return False
    
    try:
        from tidal_protocol_simulation import (
            TidalProtocol, MonteCarloSimulator, Asset, 
            ProtocolState, UniswapV3Pool
        )
        print("✓ Tidal protocol simulation imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import tidal protocol simulation: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of the simulation"""
    try:
        from tidal_protocol_simulation import TidalProtocol, Asset
        
        # Create protocol instance
        protocol = TidalProtocol()
        print("✓ Protocol instance created successfully")
        
        # Test basic calculations
        utilization = protocol.calculate_utilization(Asset.ETH)
        print(f"✓ ETH utilization calculated: {utilization:.2%}")
        
        # Test price shock
        protocol.apply_price_shock(Asset.ETH, -0.10)
        new_price = protocol.asset_prices[Asset.ETH]
        expected_price = 3000 * 0.9  # 10% drop from $3000
        print(f"✓ Price shock applied: ETH price changed to ${new_price:.2f}")
        
        # Test state retrieval
        state = protocol.get_protocol_state()
        print("✓ Protocol state retrieved successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_monte_carlo():
    """Test Monte Carlo simulation with minimal runs"""
    try:
        from tidal_protocol_simulation import TidalProtocol, MonteCarloSimulator
        
        protocol = TidalProtocol()
        simulator = MonteCarloSimulator(protocol, n_simulations=10)
        
        print("✓ Monte Carlo simulator created successfully")
        
        # Run minimal simulation
        results = simulator.run_simulation()
        print(f"✓ Monte Carlo simulation completed: {len(results)} results")
        
        # Test summary statistics
        summary = simulator.generate_summary_statistics()
        print("✓ Summary statistics generated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Monte Carlo test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Testing Tidal Protocol Simulation")
    print("="*40)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Monte Carlo Simulation", test_monte_carlo)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if test_func():
            print(f"✓ {test_name} PASSED")
            passed += 1
        else:
            print(f"✗ {test_name} FAILED")
    
    print(f"\n" + "="*40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The simulation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    main()
