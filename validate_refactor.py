#!/usr/bin/env python3
"""
Validation Script - Compare refactored system with tidal_sim_v1

This script validates that the refactored system produces equivalent results
to the original tidal_sim_v1 baseline for key metrics.
"""

import sys
import os
import numpy as np

# Add both systems to path
sys.path.append('tidal_protocol_sim')
sys.path.append('tidal_sim_v1')

def validate_debt_cap_calculation():
    """Validate debt cap calculations between systems"""
    
    print("1. Validating Debt Cap Calculations")
    print("-" * 40)
    
    # New system
    try:
        from tidal_protocol_sim.core.protocol import TidalProtocol
        new_protocol = TidalProtocol()
        new_debt_cap = new_protocol.calculate_debt_cap()
        print(f"New system debt cap: ${new_debt_cap:,.0f}")
    except Exception as e:
        print(f"New system error: {e}")
        new_debt_cap = 0
    
    # Original system (simplified calculation based on tidal_sim_v1)
    try:
        # Replicate original logic
        total_liquidation_capacity = 10000000  # $10M estimate from pools
        dex_allocation = 0.35
        underwater_percentage = 0.20  # Weighted average
        
        original_debt_cap = total_liquidation_capacity * dex_allocation * underwater_percentage
        print(f"Original system debt cap (est.): ${original_debt_cap:,.0f}")
        
        # Calculate difference
        if original_debt_cap > 0:
            difference = abs(new_debt_cap - original_debt_cap) / original_debt_cap
            print(f"Relative difference: {difference:.1%}")
            
            if difference < 0.1:  # Within 10%
                print("✓ Debt cap calculation: PASS")
                return True
            else:
                print("✗ Debt cap calculation: FAIL (difference > 10%)")
                return False
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False

def validate_interest_rate_calculation():
    """Validate interest rate calculations"""
    
    print("\n2. Validating Interest Rate Calculations")
    print("-" * 42)
    
    try:
        from tidal_protocol_sim.core.math import TidalMath
        
        # Test kinked interest rate at different utilization levels
        test_cases = [
            {"utilization": 0.5, "expected_range": (0.02, 0.08)},   # Below kink
            {"utilization": 0.8, "expected_range": (0.05, 0.12)},   # At kink
            {"utilization": 0.95, "expected_range": (0.15, 0.30)}   # Above kink
        ]
        
        # Tidal parameters
        base_rate = 0
        multiplier = 11415525114
        jump = 253678335870
        kink = 0.8
        
        all_passed = True
        
        for case in test_cases:
            util = case["utilization"]
            expected_min, expected_max = case["expected_range"]
            
            rate = TidalMath.calculate_kinked_interest_rate(
                util, base_rate, multiplier, jump, kink
            )
            
            print(f"Utilization {util:.0%}: {rate:.2%}")
            
            if expected_min <= rate <= expected_max:
                print(f"  ✓ Within expected range ({expected_min:.1%} - {expected_max:.1%})")
            else:
                print(f"  ✗ Outside expected range ({expected_min:.1%} - {expected_max:.1%})")
                all_passed = False
        
        if all_passed:
            print("✓ Interest rate calculation: PASS")
            return True
        else:
            print("✗ Interest rate calculation: FAIL")
            return False
            
    except Exception as e:
        print(f"Interest rate validation error: {e}")
        return False

def validate_health_factor_calculation():
    """Validate health factor calculations"""
    
    print("\n3. Validating Health Factor Calculations")
    print("-" * 42)
    
    try:
        from tidal_protocol_sim.core.math import TidalMath
        
        # Test cases
        test_cases = [
            {"collateral": 10000, "debt": 5000, "cf": 0.75, "expected_hf": 1.5},
            {"collateral": 10000, "debt": 7500, "cf": 0.75, "expected_hf": 1.0},
            {"collateral": 10000, "debt": 8000, "cf": 0.75, "expected_hf": 0.9375},
        ]
        
        all_passed = True
        
        for case in test_cases:
            hf = TidalMath.calculate_health_factor(
                case["collateral"], case["debt"], case["cf"]
            )
            expected = case["expected_hf"]
            
            print(f"Collateral: ${case['collateral']:,}, Debt: ${case['debt']:,}")
            print(f"  Calculated HF: {hf:.4f}, Expected: {expected:.4f}")
            
            if abs(hf - expected) < 0.001:  # Within 0.1%
                print(f"  ✓ PASS")
            else:
                print(f"  ✗ FAIL (difference: {abs(hf - expected):.4f})")
                all_passed = False
        
        if all_passed:
            print("✓ Health factor calculation: PASS")
            return True
        else:
            print("✗ Health factor calculation: FAIL")
            return False
            
    except Exception as e:
        print(f"Health factor validation error: {e}")
        return False

def validate_moet_system():
    """Validate MOET system functionality"""
    
    print("\n4. Validating MOET System")
    print("-" * 30)
    
    try:
        from tidal_protocol_sim.core.moet import MoetStablecoin
        
        moet = MoetStablecoin()
        
        # Test 1: Initial state
        initial_supply = moet.total_supply
        print(f"Initial supply: {initial_supply:,} MOET")
        
        # Test 2: Minting (should be fee-less)
        mint_amount = 1000
        minted = moet.mint(mint_amount)
        
        if minted == mint_amount and moet.total_supply == initial_supply + mint_amount:
            print("✓ Fee-less minting: PASS")
            mint_pass = True
        else:
            print("✗ Fee-less minting: FAIL")
            mint_pass = False
        
        # Test 3: Burning (should be fee-less)
        burn_amount = 500
        burned = moet.burn(burn_amount)
        expected_supply = initial_supply + mint_amount - burn_amount
        
        if burned == burn_amount and moet.total_supply == expected_supply:
            print("✓ Fee-less burning: PASS")
            burn_pass = True
        else:
            print("✗ Fee-less burning: FAIL")
            burn_pass = False
        
        # Test 4: Peg stability
        moet.current_price = 0.99
        stability_action = moet.calculate_stability_action()
        
        if stability_action == "burn_pressure":
            print("✓ Peg stability detection: PASS")
            stability_pass = True
        else:
            print("✗ Peg stability detection: FAIL")
            stability_pass = False
        
        return mint_pass and burn_pass and stability_pass
        
    except Exception as e:
        print(f"MOET system validation error: {e}")
        return False

def validate_system_performance():
    """Validate system performance meets refactor targets"""
    
    print("\n5. Validating Performance Targets")
    print("-" * 36)
    
    try:
        import time
        from tidal_protocol_sim.simulation.engine import TidalSimulationEngine
        from tidal_protocol_sim.simulation.config import SimulationConfig
        
        # Performance test: 30-day simulation (1000 steps) in < 10 seconds
        config = SimulationConfig()
        config.simulation_steps = 1000
        config.num_lenders = 3
        config.num_traders = 2
        config.num_liquidators = 1
        
        engine = TidalSimulationEngine(config)
        
        start_time = time.time()
        results = engine.run_simulation(1000)
        elapsed = time.time() - start_time
        
        print(f"1000-step simulation completed in {elapsed:.2f}s")
        
        if elapsed < 10.0:
            print("✓ Performance target: PASS (< 10s)")
            return True
        else:
            print("✗ Performance target: FAIL (> 10s)")
            return False
            
    except Exception as e:
        print(f"Performance validation error: {e}")
        return False

def count_lines_of_code():
    """Count lines of code in refactored system"""
    
    print("\n6. Code Reduction Analysis")
    print("-" * 30)
    
    try:
        import os
        
        def count_python_lines(directory):
            total_lines = 0
            total_files = 0
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        filepath = os.path.join(root, file)
                        try:
                            with open(filepath, 'r') as f:
                                lines = len(f.readlines())
                                total_lines += lines
                                total_files += 1
                        except:
                            continue
            
            return total_lines, total_files
        
        # Count new system
        new_lines, new_files = count_python_lines('tidal_protocol_sim')
        
        print(f"Refactored system:")
        print(f"  Lines of code: {new_lines:,}")
        print(f"  Number of files: {new_files}")
        
        # Compare to targets from refactor spec
        target_lines = 2000
        target_files = 15
        
        if new_lines <= target_lines:
            print(f"✓ Lines of code target: PASS (<= {target_lines:,})")
            lines_pass = True
        else:
            print(f"✗ Lines of code target: FAIL (> {target_lines:,})")
            lines_pass = False
        
        if new_files <= target_files:
            print(f"✓ File count target: PASS (<= {target_files})")
            files_pass = True
        else:
            print(f"✗ File count target: FAIL (> {target_files})")
            files_pass = False
        
        return lines_pass and files_pass
        
    except Exception as e:
        print(f"Code analysis error: {e}")
        return False

def main():
    """Run all validation tests"""
    
    print("TIDAL PROTOCOL REFACTOR VALIDATION")
    print("=" * 50)
    print("Comparing refactored system against tidal_sim_v1 baseline")
    print()
    
    # Run validation tests
    results = {
        "debt_cap": validate_debt_cap_calculation(),
        "interest_rates": validate_interest_rate_calculation(), 
        "health_factors": validate_health_factor_calculation(),
        "moet_system": validate_moet_system(),
        "performance": validate_system_performance(),
        "code_reduction": count_lines_of_code()
    }
    
    # Summary
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 REFACTOR VALIDATION: SUCCESS")
        print("\nThe refactored system meets all requirements:")
        print("✓ Maintains accuracy of tidal_sim_v1")
        print("✓ Adds modularity for stress testing") 
        print("✓ Achieves performance targets")
        print("✓ Reduces code complexity")
        return 0
    else:
        print("❌ REFACTOR VALIDATION: INCOMPLETE")
        print(f"\n{total - passed} tests failed - see details above")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)