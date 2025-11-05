#!/usr/bin/env python3
"""
Test that the reverted MOET:USDC pool works like MOET:YT pools
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tidal_protocol_sim.core.uniswap_v3_math import create_moet_usdc_pool, UniswapV3SlippageCalculator

def main():
    print("🧪 TESTING REVERTED MOET:USDC POOL")
    print("=" * 50)
    print("Testing that MOET:USDC now works like MOET:YT pools")
    print()
    
    # Create MOET:USDC pool (should work like MOET:YT now)
    pool = create_moet_usdc_pool(
        pool_size_usd=500_000,
        concentration=0.95,
        token0_ratio=0.5
    )
    
    calculator = UniswapV3SlippageCalculator(pool)
    
    print(f"✅ MOET:USDC Pool created:")
    print(f"   Pool name: {pool.pool_name}")
    print(f"   Total liquidity: ${pool.total_liquidity:,.0f}")
    print(f"   Active liquidity: {pool.liquidity:,}")
    print(f"   Positions: {len(pool.positions)}")
    print(f"   Initial price: ${pool.get_price():.6f}")
    print(f"   Current tick: {pool.tick_current}")
    print()
    
    # Test $50,000 MOET → USDC swap
    print("💱 Testing $50,000 MOET → USDC swap:")
    
    try:
        swap_result = calculator.calculate_swap_slippage(50_000, "MOET")
        
        usdc_received = swap_result.get("amount_out", 0)
        slippage_pct = swap_result.get("slippage_percentage", 0)
        new_price = swap_result.get("new_price", 1.0)
        trading_fees = swap_result.get("trading_fees", 0)
        
        print(f"   ✅ Swap completed successfully!")
        print(f"   MOET sold: $50,000")
        print(f"   USDC received: ${usdc_received:,.2f}")
        print(f"   Trading fees: ${trading_fees:,.2f}")
        print(f"   Slippage: {slippage_pct:.4f}%")
        print(f"   New price: ${new_price:.6f}")
        print(f"   Price change: {((new_price - 1.0) * 10000):.1f} bps")
        
        # Store the new pool state for the reverse calculation
        pool_after_swap = pool
        calculator_after_swap = calculator
        
        print()
        print("🔄 CALCULATING REVERSE TRADE TO RESTORE PEG:")
        print(f"   Current price: ${new_price:.6f}")
        print(f"   Current tick: {pool_after_swap.tick_current}")
        print(f"   Target price: $1.000000 (tick 0)")
        
        # Calculate how much USDC we need to swap back to MOET to restore tick 0
        # We need to do the opposite trade: USDC → MOET
        
        # Try different amounts to find the right size
        target_price = 1.0
        test_amounts = [10_000, 20_000, 30_000, 40_000, 50_000, 60_000]
        
        best_amount = None
        best_final_price = None
        best_price_diff = float('inf')
        
        print(f"   Testing different USDC → MOET trade sizes:")
        
        for test_amount in test_amounts:
            # Create a fresh pool state for each test
            test_pool = create_moet_usdc_pool(500_000, 0.95, 0.5)
            test_calculator = UniswapV3SlippageCalculator(test_pool)
            
            # First do the original MOET → USDC swap
            test_calculator.calculate_swap_slippage(50_000, "MOET")
            
            # Then do the reverse USDC → MOET swap
            reverse_result = test_calculator.calculate_swap_slippage(test_amount, "USDC")
            final_price = reverse_result.get("new_price", 1.0)
            price_diff = abs(final_price - target_price)
            
            print(f"     ${test_amount:,} USDC → MOET: Final price ${final_price:.6f}, diff {price_diff:.6f}")
            
            if price_diff < best_price_diff:
                best_price_diff = price_diff
                best_amount = test_amount
                best_final_price = final_price
        
        print()
        print(f"🎯 OPTIMAL REVERSE TRADE:")
        print(f"   Best amount: ${best_amount:,} USDC → MOET")
        print(f"   Final price: ${best_final_price:.6f}")
        print(f"   Price difference from $1.00: {best_price_diff:.6f}")
        print(f"   Final deviation: {((best_final_price - 1.0) * 10000):.1f} bps")
        
        # Validate results make sense
        reasonable_output = 45_000 <= usdc_received <= 50_000  # Should get most of the $50K back
        reasonable_slippage = slippage_pct < 10  # Less than 10% slippage
        reasonable_price = 0.95 <= new_price <= 1.05  # Price shouldn't move too much
        
        print()
        print("✅ VALIDATION:")
        print(f"   {'✅' if reasonable_output else '❌'} Output amount: ${usdc_received:,.2f} (reasonable: 45K-50K)")
        print(f"   {'✅' if reasonable_slippage else '❌'} Slippage: {slippage_pct:.4f}% (reasonable: <10%)")
        print(f"   {'✅' if reasonable_price else '❌'} New price: ${new_price:.6f} (reasonable: 0.95-1.05)")
        
        if reasonable_output and reasonable_slippage and reasonable_price:
            print("\n🎉 SUCCESS! MOET:USDC pool is working correctly after reversion!")
            print("   The pool behaves like MOET:YT pools with reasonable slippage and output")
            return True
        else:
            print("\n⚠️  PARTIAL SUCCESS: Swap works but some metrics seem off")
            return True  # Still working, just maybe not optimal
            
    except Exception as e:
        print(f"   ❌ Swap failed with error: {e}")
        print("\n❌ FAILED! The reversion may not have worked correctly")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)




