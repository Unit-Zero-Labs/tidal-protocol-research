#!/usr/bin/env python3
"""
Test Pool Exhaustion Directly
Test if the pool can actually be exhausted with the current fix
"""

from tidal_protocol_sim.core.yield_tokens import YieldTokenPool

def test_pool_exhaustion_directly():
    """Test if we can exhaust the pool with direct calls"""
    
    print("🧪 TESTING POOL EXHAUSTION DIRECTLY")
    print("=" * 50)
    
    # Create pool with $250K MOET reserve
    pool = YieldTokenPool(250_000, 0.95)
    
    print(f"Initial pool: $500K ($250K MOET + $250K YT)")
    print(f"Testing if we can extract more than $250K MOET...")
    print()
    
    total_extracted = 0
    trade_count = 0
    
    # Try to extract exactly what our scenarios extracted: ~$370K
    target_extraction = 370_000
    trade_size = 15_000  # $15K per trade
    
    while total_extracted < target_extraction and trade_count < 50:
        trade_count += 1
        
        print(f"Trade {trade_count}: Attempting to sell ${trade_size:,} YT for MOET")
        
        result = pool.execute_yield_token_sale(trade_size)
        
        if result <= 0:
            print(f"   ❌ TRADE FAILED - Pool exhausted after ${total_extracted:,}")
            print(f"   ✅ Pool correctly limited extraction at {(total_extracted/250_000)*100:.1f}% utilization")
            break
        else:
            total_extracted += result
            print(f"   ✅ Received ${result:,.2f} MOET (Total: ${total_extracted:,.2f})")
            
            # Check if we've exceeded the impossible threshold
            if total_extracted > 250_000:
                utilization = (total_extracted / 250_000) * 100
                print(f"   🚨 WARNING: Extracted ${total_extracted:,} > $250K reserve ({utilization:.1f}% utilization)")
                
                if total_extracted > 350_000:  # Way beyond limits
                    print(f"   ❌ CRITICAL: Pool is providing infinite liquidity!")
                    break
    
    print()
    print(f"🎯 FINAL RESULTS:")
    print(f"   Total Extracted: ${total_extracted:,.2f}")
    print(f"   Pool Utilization: {(total_extracted/250_000)*100:.1f}%")
    print(f"   Trades Completed: {trade_count}")
    
    if total_extracted <= 250_000:
        print(f"   ✅ Pool behavior is CORRECT - stayed within limits")
    else:
        print(f"   ❌ Pool behavior is BROKEN - infinite liquidity bug still exists")

if __name__ == "__main__":
    test_pool_exhaustion_directly()
