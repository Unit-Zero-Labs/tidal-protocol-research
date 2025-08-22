#!/usr/bin/env python3
"""
Test the proper constant product formula slippage calculation
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.math.tidal_math import TidalMath
from src.core.markets.tidal_protocol import TidalLiquidityPool
from src.core.simulation.primitives import Asset


def test_constant_product_slippage():
    """Test the constant product formula slippage calculation"""
    print("🧮 CONSTANT PRODUCT FORMULA SLIPPAGE TEST")
    print("=" * 60)
    
    # Test parameters based on MOET/ETH pool
    # $1.25M MOET, 284.09 ETH (~$1.25M at $4400/ETH) - $2.5M total liquidity
    moet_reserve = 1250000.0
    eth_reserve = 284.09
    
    print(f"Pool Reserves:")
    print(f"  MOET: {moet_reserve:,.0f}")
    print(f"  ETH: {eth_reserve:.2f}")
    print(f"  Expected Price: {moet_reserve/eth_reserve:.2f} MOET per ETH")
    print()
    
    # Test different swap sizes
    test_swaps = [1.0, 5.0, 10.0, 25.0, 50.0]  # ETH amounts
    
    print("Slippage Analysis:")
    print(f"{'ETH In':<8} {'MOET Out':<12} {'Actual Price':<12} {'Slippage %':<10} {'Fee':<8}")
    print("-" * 60)
    
    for eth_in in test_swaps:
        amount_out, fee_amount, slippage_percent, actual_price = TidalMath.calculate_constant_product_swap(
            amount_in=eth_in,
            reserve_in=eth_reserve,
            reserve_out=moet_reserve,
            fee_rate=0.003
        )
        
        print(f"{eth_in:<8.1f} {amount_out:<12,.0f} {actual_price:<12.2f} {slippage_percent:<10.2f} ${fee_amount*4400:<8.0f}")
    
    print()
    
    # Test Uniswap V3 concentrated liquidity
    print("🌊 UNISWAP V3 CONCENTRATED LIQUIDITY TEST")
    print("=" * 60)
    
    current_price = moet_reserve / eth_reserve
    price_std_dev = 0.10  # 10% standard deviation
    
    print(f"Current Price: {current_price:.2f} MOET per ETH")
    print(f"Price Range (±1σ): {current_price * (1-price_std_dev):.2f} - {current_price * (1+price_std_dev):.2f}")
    print()
    
    print("Concentrated Liquidity vs Normal AMM:")
    print(f"{'ETH In':<8} {'Normal Out':<12} {'Concentrated Out':<16} {'Improvement':<12}")
    print("-" * 60)
    
    for eth_in in test_swaps:
        # Normal AMM
        normal_out, _, normal_slippage, _ = TidalMath.calculate_constant_product_swap(
            eth_in, eth_reserve, moet_reserve, 0.003
        )
        
        # Concentrated liquidity
        conc_out, _, conc_slippage, _ = TidalMath.calculate_uniswap_v3_concentrated_liquidity(
            eth_in, eth_reserve, moet_reserve, current_price, price_std_dev, 0.003
        )
        
        improvement = ((conc_out - normal_out) / normal_out * 100) if normal_out > 0 else 0
        
        print(f"{eth_in:<8.1f} {normal_out:<12,.0f} {conc_out:<16,.0f} {improvement:<12.1f}%")


def test_tidal_liquidity_pools():
    """Test the updated Tidal liquidity pools"""
    print("\n🌊 TIDAL PROTOCOL UPDATED POOLS TEST")
    print("=" * 60)
    
    # Updated prices
    prices = {
        Asset.ETH: 4400.0,
        Asset.BTC: 118000.0,
        Asset.FLOW: 0.40,
        Asset.USDC: 1.0,
        Asset.MOET: 1.0
    }
    
    print("Updated Prices:")
    for asset, price in prices.items():
        print(f"  {asset.value}: ${price:,.2f}")
    print()
    
    # Test each pool
    pools = {
        "MOET/ETH": TidalLiquidityPool(
            pair_assets=(Asset.MOET, Asset.ETH),
            reserves={Asset.MOET: 1250000, Asset.ETH: 284.09}
        ),
        "MOET/BTC": TidalLiquidityPool(
            pair_assets=(Asset.MOET, Asset.BTC),
            reserves={Asset.MOET: 1250000, Asset.BTC: 10.59}
        ),
        "MOET/FLOW": TidalLiquidityPool(
            pair_assets=(Asset.MOET, Asset.FLOW),
            reserves={Asset.MOET: 1250000, Asset.FLOW: 3125000}
        ),
        "MOET/USDC": TidalLiquidityPool(
            pair_assets=(Asset.MOET, Asset.USDC),
            reserves={Asset.MOET: 1250000, Asset.USDC: 1250000}
        )
    }
    
    print("Pool Analysis:")
    print(f"{'Pool':<12} {'Liquidity':<12} {'Liquidation Cap':<15} {'Price':<12}")
    print("-" * 60)
    
    for pool_name, pool in pools.items():
        total_liquidity = 2500000  # $2.5M total liquidity
        liq_cap = pool.get_liquidation_capacity()
        
        # Calculate pool price
        reserves = list(pool.reserves.values())
        price = reserves[0] / reserves[1] if len(reserves) == 2 and reserves[1] > 0 else 0
        
        print(f"{pool_name:<12} ${total_liquidity/1e6:<11.1f}M ${liq_cap:<14,.0f} {price:<12.2f}")
    
    print()
    
    # Test liquidation capacity calculation
    total_liq_cap = sum(pool.get_liquidation_capacity() for pool in pools.values())
    print(f"Total Liquidation Capacity: ${total_liq_cap:,.0f}")
    print(f"Debt Cap (35% allocation): ${total_liq_cap * 0.35:,.0f}")


def main():
    """Run all slippage and pool tests"""
    test_constant_product_slippage()
    test_tidal_liquidity_pools()
    
    print("\n✅ All tests completed!")
    print("\nKey Improvements:")
    print("• Proper constant product formula: x * y = k")
    print("• Accurate slippage calculation: (P(actual) - P(expected)) / P(expected)")
    print("• Uniswap V3 concentrated liquidity with normal distribution")
    print("• Updated prices: ETH=$4400, BTC=$118K, FLOW=$0.40")
    print("• All pools have $2.5M total liquidity each")


if __name__ == "__main__":
    main()
