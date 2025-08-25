#!/usr/bin/env python3
"""
Test High Tide Math Verification

Quick test to verify that the High Tide agent math is working correctly
with proper collateral factors and health factor calculations.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_high_tide_math():
    """Test the High Tide agent math calculations"""
    
    print("=" * 60)
    print("HIGH TIDE MATH VERIFICATION")
    print("=" * 60)
    
    try:
        from tidal_protocol_sim.agents.high_tide_agent import HighTideAgent
        from tidal_protocol_sim.core.protocol import Asset
        
        # Test parameters
        btc_initial_price = 100_000.0
        btc_collateral_factor = 0.80
        
        print(f"Initial BTC Price: ${btc_initial_price:,.0f}")
        print(f"BTC Collateral Factor: {btc_collateral_factor:.1%}")
        print(f"Collateral Deposited: 1.0 BTC")
        print(f"Collateral Value: ${btc_initial_price:,.0f}")
        print(f"Effective Collateral: ${btc_initial_price * btc_collateral_factor:,.0f}")
        print()
        
        # Test different risk profiles
        test_cases = [
            ("Conservative", 2.2, 2.0),
            ("Moderate", 1.6, 1.4), 
            ("Aggressive", 1.4, 1.3)
        ]
        
        print("Testing Agent Math:")
        print("-" * 40)
        
        for profile, target_hf, maintenance_hf in test_cases:
            print(f"\n{profile} Agent:")
            print(f"  Target Initial HF: {target_hf:.1f}")
            print(f"  Maintenance HF: {maintenance_hf:.1f}")
            
            # Create agent
            agent = HighTideAgent(f"test_{profile.lower()}", target_hf, maintenance_hf)
            
            # Check calculations
            effective_collateral = btc_initial_price * btc_collateral_factor
            expected_debt = effective_collateral / target_hf
            actual_debt = agent.state.moet_debt
            
            print(f"  Expected Debt: ${expected_debt:,.0f}")
            print(f"  Actual Debt: ${actual_debt:,.0f}")
            print(f"  Math Check: {'✅ PASS' if abs(expected_debt - actual_debt) < 0.01 else '❌ FAIL'}")
            
            # Calculate actual initial HF
            asset_prices = {Asset.BTC: btc_initial_price, Asset.MOET: 1.0}
            agent._update_health_factor(asset_prices)
            actual_hf = agent.state.health_factor
            
            print(f"  Calculated HF: {actual_hf:.2f}")
            print(f"  HF Check: {'✅ PASS' if abs(actual_hf - target_hf) < 0.01 else '❌ FAIL'}")
            
        print()
        print("=" * 60)
        print("VERIFICATION COMPLETE")
        print("=" * 60)
        
        # Now test agent summary table
        print("\nTesting Agent Summary Table Math:")
        print("-" * 40)
        
        from tidal_protocol_sim.analysis.agent_summary_table import AgentSummaryTableGenerator
        
        # Create mock results data
        mock_results = {
            "agent_outcomes": [
                {
                    "agent_id": "test_agent_1",
                    "risk_profile": "conservative",
                    "target_health_factor": 2.2,
                    "final_health_factor": 1.8,
                    "initial_debt": 80000 / 2.2,  # $36,364
                    "final_debt": 37000,
                    "interest_accrued": 636,
                    "total_yield_sold": 2000,
                    "yield_token_value": 34000,
                    "net_position_value": 75000,
                    "cost_of_liquidation": 25000,
                    "survived": True,
                    "rebalancing_events": 2,
                    "emergency_liquidations": 0
                }
            ],
            "btc_price_history": [100000, 99000, 98000, 77000],  # Decline to $77k
            "agent_health_history": [
                {
                    "minute": 0,
                    "agents": [{"agent_id": "test_agent_1"}]
                }
            ]
        }
        
        generator = AgentSummaryTableGenerator()
        table = generator.generate_agent_summary_table(mock_results, btc_initial_price)
        
        if not table.empty:
            print("✅ Agent summary table generated successfully")
            print("\nSample row:")
            for col in table.columns:
                if len(table) > 0:
                    print(f"  {col}: {table.iloc[0][col]}")
        else:
            print("❌ Agent summary table generation failed")
            
        return True
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_high_tide_math()
    sys.exit(0 if success else 1)
