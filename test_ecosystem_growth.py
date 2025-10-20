#!/usr/bin/env python3
"""
Test script for Ecosystem Growth simulation

This script demonstrates how to run a full year simulation with ecosystem growth enabled.
The simulation will gradually add agents over time to reach $150M in BTC deposits.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from sim_tests.full_year_sim import FullYearSimulation, FullYearSimConfig

def main():
    """Run ecosystem growth simulation"""
    
    print("🌱 ECOSYSTEM GROWTH SIMULATION")
    print("=" * 50)
    print()
    print("This simulation will:")
    print("• Start with 125 agents (standard simulation)")
    print("• Gradually add new agents over the year")
    print("• Target $150M in total BTC deposits by year end")
    print("• Each new agent deposits 1 BTC")
    print("• Growth starts after 30 days")
    print("• Results saved in separate 'Ecosystem_Growth' folder")
    print()
    
    # Create configuration
    config = FullYearSimConfig()
    
    # Create simulation
    simulation = FullYearSimulation(config)
    
    # Enable ecosystem growth
    simulation.enable_ecosystem_growth(
        target_btc_deposits=150_000_000,  # $150M target
        growth_start_delay_days=30,       # Start growth after 30 days
        growth_acceleration_factor=1.2    # Exponential growth factor
    )
    
    print("\n🚀 Starting ecosystem growth simulation...")
    print("This will take 10-15 minutes and generate comprehensive charts.")
    print()
    
    try:
        # Run the simulation
        results = simulation.run_test()
        
        # Print ecosystem growth summary
        sim_results = results.get("simulation_results", {})
        growth_summary = sim_results.get("ecosystem_growth_summary", {})
        
        if growth_summary:
            print("\n🌱 ECOSYSTEM GROWTH RESULTS:")
            print(f"   Final agent count: {growth_summary.get('final_agent_count', 0):,}")
            print(f"   Final BTC deposits: {growth_summary.get('final_btc_deposits', 0):.1f} BTC")
            print(f"   Final USD value: ${growth_summary.get('final_usd_value', 0):,.0f}")
            print(f"   Target achievement: {growth_summary.get('target_achievement', 0)*100:.1f}%")
            print(f"   New agents added: {growth_summary.get('total_new_agents', 0):,}")
        
        # Print agent performance summary
        agent_perf = results.get("agent_performance", {}).get("summary", {})
        if agent_perf:
            print(f"\n📊 AGENT PERFORMANCE:")
            print(f"   Survival rate: {agent_perf.get('survival_rate', 0)*100:.1f}%")
            print(f"   Average APY: {agent_perf.get('average_apy', 0)*100:.1f}%")
            print(f"   Total slippage: ${agent_perf.get('total_slippage', 0):,.2f}")
        
        # Print arbitrage summary
        moet_system = sim_results.get("moet_system_state", {})
        if moet_system:
            arb_summary = moet_system.get("arbitrage_agents_summary", [])
            if arb_summary and isinstance(arb_summary, list):
                total_volume = sum(agent.get('total_volume_traded', 0) for agent in arb_summary)
                total_attempts = sum(agent.get('total_attempts', 0) for agent in arb_summary)
                print(f"\n⚡ ARBITRAGE ACTIVITY:")
                print(f"   Total attempts: {total_attempts:,}")
                print(f"   Total volume: ${total_volume:,.0f}")
                print(f"   Agents active: {len(arb_summary)}")
        
        print(f"\n✅ Simulation completed successfully!")
        print(f"📁 Results saved in: tidal_protocol_sim/results/{config.test_name}_Ecosystem_Growth/")
        print(f"📊 Charts available in the charts/ subdirectory")
        
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

