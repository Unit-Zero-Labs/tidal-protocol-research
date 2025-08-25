#!/usr/bin/env python3
"""
High Tide Scenario Demo

Quick demonstration of the High Tide scenario with BTC price decline and active rebalancing.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from tidal_protocol_sim.simulation.high_tide_engine import HighTideSimulationEngine, HighTideConfig
    from tidal_protocol_sim.stress_testing.runner import StressTestRunner
    from tidal_protocol_sim.simulation.config import SimulationConfig
    from tidal_protocol_sim.analysis.agent_summary_table import AgentSummaryTableGenerator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def run_high_tide_demo():
    """Run a demonstration of the High Tide scenario"""
    
    print("=" * 60)
    print("HIGH TIDE SCENARIO DEMONSTRATION")
    print("=" * 60)
    print()
    
    print("Setting up High Tide simulation...")
    print("• 20 agents with varied risk profiles")
    print("• BTC decline from $100,000 to ~$80,000")
    print("• Active yield token rebalancing enabled")
    print("• 60-minute simulation duration")
    print()
    
    try:
        # Create High Tide configuration
        config = HighTideConfig()
        config.num_high_tide_agents = 20
        config.btc_decline_duration = 60
        config.btc_final_price_range = (75_000, 85_000)
        
        print("Creating High Tide simulation engine...")
        engine = HighTideSimulationEngine(config)
        
        print(f"Initialized {len(engine.high_tide_agents)} High Tide agents:")
        
        # Show agent distribution
        risk_counts = {"conservative": 0, "moderate": 0, "aggressive": 0}
        for agent in engine.high_tide_agents:
            risk_counts[agent.risk_profile] += 1
            
        for profile, count in risk_counts.items():
            print(f"  • {profile.title()}: {count} agents")
        
        print()
        print("Running High Tide simulation...")
        print("-" * 40)
        
        # Run the simulation
        results = engine.run_high_tide_simulation()
        
        # Generate position tracker CSV if available
        if hasattr(engine, 'position_tracker') and engine.position_tracker.tracking_data:
            import pandas as pd
            from pathlib import Path
            
            output_dir = Path("tidal_protocol_sim/results/High_Tide_BTC_Decline")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame(engine.position_tracker.tracking_data)
            tracker_file = output_dir / f"agent_position_tracker_{engine.position_tracker.agent_id}.csv"
            df.to_csv(tracker_file, index=False)
            print(f"💾 Position tracker saved: {tracker_file}")
            
            # Print tracking summary
            engine.position_tracker.print_tracking_summary()
        
        print()
        print("=" * 60)
        print("HIGH TIDE SIMULATION RESULTS")
        print("=" * 60)
        
        # Display key results
        btc_stats = results.get("btc_decline_statistics", {})
        survival_stats = results.get("survival_statistics", {})
        cost_analysis = results.get("cost_analysis", {})
        yield_activity = results.get("yield_token_activity", {})
        
        print(f"\n🏦 BTC Price Movement:")
        print(f"   Initial Price: ${btc_stats.get('initial_price', 0):,.0f}")
        print(f"   Final Price: ${btc_stats.get('final_price', 0):,.0f}")
        print(f"   Total Decline: {btc_stats.get('total_decline_percent', 0):.1f}%")
        print(f"   Duration: {btc_stats.get('duration_minutes', 0)} minutes")
        
        print(f"\n👥 Agent Survival:")
        print(f"   Total Agents: {survival_stats.get('total_agents', 0)}")
        print(f"   Survivors: {survival_stats.get('survivors', 0)}")
        print(f"   Survival Rate: {survival_stats.get('survival_rate', 0)*100:.1f}%")
        
        survival_by_profile = survival_stats.get('survival_by_risk_profile', {})
        for profile, count in survival_by_profile.items():
            total = risk_counts.get(profile, 0)
            rate = (count / total * 100) if total > 0 else 0
            print(f"     • {profile.title()}: {count}/{total} ({rate:.1f}%)")
        
        print(f"\n💰 Cost Analysis:")
        print(f"   Total Cost of Liquidation: ${cost_analysis.get('total_cost_of_liquidation', 0):,.0f}")
        print(f"   Average Cost per Agent: ${cost_analysis.get('average_cost_per_agent', 0):,.0f}")
        
        cost_by_profile = cost_analysis.get('cost_by_risk_profile', {})
        for profile, data in cost_by_profile.items():
            if data['agent_count'] > 0:
                print(f"     • {profile.title()}: ${data['average_cost']:,.0f} avg")
        
        print(f"\n🔄 Yield Token Activity:")
        print(f"   Total Purchases: ${yield_activity.get('total_purchases', 0):,.0f}")
        print(f"   Rebalancing Sales: ${yield_activity.get('total_rebalancing_sales', 0):,.0f}")
        print(f"   Total Trades: {yield_activity.get('total_trades', 0)}")
        print(f"   Rebalancing Events: {yield_activity.get('rebalancing_events', 0)}")
        
        print(f"\n📊 Charts and Analysis:")
        print(f"   Results saved to: tidal_protocol_sim/results/High_Tide_BTC_Decline/")
        print(f"   Generated {len(results.get('charts', []))} visualization charts")
        
        # Generate and display agent summary table
        print(f"\n📋 Agent Summary Table:")
        print("-" * 40)
        
        try:
            table_generator = AgentSummaryTableGenerator()
            agent_table = table_generator.generate_agent_summary_table(results)
            
            if not agent_table.empty:
                table_generator.print_agent_summary_table(agent_table, max_width=180)
            else:
                print("   No agent summary data available")
                
        except Exception as e:
            print(f"   Error generating agent summary table: {e}")
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print()
        print("The High Tide scenario demonstrates active position management")
        print("with yield token rebalancing to minimize liquidation costs during")
        print("BTC price declines. Compare these results with traditional")
        print("Aave-style liquidation mechanisms.")
        print()
        
        return results
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_stress_test_comparison():
    """Run High Tide scenario through the stress test framework"""
    
    print("\n" + "=" * 60)
    print("STRESS TEST FRAMEWORK INTEGRATION")
    print("=" * 60)
    
    try:
        # Create stress test runner
        runner = StressTestRunner(auto_save=True)
        
        print("Running High Tide scenario through stress test framework...")
        
        # Run the High Tide scenario
        results = runner.run_targeted_scenario("High_Tide_BTC_Decline")
        
        print("✅ High Tide scenario completed through stress test framework")
        print(f"📁 Results automatically saved with charts and analysis")
        
        return results
        
    except Exception as e:
        print(f"❌ Stress test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the direct demo first
    demo_results = run_high_tide_demo()
    
    # Then demonstrate stress test integration
    if demo_results:
        stress_results = run_stress_test_comparison()
        
        if stress_results:
            print("\n🎉 High Tide scenario implementation complete!")
            print("Ready for Monte Carlo analysis and comparative studies.")
