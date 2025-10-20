#!/usr/bin/env python3
"""
Debug script to check arbitrage agent data collection
"""

import json
from pathlib import Path

def check_arbitrage_data():
    """Check the latest simulation results for arbitrage data"""
    
    results_dir = Path("tidal_protocol_sim/results/Full_Year_2024_BTC_Simulation_10min_leverage")
    json_files = list(results_dir.glob("*.json"))
    
    if not json_files:
        print("❌ No JSON files found")
        return
    
    # Get the latest file
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    print(f"📁 Checking: {latest_file.name}")
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
        
        print("\n🔍 CHECKING DATA STRUCTURE:")
        
        # Check if moet_system_state exists
        if "moet_system_state" in data:
            moet_state = data["moet_system_state"]
            print("✅ moet_system_state found")
            
            # Check for arbitrage_agents_summary
            if "arbitrage_agents_summary" in moet_state:
                agents = moet_state["arbitrage_agents_summary"]
                print(f"✅ arbitrage_agents_summary found with {len(agents)} agents")
                
                for i, agent in enumerate(agents):
                    print(f"   Agent {i+1}: {agent.get('agent_id', 'unknown')}")
                    print(f"     Total attempts: {agent.get('total_attempts', 0)}")
                    print(f"     Total profit: ${agent.get('total_profit', 0):.2f}")
                    
            else:
                print("❌ arbitrage_agents_summary NOT found in moet_system_state")
                print("   Available keys:", list(moet_state.keys()))
                
        else:
            print("❌ moet_system_state NOT found")
            print("   Available top-level keys:", list(data.keys()))
            
        # Check for arbitrage agents in agent_outcomes
        if "agent_outcomes" in data:
            outcomes = data["agent_outcomes"]
            arbitrage_outcomes = [o for o in outcomes if o.get("agent_type") == "moet_arbitrage_agent"]
            print(f"\n📊 Found {len(arbitrage_outcomes)} arbitrage agents in agent_outcomes")
            
            for outcome in arbitrage_outcomes:
                print(f"   {outcome.get('agent_id', 'unknown')}: ${outcome.get('total_profit', 0):.2f} profit")
        
        # Check tracking_data
        if "moet_system_state" in data and "tracking_data" in data["moet_system_state"]:
            tracking = data["moet_system_state"]["tracking_data"]
            print(f"\n📈 TRACKING DATA:")
            print(f"   Reserve history entries: {len(tracking.get('reserve_history', []))}")
            print(f"   Arbitrage history entries: {len(tracking.get('arbitrage_history', []))}")
            print(f"   Pool price history entries: {len(tracking.get('pool_price_history', []))}")
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    check_arbitrage_data()

