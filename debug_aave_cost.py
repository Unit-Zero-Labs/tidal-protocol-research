#!/usr/bin/env python3
"""
Debug script to understand AAVE cost calculation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tidal_protocol_sim.simulation.aave_engine import AaveSimulationEngine, AaveConfig

def main():
    print("=" * 60)
    print("DEBUGGING AAVE COST CALCULATION")
    print("=" * 60)
    
    # Run a single AAVE simulation
    config = AaveConfig()
    config.num_aave_agents = 20  # Fixed number for debugging
    config.btc_decline_duration = 60
    config.btc_final_price_range = (75_000, 80_000)
    
    engine = AaveSimulationEngine(config)
    results = engine.run_aave_simulation()
    
    print("\n🔍 DEBUGGING COST CALCULATION:")
    print("-" * 40)
    
    # Get cost analysis
    cost_analysis = results.get("cost_analysis", {})
    agent_outcomes = results.get("agent_outcomes", [])
    liquidation_events = results.get("liquidation_activity", {}).get("liquidation_events", [])
    
    print(f"Total agents: {len(agent_outcomes)}")
    print(f"Total liquidation events: {len(liquidation_events)}")
    
    # Show liquidation events details
    total_penalties_from_events = 0
    print(f"\n📊 Liquidation Events:")
    for i, event in enumerate(liquidation_events):
        penalty = event.get("liquidation_bonus_value", 0)
        total_penalties_from_events += penalty
        print(f"  Event {i+1}: Agent {event.get('agent_id', 'unknown')}, Penalty: ${penalty:,.2f}")
    
    print(f"\nTotal penalties from events: ${total_penalties_from_events:,.2f}")
    
    # Show agent outcomes
    liquidated_agents = []
    total_penalties_from_agents = 0
    print(f"\n📊 Agent Outcomes:")
    for outcome in agent_outcomes:
        agent_id = outcome.get("agent_id", "unknown")
        liquidation_count = outcome.get("liquidation_events", 0)
        penalty = outcome.get("liquidation_penalties", 0)
        
        if liquidation_count > 0:
            liquidated_agents.append({
                "id": agent_id,
                "penalty": penalty,
                "events": liquidation_count
            })
            total_penalties_from_agents += penalty
            print(f"  {agent_id}: {liquidation_count} events, Penalty: ${penalty:,.2f}")
    
    print(f"\nTotal penalties from agent outcomes: ${total_penalties_from_agents:,.2f}")
    print(f"Liquidated agents count: {len(liquidated_agents)}")
    
    # Show cost analysis results
    print(f"\n📊 Cost Analysis Results:")
    print(f"  average_cost_per_agent: ${cost_analysis.get('average_cost_per_agent', 0):,.2f}")
    print(f"  average_cost_per_all_agents: ${cost_analysis.get('average_cost_per_all_agents', 0):,.2f}")
    print(f"  total_cost_of_liquidation: ${cost_analysis.get('total_cost_of_liquidation', 0):,.2f}")
    
    # Manual calculation
    if liquidated_agents:
        manual_avg = sum(agent["penalty"] for agent in liquidated_agents) / len(liquidated_agents)
        print(f"\n🧮 Manual Calculation:")
        print(f"  Sum of penalties: ${sum(agent['penalty'] for agent in liquidated_agents):,.2f}")
        print(f"  Number of liquidated agents: {len(liquidated_agents)}")
        print(f"  Manual average: ${manual_avg:,.2f}")
    else:
        print(f"\n🧮 No liquidations occurred in this simulation")

if __name__ == "__main__":
    main()
