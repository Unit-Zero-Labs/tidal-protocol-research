#!/usr/bin/env python3
"""
Comprehensive Single Agent Tracking Test for High Tide Simulation

This test tracks the complete state of a single High Tide agent through the simulation,
including all rebalancing events, state changes, and final outcomes.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import random

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tidal_protocol_sim.engine.high_tide_vault_engine import HighTideVaultEngine, HighTideConfig
from tidal_protocol_sim.agents.high_tide_agent import HighTideAgent
from tidal_protocol_sim.core.protocol import TidalProtocol, Asset


class AgentStateTracker:
    """Tracks complete state of a single agent through the simulation"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.initial_state = {}
        self.final_state = {}
        self.state_history = []
        self.rebalancing_events = []
        self.btc_price_history = []
        self.health_factor_history = []
        self.debt_history = []
        self.collateral_history = []
        self.yield_token_history = []
        self.net_position_history = []
        
    def capture_initial_state(self, agent: HighTideAgent, btc_price: float):
        """Capture the initial state of the agent AFTER yield token minting"""
        # Calculate effective collateral value
        btc_collateral_factor = 0.80
        effective_collateral = agent.state.btc_amount * btc_price * btc_collateral_factor
        
        # Calculate initial yield token value (should be available after minute 0 minting)
        initial_yield_token_value = 0.0
        if hasattr(agent, 'state') and hasattr(agent.state, 'yield_token_manager'):
            yt_manager = agent.state.yield_token_manager
            if hasattr(yt_manager, 'calculate_total_value'):
                initial_yield_token_value = yt_manager.calculate_total_value(0)  # Minute 0
        
        initial_net_position = (agent.state.btc_amount * btc_price) + (initial_yield_token_value - agent.state.moet_debt)
        
        self.initial_state = {
            "agent_id": agent.agent_id,
            "initial_health_factor": agent.state.initial_health_factor,
            "target_health_factor": agent.state.target_health_factor,
            "btc_amount": agent.state.btc_amount,
            "initial_btc_price": btc_price,
            "initial_collateral_value": agent.state.btc_amount * btc_price,
            "initial_moet_debt": agent.state.moet_debt,
            "initial_effective_collateral": effective_collateral,
            "initial_health_factor_calculated": agent.state.health_factor,
            "initial_yield_token_value": initial_yield_token_value,
            "initial_net_position": initial_net_position,
            "timestamp": 0
        }
        
        # Initialize history with initial values
        self.btc_price_history.append(btc_price)
        self.health_factor_history.append(agent.state.health_factor)
        self.debt_history.append(agent.state.moet_debt)
        self.collateral_history.append(agent.state.btc_amount * btc_price)
        self.yield_token_history.append(initial_yield_token_value)
        self.net_position_history.append(initial_net_position)
        
    def capture_state_snapshot(self, agent: HighTideAgent, btc_price: float, timestamp: int):
        """Capture a snapshot of the agent's state at a specific timestamp"""
        # Calculate effective collateral and net position
        btc_collateral_factor = 0.80
        effective_collateral = agent.state.btc_amount * btc_price * btc_collateral_factor
        current_yield_token_value = 0.0
        
        # Get yield token value if available
        if hasattr(agent, 'state') and hasattr(agent.state, 'yield_token_manager'):
            yt_manager = agent.state.yield_token_manager
            if hasattr(yt_manager, 'calculate_total_value'):
                current_yield_token_value = yt_manager.calculate_total_value(timestamp)
        
        net_position = (agent.state.btc_amount * btc_price) + (current_yield_token_value - agent.state.moet_debt)
        
        snapshot = {
            "timestamp": timestamp,
            "btc_price": btc_price,
            "btc_amount": agent.state.btc_amount,
            "collateral_value": agent.state.btc_amount * btc_price,
            "moet_debt": agent.state.moet_debt,
            "effective_collateral": effective_collateral,
            "health_factor": agent.state.health_factor,
            "yield_token_value": current_yield_token_value,
            "net_position": net_position,
            "rebalancing_triggered": False,
            "rebalancing_event": None
        }
        
        # Update yield token value if available
        if hasattr(agent, 'state') and hasattr(agent.state, 'yield_token_manager'):
            yt_manager = agent.state.yield_token_manager
            if hasattr(yt_manager, 'get_portfolio_summary'):
                portfolio = yt_manager.get_portfolio_summary(timestamp)
                snapshot["yield_token_value"] = portfolio.get("total_current_value", 0.0)
                snapshot["yield_tokens_held"] = portfolio.get("total_yield_tokens", 0.0)
                snapshot["yield_tokens_sold"] = portfolio.get("total_yield_sold", 0.0)
                snapshot["total_accrued_yield"] = portfolio.get("total_accrued_yield", 0.0)
        
        self.state_history.append(snapshot)
        
        # Update history arrays
        self.btc_price_history.append(btc_price)
        self.health_factor_history.append(agent.state.health_factor)
        self.debt_history.append(agent.state.moet_debt)
        self.collateral_history.append(agent.state.btc_amount * btc_price)
        self.yield_token_history.append(snapshot["yield_token_value"])
        self.net_position_history.append(net_position)
        
        return snapshot
    
    def capture_rebalancing_event(self, event_data: Dict, timestamp: int):
        """Capture details of a rebalancing event"""
        rebalancing_event = {
            "timestamp": timestamp,
            "event_type": "rebalancing",
            "trigger_health_factor": event_data.get("trigger_health_factor", 0.0),
            "target_health_factor": event_data.get("target_health_factor", 0.0),
            "yield_tokens_sold": event_data.get("yield_tokens_sold", 0.0),
            "moet_received": event_data.get("moet_received", 0.0),
            "debt_paid_down": event_data.get("debt_paid_down", 0.0),
            "slippage_cost": event_data.get("slippage_cost", 0.0),
            "trading_fee": event_data.get("trading_fee", 0.0),
            "health_factor_before": event_data.get("health_factor_before", 0.0),
            "health_factor_after": event_data.get("health_factor_after", 0.0),
            "debt_before": event_data.get("debt_before", 0.0),
            "debt_after": event_data.get("debt_after", 0.0),
            "yield_token_value_before": event_data.get("yield_token_value_before", 0.0),
            "yield_token_value_after": event_data.get("yield_token_value_after", 0.0),
            "net_position_before": event_data.get("net_position_before", 0.0),
            "net_position_after": event_data.get("net_position_after", 0.0)
        }
        
        self.rebalancing_events.append(rebalancing_event)
        
        # Mark the corresponding state snapshot as having a rebalancing event
        if self.state_history:
            self.state_history[-1]["rebalancing_triggered"] = True
            self.state_history[-1]["rebalancing_event"] = rebalancing_event
    
    def capture_final_state(self, agent: HighTideAgent, btc_price: float, timestamp: int):
        """Capture the final state of the agent"""
        final_yield_token_value = 0.0
        final_yield_tokens_held = 0.0
        final_yield_tokens_sold = 0.0
        final_total_accrued_yield = 0.0
        
        if hasattr(agent, 'state') and hasattr(agent.state, 'yield_token_manager'):
            yt_manager = agent.state.yield_token_manager
            if hasattr(yt_manager, 'get_portfolio_summary'):
                portfolio = yt_manager.get_portfolio_summary(timestamp)
                final_yield_token_value = portfolio.get("total_current_value", 0.0)
                final_yield_tokens_held = portfolio.get("total_yield_tokens", 0.0)
                final_yield_tokens_sold = portfolio.get("total_yield_sold", 0.0)
                final_total_accrued_yield = portfolio.get("total_accrued_yield", 0.0)
        
        # Calculate final values
        btc_collateral_factor = 0.80
        final_effective_collateral = agent.state.btc_amount * btc_price * btc_collateral_factor
        final_net_position = (agent.state.btc_amount * btc_price) + (final_yield_token_value - agent.state.moet_debt)
        
        self.final_state = {
            "agent_id": agent.agent_id,
            "final_health_factor": agent.state.health_factor,
            "final_btc_price": btc_price,
            "final_btc_amount": agent.state.btc_amount,
            "final_collateral_value": agent.state.btc_amount * btc_price,
            "final_moet_debt": agent.state.moet_debt,
            "final_effective_collateral": final_effective_collateral,
            "final_yield_token_value": final_yield_token_value,
            "final_yield_tokens_held": final_yield_tokens_held,
            "final_yield_tokens_sold": final_yield_tokens_sold,
            "final_total_accrued_yield": final_total_accrued_yield,
            "final_net_position": final_net_position,
            "total_rebalancing_events": len(self.rebalancing_events),
            "total_yield_tokens_sold": sum(event["yield_tokens_sold"] for event in self.rebalancing_events),
            "total_moet_received": sum(event["moet_received"] for event in self.rebalancing_events),
            "total_debt_paid_down": sum(event["debt_paid_down"] for event in self.rebalancing_events),
            "total_slippage_costs": sum(event["slippage_cost"] for event in self.rebalancing_events),
            "total_trading_fees": sum(event["trading_fee"] for event in self.rebalancing_events),
            "survived": agent.state.health_factor > 1.0,
            "timestamp": timestamp
        }
    
    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the agent's journey"""
        return {
            "agent_id": self.agent_id,
            "initial_state": self.initial_state,
            "final_state": self.final_state,
            "state_history": self.state_history,
            "rebalancing_events": self.rebalancing_events,
            "time_series_data": {
                "btc_price_history": self.btc_price_history,
                "health_factor_history": self.health_factor_history,
                "debt_history": self.debt_history,
                "collateral_history": self.collateral_history,
                "yield_token_history": self.yield_token_history,
                "net_position_history": self.net_position_history
            },
            "performance_metrics": self._calculate_performance_metrics()
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from the tracked data"""
        if not self.state_history:
            return {}
        
        initial_state = self.initial_state
        final_state = self.final_state
        
        # Calculate changes
        health_factor_change = final_state["final_health_factor"] - initial_state["initial_health_factor"]
        debt_change = final_state["final_moet_debt"] - initial_state["initial_moet_debt"]
        collateral_value_change = final_state["final_collateral_value"] - initial_state["initial_collateral_value"]
        net_position_change = final_state["final_net_position"] - initial_state["initial_net_position"]
        
        # Calculate costs
        total_rebalancing_cost = sum(event["slippage_cost"] + event["trading_fee"] for event in self.rebalancing_events)
        total_yield_opportunity_cost = sum(event["yield_tokens_sold"] for event in self.rebalancing_events)
        
        # Calculate efficiency metrics
        rebalancing_efficiency = 0.0
        if total_rebalancing_cost > 0:
            rebalancing_efficiency = (sum(event["debt_paid_down"] for event in self.rebalancing_events) / total_rebalancing_cost)
        
        return {
            "health_factor_change": health_factor_change,
            "debt_change": debt_change,
            "collateral_value_change": collateral_value_change,
            "net_position_change": net_position_change,
            "total_rebalancing_cost": total_rebalancing_cost,
            "total_yield_opportunity_cost": total_yield_opportunity_cost,
            "rebalancing_efficiency": rebalancing_efficiency,
            "survival_status": final_state["survived"],
            "total_rebalancing_events": len(self.rebalancing_events),
            "average_health_factor": np.mean(self.health_factor_history) if self.health_factor_history else 0.0,
            "min_health_factor": min(self.health_factor_history) if self.health_factor_history else 0.0,
            "max_health_factor": max(self.health_factor_history) if self.health_factor_history else 0.0
        }


class ComprehensiveAgentTrackingTest:
    """Main test class for comprehensive single agent tracking"""
    
    def __init__(self):
        self.config = self._create_test_config()
        self.tracker = None
        self.results = {}
        
    def _create_test_config(self) -> HighTideConfig:
        """Create test configuration"""
        config = HighTideConfig()
        
        # Single agent configuration
        config.num_high_tide_agents = 1
        
        # BTC price decline scenario
        config.btc_decline_duration = 60  # 60 minutes
        config.btc_initial_price = 100_000.0
        config.btc_final_price = 76_342.50  # 23.66% decline
        
        # Pool configurations
        config.moet_btc_pool_size = 250_000
        config.moet_btc_concentration = 0.80
        config.moet_yield_pool_size = 250_000
        config.yield_token_concentration = 0.95
        
        # Yield token parameters
        config.use_direct_minting_for_initial = True
        
        return config
    
    def run_comprehensive_tracking_test(self, initial_hf: float = 1.25, target_hf: float = 1.05) -> Dict[str, Any]:
        """Run comprehensive tracking test for a single agent"""
        
        print("=" * 80)
        print("COMPREHENSIVE SINGLE AGENT TRACKING TEST")
        print("=" * 80)
        print(f"Initial Health Factor: {initial_hf}")
        print(f"Target Health Factor: {target_hf}")
        print(f"BTC Price Decline: ${self.config.btc_initial_price:,.0f} → ${self.config.btc_final_price:,.0f}")
        print()
        
        # Create custom agent
        agent = HighTideAgent(
            agent_id="tracking_test_agent",
            initial_hf=initial_hf,
            target_hf=target_hf
        )
        
        # Initialize tracker
        self.tracker = AgentStateTracker(agent.agent_id)
        
        # Create engine
        engine = HighTideVaultEngine(self.config)
        
        # Replace the default agent with our custom agent
        engine.high_tide_agents = [agent]
        engine.agents = {agent.agent_id: agent}
        
        # Connect agent to yield token pool
        if hasattr(agent, 'state') and hasattr(agent.state, 'yield_token_manager'):
            agent.state.yield_token_manager.yield_token_pool = engine.yield_token_pool
        
        # Run simulation with detailed tracking
        print("🚀 Running simulation with detailed tracking...")
        self._run_simulation_with_tracking(engine, agent)
        
        # Capture final state
        self.tracker.capture_final_state(agent, self.config.btc_final_price, self.config.btc_decline_duration)
        
        # Generate comprehensive results
        self.results = self.tracker.get_comprehensive_summary()
        
        # Generate reports and visualizations
        self._generate_comprehensive_report()
        self._generate_detailed_charts()
        self._save_detailed_data()
        
        print("✅ Comprehensive tracking test completed!")
        return self.results
    
    def _run_simulation_with_tracking(self, engine: HighTideVaultEngine, agent: HighTideAgent):
        """Run simulation with detailed state tracking"""
        
        # Store original methods
        original_process_agents = engine._process_high_tide_agents
        original_record_metrics = engine._record_high_tide_metrics
        
        def tracked_process_agents(minute: int):
            # Run the original agent processing
            result = original_process_agents(minute)
            
            # Capture state snapshot after agent processing
            current_btc_price = engine.state.current_prices.get(Asset.BTC, 100_000.0)
            self.tracker.capture_state_snapshot(agent, current_btc_price, minute)
            
            # Capture initial state after minute 0 (when yield tokens are minted)
            if minute == 0:
                self.tracker.capture_initial_state(agent, current_btc_price)
                print(f"📊 Initial State Captured (after yield token minting):")
                print(f"   Health Factor: {agent.state.health_factor:.4f}")
                print(f"   BTC Amount: {agent.state.btc_amount:.6f}")
                print(f"   Collateral Value: ${agent.state.btc_amount * current_btc_price:,.2f}")
                print(f"   MOET Debt: ${agent.state.moet_debt:,.2f}")
                
                # Calculate initial yield token value
                initial_yield_token_value = 0.0
                if hasattr(agent, 'state') and hasattr(agent.state, 'yield_token_manager'):
                    yt_manager = agent.state.yield_token_manager
                    if hasattr(yt_manager, 'calculate_total_value'):
                        initial_yield_token_value = yt_manager.calculate_total_value(minute)
                
                initial_net_position = (agent.state.btc_amount * current_btc_price) + (initial_yield_token_value - agent.state.moet_debt)
                print(f"   Yield Token Value: ${initial_yield_token_value:,.2f}")
                print(f"   Net Position: ${initial_net_position:,.2f}")
                print()
            
            # Check for rebalancing events in the agent's state
            if hasattr(agent.state, 'rebalancing_events') and agent.state.rebalancing_events:
                # Get the latest rebalancing event
                latest_event = agent.state.rebalancing_events[-1] if agent.state.rebalancing_events else None
                if latest_event and latest_event.get('minute') == minute:
                    # Convert the event data to match our expected format
                    event_data = {
                        "trigger_health_factor": latest_event.get("health_factor_before", 0.0),
                        "target_health_factor": agent.state.target_health_factor,
                        "yield_tokens_sold": latest_event.get("yield_tokens_sold_value", 0.0),
                        "moet_received": latest_event.get("moet_raised", 0.0),
                        "debt_paid_down": latest_event.get("debt_repaid", 0.0),
                        "slippage_cost": latest_event.get("slippage_cost", 0.0),
                        "trading_fee": 0.0,  # Not tracked in current implementation
                        "health_factor_before": latest_event.get("health_factor_before", 0.0),
                        "health_factor_after": agent.state.health_factor,
                        "debt_before": latest_event.get("debt_before", 0.0),
                        "debt_after": agent.state.moet_debt
                    }
                    self.tracker.capture_rebalancing_event(event_data, minute)
            
            return result
        
        def tracked_record_metrics(minute: int):
            # Run the original metrics recording
            original_record_metrics(minute)
            
            # Additional tracking if needed
            pass
        
        # Replace methods
        engine._process_high_tide_agents = tracked_process_agents
        engine._record_high_tide_metrics = tracked_record_metrics
        
        # Run the simulation
        try:
            engine.run_simulation()
        finally:
            # Restore original methods
            engine._process_high_tide_agents = original_process_agents
            engine._record_high_tide_metrics = original_record_metrics
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive text report"""
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE AGENT TRACKING REPORT")
        print("=" * 80)
        
        initial = self.results["initial_state"]
        final = self.results["final_state"]
        metrics = self.results["performance_metrics"]
        
        print(f"\n📋 AGENT SUMMARY")
        print(f"Agent ID: {initial['agent_id']}")
        print(f"Survived: {'✅ YES' if final['survived'] else '❌ NO'}")
        print(f"Total Rebalancing Events: {final['total_rebalancing_events']}")
        
        print(f"\n💰 INITIAL STATE")
        print(f"Initial Health Factor: {initial['initial_health_factor']:.4f}")
        print(f"Target Health Factor: {initial['target_health_factor']:.4f}")
        print(f"BTC Amount: {initial['btc_amount']:.6f}")
        print(f"Initial BTC Price: ${initial['initial_btc_price']:,.2f}")
        print(f"Initial Collateral Value: ${initial['initial_collateral_value']:,.2f}")
        print(f"Initial MOET Debt: ${initial['initial_moet_debt']:,.2f}")
        print(f"Initial Net Position: ${initial['initial_net_position']:,.2f}")
        
        print(f"\n💰 FINAL STATE")
        print(f"Final Health Factor: {final['final_health_factor']:.4f}")
        print(f"Final BTC Price: ${final['final_btc_price']:,.2f}")
        print(f"Final Collateral Value: ${final['final_collateral_value']:,.2f}")
        print(f"Final MOET Debt: ${final['final_moet_debt']:,.2f}")
        print(f"Final Yield Token Value: ${final['final_yield_token_value']:,.2f}")
        print(f"Final Net Position: ${final['final_net_position']:,.2f}")
        
        print(f"\n📊 PERFORMANCE METRICS")
        print(f"Health Factor Change: {metrics['health_factor_change']:+.4f}")
        print(f"Debt Change: ${metrics['debt_change']:+,.2f}")
        print(f"Net Position Change: ${metrics['net_position_change']:+,.2f}")
        print(f"Total Rebalancing Cost: ${metrics['total_rebalancing_cost']:,.2f}")
        print(f"Total Yield Opportunity Cost: ${metrics['total_yield_opportunity_cost']:,.2f}")
        print(f"Rebalancing Efficiency: {metrics['rebalancing_efficiency']:.4f}")
        print(f"Average Health Factor: {metrics['average_health_factor']:.4f}")
        print(f"Min Health Factor: {metrics['min_health_factor']:.4f}")
        print(f"Max Health Factor: {metrics['max_health_factor']:.4f}")
        
        print(f"\n🔄 REBALANCING EVENTS")
        if self.results["rebalancing_events"]:
            for i, event in enumerate(self.results["rebalancing_events"], 1):
                print(f"Event {i} (Minute {event['timestamp']}):")
                print(f"  Trigger HF: {event['trigger_health_factor']:.4f}")
                print(f"  Target HF: {event['target_health_factor']:.4f}")
                print(f"  Yield Tokens Sold: {event['yield_tokens_sold']:,.2f}")
                print(f"  MOET Received: ${event['moet_received']:,.2f}")
                print(f"  Debt Paid Down: ${event['debt_paid_down']:,.2f}")
                print(f"  Slippage Cost: ${event['slippage_cost']:,.2f}")
                print(f"  Trading Fee: ${event['trading_fee']:,.2f}")
                print(f"  HF Before: {event['health_factor_before']:.4f}")
                print(f"  HF After: {event['health_factor_after']:.4f}")
                print()
        else:
            print("No rebalancing events occurred.")
        
        print(f"\n📈 TIME SERIES SUMMARY")
        print(f"Total Time Steps: {len(self.results['time_series_data']['btc_price_history'])}")
        print(f"BTC Price Range: ${min(self.results['time_series_data']['btc_price_history']):,.2f} - ${max(self.results['time_series_data']['btc_price_history']):,.2f}")
        print(f"Health Factor Range: {min(self.results['time_series_data']['health_factor_history']):.4f} - {max(self.results['time_series_data']['health_factor_history']):.4f}")
        print(f"Debt Range: ${min(self.results['time_series_data']['debt_history']):,.2f} - ${max(self.results['time_series_data']['debt_history']):,.2f}")
    
    def _generate_detailed_charts(self):
        """Generate detailed visualization charts"""
        
        output_dir = Path("tidal_protocol_sim/results/Agent_Tracking_Test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📊 Generating detailed charts...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive dashboard
        self._create_comprehensive_dashboard(output_dir)
        self._create_rebalancing_events_chart(output_dir)
        self._create_health_factor_evolution_chart(output_dir)
        self._create_portfolio_evolution_chart(output_dir)
        
        print(f"📊 Charts saved to: {output_dir}")
    
    def _create_comprehensive_dashboard(self, output_dir: Path):
        """Create comprehensive dashboard with multiple subplots"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive Agent Tracking Dashboard', fontsize=16, fontweight='bold')
        
        time_steps = list(range(len(self.results['time_series_data']['btc_price_history'])))
        
        # Chart 1: BTC Price and Health Factor
        ax1_twin = ax1.twinx()
        line1 = ax1.plot(time_steps, self.results['time_series_data']['btc_price_history'], 'b-', label='BTC Price', linewidth=2)
        line2 = ax1_twin.plot(time_steps, self.results['time_series_data']['health_factor_history'], 'r-', label='Health Factor', linewidth=2)
        
        # Mark rebalancing events
        for event in self.results['rebalancing_events']:
            ax1.axvline(x=event['timestamp'], color='orange', linestyle='--', alpha=0.7)
            ax1_twin.axvline(x=event['timestamp'], color='orange', linestyle='--', alpha=0.7)
        
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('BTC Price ($)', color='b')
        ax1_twin.set_ylabel('Health Factor', color='r')
        ax1.set_title('BTC Price and Health Factor Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Debt and Collateral Value
        ax2.plot(time_steps, self.results['time_series_data']['debt_history'], 'r-', label='MOET Debt', linewidth=2)
        ax2.plot(time_steps, self.results['time_series_data']['collateral_history'], 'g-', label='Collateral Value', linewidth=2)
        
        # Mark rebalancing events
        for event in self.results['rebalancing_events']:
            ax2.axvline(x=event['timestamp'], color='orange', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Value ($)')
        ax2.set_title('Debt and Collateral Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Yield Token Value and Net Position
        ax3.plot(time_steps, self.results['time_series_data']['yield_token_history'], 'purple', label='Yield Token Value', linewidth=2)
        ax3.plot(time_steps, self.results['time_series_data']['net_position_history'], 'orange', label='Net Position', linewidth=2)
        
        # Mark rebalancing events
        for event in self.results['rebalancing_events']:
            ax3.axvline(x=event['timestamp'], color='red', linestyle='--', alpha=0.7)
        
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Value ($)')
        ax3.set_title('Yield Token and Net Position Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Rebalancing Events Summary
        if self.results['rebalancing_events']:
            event_timestamps = [event['timestamp'] for event in self.results['rebalancing_events']]
            yield_tokens_sold = [event['yield_tokens_sold'] for event in self.results['rebalancing_events']]
            moet_received = [event['moet_received'] for event in self.results['rebalancing_events']]
            
            ax4_twin = ax4.twinx()
            bars1 = ax4.bar(event_timestamps, yield_tokens_sold, alpha=0.7, label='Yield Tokens Sold', color='blue')
            bars2 = ax4_twin.bar(event_timestamps, moet_received, alpha=0.7, label='MOET Received', color='green')
            
            ax4.set_xlabel('Time (minutes)')
            ax4.set_ylabel('Yield Tokens Sold', color='blue')
            ax4_twin.set_ylabel('MOET Received ($)', color='green')
            ax4.set_title('Rebalancing Events Summary')
        else:
            ax4.text(0.5, 0.5, 'No Rebalancing Events', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Rebalancing Events Summary')
        
        plt.tight_layout()
        plt.savefig(output_dir / "comprehensive_agent_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_rebalancing_events_chart(self, output_dir: Path):
        """Create detailed rebalancing events chart"""
        
        if not self.results['rebalancing_events']:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Rebalancing Events Analysis', fontsize=16, fontweight='bold')
        
        events = self.results['rebalancing_events']
        event_numbers = list(range(1, len(events) + 1))
        
        # Chart 1: Yield Tokens Sold and MOET Received
        ax1_twin = ax1.twinx()
        bars1 = ax1.bar(event_numbers, [e['yield_tokens_sold'] for e in events], alpha=0.7, label='Yield Tokens Sold', color='blue')
        bars2 = ax1_twin.bar(event_numbers, [e['moet_received'] for e in events], alpha=0.7, label='MOET Received', color='green')
        
        ax1.set_xlabel('Rebalancing Event')
        ax1.set_ylabel('Yield Tokens Sold', color='blue')
        ax1_twin.set_ylabel('MOET Received ($)', color='green')
        ax1.set_title('Yield Tokens Sold vs MOET Received')
        ax1.set_xticks(event_numbers)
        
        # Chart 2: Health Factor Before and After
        ax2.bar([x - 0.2 for x in event_numbers], [e['health_factor_before'] for e in events], 0.4, label='HF Before', alpha=0.7, color='red')
        ax2.bar([x + 0.2 for x in event_numbers], [e['health_factor_after'] for e in events], 0.4, label='HF After', alpha=0.7, color='green')
        
        ax2.set_xlabel('Rebalancing Event')
        ax2.set_ylabel('Health Factor')
        ax2.set_title('Health Factor Before and After Rebalancing')
        ax2.legend()
        ax2.set_xticks(event_numbers)
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Costs Breakdown
        slippage_costs = [e['slippage_cost'] for e in events]
        trading_fees = [e['trading_fee'] for e in events]
        total_costs = [s + t for s, t in zip(slippage_costs, trading_fees)]
        
        ax3.bar([x - 0.2 for x in event_numbers], slippage_costs, 0.4, label='Slippage Cost', alpha=0.7, color='orange')
        ax3.bar([x + 0.2 for x in event_numbers], trading_fees, 0.4, label='Trading Fee', alpha=0.7, color='purple')
        
        ax3.set_xlabel('Rebalancing Event')
        ax3.set_ylabel('Cost ($)')
        ax3.set_title('Rebalancing Costs Breakdown')
        ax3.legend()
        ax3.set_xticks(event_numbers)
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Debt Reduction
        debt_before = [e['debt_before'] for e in events]
        debt_after = [e['debt_after'] for e in events]
        debt_reduction = [b - a for b, a in zip(debt_before, debt_after)]
        
        ax4.bar(event_numbers, debt_reduction, alpha=0.7, label='Debt Reduction', color='green')
        ax4.set_xlabel('Rebalancing Event')
        ax4.set_ylabel('Debt Reduction ($)')
        ax4.set_title('Debt Reduction per Rebalancing Event')
        ax4.set_xticks(event_numbers)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "rebalancing_events_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_health_factor_evolution_chart(self, output_dir: Path):
        """Create health factor evolution chart"""
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Health Factor Evolution Analysis', fontsize=16, fontweight='bold')
        
        time_steps = list(range(len(self.results['time_series_data']['health_factor_history'])))
        hf_history = self.results['time_series_data']['health_factor_history']
        
        # Chart 1: Health Factor over time
        ax1.plot(time_steps, hf_history, 'b-', linewidth=2, label='Health Factor')
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Liquidation Threshold')
        ax1.axhline(y=self.results['initial_state']['target_health_factor'], color='green', linestyle='--', alpha=0.7, label='Target Health Factor')
        
        # Mark rebalancing events
        for event in self.results['rebalancing_events']:
            ax1.axvline(x=event['timestamp'], color='orange', linestyle=':', alpha=0.7)
            ax1.annotate(f"Rebalance {event['timestamp']}", 
                        xy=(event['timestamp'], event['health_factor_after']), 
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Health Factor')
        ax1.set_title('Health Factor Evolution with Rebalancing Events')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Health Factor distribution (filter out infinite values)
        finite_hf_history = [hf for hf in hf_history if hf != float('inf') and hf != float('-inf')]
        if finite_hf_history:
            ax2.hist(finite_hf_history, bins=20, alpha=0.7, color='blue', edgecolor='black')
        else:
            ax2.text(0.5, 0.5, 'No finite health factor data', ha='center', va='center', transform=ax2.transAxes)
        ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Liquidation Threshold')
        ax2.axvline(x=self.results['initial_state']['target_health_factor'], color='green', linestyle='--', alpha=0.7, label='Target Health Factor')
        if finite_hf_history:
            mean_hf = np.mean(finite_hf_history)
            ax2.axvline(x=mean_hf, color='orange', linestyle='-', alpha=0.7, label=f'Mean: {mean_hf:.3f}')
        
        ax2.set_xlabel('Health Factor')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Health Factor Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "health_factor_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_portfolio_evolution_chart(self, output_dir: Path):
        """Create portfolio evolution chart"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Portfolio Evolution Analysis', fontsize=16, fontweight='bold')
        
        time_steps = list(range(len(self.results['time_series_data']['btc_price_history'])))
        
        # Chart 1: Portfolio Components
        ax1.plot(time_steps, self.results['time_series_data']['collateral_history'], 'g-', label='Collateral Value', linewidth=2)
        ax1.plot(time_steps, self.results['time_series_data']['debt_history'], 'r-', label='MOET Debt', linewidth=2)
        ax1.plot(time_steps, self.results['time_series_data']['yield_token_history'], 'purple', label='Yield Token Value', linewidth=2)
        
        # Mark rebalancing events
        for event in self.results['rebalancing_events']:
            ax1.axvline(x=event['timestamp'], color='orange', linestyle='--', alpha=0.7)
        
        ax1.set_xlabel('Time (minutes)')
        ax1.set_ylabel('Value ($)')
        ax1.set_title('Portfolio Components Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Net Position Evolution
        ax2.plot(time_steps, self.results['time_series_data']['net_position_history'], 'orange', linewidth=2, label='Net Position')
        
        # Mark rebalancing events
        for event in self.results['rebalancing_events']:
            ax2.axvline(x=event['timestamp'], color='red', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Net Position Value ($)')
        ax2.set_title('Net Position Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: BTC Price Impact
        btc_prices = self.results['time_series_data']['btc_price_history']
        ax3.plot(time_steps, btc_prices, 'b-', linewidth=2, label='BTC Price')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('BTC Price ($)')
        ax3.set_title('BTC Price Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Portfolio Value Breakdown (Final State)
        final_state = self.results['final_state']
        components = ['Collateral', 'Yield Tokens', 'Debt']
        values = [
            final_state['final_collateral_value'],
            final_state['final_yield_token_value'],
            -final_state['final_moet_debt']  # Negative for debt
        ]
        colors = ['green', 'purple', 'red']
        
        bars = ax4.bar(components, values, color=colors, alpha=0.7)
        ax4.set_ylabel('Value ($)')
        ax4.set_title('Final Portfolio Value Breakdown')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.annotate(f'${value:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(output_dir / "portfolio_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_detailed_data(self):
        """Save detailed data to JSON and CSV files"""
        
        output_dir = Path("tidal_protocol_sim/results/Agent_Tracking_Test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive JSON data
        json_path = output_dir / "comprehensive_agent_tracking_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📁 Detailed JSON data saved to: {json_path}")
        
        # Save state history CSV
        if self.results['state_history']:
            state_df = pd.DataFrame(self.results['state_history'])
            csv_path = output_dir / "agent_state_history.csv"
            state_df.to_csv(csv_path, index=False)
            print(f"📁 State history CSV saved to: {csv_path}")
        
        # Save rebalancing events CSV
        if self.results['rebalancing_events']:
            events_df = pd.DataFrame(self.results['rebalancing_events'])
            events_csv_path = output_dir / "rebalancing_events.csv"
            events_df.to_csv(events_csv_path, index=False)
            print(f"📁 Rebalancing events CSV saved to: {events_csv_path}")
        
        # Save time series data CSV
        time_series_data = self.results['time_series_data']
        time_series_df = pd.DataFrame({
            'timestamp': list(range(len(time_series_data['btc_price_history']))),
            'btc_price': time_series_data['btc_price_history'],
            'health_factor': time_series_data['health_factor_history'],
            'debt': time_series_data['debt_history'],
            'collateral_value': time_series_data['collateral_history'],
            'yield_token_value': time_series_data['yield_token_history'],
            'net_position': time_series_data['net_position_history']
        })
        time_series_csv_path = output_dir / "time_series_data.csv"
        time_series_df.to_csv(time_series_csv_path, index=False)
        print(f"📁 Time series data CSV saved to: {time_series_csv_path}")


def main():
    """Main execution function"""
    print("Comprehensive Single Agent Tracking Test")
    print("=" * 50)
    print()
    print("This test will:")
    print("• Track a single High Tide agent through the simulation")
    print("• Capture complete state at each time step")
    print("• Record all rebalancing events in detail")
    print("• Generate comprehensive reports and visualizations")
    print()
    
    # Create and run test
    test = ComprehensiveAgentTrackingTest()
    
    # Test with different scenarios
    scenarios = [
        {"initial_hf": 1.25, "target_hf": 1.05, "name": "Moderate"},
        {"initial_hf": 1.15, "target_hf": 1.01, "name": "Aggressive"},
        {"initial_hf": 1.35, "target_hf": 1.10, "name": "Conservative"}
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Running {scenario['name']} Scenario")
        print(f"Initial HF: {scenario['initial_hf']}, Target HF: {scenario['target_hf']}")
        print(f"{'='*60}")
        
        results = test.run_comprehensive_tracking_test(
            initial_hf=scenario['initial_hf'],
            target_hf=scenario['target_hf']
        )
        
        print(f"\n✅ {scenario['name']} scenario completed!")
        print(f"Final Health Factor: {results['final_state']['final_health_factor']:.4f}")
        print(f"Survived: {'Yes' if results['final_state']['survived'] else 'No'}")
        print(f"Rebalancing Events: {results['final_state']['total_rebalancing_events']}")
        print(f"Total Cost: ${results['performance_metrics']['total_rebalancing_cost']:,.2f}")
    
    print("\n🎉 All tracking tests completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
