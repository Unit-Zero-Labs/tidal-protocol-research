#!/usr/bin/env python3
"""
AAVE Protocol Engine

Pure AAVE Protocol implementation with traditional liquidation mechanics
(50% collateral seizure + 5% bonus) when health factor crosses 1.0.
No Tidal dependencies - inherits only from BaseLendingEngine.
"""

import random
from typing import Dict, List, Optional
from .base_lending_engine import BaseLendingEngine, BaseLendingConfig
from .btc_price_manager import BTCPriceDeclineManager
from ..core.protocol import Asset
from ..core.yield_tokens import YieldTokenPool
from ..agents.aave_agent import AaveAgent, create_aave_agents
from ..agents.base_agent import BaseAgent, AgentAction
from ..analysis.agent_position_tracker import AgentPositionTracker


class AaveConfig(BaseLendingConfig):
    """Pure AAVE configuration - no Tidal dependencies"""
    
    def __init__(self):
        super().__init__()
        self.scenario_name = "AAVE_Protocol"
        
        # AAVE liquidation parameters
        self.liquidation_threshold = 0.85
        self.liquidation_bonus = 0.05  # 5%
        self.liquidation_percentage = 0.5  # 50%
        
        # Yield token parameters (for fair comparison)
        self.yield_apr = 0.10  # 10% APR
        self.moet_btc_pool_size = 500_000  # Same as Tidal for comparison
        
        # BTC price decline parameters (for fair comparison)
        self.btc_initial_price = 100_000.0
        self.btc_decline_duration = 60  # 60 minutes
        self.btc_final_price_range = (75_000.0, 85_000.0)  # 15-25% decline
        
        # Agent configuration
        self.num_aave_agents = 20
        self.monte_carlo_agent_variation = True
        
        # Disable rebalancing for AAVE scenario
        self.rebalancing_enabled = False
        
        # Override simulation parameters
        self.simulation_steps = self.btc_decline_duration
        self.price_update_frequency = 1  # Update every minute


class AaveProtocolEngine(BaseLendingEngine):
    """Pure AAVE Protocol implementation"""
    
    def __init__(self, config: AaveConfig):
        super().__init__(config)
        self.aave_config = config
        
        # AAVE uses traditional AMM pools, not Uniswap V3
        self._setup_aave_liquidation_pools()
        
        # Initialize AAVE specific components
        self.yield_token_pool = YieldTokenPool(config.moet_btc_pool_size)
        self.btc_price_manager = BTCPriceDeclineManager(
            initial_price=config.btc_initial_price,
            duration=config.btc_decline_duration,
            final_price_range=config.btc_final_price_range
        )
        
        # Replace agents with AAVE agents
        self.aave_agents = create_aave_agents(
            config.num_aave_agents,
            config.monte_carlo_agent_variation
        )
        
        # Add AAVE agents to main agents dict
        self.agents = {}  # Clear base agents
        for agent in self.aave_agents:
            self.agents[agent.agent_id] = agent
            
        # Initialize position tracker
        self.position_tracker = AgentPositionTracker(self.aave_agents[0].agent_id)
        self.position_tracker.start_tracking()
            
        # Initialize AAVE agent positions
        self._setup_aave_positions()
        
        # Enhanced tracking
        self.yield_token_trades = []
        self.agent_health_history = []
        self.btc_price_history = []
        
    def _setup_aave_liquidation_pools(self):
        """Setup AAVE's actual liquidation mechanisms"""
        # Traditional constant product AMM
        # 50% liquidation + 5% bonus
        self.liquidation_threshold = self.aave_config.liquidation_threshold
        self.liquidation_bonus = self.aave_config.liquidation_bonus
        self.liquidation_percentage = self.aave_config.liquidation_percentage
        
    def _setup_aave_positions(self):
        """Set up initial AAVE agent positions"""
        for agent in self.aave_agents:
            # Update protocol with agent's BTC collateral
            btc_pool = self.protocol.asset_pools[Asset.BTC]
            btc_pool.total_supplied += agent.state.btc_amount
            
            # Update protocol with agent's MOET debt
            self.protocol.moet_system.mint(agent.state.moet_debt)
            
            # Initialize agent's health factor
            self._update_agent_health_factor(agent)
            
    def run_simulation(self, steps: int = None) -> Dict:
        """Run AAVE simulation with BTC price decline and traditional liquidations"""
        if steps is None:
            steps = self.aave_config.btc_decline_duration
            
        print(f"Starting AAVE simulation with {len(self.aave_agents)} agents")
        print(f"BTC decline from ${self.btc_price_manager.initial_price:,.0f} to ~${self.btc_price_manager.target_final_price:,.0f}")
        print("AAVE liquidation: 50% collateral + 5% bonus when HF ≤ 1.0")
        
        for minute in range(steps):
            self.current_step = minute
            
            # Update BTC price (same as High Tide)
            new_btc_price = self.btc_price_manager.update_btc_price(minute)
            self.state.current_prices[Asset.BTC] = new_btc_price
            self.btc_price_history.append(new_btc_price)
            
            # Update protocol state
            self.protocol.current_block = minute
            self.protocol.accrue_interest()
            
            # Update agent debt interest
            self._update_agent_debt_interest(minute)
            
            # Process AAVE agent actions
            swap_data = self._process_aave_agents(minute)
            
            # Check for AAVE liquidations (key difference from High Tide)
            self._check_aave_liquidations(minute)
            
            # Record position tracking data
            tracked_agent = self._get_tracked_agent()
            if tracked_agent:
                agent_swap_data = swap_data.get(tracked_agent.agent_id, {})
                self.position_tracker.record_minute_data(
                    minute, new_btc_price, tracked_agent, self, agent_swap_data
                )
            
            # Record metrics
            self._record_aave_metrics(minute)
            
            if minute % 10 == 0:
                print(f"Minute {minute}: BTC = ${new_btc_price:,.0f}, Active agents: {self._count_active_agents()}")
                
        return self._generate_aave_results()
        
    def _update_agent_debt_interest(self, minute: int):
        """Update debt interest for all AAVE agents"""
        btc_pool = self.protocol.asset_pools.get(Asset.BTC)
        if not btc_pool:
            return
            
        borrow_rate = btc_pool.calculate_borrow_rate()
        
        for agent in self.aave_agents:
            if agent.active:
                agent.update_debt_interest(minute, borrow_rate)
    
    def _process_aave_agents(self, minute: int) -> Dict[str, Dict]:
        """Process AAVE agent actions for current minute"""
        swap_data = {}
        
        for agent in self.aave_agents:
            if not agent.active:
                continue
                
            # Get agent's decision
            protocol_state = self._get_protocol_state()
            protocol_state["current_step"] = minute
            
            action_type, params = agent.decide_action(protocol_state, self.state.current_prices)
            
            # Execute action and capture swap data
            success, agent_swap_data = self._execute_aave_action(agent, action_type, params, minute)
            
            # Store swap data for tracking
            if agent_swap_data:
                swap_data[agent.agent_id] = agent_swap_data
            
            # Record action
            self._record_agent_action(agent.agent_id, action_type, params)
            
        return swap_data
            
    def _execute_aave_action(self, agent: AaveAgent, action_type: AgentAction, params: dict, minute: int) -> tuple:
        """Execute AAVE specific actions (only initial yield token purchase)"""
        if action_type == AgentAction.SWAP:
            swap_type = params.get("action_type", "")
            
            if swap_type == "buy_yield_tokens":
                success = self._execute_yield_token_purchase(agent, params, minute)
                return success, None
                
        # AAVE agents don't do any other swaps (no rebalancing)
        return False, None
        
    def _execute_yield_token_purchase(self, agent: AaveAgent, params: dict, minute: int) -> bool:
        """Execute yield token purchase for agent (same as High Tide)"""
        moet_amount = params.get("moet_amount", 0.0)
        
        if moet_amount <= 0:
            return False
            
        success = agent.execute_yield_token_purchase(moet_amount, minute)
        
        if success:
            self.yield_token_pool.execute_yield_token_purchase(moet_amount)
            
            self.yield_token_trades.append({
                "minute": minute,
                "agent_id": agent.agent_id,
                "action": "purchase",
                "moet_amount": moet_amount,
                "agent_health_factor": agent.state.health_factor
            })
            
        return success
        
    def _execute_aave_liquidation(self, agent, collateral_asset, debt_amount):
        """Traditional AAVE liquidation: 50% collateral + 5% bonus"""
        liquidation_amount = debt_amount * self.liquidation_percentage  # 50% max
        bonus = liquidation_amount * self.liquidation_bonus  # 5% bonus
        return liquidation_amount + bonus
        
    def _check_aave_liquidations(self, minute: int):
        """Check for AAVE-style liquidations (HF ≤ 1.0)"""
        for agent in self.aave_agents:
            if not agent.active:
                continue
                
            # Update health factor
            agent._update_health_factor(self.state.current_prices)
            
            # Check if liquidation is needed (HF ≤ 1.0)
            if agent.state.health_factor <= 1.0:
                liquidation_event = agent.execute_aave_liquidation(minute, self.state.current_prices)
                
                if liquidation_event:
                    self.liquidation_events.append(liquidation_event)
        
    def _get_tracked_agent(self) -> Optional[AaveAgent]:
        """Get the agent being tracked for position analysis"""
        for agent in self.aave_agents:
            if agent.agent_id == self.position_tracker.agent_id:
                return agent
        return None
        
    def _count_active_agents(self) -> int:
        """Count number of active AAVE agents"""
        return sum(1 for agent in self.aave_agents if agent.active)
        
    def _record_aave_metrics(self, minute: int):
        """Record AAVE specific metrics"""
        # Base metrics
        super()._record_metrics()
        
        # AAVE specific metrics
        agent_health_data = []
        for agent in self.aave_agents:
            portfolio = agent.get_detailed_portfolio_summary(self.state.current_prices, minute)
            agent_health_data.append({
                "agent_id": agent.agent_id,
                "health_factor": agent.state.health_factor,
                "risk_profile": agent.risk_profile,
                "target_hf": agent.state.target_health_factor,
                "initial_hf": agent.state.initial_health_factor,
                "cost_of_liquidation": portfolio["cost_of_liquidation"],
                "net_position_value": portfolio["net_position_value"],
                "yield_token_value": portfolio["yield_token_portfolio"]["total_current_value"],
                "liquidation_events": portfolio["liquidation_events_count"],
                "liquidation_penalties": portfolio["liquidation_penalties"],
                "remaining_collateral": portfolio["btc_amount"]
            })
            
        self.agent_health_history.append({
            "minute": minute,
            "btc_price": self.state.current_prices[Asset.BTC],
            "agents": agent_health_data
        })
        
    def _generate_aave_results(self) -> dict:
        """Generate comprehensive AAVE simulation results"""
        base_results = super()._generate_results()
        
        # Calculate AAVE specific metrics
        final_minute = self.aave_config.btc_decline_duration - 1
        
        # Agent outcomes
        agent_outcomes = []
        total_cost_of_liquidation = 0.0
        total_liquidation_penalties = 0.0
        survival_by_risk_profile = {"conservative": 0, "moderate": 0, "aggressive": 0}
        
        for agent in self.aave_agents:
            agent._update_health_factor(self.state.current_prices)
            
            portfolio = agent.get_detailed_portfolio_summary(self.state.current_prices, final_minute)
            
            outcome = {
                "agent_id": agent.agent_id,
                "risk_profile": agent.risk_profile,
                "target_health_factor": agent.state.target_health_factor,
                "initial_health_factor": agent.state.initial_health_factor,
                "final_health_factor": agent.state.health_factor,
                "cost_of_liquidation": portfolio["cost_of_liquidation"],
                "net_position_value": portfolio["net_position_value"],
                "total_yield_earned": portfolio["yield_token_portfolio"]["total_accrued_yield"],
                "total_yield_sold": 0.0,  # AAVE agents don't sell yield tokens
                "liquidation_events": len(agent.get_liquidation_history()),
                "survived": agent.active and agent.state.health_factor > 1.0,
                "liquidation_penalties": portfolio["liquidation_penalties"],
                "yield_token_value": portfolio["yield_token_portfolio"]["total_current_value"],
                "remaining_collateral": portfolio["btc_amount"]
            }
            
            agent_outcomes.append(outcome)
            total_cost_of_liquidation += outcome["cost_of_liquidation"]
            total_liquidation_penalties += outcome["liquidation_penalties"]
            
            if outcome["survived"]:
                survival_by_risk_profile[agent.risk_profile] += 1
                
        # AAVE specific results
        aave_results = {
            "scenario_type": "AAVE_BTC_Decline",
            "btc_decline_statistics": self.btc_price_manager.get_decline_statistics(),
            "agent_outcomes": agent_outcomes,
            "survival_statistics": {
                "total_agents": len(self.aave_agents),
                "survivors": sum(1 for outcome in agent_outcomes if outcome["survived"]),
                "survival_rate": sum(1 for outcome in agent_outcomes if outcome["survived"]) / len(self.aave_agents),
                "survival_by_risk_profile": survival_by_risk_profile
            },
            "cost_analysis": {
                "total_cost_of_liquidation": total_cost_of_liquidation,
                "average_cost_per_agent": total_cost_of_liquidation / len(self.aave_agents),
                "total_liquidation_penalties": total_liquidation_penalties,
                "average_liquidation_penalty": total_liquidation_penalties / len(self.aave_agents)
            },
            "liquidation_activity": {
                "total_liquidation_events": len(self.liquidation_events),
                "total_collateral_seized": sum(event.get("btc_value_seized", 0) for event in self.liquidation_events),
                "total_penalties_collected": sum(event.get("liquidation_bonus_value", 0) for event in self.liquidation_events),
                "liquidation_events": self.liquidation_events
            },
            "yield_token_activity": {
                "total_purchases": sum(trade["moet_amount"] for trade in self.yield_token_trades if trade["action"] == "purchase"),
                "total_rebalancing_sales": 0.0,  # No rebalancing in AAVE
                "total_trades": len(self.yield_token_trades),
                "rebalancing_events": 0  # No rebalancing in AAVE
            },
            "agent_health_history": self.agent_health_history,
            "btc_price_history": self.btc_price_history,
            "protocol_metrics": {
                "initial_btc_price": self.btc_price_manager.initial_price,
                "final_btc_price": self.btc_price_manager.get_decline_statistics().get("final_price", 0),
                "simulation_duration": final_minute + 1,
                "liquidation_mechanism": "AAVE_style"
            }
        }
        
        # Merge with base results
        base_results.update(aave_results)
        
        # Generate position tracking results
        if hasattr(self, 'position_tracker') and self.position_tracker.tracking_data:
            base_results["position_tracking"] = {
                "tracked_agent_id": self.position_tracker.agent_id,
                "tracking_summary": self.position_tracker.get_rebalancing_summary(),
                "minute_by_minute_data": self.position_tracker.tracking_data
            }
        
        return base_results
