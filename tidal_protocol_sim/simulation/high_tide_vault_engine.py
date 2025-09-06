#!/usr/bin/env python3
"""
High Tide Vault Engine

High Tide Yield Vaults built on Tidal Protocol with rebalancing mechanisms.
Inherits sophisticated Uniswap V3 math from TidalProtocolEngine.
"""

import random
from typing import Dict, List, Optional
from .tidal_engine import TidalProtocolEngine, TidalConfig
from ..core.protocol import Asset
from ..core.yield_tokens import YieldTokenPool
from ..core.uniswap_v3_math import create_yield_token_pool, UniswapV3SlippageCalculator
from ..analysis.lp_curve_analysis import LPCurveTracker
from ..agents.high_tide_agent import HighTideAgent, create_high_tide_agents
from ..agents.base_agent import BaseAgent, AgentAction
from ..analysis.agent_position_tracker import AgentPositionTracker


class HighTideConfig(TidalConfig):
    """Configuration for High Tide scenario"""
    
    def __init__(self):
        super().__init__()
        self.scenario_name = "High_Tide_Vault"
        
        # High Tide specific parameters
        self.yield_apr = 0.10  # 10% APR for yield tokens
        self.moet_yield_pool_size = 250_000  # $250K each side
        self.btc_decline_duration = 60  # 60 minutes
        self.rebalancing_enabled = True
        
        # Yield Token Pool Configuration (Internal protocol trading)
        self.yield_token_concentration = 0.95  # 95% concentration for MOET:Yield Tokens
        
        # Agent configuration for High Tide
        self.num_high_tide_agents = 20
        self.monte_carlo_agent_variation = True
        
        # BTC price decline parameters
        self.btc_final_price_range = (75_000.0, 85_000.0)  # 15-25% decline
        
        # Override base simulation parameters
        self.simulation_steps = self.btc_decline_duration
        self.price_update_frequency = 1  # Update every minute


class BTCPriceDeclineManager:
    """Manages gradual BTC price decline with historical volatility"""
    
    def __init__(self, config: HighTideConfig):
        self.config = config
        self.initial_price = config.btc_initial_price
        self.duration = config.btc_decline_duration
        
        # Historical decline rates
        self.decline_rates = [-0.0054, -0.0053, -0.0046, -0.0043, -0.0040]
        self.max_decline = -0.0095
        
        # Calculate target final price
        final_price_min, final_price_max = config.btc_final_price_range
        self.target_final_price = random.uniform(final_price_min, final_price_max)
        
        # Calculate required decline per minute
        total_decline_needed = (self.target_final_price - self.initial_price) / self.initial_price
        self.base_decline_per_minute = total_decline_needed / self.duration
        
        # Track price history
        self.price_history = [self.initial_price]
        self.current_price = self.initial_price
        
    def update_btc_price(self, minute: int) -> float:
        """Update BTC price for current minute"""
        if minute == 0:
            return self.initial_price
            
        # Use historical volatility pattern
        base_decline = random.choice(self.decline_rates)
        variation = random.uniform(-0.0005, 0.0005)
        
        # Occasionally use maximum decline (5% probability)
        if random.random() < 0.05:
            decline_rate = self.max_decline
        else:
            decline_rate = base_decline + variation
            
        # Adjust to meet target final price
        progress = minute / self.duration
        if progress > 0.8:  # In final 20% of decline, converge to target
            current_decline = (self.current_price - self.initial_price) / self.initial_price
            remaining_decline_needed = (self.target_final_price - self.initial_price) / self.initial_price - current_decline
            remaining_minutes = self.duration - minute
            if remaining_minutes > 0:
                target_decline = remaining_decline_needed / remaining_minutes
                decline_rate = (decline_rate + target_decline) / 2
        
        self.current_price *= (1 + decline_rate)
        self.current_price = max(self.current_price, self.initial_price * 0.5)
        
        self.price_history.append(self.current_price)
        return self.current_price
        
    def get_decline_statistics(self) -> Dict[str, float]:
        """Get statistics about the price decline"""
        if len(self.price_history) < 2:
            return {}
            
        final_price = self.price_history[-1]
        total_decline = (final_price - self.initial_price) / self.initial_price
        
        return {
            "initial_price": self.initial_price,
            "final_price": final_price,
            "total_decline_percent": total_decline * 100,
            "target_final_price": self.target_final_price,
            "duration_minutes": len(self.price_history) - 1
        }


class HighTideVaultEngine(TidalProtocolEngine):
    """High Tide Yield Vaults built on Tidal Protocol"""
    
    def __init__(self, config: HighTideConfig):
        # Initialize with High Tide config - gets all Uniswap V3 functionality from Tidal
        super().__init__(config)
        self.high_tide_config = config
        
        # Add yield token pools ON TOP of existing Tidal functionality
        self._setup_yield_token_pools()
        
        # Initialize High Tide specific components
        self.yield_token_pool = YieldTokenPool(config.moet_btc_pool_size)
        self.btc_price_manager = BTCPriceDeclineManager(config)
        
        # Replace agents with High Tide agents
        self.high_tide_agents = create_high_tide_agents(
            config.num_high_tide_agents,
            config.monte_carlo_agent_variation
        )
        
        # Add High Tide agents to main agents dict
        self.agents = {}  # Clear base agents
        for agent in self.high_tide_agents:
            self.agents[agent.agent_id] = agent
            
        # Initialize position tracker
        self.position_tracker = AgentPositionTracker(self.high_tide_agents[0].agent_id)
        self.position_tracker.start_tracking()
            
        # Initialize High Tide agent positions
        self._setup_high_tide_positions()
        
        # Enhanced tracking
        self.rebalancing_events = []
        self.yield_token_trades = []
        self.agent_health_history = []
        self.btc_price_history = []
        
    def _setup_yield_token_pools(self):
        """Add yield token functionality to existing Tidal base"""
        yield_pool_size = getattr(self.config, 'moet_yield_pool_size', 250_000) * 2
        btc_price = getattr(self.config, 'btc_initial_price', 100_000.0)
        
        self.yield_token_pool_v3 = create_yield_token_pool(
            yield_pool_size, btc_price, self.config.yield_token_concentration
        )
        
        self.yield_token_slippage_calculator = UniswapV3SlippageCalculator(
            self.yield_token_pool_v3
        )
        
        # Initialize LP curve tracking for yield tokens
        self.moet_yield_tracker = LPCurveTracker(
            yield_pool_size, self.config.yield_token_concentration, 
            "MOET:Yield_Token", btc_price
        )
        
    def _setup_high_tide_positions(self):
        """Set up initial High Tide agent positions"""
        for agent in self.high_tide_agents:
            # Update protocol with agent's BTC collateral
            btc_pool = self.protocol.asset_pools[Asset.BTC]
            btc_pool.total_supplied += agent.state.btc_amount
            
            # Update protocol with agent's MOET debt
            self.protocol.moet_system.mint(agent.state.moet_debt)
            
            # Initialize agent's health factor
            self._update_agent_health_factor(agent)
            
    def run_simulation(self, steps: int = None) -> Dict:
        """Run High Tide simulation with BTC price decline"""
        if steps is None:
            steps = self.high_tide_config.btc_decline_duration
            
        print(f"Starting High Tide simulation with {len(self.high_tide_agents)} agents")
        print(f"BTC decline from ${self.btc_price_manager.initial_price:,.0f} to ~${self.btc_price_manager.target_final_price:,.0f}")
        
        for minute in range(steps):
            self.current_step = minute
            
            # Update BTC price
            new_btc_price = self.btc_price_manager.update_btc_price(minute)
            self.state.current_prices[Asset.BTC] = new_btc_price
            self.btc_price_history.append(new_btc_price)
            
            # Update protocol state
            self.protocol.current_block = minute
            self.protocol.accrue_interest()
            
            # Update agent debt interest
            self._update_agent_debt_interest(minute)
            
            # Process High Tide agent actions
            swap_data = self._process_high_tide_agents(minute)
            
            # Check for High Tide liquidations
            self._check_high_tide_liquidations(minute)
            
            # Record position tracking data
            tracked_agent = self._get_tracked_agent()
            if tracked_agent:
                agent_swap_data = swap_data.get(tracked_agent.agent_id, {})
                self.position_tracker.record_minute_data(
                    minute, new_btc_price, tracked_agent, self, agent_swap_data
                )
            
            # Record metrics
            self._record_high_tide_metrics(minute)
            
            if minute % 10 == 0:
                print(f"Minute {minute}: BTC = ${new_btc_price:,.0f}, Active agents: {self._count_active_agents()}")
                
        return self._generate_high_tide_results()
        
    def _update_agent_debt_interest(self, minute: int):
        """Update debt interest for all High Tide agents"""
        btc_pool = self.protocol.asset_pools.get(Asset.BTC)
        if not btc_pool:
            return
            
        borrow_rate = btc_pool.calculate_borrow_rate()
        
        for agent in self.high_tide_agents:
            if agent.active:
                agent.update_debt_interest(minute, borrow_rate)
    
    def _process_high_tide_agents(self, minute: int) -> Dict[str, Dict]:
        """Process High Tide agent actions for current minute"""
        swap_data = {}
        
        for agent in self.high_tide_agents:
            if not agent.active:
                continue
                
            # Get agent's decision
            protocol_state = self._get_protocol_state()
            protocol_state["current_step"] = minute
            
            action_type, params = agent.decide_action(protocol_state, self.state.current_prices)
            
            # Execute action and capture swap data
            success, agent_swap_data = self._execute_high_tide_action(agent, action_type, params, minute)
            
            # Store swap data for tracking
            if agent_swap_data:
                swap_data[agent.agent_id] = agent_swap_data
            
            # Record action
            self._record_agent_action(agent.agent_id, action_type, params)
            
        return swap_data
            
    def _execute_high_tide_action(self, agent: HighTideAgent, action_type: AgentAction, params: dict, minute: int) -> tuple:
        """Execute High Tide specific actions"""
        if action_type == AgentAction.SWAP:
            swap_type = params.get("action_type", "")
            
            if swap_type == "buy_yield_tokens":
                success = self._execute_yield_token_purchase(agent, params, minute)
                return success, None
            elif swap_type in ["sell_yield_tokens", "sell_yield_only", "emergency_sell_all_yield"]:
                success, swap_data = self._execute_yield_token_sale(agent, params, minute)
                return success, swap_data
                
        elif action_type == AgentAction.LIQUIDATE:
            success = self._execute_liquidation(agent, params)
            return success, None
            
        return False, None
        
    def _execute_yield_token_purchase(self, agent: HighTideAgent, params: dict, minute: int) -> bool:
        """Execute yield token purchase for agent"""
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
        
    def _execute_yield_token_sale(self, agent: HighTideAgent, params: dict, minute: int) -> tuple:
        """Execute yield token sale for rebalancing"""
        amount_needed = params.get("amount_needed", 0.0)
        swap_type = params.get("action_type", "sell_yield_tokens")
        
        if amount_needed <= 0 and swap_type != "emergency_sell_all_yield":
            return False, None
            
        # Execute sale through agent
        if swap_type == "emergency_sell_all_yield":
            moet_raised = agent.execute_yield_token_sale(float('inf'), minute, yield_only=False)
        else:
            yield_only = swap_type == "sell_yield_only"
            moet_raised = agent.execute_yield_token_sale(amount_needed, minute, yield_only=yield_only)
        
        if moet_raised > 0:
            # Use MOET directly to pay down debt (no BTC swap needed)
            debt_repayment = min(moet_raised, agent.state.moet_debt)
            agent.state.moet_debt -= debt_repayment
            
            # Record pool activity
            self.moet_yield_tracker.record_snapshot(
                pool_state={
                    "token0_reserve": self.yield_token_pool.moet_reserve,
                    "token1_reserve": self.yield_token_pool.yield_token_reserve,
                    "liquidity": (self.yield_token_pool.moet_reserve + self.yield_token_pool.yield_token_reserve) / 2
                },
                minute=minute,
                trade_amount=moet_raised,
                trade_type="yield_token_sale"
            )
            
            # Update yield token pool
            self.yield_token_pool.execute_yield_token_sale(moet_raised)
            
            # Calculate slippage using proper Uniswap V3 math
            slippage_result = self.yield_token_slippage_calculator.calculate_swap_slippage(
                amount_needed, "Yield_Token"
            )
            
            self.yield_token_slippage_calculator.update_pool_state(slippage_result)
            
            slippage_cost = slippage_result["slippage_amount"] + slippage_result["trading_fees"]
            
            # Create swap data for tracking
            swap_data = {
                "yt_swapped": amount_needed,
                "moet_received": moet_raised,
                "debt_repayment": debt_repayment,
                "swap_type": swap_type,
                "slippage_cost": slippage_cost,
                "slippage_percentage": slippage_result["slippage_percent"],
                "price_impact": slippage_result["price_impact"]
            }
            
            # Record rebalancing event
            self.rebalancing_events.append({
                "minute": minute,
                "agent_id": agent.agent_id,
                "moet_raised": moet_raised,
                "amount_needed": amount_needed,
                "debt_repayment": debt_repayment,
                "health_factor_before": agent.state.health_factor,
                "rebalancing_type": swap_type,
                "slippage_cost": slippage_cost
            })
            
            # Record trade
            self.yield_token_trades.append({
                "minute": minute,
                "agent_id": agent.agent_id,
                "action": "rebalancing_sale",
                "moet_amount": moet_raised,
                "debt_repayment": debt_repayment,
                "agent_health_factor": agent.state.health_factor,
                "slippage_cost": slippage_cost
            })
            
            return True, swap_data
            
        return False, None
        
    def _get_tracked_agent(self) -> Optional[HighTideAgent]:
        """Get the agent being tracked for position analysis"""
        for agent in self.high_tide_agents:
            if agent.agent_id == self.position_tracker.agent_id:
                return agent
        return None
        
    def _check_high_tide_liquidations(self, minute: int):
        """Check for High Tide liquidations (HF ≤ 1.0)"""
        for agent in self.high_tide_agents:
            if not agent.active:
                continue
                
            # Update health factor
            agent._update_health_factor(self.state.current_prices)
            
            # Check if liquidation is needed (HF ≤ 1.0)
            if agent.state.health_factor <= 1.0:
                liquidation_event = agent.execute_high_tide_liquidation(minute, self.state.current_prices)
                
                if liquidation_event:
                    self.liquidation_events.append(liquidation_event)
        
    def _count_active_agents(self) -> int:
        """Count number of active High Tide agents"""
        return sum(1 for agent in self.high_tide_agents if agent.active)
        
    def _record_high_tide_metrics(self, minute: int):
        """Record High Tide specific metrics"""
        # Base metrics
        super()._record_metrics()
        
        # High Tide specific metrics
        agent_health_data = []
        for agent in self.high_tide_agents:
            portfolio = agent.get_detailed_portfolio_summary(self.state.current_prices, minute)
            agent_health_data.append({
                "agent_id": agent.agent_id,
                "health_factor": agent.state.health_factor,
                "risk_profile": agent.risk_profile,
                "target_hf": agent.state.target_health_factor,
                "initial_hf": agent.state.initial_health_factor,
                "cost_of_rebalancing": portfolio["cost_of_rebalancing"],
                "net_position_value": portfolio["net_position_value"],
                "yield_token_value": portfolio["yield_token_portfolio"]["total_current_value"],
                "total_yield_sold": portfolio["total_yield_sold"],
                "rebalancing_events": portfolio["rebalancing_events_count"]
            })
            
        self.agent_health_history.append({
            "minute": minute,
            "btc_price": self.state.current_prices[Asset.BTC],
            "agents": agent_health_data
        })
        
    def _generate_high_tide_results(self) -> dict:
        """Generate comprehensive High Tide simulation results"""
        base_results = super()._generate_results()
        
        # Calculate High Tide specific metrics
        final_minute = self.high_tide_config.btc_decline_duration - 1
        
        # Agent outcomes
        agent_outcomes = []
        total_cost_of_rebalancing = 0.0
        survival_by_risk_profile = {"conservative": 0, "moderate": 0, "aggressive": 0}
        
        for agent in self.high_tide_agents:
            agent._update_health_factor(self.state.current_prices)
            
            portfolio = agent.get_detailed_portfolio_summary(
                self.state.current_prices, 
                final_minute
            )
            
            outcome = {
                "agent_id": agent.agent_id,
                "risk_profile": agent.risk_profile,
                "target_health_factor": agent.state.target_health_factor,
                "initial_health_factor": agent.state.initial_health_factor,
                "final_health_factor": agent.state.health_factor,
                "cost_of_rebalancing": portfolio["cost_of_rebalancing"],
                "net_position_value": portfolio["net_position_value"],
                "total_yield_earned": portfolio["yield_token_portfolio"]["total_accrued_yield"],
                "total_yield_sold": portfolio["total_yield_sold"],
                "rebalancing_events": len(agent.get_rebalancing_history()),
                "survived": agent.state.health_factor > 1.0,
                "yield_token_value": portfolio["yield_token_portfolio"]["total_current_value"]
            }
            
            agent_outcomes.append(outcome)
            total_cost_of_rebalancing += outcome["cost_of_rebalancing"]
            
            if outcome["survived"]:
                survival_by_risk_profile[agent.risk_profile] += 1
                
        # High Tide specific results
        high_tide_results = {
            "scenario_type": "High_Tide_BTC_Decline",
            "btc_decline_statistics": self.btc_price_manager.get_decline_statistics(),
            "agent_outcomes": agent_outcomes,
            "survival_statistics": {
                "total_agents": len(self.high_tide_agents),
                "survivors": sum(1 for outcome in agent_outcomes if outcome["survived"]),
                "survival_rate": sum(1 for outcome in agent_outcomes if outcome["survived"]) / len(self.high_tide_agents),
                "survival_by_risk_profile": survival_by_risk_profile
            },
            "cost_analysis": {
                "total_cost_of_rebalancing": total_cost_of_rebalancing,
                "average_cost_per_agent": total_cost_of_rebalancing / len(self.high_tide_agents)
            },
            "yield_token_activity": {
                "total_purchases": sum(trade["moet_amount"] for trade in self.yield_token_trades if trade["action"] == "purchase"),
                "total_rebalancing_sales": sum(trade["moet_amount"] for trade in self.yield_token_trades if trade["action"] == "rebalancing_sale"),
                "total_trades": len(self.yield_token_trades),
                "rebalancing_events": len(self.rebalancing_events)
            },
            "agent_health_history": self.agent_health_history,
            "btc_price_history": self.btc_price_history,
            "rebalancing_events": self.rebalancing_events,
            "yield_token_trades": self.yield_token_trades
        }
        
        # Merge with base results
        base_results.update(high_tide_results)
        
        # Generate position tracking results
        if hasattr(self, 'position_tracker') and self.position_tracker.tracking_data:
            base_results["position_tracking"] = {
                "tracked_agent_id": self.position_tracker.agent_id,
                "tracking_summary": self.position_tracker.get_rebalancing_summary(),
                "minute_by_minute_data": self.position_tracker.tracking_data
            }
        
        return base_results
