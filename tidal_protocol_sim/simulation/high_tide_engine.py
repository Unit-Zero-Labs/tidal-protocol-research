#!/usr/bin/env python3
"""
High Tide Simulation Engine

Specialized simulation engine for High Tide scenario with BTC price decline
and active position management.
"""

import random
from typing import Dict, List, Optional
from .engine import TidalSimulationEngine
from .config import SimulationConfig
from ..core.protocol import Asset
from ..core.yield_tokens import YieldTokenPool
from ..core.uniswap_v3_math import UniswapV3Pool, UniswapV3SlippageCalculator
from ..analysis.lp_curve_analysis import LPCurveTracker
from ..agents.high_tide_agent import HighTideAgent, create_high_tide_agents
from ..agents.base_agent import BaseAgent, AgentAction
from ..analysis.agent_position_tracker import AgentPositionTracker


class HighTideConfig(SimulationConfig):
    """Configuration for High Tide scenario"""
    
    def __init__(self):
        super().__init__()
        
        # High Tide specific parameters
        self.yield_apr = 0.10  # 10% APR for yield tokens
        self.moet_btc_pool_size = 500_000  # $500K each side (minimum viable configuration)
        self.moet_yield_pool_size = 250_000  # $250K each side (minimum viable configuration)
        self.btc_decline_duration = 60  # 60 minutes
        self.rebalancing_enabled = True
        self.comparison_mode = True  # Include Aave strategy comparison
        
        # Uniswap v3 Pool Configuration (External MOET:BTC trading)
        self.uniswap_pool_size = 500_000  # Total pool size ($500k minimum viable)
        self.moet_btc_concentration = 0.80  # 80% concentration for MOET:BTC (single peg bin)
        
        # Yield Token Pool Configuration (Internal protocol trading)
        self.yield_token_concentration = 0.95  # 95% concentration for MOET:Yield Tokens (single peg bin)
        
        # Agent configuration for High Tide
        self.num_high_tide_agents = 20  # Base number, can be randomized
        self.monte_carlo_agent_variation = True
        
        # BTC price decline parameters
        self.btc_initial_price = 100_000.0
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
        
        # Historical decline rates from 2025 data
        self.decline_rates = [-0.0054, -0.0053, -0.0046, -0.0043, -0.0040]  # -0.54% to -0.40%
        self.max_decline = -0.0095  # -0.95% maximum decline
        
        # Calculate target final price
        final_price_min, final_price_max = config.btc_final_price_range
        self.target_final_price = random.uniform(final_price_min, final_price_max)
        
        # Calculate required decline per minute
        total_decline_needed = (self.target_final_price - self.initial_price) / self.initial_price
        self.base_decline_per_minute = total_decline_needed / self.duration
        
        # Track price history
        self.price_history = [self.initial_price]
        self.current_price = self.initial_price
        
    def calculate_btc_price_change(self, minute: int) -> float:
        """Calculate BTC price change for current minute"""
        # Use historical volatility pattern
        base_decline = random.choice(self.decline_rates)
        
        # Add small random variation (±0.05%)
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
                decline_rate = (decline_rate + target_decline) / 2  # Blend with target
        
        return decline_rate
        
    def update_btc_price(self, minute: int) -> float:
        """Update BTC price for current minute"""
        if minute == 0:
            return self.initial_price
            
        decline_rate = self.calculate_btc_price_change(minute)
        self.current_price *= (1 + decline_rate)
        
        # Ensure we don't go below minimum reasonable price
        self.current_price = max(self.current_price, self.initial_price * 0.5)
        
        self.price_history.append(self.current_price)
        return self.current_price
        
    def get_price_history(self) -> List[float]:
        """Get complete price history"""
        return self.price_history.copy()
        
    def get_decline_statistics(self) -> Dict[str, float]:
        """Get statistics about the price decline"""
        if len(self.price_history) < 2:
            return {}
            
        final_price = self.price_history[-1]
        total_decline = (final_price - self.initial_price) / self.initial_price
        
        # Calculate minute-to-minute volatility
        minute_changes = []
        for i in range(1, len(self.price_history)):
            change = (self.price_history[i] - self.price_history[i-1]) / self.price_history[i-1]
            minute_changes.append(change)
            
        avg_decline_per_minute = sum(minute_changes) / len(minute_changes) if minute_changes else 0.0
        volatility = (sum([(change - avg_decline_per_minute)**2 for change in minute_changes]) / len(minute_changes))**0.5 if minute_changes else 0.0
        
        return {
            "initial_price": self.initial_price,
            "final_price": final_price,
            "total_decline_percent": total_decline * 100,
            "target_final_price": self.target_final_price,
            "average_decline_per_minute": avg_decline_per_minute * 100,
            "volatility": volatility * 100,
            "duration_minutes": len(self.price_history) - 1
        }


class HighTideSimulationEngine(TidalSimulationEngine):
    """Specialized simulation engine for High Tide scenario"""
    
    def __init__(self, config: HighTideConfig):
        # Initialize with High Tide config
        self.high_tide_config = config
        super().__init__(config)
        
        # Initialize High Tide specific components
        self.yield_token_pool = YieldTokenPool(config.moet_btc_pool_size)
        self.btc_price_manager = BTCPriceDeclineManager(config)
        
        # Initialize Uniswap v3 pool for slippage calculations
        # Default to $500k pool with proper MOET:BTC ratio
        pool_size = getattr(config, 'uniswap_pool_size', 500_000)  # Total pool size
        btc_price = getattr(config, 'btc_initial_price', 100_000.0)  # Get BTC price from config
        
        # Get concentration parameters first
        self.moet_btc_concentration = getattr(config, 'moet_btc_concentration', 0.80)  # 80% concentration for MOET:BTC
        self.yield_token_concentration = getattr(config, 'yield_token_concentration', 0.05)  # 95% concentration for yield tokens
        
        # Import the corrected pool creation function
        from ..core.uniswap_v3_math import create_moet_btc_pool
        self.uniswap_pool = create_moet_btc_pool(pool_size, btc_price, self.moet_btc_concentration)
        self.slippage_calculator = UniswapV3SlippageCalculator(self.uniswap_pool)
        
        # Initialize LP curve tracking for both pools
        self.moet_btc_tracker = LPCurveTracker(pool_size, self.moet_btc_concentration, "MOET:BTC", btc_price)
        
        # Initialize yield token pool tracker
        yield_pool_size = getattr(config, 'moet_yield_pool_size', 250_000) * 2  # Total pool size
        self.moet_yield_tracker = LPCurveTracker(yield_pool_size, self.yield_token_concentration, "MOET:Yield_Token", btc_price)
        
        # Initialize concentrated liquidity pools for advanced analysis
        from ..core.uniswap_v3_math import create_moet_btc_pool, create_yield_token_pool
        self.moet_btc_concentrated_pool = create_moet_btc_pool(pool_size, btc_price, self.moet_btc_concentration)
        self.yield_token_concentrated_pool = create_yield_token_pool(yield_pool_size, btc_price, self.yield_token_concentration)

        # Replace agents with High Tide agents
        self.high_tide_agents = create_high_tide_agents(
            config.num_high_tide_agents,
            config.monte_carlo_agent_variation
        )
        
        # Add High Tide agents to main agents dict
        for agent in self.high_tide_agents:
            self.agents[agent.agent_id] = agent
            
        # Initialize position tracker for first agent
        self.position_tracker = AgentPositionTracker(self.high_tide_agents[0].agent_id)
        self.position_tracker.start_tracking()
            
        # Initialize High Tide agent positions
        self._setup_high_tide_positions()
        
        # Enhanced tracking for High Tide scenario
        self.rebalancing_events = []
        self.yield_token_trades = []
        self.agent_health_history = []
        self.btc_price_history = []
        
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
            
    def run_high_tide_simulation(self) -> Dict:
        """Run High Tide simulation with BTC price decline"""
        print(f"Starting High Tide simulation with {len(self.high_tide_agents)} agents")
        print(f"BTC decline from ${self.btc_price_manager.initial_price:,.0f} to ~${self.btc_price_manager.target_final_price:,.0f}")
        
        for minute in range(self.high_tide_config.btc_decline_duration):
            self.current_step = minute
            
            # Update BTC price
            new_btc_price = self.btc_price_manager.update_btc_price(minute)
            self.state.current_prices[Asset.BTC] = new_btc_price
            self.btc_price_history.append(new_btc_price)
            
            # Update protocol state
            self.protocol.current_block = minute
            self.protocol.accrue_interest()
            
            # Update agent debt interest based on BTC pool utilization
            self._update_agent_debt_interest(minute)
            
            # Process High Tide agent actions
            swap_data = self._process_high_tide_agents(minute)
            
            # Check for traditional liquidations (fallback)
            self._check_liquidations()
            
            # Record position tracking data
            tracked_agent = self._get_tracked_agent()
            if tracked_agent:
                agent_swap_data = swap_data.get(tracked_agent.agent_id, {})
                self.position_tracker.record_minute_data(
                    minute, new_btc_price, tracked_agent, self, agent_swap_data
                )
            
            # Record metrics
            self._record_high_tide_metrics(minute)
            
            if minute % 10 == 0:  # Progress update every 10 minutes
                print(f"Minute {minute}: BTC = ${new_btc_price:,.0f}, Active agents: {self._count_active_agents()}")
                
        # Generate comprehensive results
        return self._generate_high_tide_results()
        
    def _update_agent_debt_interest(self, minute: int):
        """Update debt interest for all High Tide agents based on BTC pool utilization"""
        # Get BTC pool borrow rate
        btc_pool = self.protocol.asset_pools.get(Asset.BTC)
        if not btc_pool:
            return
            
        borrow_rate = btc_pool.calculate_borrow_rate()
        
        # Update each agent's debt
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
            elif swap_type == "sell_yield_tokens":
                success, swap_data = self._execute_yield_token_sale(agent, params, minute)
                return success, swap_data
            elif swap_type == "sell_yield_only":
                success, swap_data = self._execute_yield_only_sale(agent, params, minute)
                return success, swap_data
            elif swap_type == "emergency_sell_all_yield":
                success, swap_data = self._execute_emergency_yield_sale(agent, params, minute)
                return success, swap_data
                
        elif action_type == AgentAction.LIQUIDATE:
            # Traditional liquidation fallback
            success = self._execute_liquidation(agent, params)
            return success, None
            
        return False, None
        
    def _execute_yield_token_purchase(self, agent: HighTideAgent, params: dict, minute: int) -> bool:
        """Execute yield token purchase for agent"""
        moet_amount = params.get("moet_amount", 0.0)
        
        if moet_amount <= 0:
            return False
            
        # Execute purchase through agent
        success = agent.execute_yield_token_purchase(moet_amount, minute)
        
        if success:
            # Update yield token pool
            self.yield_token_pool.execute_yield_token_purchase(moet_amount)
            
            # Record trade
            self.yield_token_trades.append({
                "minute": minute,
                "agent_id": agent.agent_id,
                "action": "purchase",
                "moet_amount": moet_amount,
                "agent_health_factor": agent.state.health_factor
            })
            
        return success
        
    def _execute_yield_token_sale(self, agent: HighTideAgent, params: dict, minute: int) -> tuple:
        """Execute yield token sale for rebalancing - MOET directly pays down debt"""
        amount_needed = params.get("amount_needed", 0.0)
        
        if amount_needed <= 0:
            return False, None
            
        # Execute sale through agent (gets MOET from yield tokens)
        moet_raised = agent.execute_yield_token_sale(amount_needed, minute, yield_only=False)
        
        if moet_raised > 0:
            # CORRECTED: Use MOET directly to pay down debt (no BTC swap needed)
            # Since debt is in MOET and we received MOET, direct 1:1 repayment
            debt_repayment = min(moet_raised, agent.state.moet_debt)
            agent.state.moet_debt -= debt_repayment
            
            # Record MOET:Yield Token pool activity (the only pool used for rebalancing)
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
            
            # Create swap data for tracking (no BTC swap, no slippage)
            swap_data = {
                "yt_swapped": amount_needed,
                "moet_received": moet_raised,
                "debt_repayment": debt_repayment,
                "swap_type": "rebalancing"
            }
            
            # Record rebalancing event (no BTC swap slippage)
            self.rebalancing_events.append({
                "minute": minute,
                "agent_id": agent.agent_id,
                "moet_raised": moet_raised,
                "amount_needed": amount_needed,
                "debt_repayment": debt_repayment,
                "health_factor_before": agent.state.health_factor,
                "rebalancing_type": "full_sale"
            })
            
            # Record trade (no BTC swap)
            self.yield_token_trades.append({
                "minute": minute,
                "agent_id": agent.agent_id,
                "action": "rebalancing_sale",
                "moet_amount": moet_raised,
                "debt_repayment": debt_repayment,
                "agent_health_factor": agent.state.health_factor
            })
            
            return True, swap_data
            
        return False, None
        
    def _execute_yield_only_sale(self, agent: HighTideAgent, params: dict, minute: int) -> tuple:
        """Execute sale of only accrued yield - MOET directly pays down debt"""
        amount_needed = params.get("amount_needed", 0.0)
        
        # Execute yield-only sale through agent
        moet_raised = agent.execute_yield_token_sale(amount_needed, minute, yield_only=True)
        
        if moet_raised > 0:
            # CORRECTED: Use MOET directly to pay down debt (no BTC swap)
            debt_repayment = min(moet_raised, agent.state.moet_debt)
            agent.state.moet_debt -= debt_repayment
            
            # Record MOET:Yield Token pool activity (the only pool used for rebalancing)
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
            
            # Create swap data for tracking (no BTC swap)
            swap_data = {
                "yt_swapped": amount_needed,
                "moet_received": moet_raised,
                "debt_repayment": debt_repayment,
                "swap_type": "yield_only"
            }
            
            # Record rebalancing event (no BTC swap)
            self.rebalancing_events.append({
                "minute": minute,
                "agent_id": agent.agent_id,
                "moet_raised": moet_raised,
                "amount_needed": amount_needed,
                "debt_repayment": debt_repayment,
                "health_factor_before": agent.state.health_factor,
                "rebalancing_type": "yield_only"
            })
            
            return True, swap_data
            
        return False, None
        
    def _execute_emergency_yield_sale(self, agent: HighTideAgent, params: dict, minute: int) -> tuple:
        """Execute emergency sale of ALL remaining yield tokens - MOET directly pays down debt"""
        amount_needed = params.get("amount_needed", 0.0)
        
        if amount_needed <= 0:
            return False, None
            
        # Execute emergency sale through agent (sell everything)
        moet_raised = agent.execute_yield_token_sale(float('inf'), minute, yield_only=False)
        
        if moet_raised > 0:
            # CORRECTED: Use MOET directly to pay down debt (no BTC swap)
            debt_repayment = min(moet_raised, agent.state.moet_debt)
            agent.state.moet_debt -= debt_repayment
            
            # Record MOET:Yield Token pool activity (the only pool used for rebalancing)
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
            
            # Record emergency rebalancing event (no BTC swap)
            self.rebalancing_events.append({
                "minute": minute,
                "agent_id": agent.agent_id,
                "moet_raised": moet_raised,
                "amount_needed": amount_needed,
                "debt_repayment": debt_repayment,
                "health_factor_before": agent.state.health_factor,
                "rebalancing_type": "emergency_all_yield"
            })
            
            # Create swap data for tracking (no BTC swap)
            swap_data = {
                "yt_swapped": agent.state.yield_token_manager.calculate_total_value(minute),  # All yield tokens
                "moet_received": moet_raised,
                "debt_repayment": debt_repayment,
                "swap_type": "emergency"
            }
            
            # Record trade (no BTC swap)
            self.yield_token_trades.append({
                "minute": minute,
                "agent_id": agent.agent_id,
                "action": "emergency_liquidation_sale",
                "moet_amount": moet_raised,
                "debt_repayment": debt_repayment,
                "agent_health_factor": agent.state.health_factor
            })
            
            return True, swap_data
            
        return False, None
        
    def _get_tracked_agent(self) -> Optional[HighTideAgent]:
        """Get the agent being tracked for position analysis"""
        for agent in self.high_tide_agents:
            if agent.agent_id == self.position_tracker.agent_id:
                return agent
        return None
        
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
            # Force update health factor with final prices before collecting results
            agent._update_health_factor(self.state.current_prices)
            
            portfolio = agent.get_detailed_portfolio_summary(
                self.state.current_prices, 
                final_minute,
                pool_size_usd=self.uniswap_pool.token0_reserve + self.uniswap_pool.token1_reserve,
                concentrated_range=self.moet_btc_concentration
            )
            
            outcome = {
                "agent_id": agent.agent_id,
                "risk_profile": agent.risk_profile,
                "target_health_factor": agent.state.target_health_factor,
                "initial_health_factor": agent.state.initial_health_factor,  # Add this for table
                "final_health_factor": agent.state.health_factor,
                "cost_of_rebalancing": portfolio["cost_of_rebalancing"],
                "net_position_value": portfolio["net_position_value"],
                "total_yield_earned": portfolio["yield_token_portfolio"]["total_accrued_yield"],
                "total_yield_sold": portfolio["total_yield_sold"],
                "rebalancing_events": len(agent.get_rebalancing_history()),
                "survived": agent.state.health_factor > 1.0,
                "emergency_liquidations": portfolio["emergency_liquidations"],
                "yield_token_value": portfolio["yield_token_portfolio"]["total_current_value"],
                "initial_debt": portfolio["initial_moet_debt"],
                "final_debt": portfolio["current_moet_debt"],
                "interest_accrued": portfolio["total_interest_accrued"]
            }
            
            agent_outcomes.append(outcome)
            total_cost_of_rebalancing += outcome["cost_of_rebalancing"]
            
            if outcome["survived"]:
                survival_by_risk_profile[agent.risk_profile] += 1
                
        # BTC price statistics
        btc_stats = self.btc_price_manager.get_decline_statistics()
        
        # Yield token trading summary
        total_yield_purchases = sum(trade["moet_amount"] for trade in self.yield_token_trades if trade["action"] == "purchase")
        total_rebalancing_sales = sum(trade["moet_amount"] for trade in self.yield_token_trades if trade["action"] == "rebalancing_sale")
        
        # High Tide specific results
        high_tide_results = {
            "scenario_type": "High_Tide_BTC_Decline",
            "btc_decline_statistics": btc_stats,
            "agent_outcomes": agent_outcomes,
            "survival_statistics": {
                "total_agents": len(self.high_tide_agents),
                "survivors": sum(1 for outcome in agent_outcomes if outcome["survived"]),
                "survival_rate": sum(1 for outcome in agent_outcomes if outcome["survived"]) / len(self.high_tide_agents),
                "survival_by_risk_profile": survival_by_risk_profile
            },
            "cost_analysis": {
                "total_cost_of_rebalancing": total_cost_of_rebalancing,
                "average_cost_per_agent": total_cost_of_rebalancing / len(self.high_tide_agents),
                "cost_by_risk_profile": self._calculate_cost_by_risk_profile(agent_outcomes)
            },
            "yield_token_activity": {
                "total_purchases": total_yield_purchases,
                "total_rebalancing_sales": total_rebalancing_sales,
                "total_trades": len(self.yield_token_trades),
                "rebalancing_events": len(self.rebalancing_events)
            },
            "agent_health_history": self.agent_health_history,
            "btc_price_history": self.btc_price_history,
            "rebalancing_events": self.rebalancing_events,
            "yield_token_trades": self.yield_token_trades,
            "moet_btc_lp_snapshots": [
                {
                    "minute": s.minute,
                    "moet_reserve": s.moet_reserve,
                    "btc_reserve": s.btc_reserve,
                    "price": s.price,
                    "liquidity": s.liquidity,
                    "concentration_range": s.concentration_range,
                    "trade_amount": s.trade_amount,
                    "trade_type": s.trade_type
                } for s in self.moet_btc_tracker.get_snapshots()
            ],
            "moet_yield_lp_snapshots": [
                {
                    "minute": s.minute,
                    "moet_reserve": s.moet_reserve,
                    "btc_reserve": s.btc_reserve,  # Actually yield token reserve
                    "price": s.price,
                    "liquidity": s.liquidity,
                    "concentration_range": s.concentration_range,
                    "trade_amount": s.trade_amount,
                    "trade_type": s.trade_type
                } for s in self.moet_yield_tracker.get_snapshots()
            ],
            "protocol_metrics": {
                "initial_btc_price": self.btc_price_manager.initial_price,
                "final_btc_price": btc_stats.get("final_price", 0),
                "simulation_duration": final_minute + 1
            }
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
        
    def _calculate_cost_by_risk_profile(self, agent_outcomes: List[dict]) -> dict:
        """Calculate cost of rebalancing by risk profile"""
        profile_costs = {"conservative": [], "moderate": [], "aggressive": []}
        
        for outcome in agent_outcomes:
            profile = outcome["risk_profile"]
            profile_costs[profile].append(outcome["cost_of_rebalancing"])
            
        result = {}
        for profile, costs in profile_costs.items():
            if costs:
                result[profile] = {
                    "average_cost": sum(costs) / len(costs),
                    "total_cost": sum(costs),
                    "agent_count": len(costs)
                }
            else:
                result[profile] = {"average_cost": 0.0, "total_cost": 0.0, "agent_count": 0}
                
        return result
