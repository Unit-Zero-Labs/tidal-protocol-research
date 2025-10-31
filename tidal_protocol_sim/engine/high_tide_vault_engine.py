#!/usr/bin/env python3
"""
High Tide Vault Engine

High Tide Yield Vaults built on Tidal Protocol with rebalancing mechanisms.
Inherits sophisticated Uniswap V3 math from TidalProtocolEngine.
"""

import random
from typing import Dict, List, Optional
from .tidal_engine import TidalProtocolEngine, TidalConfig
from .btc_price_manager import BTCPriceDeclineManager
from ..core.protocol import Asset
from ..core.yield_tokens import YieldTokenPool
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
        self.moet_yield_pool_size = 250_000  # $250K each side
        self.btc_decline_duration = 60  # 60 minutes
        
        # Yield Token Pool Configuration (Internal protocol trading)
        self.yield_token_concentration = 0.95  # 95% concentration for MOET:Yield Tokens
        self.yield_token_ratio = 0.5  # Default to 50/50 MOET:YT ratio (will be overridden by config)
        
        # Agent configuration for High Tide
        self.num_high_tide_agents = 20
        self.monte_carlo_agent_variation = True
        
        # BTC price decline parameters
        self.btc_final_price_range = (75_000.0, 85_000.0)  # 15-25% decline
        
        # Yield Token Creation Method
        self.use_direct_minting_for_initial = True  # True = 1:1 minting at minute 0, False = Uniswap purchases
        
        # Advanced MOET system toggle
        self.enable_advanced_moet_system = False  # Default to legacy system
        
        # Enhanced MOET System Configuration (for arbitrage agents)
        self.num_arbitrage_agents = 0  # Number of MOET arbitrage agents for peg maintenance
        self.arbitrage_agent_balance = 100_000.0  # Initial balance per arbitrage agent
        
        # Override base simulation parameters
        self.simulation_steps = self.btc_decline_duration


class HighTideVaultEngine(TidalProtocolEngine):
    """High Tide Yield Vaults built on Tidal Protocol"""
    
    def __init__(self, config: HighTideConfig):
        # Initialize with High Tide config - gets all Uniswap V3 functionality from Tidal
        super().__init__(config)
        self.high_tide_config = config
                
        # Initialize High Tide specific components
        # Convert old interface to new interface
        total_pool_size = config.moet_yield_pool_size * 2  # Convert from single-side to total
        token0_ratio = getattr(config, 'yield_token_ratio', 0.5)  # Use configured ratio or default to 50/50
        
        self.yield_token_pool = YieldTokenPool(
            total_pool_size=total_pool_size,
            token0_ratio=token0_ratio,
            concentration=config.yield_token_concentration
        )
        # BTC Price Manager is used for BTC price decline
        
        self.btc_price_manager = BTCPriceDeclineManager(
            initial_price=config.btc_initial_price,
            duration=config.btc_decline_duration,
            final_price_range=config.btc_final_price_range
        )
        
        # Replace agents with High Tide agents
        self.high_tide_agents = create_high_tide_agents(
            config.num_high_tide_agents,
            config.monte_carlo_agent_variation,
            self.yield_token_pool
        )
        
        # Add High Tide agents to main agents dict
        # CRITICAL FIX: Preserve arbitrage agents if advanced MOET system is enabled
        if getattr(config, 'enable_advanced_moet_system', False):
            # Check if arbitrage agents were created by parent class
            if hasattr(self, 'arbitrage_agents') and self.arbitrage_agents:
                print(f"🤖 Preserving {len(self.arbitrage_agents)} arbitrage agents in High Tide engine")
                # Keep existing arbitrage agents and add High Tide agents
                for agent in self.high_tide_agents:
                    self.agents[agent.agent_id] = agent
                    agent.engine = self
            else:
                print("⚠️  Advanced MOET system enabled but no arbitrage agents found. Creating them now...")
                # Clear base agents and add High Tide agents first
                self.agents = {}
                for agent in self.high_tide_agents:
                    self.agents[agent.agent_id] = agent
                    agent.engine = self
                # Then create arbitrage agents
                self._setup_arbitrage_agents()
        else:
            # Clear base agents and use only High Tide agents (legacy behavior)
            self.agents = {}
            for agent in self.high_tide_agents:
                self.agents[agent.agent_id] = agent
                # CRITICAL FIX: Set engine reference for real swap recording
                agent.engine = self
            
        # Initialize LP curve tracking for yield tokens using YieldTokenPool data
        yield_pool_size = config.moet_btc_pool_size * 2
        self.moet_yield_tracker = LPCurveTracker(
            yield_pool_size, config.yield_token_concentration, 
            "MOET:Yield_Token", config.btc_initial_price
        )
        
        # Initialize position tracker
        self.position_tracker = AgentPositionTracker(self.high_tide_agents[0].agent_id)
        self.position_tracker.start_tracking()
        
        # CRITICAL FIX: Initialize tracking lists for real swap data
        self.rebalancing_events = []
        self.yield_token_trades = []
        self.agent_health_history = []
        self.arbitrage_events = []  # Track arbitrage events
            
        # Initialize High Tide agent positions
        self._setup_high_tide_positions()
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
        
        # Initialize MOET reserves after all agents are set up
        if self.protocol.enable_advanced_moet:
            total_agent_debt = sum(agent.state.moet_debt for agent in self.high_tide_agents)
            self.protocol.initialize_moet_reserves(total_agent_debt)
            # Get actual initialized reserves from the system
            actual_reserves = self.protocol.moet_system.redeemer.reserve_state.total_reserves
            reserve_ratio = actual_reserves / total_agent_debt if total_agent_debt > 0 else 0
            print(f"🏦 Initialized MOET reserves: ${actual_reserves:,.0f} ({reserve_ratio:.1%} of ${total_agent_debt:,.0f} total debt)")
            
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
            
            # Process MOET system updates (bond auctions, interest rate calculations)
            moet_update_results = self.protocol.process_moet_system_update(minute)
            if moet_update_results.get('advanced_system_enabled') and minute % 60 == 0:  # Log hourly
                if moet_update_results.get('bond_auction_triggered'):
                    print(f"🔔 Bond auction triggered at minute {minute}")
                if moet_update_results.get('bond_auction_completed'):
                    auction = moet_update_results['completed_auction']
                    print(f"✅ Bond auction completed: ${auction['amount_filled']:,.0f} at {auction['final_apr']:.2%} APR")
                if moet_update_results.get('interest_rate_updated'):
                    print(f"📈 MOET rate updated: {moet_update_results['new_interest_rate']:.2%}")
            
            # Update agent debt interest
            self._update_agent_debt_interest(minute)
            
            # Process High Tide agent actions
            swap_data = self._process_high_tide_agents(minute)
            
            # Process arbitrage agents (if advanced MOET system enabled)
            if hasattr(self, 'arbitrage_agents'):
                self._process_arbitrage_agents(minute)
            
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
            
            # Monitor MOET peg (for price tracking)
            if hasattr(self.__class__, '_monitor_moet_peg'):
                if minute % 1440 == 0:  # Debug log once per day
                    print(f"🔍 DEBUG: Calling _monitor_moet_peg at minute {minute}")
                self._monitor_moet_peg(minute)
            else:
                if minute % 1440 == 0:  # Debug log once per day
                    print(f"❌ DEBUG: _monitor_moet_peg method not found at minute {minute}")
            
            if minute % 10 == 0:
                print(f"Minute {minute}: BTC = ${new_btc_price:,.0f}, Active agents: {self._count_active_agents()}")
                
        return self._generate_high_tide_results()
        
    def _update_agent_debt_interest(self, minute: int):
        """Update debt interest for all High Tide agents using MOET system"""
        # Use the new MOET borrow rate system
        moet_borrow_rate = self.protocol.get_moet_borrow_rate()
        
        for agent in self.high_tide_agents:
            if agent.active:
                agent.update_debt_interest(minute, moet_borrow_rate)
    
    def _process_high_tide_agents(self, minute: int) -> Dict[str, Dict]:
        """Process High Tide agent actions for current minute with intra-loop rebalancing"""
        swap_data = {}
        agents_processed = 0
        
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
            
            agents_processed += 1
            
            # INTRA-LOOP REBALANCING: Check pool health after each agent action
            # Only check if we have a pool rebalancer and this was a leverage increase
            # SKIP if emergency rebalancing is disabled (e.g., during flash crash)
            if (hasattr(self, 'pool_rebalancer') and 
                action_type == AgentAction.BORROW and 
                params.get("leverage_increase", False) and
                not getattr(self.state, 'disable_emergency_rebalancing', False)):
                
                # Check if pool needs emergency rebalancing
                if self._should_trigger_emergency_rebalancing():
                    print(f"🚨 EMERGENCY REBALANCING triggered after agent {agents_processed} at minute {minute}")
                    self._execute_emergency_rebalancing(minute)
            
        return swap_data
    
    def _process_arbitrage_agents(self, minute: int) -> Dict[str, Dict]:
        """Process arbitrage agent actions with competition mechanism"""
        if not hasattr(self, 'arbitrage_agents'):
            return {}
        
        # Use the new competition-based processing from parent class
        swap_data = self._process_arbitrage_agents_with_competition(minute)
        
        # Store arbitrage events for tracking (like rebalancing events)
        for swap_key, swap_info in swap_data.items():
            arbitrage_event = {
                "minute": minute,
                "agent_id": swap_info.get("agent_id", "unknown"),
                "type": swap_info.get("type", "unknown"),
                "volume": swap_info.get("volume", 0),
                "profit": swap_info.get("profit", 0),
                "fees_generated": swap_info.get("fees_generated", 0),
                "success": True
            }
            self.arbitrage_events.append(arbitrage_event)
        
        return swap_data
    
    def _should_trigger_emergency_rebalancing(self) -> bool:
        """Check if emergency rebalancing should be triggered based on pool health"""
        try:
            # Check if yield token pool exists and has liquidity
            if not hasattr(self, 'yield_token_pool') or not self.yield_token_pool:
                return False
            
            # Get current pool liquidity from Uniswap V3 pool
            uniswap_pool = self.yield_token_pool.uniswap_pool
            if not hasattr(uniswap_pool, 'liquidity'):
                return False
            
            # Check if liquidity is critically low (less than 10% of initial)
            current_liquidity = uniswap_pool.liquidity
            initial_liquidity = 500_000_000_000  # Based on pool initialization
            
            liquidity_threshold = initial_liquidity * 0.1  # 10% threshold
            
            if current_liquidity < liquidity_threshold:
                print(f"⚠️  Pool liquidity critically low: {current_liquidity:,.0f} < {liquidity_threshold:,.0f}")
                return True
            
            # Also check if pool reserves are critically low
            moet_reserve = getattr(uniswap_pool, 'token0_reserve', 0)
            yt_reserve = getattr(uniswap_pool, 'token1_reserve', 0)
            
            if moet_reserve < 1000 or yt_reserve < 1000:  # Less than $1000 in either token
                print(f"⚠️  Pool reserves critically low: MOET={moet_reserve:.0f}, YT={yt_reserve:.0f}")
                return True
                
            return False
            
        except Exception as e:
            print(f"❌ Error checking pool health: {str(e)}")
            return False
    
    def _execute_emergency_rebalancing(self, minute: int):
        """Execute emergency pool rebalancing during agent processing"""
        try:
            # Import here to avoid circular imports
            from tidal_protocol_sim.core.yield_tokens import calculate_true_yield_token_price
            from tidal_protocol_sim.engine.state import Asset
            
            # Calculate current yield token prices and deviations
            # CHECK FOR ORACLE OVERRIDE (for flash crash scenarios)
            if hasattr(self.state, 'oracle_override_active') and self.state.oracle_override_active:
                true_yt_price = self.state.oracle_yt_price_override
            else:
                true_yt_price = calculate_true_yield_token_price(minute, 0.10, 1.0)
            
            pool_yt_price = self.yield_token_pool.uniswap_pool.get_price()
            deviation_bps = abs((pool_yt_price - true_yt_price) / true_yt_price) * 10000
            
            protocol_state = {
                "current_minute": minute,
                "true_yield_token_price": true_yt_price,
                "oracle_yt_price": true_yt_price,  # Include for rebalancer consistency
                "pool_yield_token_price": pool_yt_price,
                "deviation_bps": deviation_bps,
                "emergency_rebalancing": True  # Flag to indicate this is emergency
            }
            
            # Get current BTC price
            current_btc_price = self.state.current_prices.get(Asset.BTC, 50000)
            asset_prices = {Asset.BTC: current_btc_price}
            
            # Execute emergency rebalancing
            rebalancing_events = self.pool_rebalancer.process_rebalancing(protocol_state, asset_prices)
            
            if rebalancing_events:
                print(f"✅ Emergency rebalancing executed: {len(rebalancing_events)} events")
            else:
                print("⚠️  Emergency rebalancing attempted but no actions taken")
                
        except Exception as e:
            print(f"❌ Emergency rebalancing failed: {str(e)}")
            
    def _execute_high_tide_action(self, agent: HighTideAgent, action_type: AgentAction, params: dict, minute: int) -> tuple:
        """Execute High Tide specific actions"""
        if action_type == AgentAction.SWAP:
            swap_type = params.get("action_type", "")
            
            if swap_type == "buy_yield_tokens":
                success = self._execute_yield_token_purchase(agent, params, minute)
                return success, None
            elif swap_type in ["sell_yield_tokens", "emergency_sell_all_yield"]:
                success, swap_data = self._execute_yield_token_sale(agent, params, minute)
                return success, swap_data
        
        # NEW: Handle deleveraging actions
        elif action_type in ["delever_hf", "delever_weekly"]:
            success = self._execute_deleveraging_action(agent, action_type, params, minute)
            return success, None
                
        elif action_type == AgentAction.BORROW:
            if params.get("leverage_increase", False):
                success = self._execute_leverage_increase_borrow(agent, params, minute)
                return success, None
            else:
                # Handle regular borrows
                return super()._execute_agent_action(agent, action_type, params)
                
        elif action_type == AgentAction.LIQUIDATE:
            success = self._execute_liquidation(agent, params)
            return success, None
            
        return False, None
        
    def _execute_leverage_increase_borrow(self, agent: HighTideAgent, params: dict, minute: int) -> bool:
        """Execute additional MOET borrowing for leverage increase"""
        amount = params.get("amount", 0.0)
        
        if amount <= 0:
            return False
        
        # Reduced logging for ecosystem growth
        if minute % 1440 == 0 or len(self.high_tide_agents) <= 150:
            print(f"           📈 ENGINE: Executing leverage increase borrow for {agent.agent_id}")
            print(f"           Amount: ${amount:,.2f} MOET at minute {minute}")
        
        # Borrow additional MOET
        if self.protocol.borrow(agent.agent_id, amount):
            agent.state.borrowed_balances[Asset.MOET] += amount
            agent.state.moet_debt += amount
            agent.state.token_balances[Asset.MOET] += amount
            
            if minute % 1440 == 0 or len(self.high_tide_agents) <= 150:  # Reduced logging
                print(f"           ✅ Borrow successful - New debt: ${agent.state.moet_debt:,.2f}, MOET balance: ${agent.state.token_balances[Asset.MOET]:,.2f}")
            
            # Determine if we should use direct minting (minute 0 + config enabled)
            use_direct_minting = (minute == 0 and self.high_tide_config.use_direct_minting_for_initial)
            
            # Use new MOET to buy more yield tokens
            # Reduced logging for ecosystem growth - only log occasionally
            if minute % 1440 == 0 or len(self.high_tide_agents) <= 150:  # Daily or if small agent count
                print(f"           🚀 LEVERAGE INCREASE: Purchasing ${amount:,.2f} worth of yield tokens with borrowed MOET")
            yt_purchase_success = agent.execute_yield_token_purchase(amount, minute, use_direct_minting)
            
            if minute % 1440 == 0 or len(self.high_tide_agents) <= 150:  # Reduced logging
                if yt_purchase_success:
                    print(f"           ✅ Leverage increase YT purchase successful")
                else:
                    print(f"           ❌ Leverage increase YT purchase failed")
            
            # Record leverage increase event
            if yt_purchase_success:
                agent.state.leverage_increase_events.append({
                    "minute": minute,
                    "moet_borrowed": amount,
                    "health_factor_before": agent.state.health_factor
                })
            
            return yt_purchase_success
        
        print(f"           ❌ Borrow failed")
        return False
        
    def _execute_yield_token_purchase(self, agent: HighTideAgent, params: dict, minute: int) -> bool:
        """Execute yield token purchase for agent"""
        moet_amount = params.get("moet_amount", 0.0)
        
        if moet_amount <= 0:
            return False
        
        # Determine if we should use direct minting (minute 0 + config enabled)
        use_direct_minting = (minute == 0 and self.high_tide_config.use_direct_minting_for_initial)
        
        success = agent.execute_yield_token_purchase(moet_amount, minute, use_direct_minting)
        
        if success:
            # Always update pool state to maintain synchronization
            # For direct minting, we need to update the pool's internal reserves
            if use_direct_minting:
                # For direct minting, update pool reserves to reflect the 1:1 minting
                # This ensures pool state stays synchronized with agent state
                self.yield_token_pool.moet_reserve += moet_amount
                self.yield_token_pool.yield_token_reserve += moet_amount
            else:
                # For regular purchases, use the pool's execute method
                self.yield_token_pool.execute_yield_token_purchase(moet_amount)
            
            # Calculate yield tokens received for tracking
            if use_direct_minting:
                yield_tokens_received = moet_amount  # 1:1 rate
            else:
                yield_tokens_received = self.yield_token_pool.quote_yield_token_purchase(moet_amount)
            
            self.yield_token_trades.append({
                "minute": minute,
                "agent_id": agent.agent_id,
                "action": "purchase",
                "moet_amount": moet_amount,
                "yield_tokens_received": yield_tokens_received,
                "agent_health_factor": agent.state.health_factor,
                "use_direct_minting": use_direct_minting
            })
            
        return success
        
    def _execute_yield_token_sale(self, agent: HighTideAgent, params: dict, minute: int) -> tuple:
        """Execute yield token sale for rebalancing"""
        moet_amount_needed = params.get("moet_needed", params.get("moet_amount_needed", 0.0))  # Fix param name
        swap_type = params.get("swap_type", params.get("action_type", "sell_yield_tokens"))  # Fix param name
        
        if moet_amount_needed <= 0 and swap_type != "emergency_sell_all_yield":
            return False, None
            
        # Execute sale through agent
        if swap_type == "emergency_sell_all_yield":
            moet_raised = agent.execute_yield_token_sale(float('inf'), minute)
        else:
            moet_raised = agent.execute_yield_token_sale(moet_amount_needed, minute)
        
        if moet_raised > 0:
            
            # CRITICAL FIX: Don't double-execute the swap! The agent already did the real swap.
            # The engine should just record the event and update tracking.
            
            # Calculate debt repayment (agent already handled this, but we need it for tracking)
            debt_repayment = min(moet_raised, agent.state.moet_debt)
            
            # Calculate slippage for tracking
            slippage_cost = moet_amount_needed - moet_raised  # Simple slippage calculation
            
            # Record the rebalancing event for engine tracking
            rebalancing_event = {
                "agent_id": agent.agent_id,
                "minute": minute,
                "moet_needed": moet_amount_needed,
                "moet_raised": moet_raised,
                "swap_type": swap_type,
                "slippage_cost": slippage_cost
            }
            self.rebalancing_events.append(rebalancing_event)
            
            # Record pool activity for tracking (but don't double-execute)
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
            
            # Create swap data for tracking
            swap_data = {
                "yt_swapped": moet_amount_needed,  # Amount we tried to get
                "moet_received": moet_raised,
                "debt_repayment": debt_repayment,
                "swap_type": swap_type,
                "slippage_cost": slippage_cost,
                "slippage_percentage": (slippage_cost / moet_amount_needed) * 100 if moet_amount_needed > 0 else 0,
                "price_impact": 0.1  # Placeholder
            }
            
            # Record rebalancing event
            self.rebalancing_events.append({
                "minute": minute,
                "agent_id": agent.agent_id,
                "moet_raised": moet_raised,
                "moet_amount_needed": moet_amount_needed,
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
    
    def _execute_deleveraging_action(self, agent: HighTideAgent, action_type: str, params: dict, minute: int) -> bool:
        """Execute deleveraging action (HF threshold or weekly)"""
        yt_amount = params.get("yt_amount", 0)
        reason = params.get("reason", "unknown")
        
        if yt_amount <= 0:
            return False
        
        print(f"🔻 Engine executing {action_type} for {agent.agent_id}: ${yt_amount:,.0f} YT ({reason})")
        
        # Execute deleveraging through agent
        success = agent.execute_deleveraging(params, minute)
        
        if success:
            # Record deleveraging event for engine tracking
            deleveraging_event = {
                "agent_id": agent.agent_id,
                "minute": minute,
                "action_type": action_type,
                "yt_amount": yt_amount,
                "reason": reason,
                "health_factor_before": agent.state.health_factor
            }
            
            # Add to engine tracking (create list if doesn't exist)
            if not hasattr(self, 'deleveraging_events'):
                self.deleveraging_events = []
            self.deleveraging_events.append(deleveraging_event)
            
            print(f"   ✅ Deleveraging recorded in engine")
            
        return success
    
    def record_agent_rebalancing_event(self, agent_id: str, minute: int, moet_raised: float, 
                                     debt_repayment: float, slippage_cost: float, health_factor_before: float):
        """CRITICAL FIX: Method for agents to record real rebalancing events in engine"""
        self.rebalancing_events.append({
            "minute": minute,
            "agent_id": agent_id,
            "moet_raised": moet_raised,
            "moet_amount_needed": moet_raised,  # Approximate
            "debt_repayment": debt_repayment,
            "health_factor_before": health_factor_before,
            "rebalancing_type": "yield_token_sale",
            "slippage_cost": slippage_cost
        })
        
        # Also record in yield token trades
        self.yield_token_trades.append({
            "minute": minute,
            "agent_id": agent_id,
            "action": "rebalancing_sale",
            "moet_amount": moet_raised,
            "debt_repayment": debt_repayment,
            "agent_health_factor": health_factor_before,
            "slippage_cost": slippage_cost
        })
        
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
                liquidation_event = agent.execute_high_tide_liquidation(minute, self.state.current_prices, self)
                
                if liquidation_event:
                    self.liquidation_events.append(liquidation_event)
        
    def _count_active_agents(self) -> int:
        """Count number of active High Tide agents"""
        return sum(1 for agent in self.high_tide_agents if agent.active)
        
    def _record_high_tide_metrics(self, minute: int):
        """Record High Tide specific metrics"""
        # Base metrics
        super()._record_metrics()
        
        # PERFORMANCE OPTIMIZATION: Only record detailed agent health daily
        # This reduces memory usage from 12.6 GB to 8.8 MB (1,440x improvement)
        if minute % 1440 == 0:  # Every 24 hours (1440 minutes)
            print(f"📊 Recording daily agent health snapshot at minute {minute} (day {minute//1440 + 1})")
            
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
                    "rebalancing_events": portfolio["rebalancing_events_count"],
                    "btc_amount": portfolio["btc_amount"]  # Track BTC holdings for capital preservation analysis
                })
                
            self.agent_health_history.append({
                "minute": minute,
                "btc_price": self.state.current_prices[Asset.BTC],
                "agents": agent_health_data
            })
            
            # Record pool state snapshots (for Enhanced Redeemer charts)
            self._record_pool_state_snapshot(minute)
    
    def _record_pool_state_snapshot(self, minute: int):
        """Record pool state snapshots for Enhanced Redeemer analysis"""
        if not hasattr(self, 'pool_state_snapshots'):
            self.pool_state_snapshots = []
        
        snapshot = {"minute": minute}
        
        # Capture new pool structure if advanced MOET system is enabled
        if hasattr(self, 'moet_usdc_calculator') and hasattr(self, 'moet_usdf_calculator'):
            # MOET:USDC pool state
            try:
                moet_usdc_price = self.moet_usdc_pool.get_price()
                snapshot["moet_usdc_price"] = moet_usdc_price
                snapshot["moet_usdc_liquidity"] = self.moet_usdc_pool.liquidity
            except:
                snapshot["moet_usdc_price"] = 1.0
                snapshot["moet_usdc_liquidity"] = 0
            
            # MOET:USDF pool state  
            try:
                moet_usdf_price = self.moet_usdf_pool.get_price()
                snapshot["moet_usdf_price"] = moet_usdf_price
                snapshot["moet_usdf_liquidity"] = self.moet_usdf_pool.liquidity
            except:
                snapshot["moet_usdf_price"] = 1.0
                snapshot["moet_usdf_liquidity"] = 0
            
            # USDC:BTC pool state
            try:
                usdc_btc_price = self.usdc_btc_pool.get_price()
                snapshot["usdc_btc_price"] = usdc_btc_price
                snapshot["usdc_btc_liquidity"] = self.usdc_btc_pool.liquidity
            except:
                snapshot["usdc_btc_price"] = self.state.current_prices[Asset.BTC]
                snapshot["usdc_btc_liquidity"] = 0
            
            # USDF:BTC pool state
            try:
                usdf_btc_price = self.usdf_btc_pool.get_price()
                snapshot["usdf_btc_price"] = usdf_btc_price
                snapshot["usdf_btc_liquidity"] = self.usdf_btc_pool.liquidity
            except:
                snapshot["usdf_btc_price"] = self.state.current_prices[Asset.BTC]
                snapshot["usdf_btc_liquidity"] = 0
        
        # Legacy pool state (for backward compatibility)
        if hasattr(self, 'slippage_calculator'):
            try:
                legacy_price = self.slippage_calculator.pool.get_price()
                snapshot["legacy_pool_price"] = legacy_price
                snapshot["legacy_pool_liquidity"] = self.slippage_calculator.pool.liquidity
            except:
                snapshot["legacy_pool_price"] = self.state.current_prices[Asset.BTC]
                snapshot["legacy_pool_liquidity"] = 0
        
        self.pool_state_snapshots.append(snapshot)
    
    def _collect_moet_system_data(self) -> dict:
        """Collect MOET system data for redeemer and arbitrage charts"""
        
        # Check if advanced MOET system is enabled
        config_enabled = getattr(self.config, 'enable_advanced_moet_system', False)
        ht_config_enabled = getattr(self.high_tide_config, 'enable_advanced_moet_system', False)
        advanced_enabled = config_enabled or ht_config_enabled
        
        print(f"🔧 DEBUG: Advanced MOET system check:")
        print(f"   self.config.enable_advanced_moet_system: {config_enabled}")
        print(f"   self.high_tide_config.enable_advanced_moet_system: {ht_config_enabled}")
        print(f"   Final advanced_enabled: {advanced_enabled}")
        
        # Initialize data structure
        moet_data = {
            "advanced_system_enabled": advanced_enabled,  # CRITICAL FIX: Add this flag
            "redeemer_system": {},
            "tracking_data": {
                "reserve_history": [],
                "arbitrage_history": [],
                "pool_price_history": [],
                "deficit_history": []  # Add deficit tracking for chart compatibility
            },
            "arbitrage_agents_summary": []
        }
        
        # Collect redeemer system data if advanced MOET system is enabled
        if hasattr(self.protocol, 'enhanced_redeemer') and self.protocol.enhanced_redeemer:
            redeemer = self.protocol.enhanced_redeemer
            
            moet_data["redeemer_system"] = {
                "total_mints": getattr(redeemer, 'total_mints', 0),
                "total_redemptions": getattr(redeemer, 'total_redemptions', 0),
                "total_fees_collected": getattr(redeemer, 'total_fees_collected', 0),
                "current_usdc_balance": getattr(redeemer, 'usdc_balance', 0),
                "current_usdf_balance": getattr(redeemer, 'usdf_balance', 0),
                "current_fee_rate": getattr(redeemer, 'current_fee_rate', 0.001),
                "peg_stability_score": self._calculate_peg_stability_score()
            }
        
        # CRITICAL FIX: Collect bonder system data for bond auction charts
        if advanced_enabled and hasattr(self.protocol, 'moet_system') and self.protocol.moet_system:
            moet_system = self.protocol.moet_system
            
            # Get bonder system data
            if hasattr(moet_system, 'bonder_system') and moet_system.bonder_system:
                bonder = moet_system.bonder_system
                
                moet_data["bonder_system"] = {
                    "auction_history_count": len(getattr(bonder, 'auction_history', [])),
                    "current_bond_cost_ema": getattr(bonder, 'current_bond_cost_ema', 0),
                    "pending_auction": getattr(bonder, 'pending_auction', None) is not None,
                    "recent_auctions": [
                        {
                            'timestamp': auction.timestamp,
                            'final_apr': auction.final_apr,
                            'amount_filled': auction.amount_filled,
                            'filled_completely': auction.filled_completely,
                            'target_amount': auction.target_amount
                        }
                        for auction in getattr(bonder, 'auction_history', [])[-5:]  # Last 5 auctions
                    ]
                }
            
            # Get MOET system state for tracking data
            moet_state = moet_system.get_state()
            if 'tracking_data' in moet_state:
                tracking = moet_state['tracking_data']
                moet_data["tracking_data"]["bond_apr_history"] = tracking.get("bond_apr_history", [])
                moet_data["tracking_data"]["moet_rate_history"] = tracking.get("moet_rate_history", [])
                moet_data["tracking_data"]["reserve_history"] = tracking.get("reserve_history", [])
                moet_data["tracking_data"]["deficit_history"] = tracking.get("deficit_history", [])
        
        # Collect pool state snapshots as reserve history
        if hasattr(self, 'pool_state_snapshots'):
            for i, snapshot in enumerate(self.pool_state_snapshots):
                if i % 60 == 0:  # Sample every hour
                    hour = snapshot.get("minute", 0) / 60
                    
                    # Estimate reserve balances from pool data
                    usdc_balance = 100000  # Default values if no pool data
                    usdf_balance = 100000
                    
                    if hasattr(self.protocol, 'enhanced_redeemer') and self.protocol.enhanced_redeemer:
                        usdc_balance = getattr(self.protocol.enhanced_redeemer, 'usdc_balance', 100000)
                        usdf_balance = getattr(self.protocol.enhanced_redeemer, 'usdf_balance', 100000)
                    
                    # Calculate reserve metrics for chart compatibility
                    total_reserves = usdc_balance + usdf_balance
                    target_reserves = total_reserves  # Simplified for now
                    total_moet_supply = getattr(self.protocol.moet_system, 'total_supply', 1000000) if hasattr(self.protocol, 'moet_system') else 1000000
                    reserve_ratio = (total_reserves / total_moet_supply) if total_moet_supply > 0 else 0
                    deficit = max(0, target_reserves - total_reserves)  # Calculate deficit
                    
                    moet_data["tracking_data"]["reserve_history"].append({
                        "minute": hour * 60,  # Convert back to minutes for compatibility
                        "hour": hour,
                        "usdc_balance": usdc_balance,
                        "usdf_balance": usdf_balance,
                        "actual_reserves": total_reserves,
                        "target_reserves": target_reserves,
                        "reserve_ratio": reserve_ratio,
                        "moet_usdc_price": snapshot.get("moet_usdc_price", 1.0),
                        "moet_usdf_price": snapshot.get("moet_usdf_price", 1.0)
                    })
                    
                    # Add deficit data for chart compatibility
                    moet_data["tracking_data"]["deficit_history"].append({
                        "minute": hour * 60,
                        "hour": hour,
                        "deficit": deficit
                    })
        
        # Collect arbitrage agent data
        if hasattr(self, 'arbitrage_agents') and self.arbitrage_agents:
            print(f"   ✅ Collecting data from {len(self.arbitrage_agents)} arbitrage agents")
            for agent in self.arbitrage_agents:
                # Use the new detailed summary method
                agent_summary = agent.get_detailed_portfolio_summary()
                print(f"   Agent {agent.agent_id}: {agent_summary.get('total_attempts', 0)} attempts, {agent_summary.get('total_arbitrage_events', 0)} executed, ${agent_summary.get('total_profit', 0):.2f} profit")
                
                # Store comprehensive data
                moet_data["arbitrage_agents_summary"].append({
                    "agent_id": agent.agent_id,
                    "agent_type": agent_summary.get("agent_type", "moet_arbitrage_agent"),
                    "initial_balance": agent_summary.get("initial_balance", 0),
                    "current_balance": agent_summary.get("current_balance", 0),
                    "net_profit": agent_summary.get("net_profit", 0),
                    "total_profit": agent_summary.get("total_profit", 0),
                    "total_attempts": agent_summary.get("total_attempts", 0),
                    "total_mint_attempts": agent_summary.get("total_mint_attempts", 0),
                    "total_redeem_attempts": agent_summary.get("total_redeem_attempts", 0),
                    "total_volume_traded": agent_summary.get("total_volume_traded", 0),
                    "total_fees_generated": agent_summary.get("total_fees_generated", 0),
                    "successful_arbitrages": agent_summary.get("successful_arbitrages", 0),
                    "failed_arbitrages": agent_summary.get("failed_arbitrages", 0),
                    "total_arbitrage_events": agent_summary.get("total_arbitrage_events", 0),
                    "execution_rate": agent_summary.get("execution_rate", 0),
                    "success_rate": agent_summary.get("success_rate", 0),
                    "average_profit": agent_summary.get("average_profit", 0),
                    "average_trade_size": agent_summary.get("average_trade_size", 0),
                    "attempts_breakdown": agent_summary.get("attempts_breakdown", {}),
                    "arbitrage_events": getattr(agent.state, 'arbitrage_events', []),
                    "arbitrage_attempts": getattr(agent.state, 'arbitrage_attempts', [])
                })
                
                # Add ALL arbitrage attempts to history (both executed and not executed)
                for attempt in getattr(agent.state, 'arbitrage_attempts', []):
                    moet_data["tracking_data"]["arbitrage_history"].append({
                        "minute": attempt.get("minute", 0),
                        "hour": attempt.get("minute", 0) / 60,
                        "agent_id": agent.agent_id,
                        "arbitrage_type": attempt.get("type", "unknown"),
                        "executed": attempt.get("executed", False),
                        "expected_profit": attempt.get("expected_profit", 0),
                        "actual_profit": attempt.get("actual_profit", 0),
                        "trade_size": attempt.get("trade_size", 0),
                        "pool_used": attempt.get("pool", "unknown"),
                        "moet_price": attempt.get("moet_price", 1.0),
                        "reason_not_executed": attempt.get("reason_not_executed", None)
                    })
        else:
            print(f"   ⚠️  No arbitrage agents found for data collection")
        
        return moet_data
    
    def _calculate_peg_stability_score(self) -> float:
        """Calculate a peg stability score based on recent price deviations"""
        if not hasattr(self, 'pool_state_snapshots') or not self.pool_state_snapshots:
            return 1.0  # Perfect stability if no data
        
        # Look at recent snapshots (last 100 or all if fewer)
        recent_snapshots = self.pool_state_snapshots[-100:]
        
        total_deviation = 0.0
        count = 0
        
        for snapshot in recent_snapshots:
            usdc_price = snapshot.get("moet_usdc_price", 1.0)
            usdf_price = snapshot.get("moet_usdf_price", 1.0)
            
            # Calculate deviation from $1.00 peg
            usdc_deviation = abs(usdc_price - 1.0)
            usdf_deviation = abs(usdf_price - 1.0)
            
            total_deviation += (usdc_deviation + usdf_deviation) / 2
            count += 1
        
        if count == 0:
            return 1.0
        
        avg_deviation = total_deviation / count
        # Convert to stability score (1.0 = perfect, 0.0 = very unstable)
        stability_score = max(0.0, 1.0 - (avg_deviation * 100))  # Scale by 100x
        
        return stability_score
    
    def _generate_high_tide_results(self) -> dict:
        """Generate comprehensive High Tide simulation results"""
        base_results = super()._generate_results()
        
        # Calculate High Tide specific metrics
        final_minute = self.high_tide_config.btc_decline_duration - 1
        
        # Agent outcomes - include both High Tide agents AND arbitrage agents
        agent_outcomes = []
        total_cost_of_rebalancing = 0.0
        survival_by_risk_profile = {"conservative": 0, "moderate": 0, "aggressive": 0}
        
        # Process High Tide agents
        for agent in self.high_tide_agents:
            agent._update_health_factor(self.state.current_prices)
            
            portfolio = agent.get_detailed_portfolio_summary(
                self.state.current_prices, 
                final_minute
            )
            
            # CRITICAL FIX: Calculate real costs from engine-level swap data instead of agent portfolio data
            real_rebalancing_cost = sum(event["slippage_cost"] for event in self.rebalancing_events 
                                       if event["agent_id"] == agent.agent_id)
            real_slippage_cost = sum(trade.get("slippage_cost", 0.0) for trade in self.yield_token_trades 
                                    if trade.get("agent_id") == agent.agent_id)
            
            # Get real rebalancing events from engine data
            agent_rebalancing_events = [event for event in self.rebalancing_events 
                                       if event["agent_id"] == agent.agent_id]
            
            # Get leverage increase events from agent state
            agent_leverage_events = agent.state.leverage_increase_events if hasattr(agent.state, 'leverage_increase_events') else []
            
            outcome = {
                "agent_id": agent.agent_id,
                "agent_type": "high_tide_agent",  # Specify agent type
                "risk_profile": agent.risk_profile,
                "target_health_factor": agent.state.target_health_factor,
                "initial_health_factor": agent.state.initial_health_factor,
                "final_health_factor": agent.state.health_factor,
                "cost_of_rebalancing": real_rebalancing_cost,  # FIXED: Real Uniswap V3 swap costs
                "total_slippage_costs": real_slippage_cost,    # FIXED: Real slippage from engine
                "net_position_value": portfolio["net_position_value"],
                "total_yield_earned": portfolio["yield_token_portfolio"]["total_accrued_yield"],
                "total_yield_sold": portfolio["total_yield_sold"],
                "rebalancing_events": len(agent_rebalancing_events),
                "rebalancing_events_list": agent_rebalancing_events,  # FIXED: Real engine events
                "leverage_increase_events": len(agent_leverage_events),  # NEW: Track leverage increases
                "leverage_increase_events_list": agent_leverage_events,  # NEW: Full event list
                "total_position_adjustments": len(agent_rebalancing_events) + len(agent_leverage_events),  # NEW: Combined metric
                "survived": agent.state.health_factor > 1.0,
                "yield_token_value": portfolio["yield_token_portfolio"]["total_current_value"],
                # Add debt tracking fields for CSV
                "initial_moet_debt": portfolio["initial_moet_debt"],
                "current_moet_debt": portfolio["current_moet_debt"],
                "total_interest_accrued": portfolio["total_interest_accrued"],
                "btc_amount": portfolio["btc_amount"],
                "yield_token_portfolio": portfolio["yield_token_portfolio"],
                # Add deleveraging data
                "deleveraging_events": portfolio.get("deleveraging_events", []),
                "deleveraging_events_count": portfolio.get("deleveraging_events_count", 0),
                "total_deleveraging_sales": portfolio.get("total_deleveraging_sales", 0),
                # Add flag to indicate this uses real engine data
                "data_source": "engine_real_swaps"
            }
            
            agent_outcomes.append(outcome)
            total_cost_of_rebalancing += outcome["cost_of_rebalancing"]
            
            if outcome["survived"]:
                survival_by_risk_profile[agent.risk_profile] += 1
        
        # Process arbitrage agents (if they exist)
        if hasattr(self, 'arbitrage_agents') and self.arbitrage_agents:
            for agent in self.arbitrage_agents:
                # Get arbitrage agent summary
                agent_summary = agent.get_summary()
                
                outcome = {
                    "agent_id": agent.agent_id,
                    "agent_type": "moet_arbitrage_agent",  # Specify agent type
                    "total_profit": agent_summary.get("total_profit", 0),
                    "successful_arbitrages": agent_summary.get("successful_arbitrages", 0),
                    "failed_arbitrages": agent_summary.get("failed_arbitrages", 0),
                    "success_rate": agent_summary.get("success_rate", 0),
                    "average_profit": agent_summary.get("average_profit", 0),
                    "total_arbitrage_events": agent_summary.get("total_arbitrage_events", 0),
                    "arbitrage_events": getattr(agent.state, 'arbitrage_events', []),
                    "initial_balance": agent.state.initial_balance,
                    "current_balance": agent.state.token_balances.get(Asset.USDC, 0),  # Using USDC as proxy
                    "survived": True,  # Arbitrage agents don't have health factors, so they always "survive"
                    "net_position_value": agent.state.token_balances.get(Asset.USDC, 0),  # Current balance as net value
                    "data_source": "arbitrage_agent"
                }
                
                agent_outcomes.append(outcome)
                
        # Collect MOET system data for redeemer charts
        moet_system_data = self._collect_moet_system_data()
        
        # High Tide specific results
        high_tide_results = {
            "scenario_type": "High_Tide_BTC_Decline",
            "btc_decline_statistics": self.btc_price_manager.get_decline_statistics(),
            "agent_outcomes": agent_outcomes,
            "moet_system_state": moet_system_data,  # Add MOET system data for charts
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
            "yield_token_trades": self.yield_token_trades,
            "arbitrage_events": self.arbitrage_events
        }
        
        # Add LP curve tracking data
        high_tide_results["lp_curve_data"] = {
            "moet_yield_tracker_snapshots": self.moet_yield_tracker.get_snapshots(),
            "pool_name": "MOET:Yield_Token",
            "concentration_range": self.config.yield_token_concentration
        }
        
        # Add peg monitoring data (for Redeemer charts)
        if hasattr(self, 'peg_monitoring'):
            high_tide_results["peg_monitoring"] = self.peg_monitoring
            high_tide_results["peg_monitoring_summary"] = self.get_peg_monitoring_summary()
        
        # Add pool state snapshots (for Enhanced Redeemer charts)
        if hasattr(self, 'pool_state_snapshots'):
            high_tide_results["pool_state_snapshots"] = self.pool_state_snapshots
        
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
