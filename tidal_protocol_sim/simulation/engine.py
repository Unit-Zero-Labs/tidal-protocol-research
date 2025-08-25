#!/usr/bin/env python3
"""
Streamlined Simulation Engine

Direct function calls instead of event system as specified in refactoring requirements.
"""

from typing import Dict, List, Optional
import random
from ..core.protocol import TidalProtocol, Asset
from ..agents.base_agent import BaseAgent, AgentAction
from ..agents.tidal_lender import TidalLender
from ..agents.trader import BasicTrader
from ..agents.liquidator import Liquidator
from .state import SimulationState
from .config import SimulationConfig


class TidalSimulationEngine:
    """Streamlined simulation runner with direct function calls"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.protocol = TidalProtocol()
        self.agents = self._initialize_agents()
        self.state = SimulationState()
        self.current_step = 0
        
        # Initialize agent positions in protocol
        self._setup_initial_positions()
        
        # Enhanced simulation metrics for comprehensive analysis
        self.metrics_history = []
        self.liquidation_events = []
        self.trade_events = []
        self.agent_actions_history = []
        self.protocol_state_history = []
        
    def _initialize_agents(self) -> Dict[str, BaseAgent]:
        """Initialize agents based on configuration"""
        agents = {}
        
        # Create TidalLender agents
        for i in range(self.config.num_lenders):
            agent_id = f"lender_{i}"
            agents[agent_id] = TidalLender(agent_id, self.config.lender_initial_balance)
        
        # Create BasicTrader agents
        for i in range(self.config.num_traders):
            agent_id = f"trader_{i}"
            agents[agent_id] = BasicTrader(agent_id, self.config.trader_initial_balance)
        
        # Create Liquidator agents
        for i in range(self.config.num_liquidators):
            agent_id = f"liquidator_{i}"
            agents[agent_id] = Liquidator(agent_id, self.config.liquidator_initial_balance)
        
        return agents
    
    def _setup_initial_positions(self):
        """Set up initial agent positions in protocol to match agent state"""
        for agent_id, agent in self.agents.items():
            # Register supplied balances with protocol
            for asset, amount in agent.state.supplied_balances.items():
                if amount > 0:
                    # Update protocol pool
                    if asset in self.protocol.asset_pools:
                        pool = self.protocol.asset_pools[asset]
                        pool.total_supplied += amount
            
            # Register borrowed balances with protocol
            for asset, amount in agent.state.borrowed_balances.items():
                if amount > 0 and asset == Asset.MOET:
                    # Update MOET system
                    self.protocol.moet_system.mint(amount)
            
            # Update agent health factors
            self._update_agent_health_factor(agent)
    
    def _update_agent_health_factor(self, agent: BaseAgent):
        """Update agent's health factor based on current state"""
        collateral_factors = {
            Asset.ETH: 0.75,
            Asset.BTC: 0.75,
            Asset.FLOW: 0.50,
            Asset.USDC: 0.90
        }
        agent.state.update_health_factor(self.state.current_prices, collateral_factors)
    
    def run_simulation(self, steps: int) -> Dict:
        """Run simulation for specified number of steps"""
        
        for step in range(steps):
            self.current_step = step
            
            # Update protocol state (interest accrual, etc.)
            self.protocol.current_block = step
            self.protocol.accrue_interest()
            
            # Process agent actions
            self._process_agent_actions()
            
            # Apply market dynamics
            if step % self.config.price_update_frequency == 0:
                self._update_market_prices()
            
            # Record metrics
            self._record_metrics()
            
            # Check for liquidations
            self._check_liquidations()
            
            if step % 100 == 0:
                print(f"Simulation step {step}/{steps}")
        
        return self._generate_results()
    
    def _process_agent_actions(self):
        """Process actions for all agents"""
        
        for agent_id, agent in self.agents.items():
            if not agent.active:
                continue
            
            # Get current protocol state
            protocol_state = self._get_protocol_state()
            asset_prices = self.state.current_prices
            
            # Agent decides action
            action_type, params = agent.decide_action(protocol_state, asset_prices)
            
            # Execute action
            if action_type != AgentAction.HOLD:
                success = self._execute_agent_action(agent, action_type, params)
                
                if success:
                    self._record_agent_action(agent_id, action_type, params)
    
    def _execute_agent_action(self, agent: BaseAgent, action_type: AgentAction, params: dict) -> bool:
        """Execute agent action through protocol"""
        
        try:
            if action_type == AgentAction.SUPPLY:
                asset = params.get("asset")
                amount = params.get("amount", 0.0)
                
                if self._check_balance(agent, asset, amount):
                    # Update agent balance
                    agent.state.token_balances[asset] -= amount
                    agent.state.supplied_balances[asset] += amount
                    
                    # Update protocol
                    return self.protocol.supply(agent.agent_id, asset, amount)
                
            elif action_type == AgentAction.BORROW:
                amount = params.get("amount", 0.0)
                
                # Check collateral and borrow
                if self.protocol.borrow(agent.agent_id, amount):
                    agent.state.borrowed_balances[Asset.MOET] += amount
                    agent.state.token_balances[Asset.MOET] += amount
                    return True
                
            elif action_type == AgentAction.REPAY:
                amount = params.get("amount", 0.0)
                
                if self._check_balance(agent, Asset.MOET, amount):
                    agent.state.token_balances[Asset.MOET] -= amount
                    agent.state.borrowed_balances[Asset.MOET] -= amount
                    return self.protocol.repay(agent.agent_id, amount)
                
            elif action_type == AgentAction.LIQUIDATE:
                return self._execute_liquidation(agent, params)
                
            elif action_type == AgentAction.SWAP:
                return self._execute_swap(agent, params)
            
        except Exception as e:
            print(f"Error executing action {action_type} for {agent.agent_id}: {e}")
            return False
        
        return False
    
    def _execute_liquidation(self, liquidator: BaseAgent, params: dict) -> bool:
        """Execute liquidation"""
        target_id = params.get("target_agent_id")
        collateral_asset = params.get("collateral_asset")
        repay_amount = params.get("repay_amount", 0.0)
        
        if target_id not in self.agents:
            return False
        
        target_agent = self.agents[target_id]
        
        # Check if liquidation is valid
        if target_agent.state.health_factor >= 1.0:
            return False
        
        # Execute liquidation
        success = self.protocol.liquidate(
            liquidator.agent_id, target_id, collateral_asset, repay_amount
        )
        
        if success:
            # Update agent states
            liquidator.state.token_balances[Asset.MOET] -= repay_amount
            
            # Calculate collateral seized
            asset_price = self.state.current_prices.get(collateral_asset, 1.0)
            collateral_value = repay_amount * 1.08  # 8% penalty
            collateral_amount = collateral_value / asset_price
            
            # Transfer collateral
            target_agent.state.supplied_balances[collateral_asset] -= collateral_amount
            liquidator.state.token_balances[collateral_asset] += collateral_amount
            
            # Update debt
            target_agent.state.borrowed_balances[Asset.MOET] -= repay_amount
            
            # Record liquidation
            self.liquidation_events.append({
                "step": self.current_step,
                "liquidator": liquidator.agent_id,
                "target": target_id,
                "asset": collateral_asset.value if hasattr(collateral_asset, 'value') else str(collateral_asset),
                "repay_amount": repay_amount,
                "collateral_seized": collateral_amount
            })
            
            return True
        
        return False
    
    def _execute_swap(self, agent: BaseAgent, params: dict) -> bool:
        """Execute swap through liquidity pools"""
        asset_in = params.get("asset_in")
        asset_out = params.get("asset_out")
        amount_in = params.get("amount_in", 0.0)
        
        if not self._check_balance(agent, asset_in, amount_in):
            return False
        
        # Find appropriate pool
        pool_key = f"MOET_{asset_in.value}" if asset_out == Asset.MOET else f"MOET_{asset_out.value}"
        
        if pool_key in self.protocol.liquidity_pools:
            pool = self.protocol.liquidity_pools[pool_key]
            amount_out, fee, slippage = pool.calculate_swap_output(amount_in, asset_in, asset_out)
            
            if amount_out > 0:
                # Update agent balances
                agent.state.token_balances[asset_in] -= amount_in
                agent.state.token_balances[asset_out] += amount_out
                
                # Update pool reserves
                pool.update_reserves(asset_in, amount_in, asset_out, amount_out)
                
                # Record trade
                self.trade_events.append({
                    "step": self.current_step,
                    "agent": agent.agent_id,
                    "asset_in": asset_in.value if hasattr(asset_in, 'value') else str(asset_in),
                    "asset_out": asset_out.value if hasattr(asset_out, 'value') else str(asset_out),
                    "amount_in": amount_in,
                    "amount_out": amount_out,
                    "slippage": slippage
                })
                
                return True
        
        return False
    
    def _check_balance(self, agent: BaseAgent, asset: Asset, amount: float) -> bool:
        """Check if agent has sufficient balance"""
        balance = agent.state.token_balances.get(asset, 0.0)
        return balance >= amount
    
    def _update_market_prices(self):
        """Update market prices with volatility"""
        
        for asset in [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]:
            current_price = self.state.current_prices[asset]
            
            # Apply random volatility
            if asset == Asset.FLOW:
                volatility = random.gauss(0, 0.03)  # 3% daily volatility
            elif asset == Asset.USDC:
                volatility = random.gauss(0, 0.001)  # 0.1% for stablecoin
            else:
                volatility = random.gauss(0, 0.02)  # 2% daily volatility
            
            new_price = current_price * (1 + volatility)
            new_price = max(new_price, current_price * 0.5)  # Floor at 50% drop
            
            self.state.current_prices[asset] = new_price
    
    def _check_liquidations(self):
        """Check all agents for liquidation eligibility"""
        
        for agent in self.agents.values():
            # Update health factors
            self._update_agent_health_factor(agent)
            
            # Mark for liquidation if unhealthy
            if agent.state.health_factor < 1.0:
                self.state.liquidatable_agents.add(agent.agent_id)
            else:
                self.state.liquidatable_agents.discard(agent.agent_id)
    
    def _get_protocol_state(self) -> dict:
        """Get current protocol state for agents"""
        
        # Calculate utilization and rates
        utilization = {}
        borrow_rates = {}
        supply_rates = {}
        
        for asset, pool in self.protocol.asset_pools.items():
            utilization[asset.value] = pool.utilization_rate
            borrow_rates[asset.value] = pool.calculate_borrow_rate()
            supply_rates[asset.value] = pool.calculate_supply_rate()
        
        return {
            "utilization": utilization,
            "borrow_rates": borrow_rates,
            "supply_rates": supply_rates,
            "debt_cap": self.protocol.calculate_debt_cap(),
            "protocol_treasury": self.protocol.protocol_treasury,
            "moet_price": self.state.current_prices[Asset.MOET]
        }
    
    def _record_metrics(self):
        """Record comprehensive simulation metrics"""
        
        protocol_state = self._get_protocol_state()
        
        # Calculate total values from agent states (more accurate)
        total_supplied = 0.0
        total_borrowed = 0.0
        
        for agent in self.agents.values():
            # Sum agent supplied balances
            for asset, amount in agent.state.supplied_balances.items():
                if asset != Asset.MOET:
                    price = self.state.current_prices.get(asset, 1.0)
                    total_supplied += amount * price
            
            # Sum agent borrowed balances
            total_borrowed += agent.state.get_total_debt_value(self.state.current_prices)
        
        # Enhanced metrics with additional details
        metrics = {
            "step": self.current_step,
            "timestamp": self.current_step,  # For time-series analysis
            "total_supplied": total_supplied,
            "total_borrowed": total_borrowed,
            "protocol_treasury": protocol_state["protocol_treasury"],
            "debt_cap": protocol_state["debt_cap"],
            "liquidatable_agents": len(self.state.liquidatable_agents),
            "asset_prices": {asset.value if hasattr(asset, 'value') else str(asset): price 
                           for asset, price in self.state.current_prices.items()},
            "utilization_rates": protocol_state["utilization"],
            "borrow_rates": protocol_state["borrow_rates"],
            "supply_rates": protocol_state["supply_rates"],
            
            # Additional detailed metrics
            "active_agents": sum(1 for agent in self.agents.values() if agent.active),
            "total_collateral_value": self._calculate_total_collateral_value(),
            "average_health_factor": self._calculate_average_health_factor(),
            "protocol_revenue": self._calculate_protocol_revenue(),
            
            # Risk metrics
            "liquidation_risk_score": len(self.state.liquidatable_agents) / len(self.agents),
            "debt_cap_utilization": total_borrowed / protocol_state["debt_cap"] if protocol_state["debt_cap"] > 0 else 0,
            
            # Asset-specific metrics
            "asset_pool_details": self._get_asset_pool_details()
        }
        
        self.metrics_history.append(metrics)
        
        # Also record detailed protocol state
        self.protocol_state_history.append({
            "step": self.current_step,
            "protocol_state": protocol_state.copy(),
            "agent_health_factors": {
                agent_id: agent.state.health_factor 
                for agent_id, agent in self.agents.items()
            }
        })
    
    def _record_agent_action(self, agent_id: str, action_type: AgentAction, params: dict):
        """Record agent action for comprehensive analysis"""
        # Clean params to ensure JSON serialization
        clean_params = {}
        for key, value in params.items():
            if hasattr(value, 'value'):  # Asset enum
                clean_params[key] = value.value
            else:
                clean_params[key] = value
        
        action_record = {
            "step": self.current_step,
            "agent_id": agent_id,
            "action_type": action_type.value if hasattr(action_type, 'value') else str(action_type),
            "parameters": clean_params,
            "agent_health_factor": self.agents[agent_id].state.health_factor,
            "timestamp": self.current_step
        }
        self.agent_actions_history.append(action_record)
    
    def _calculate_total_collateral_value(self) -> float:
        """Calculate total collateral value across all agents"""
        total_collateral = 0.0
        for agent in self.agents.values():
            for asset, amount in agent.state.supplied_balances.items():
                price = self.state.current_prices.get(asset, 1.0)
                total_collateral += amount * price
        return total_collateral
    
    def _calculate_average_health_factor(self) -> float:
        """Calculate average health factor across all agents with debt"""
        health_factors = []
        for agent in self.agents.values():
            hf = agent.state.health_factor
            # Only include agents with debt (finite health factors)
            if hf != float('inf') and hf > 0:
                health_factors.append(hf)
        
        return sum(health_factors) / len(health_factors) if health_factors else 2.0  # Safe default
    
    def _calculate_protocol_revenue(self) -> float:
        """Calculate protocol revenue from interest and fees"""
        # Calculate accumulated interest revenue
        revenue = self.protocol.protocol_treasury
        
        # Add trading fees from recent trades
        recent_trade_fees = 0.0
        for trade in self.trade_events:
            if trade.get("step", 0) >= self.current_step - 10:  # Last 10 steps
                # Estimate trading fees (0.3% of trade volume)
                trade_volume = trade.get("amount_in", 0.0) * self.state.current_prices.get(
                    trade.get("asset_in"), 1.0
                )
                recent_trade_fees += trade_volume * 0.003
        
        return revenue + recent_trade_fees
    
    def _get_asset_pool_details(self) -> Dict[str, Dict[str, float]]:
        """Get detailed information about each asset pool"""
        pool_details = {}
        for asset, pool in self.protocol.asset_pools.items():
            asset_key = asset.value if hasattr(asset, 'value') else str(asset)
            pool_details[asset_key] = {
                "total_supplied": pool.total_supplied,
                "total_borrowed": pool.total_borrowed,
                "utilization_rate": pool.utilization_rate,
                "borrow_rate": pool.calculate_borrow_rate(),
                "supply_rate": pool.calculate_supply_rate()
            }
        return pool_details
    
    def _generate_results(self) -> dict:
        """Generate final simulation results"""
        
        # Calculate final totals
        total_supplied = 0.0
        total_borrowed = 0.0
        
        for agent in self.agents.values():
            # Sum agent supplied balances
            for asset, amount in agent.state.supplied_balances.items():
                if asset != Asset.MOET:
                    price = self.state.current_prices.get(asset, 1.0)
                    total_supplied += amount * price
            
            # Sum agent borrowed balances
            total_borrowed += agent.state.get_total_debt_value(self.state.current_prices)
        
        return {
            # Core simulation data
            "metrics_history": self.metrics_history,
            "liquidation_events": self.liquidation_events,
            "trade_events": self.trade_events,
            "agent_actions_history": self.agent_actions_history,
            "protocol_state_history": self.protocol_state_history,
            
            # Final states
            "final_protocol_state": self._get_protocol_state(),
            "agent_states": {
                agent_id: agent.get_portfolio_summary(self.state.current_prices)
                for agent_id, agent in self.agents.items()
            },
            
            # Configuration and summary
            "simulation_config": {
                "steps": self.current_step,
                "num_agents": len(self.agents),
                "num_liquidations": len(self.liquidation_events),
                "num_trades": len(self.trade_events),
                "num_agent_actions": len(self.agent_actions_history),
                "initial_config": {
                    "num_lenders": self.config.num_lenders,
                    "num_traders": self.config.num_traders,
                    "num_liquidators": self.config.num_liquidators,
                    "simulation_steps": self.config.simulation_steps
                }
            },
            
            # Enhanced summary statistics
            "summary_statistics": {
                "total_liquidation_value": sum(event.get("repay_amount", 0) for event in self.liquidation_events),
                "total_trade_volume": sum(
                    event.get("amount_in", 0) * self.state.current_prices.get(event.get("asset_in"), 1.0)
                    for event in self.trade_events
                ),
                "final_total_supplied": total_supplied,  # Use calculated value from agents
                "final_total_borrowed": total_borrowed,  # Use calculated value from agents
                "final_protocol_treasury": self._calculate_protocol_revenue(),
                "min_health_factor": min((agent.state.health_factor for agent in self.agents.values() 
                                        if agent.state.health_factor != float('inf')), default=1.0),
                "max_health_factor": max((agent.state.health_factor for agent in self.agents.values() 
                                        if agent.state.health_factor != float('inf')), default=1.0),
                "avg_health_factor": self._calculate_average_health_factor(),
                "liquidation_efficiency": len(self.liquidation_events) / max(len(self.state.liquidatable_agents), 1) if self.state.liquidatable_agents else 0.0,
                "market_stress_level": self.state.get_market_stress_indicator()
            }
        }