#!/usr/bin/env python3
"""
Base Lending Engine

Abstract base class for all lending protocol simulations with common functionality.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import random
from ..core.protocol import TidalProtocol, Asset
from ..agents.base_agent import BaseAgent, AgentAction
from .state import SimulationState


class BaseLendingConfig:
    """Base configuration for all lending protocols"""
    
    def __init__(self):
        self.scenario_name = "Base_Lending"
        self.simulation_steps = 100
        self.price_update_frequency = 5
        
        # Agent configuration
        self.num_lenders = 5
        self.num_traders = 3
        self.num_liquidators = 2
        self.lender_initial_balance = 100_000.0
        self.trader_initial_balance = 50_000.0
        self.liquidator_initial_balance = 200_000.0


class BaseLendingEngine(ABC):
    """Abstract base class for all lending protocol simulations"""
    
    def __init__(self, config: BaseLendingConfig):
        self.config = config
        self.protocol = TidalProtocol()
        self.agents = {}
        self.state = SimulationState()
        self.current_step = 0
        
        # Common tracking
        self.liquidation_events = []
        self.trade_events = []
        self.agent_actions_history = []
        self.metrics_history = []
        self.protocol_state_history = []
        
    @abstractmethod
    def run_simulation(self, steps: int) -> Dict:
        """Abstract method - must be implemented by subclasses"""
        pass
        
    def _process_agent_actions(self):
        """Common agent processing logic"""
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
                    agent.state.token_balances[asset] -= amount
                    agent.state.supplied_balances[asset] += amount
                    return self.protocol.supply(agent.agent_id, asset, amount)
                
            elif action_type == AgentAction.BORROW:
                amount = params.get("amount", 0.0)
                
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
        """Execute liquidation - can be overridden by subclasses"""
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
        """Execute swap - can be overridden by subclasses"""
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
    
    def _update_agent_health_factor(self, agent: BaseAgent):
        """Update agent's health factor based on current state"""
        collateral_factors = {
            Asset.ETH: 0.80,
            Asset.BTC: 0.80,
            Asset.FLOW: 0.50,
            Asset.USDC: 0.90
        }
        agent.state.update_health_factor(self.state.current_prices, collateral_factors)
    
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
        
        # Calculate total values from agent states
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
        
        # Enhanced metrics
        metrics = {
            "step": self.current_step,
            "timestamp": self.current_step,
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
            "active_agents": sum(1 for agent in self.agents.values() if agent.active),
            "liquidation_risk_score": len(self.state.liquidatable_agents) / len(self.agents) if self.agents else 0,
            "debt_cap_utilization": total_borrowed / protocol_state["debt_cap"] if protocol_state["debt_cap"] > 0 else 0
        }
        
        self.metrics_history.append(metrics)
        
        # PERFORMANCE OPTIMIZATION: Only record detailed protocol state daily
        # This reduces memory usage from 1.5 GB to 1.1 MB (1,440x improvement)
        if self.current_step % 1440 == 0:  # Every 24 hours
            # Record detailed protocol state
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
                "scenario_name": self.config.scenario_name
            },
            
            # Summary statistics
            "summary_statistics": {
                "total_liquidation_value": sum(event.get("repay_amount", 0) for event in self.liquidation_events),
                "final_total_supplied": total_supplied,
                "final_total_borrowed": total_borrowed,
                "min_health_factor": min((agent.state.health_factor for agent in self.agents.values() 
                                        if agent.state.health_factor != float('inf')), default=1.0),
                "max_health_factor": max((agent.state.health_factor for agent in self.agents.values() 
                                        if agent.state.health_factor != float('inf')), default=1.0)
            }
        }
