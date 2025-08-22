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
        
        # Simulation metrics
        self.metrics_history = []
        self.liquidation_events = []
        self.trade_events = []
        
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
                "asset": collateral_asset.value,
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
                    "asset_in": asset_in.value,
                    "asset_out": asset_out.value,
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
            collateral_factors = {
                Asset.ETH: 0.75,
                Asset.BTC: 0.75,
                Asset.FLOW: 0.50,
                Asset.USDC: 0.90
            }
            
            agent.state.update_health_factor(self.state.current_prices, collateral_factors)
            
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
        """Record simulation metrics"""
        
        protocol_state = self._get_protocol_state()
        
        # Calculate total values
        total_supplied = sum(pool.total_supplied for pool in self.protocol.asset_pools.values())
        total_borrowed = sum(agent.state.get_total_debt_value(self.state.current_prices) 
                           for agent in self.agents.values())
        
        metrics = {
            "step": self.current_step,
            "total_supplied": total_supplied,
            "total_borrowed": total_borrowed,
            "protocol_treasury": protocol_state["protocol_treasury"],
            "debt_cap": protocol_state["debt_cap"],
            "liquidatable_agents": len(self.state.liquidatable_agents),
            "asset_prices": dict(self.state.current_prices),
            "utilization_rates": protocol_state["utilization"]
        }
        
        self.metrics_history.append(metrics)
    
    def _record_agent_action(self, agent_id: str, action_type: AgentAction, params: dict):
        """Record agent action for analysis"""
        pass  # Could implement detailed action logging here
    
    def _generate_results(self) -> dict:
        """Generate final simulation results"""
        
        return {
            "metrics_history": self.metrics_history,
            "liquidation_events": self.liquidation_events,
            "trade_events": self.trade_events,
            "final_protocol_state": self._get_protocol_state(),
            "agent_states": {
                agent_id: agent.get_portfolio_summary(self.state.current_prices)
                for agent_id, agent in self.agents.items()
            },
            "simulation_config": {
                "steps": self.current_step,
                "num_agents": len(self.agents),
                "num_liquidations": len(self.liquidation_events),
                "num_trades": len(self.trade_events)
            }
        }