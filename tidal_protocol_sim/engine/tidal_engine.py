#!/usr/bin/env python3
"""
Tidal Protocol Engine

Tidal Protocol with sophisticated Uniswap V3 mathematics for MOET:BTC pools.
This is the foundation for all Tidal-based simulations.
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
from .base_lending_engine import BaseLendingEngine, BaseLendingConfig
from ..core.uniswap_v3_math import create_moet_btc_pool, UniswapV3SlippageCalculator


class TidalConfig(BaseLendingConfig):
    """Configuration for Tidal Protocol"""
    
    def __init__(self):
        super().__init__()
        self.scenario_name = "Tidal_Protocol"
        
        # Uniswap V3 Pool Configuration
        self.moet_btc_pool_size = 500_000  # $500K total pool size
        self.moet_btc_concentration = 0.80  # 80% concentration around peg
        self.btc_initial_price = 100_000.0


class TidalProtocolEngine(BaseLendingEngine):
    """Tidal Protocol with sophisticated Uniswap V3 mathematics"""
    
    def __init__(self, config: TidalConfig):
        super().__init__(config)
        
        # Initialize Tidal-specific agents
        self.agents = self._initialize_agents()
        
        # Initialize agent positions in protocol
        self._setup_initial_positions()
        
        # Initialize Uniswap V3 pools for ALL Tidal simulations
        self._setup_uniswap_v3_pools()
    
    def _setup_uniswap_v3_pools(self):
        """Setup Uniswap V3 pools with proper math"""
        self.moet_btc_pool = create_moet_btc_pool(
            pool_size_usd=self.config.moet_btc_pool_size,
            btc_price=self.config.btc_initial_price,
            concentration=self.config.moet_btc_concentration
        )
        
        self.slippage_calculator = UniswapV3SlippageCalculator(self.moet_btc_pool)
        
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
            Asset.ETH: 0.80,
            Asset.BTC: 0.80,
            Asset.FLOW: 0.50,
            Asset.USDC: 0.90
        }
        agent.state.update_health_factor(self.state.current_prices, collateral_factors)
    
    def run_simulation(self, steps: int) -> Dict:
        """Run Tidal Protocol simulation for specified number of steps"""
        
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
                print(f"Tidal Protocol simulation step {step}/{steps}")
        
        return self._generate_results()
    
    # Agent processing inherited from BaseLendingEngine
    
    def _execute_swap(self, agent: BaseAgent, params: dict) -> bool:
        """Execute swap with Tidal's Uniswap V3 pools"""
        asset_in = params.get("asset_in")
        asset_out = params.get("asset_out")
        amount_in = params.get("amount_in", 0.0)
        
        if not self._check_balance(agent, asset_in, amount_in):
            return False
        
        # Use Uniswap V3 slippage calculator for precise swaps
        try:
            if asset_in == Asset.MOET and asset_out == Asset.BTC:
                slippage_result = self.slippage_calculator.calculate_swap_slippage(
                    amount_in, "MOET"
                )
            elif asset_in == Asset.BTC and asset_out == Asset.MOET:
                slippage_result = self.slippage_calculator.calculate_swap_slippage(
                    amount_in, "BTC"
                )
            else:
                # Fall back to base implementation for other swaps
                return super()._execute_swap(agent, params)
            
            amount_out = slippage_result["amount_out"]
            
            if amount_out > 0:
                # Update agent balances
                agent.state.token_balances[asset_in] -= amount_in
                agent.state.token_balances[asset_out] += amount_out
                
                # Update pool state
                self.slippage_calculator.update_pool_state(slippage_result)
                
                # Record trade with slippage details
                self.trade_events.append({
                    "step": self.current_step,
                    "agent": agent.agent_id,
                    "asset_in": asset_in.value if hasattr(asset_in, 'value') else str(asset_in),
                    "asset_out": asset_out.value if hasattr(asset_out, 'value') else str(asset_out),
                    "amount_in": amount_in,
                    "amount_out": amount_out,
                    "slippage": slippage_result["slippage_percent"],
                    "price_impact": slippage_result["price_impact"]
                })
                
                return True
                
        except Exception as e:
            print(f"Error in Tidal swap: {e}")
            return super()._execute_swap(agent, params)
        
        return False
    
    # Liquidation inherited from BaseLendingEngine
    
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
  