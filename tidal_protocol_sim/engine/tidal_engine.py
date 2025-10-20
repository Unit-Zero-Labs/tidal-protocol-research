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
from ..core.uniswap_v3_math import (
    create_moet_btc_pool, create_usdc_btc_pool, create_usdf_btc_pool,
    create_moet_usdc_pool, create_moet_usdf_pool, UniswapV3SlippageCalculator
)
from ..agents.moet_arbitrage_agent import create_moet_arbitrage_agents


class TidalConfig(BaseLendingConfig):
    """Configuration for Tidal Protocol"""
    
    def __init__(self):
        super().__init__()
        self.scenario_name = "Tidal_Protocol"
        
        # Uniswap V3 Pool Configuration
        self.moet_btc_pool_size = 500_000  # $500K total pool size
        self.moet_btc_concentration = 0.80  # 80% concentration around peg
        self.btc_initial_price = 100_000.0
        
        # MOET Arbitrage Agent Configuration
        self.num_arbitrage_agents = 3  # Number of arbitrage agents
        self.arbitrage_agent_balance = 50_000.0  # $50K per agent


class TidalProtocolEngine(BaseLendingEngine):
    """Tidal Protocol with sophisticated Uniswap V3 mathematics"""
    
    def __init__(self, config: TidalConfig):
        super().__init__(config)
        
        # Initialize Tidal-specific agents
        self.agents = self._initialize_agents()
        
        # Initialize agent positions in protocol
        self._setup_initial_positions()
        
        # Initialize Uniswap V3 pools (structure depends on advanced MOET system)
        self._setup_uniswap_v3_pools()
        
        # Initialize MOET arbitrage agents (if advanced MOET system enabled)
        if getattr(config, 'enable_advanced_moet_system', False):
            self._setup_arbitrage_agents()
        
        # Initialize peg monitoring
        self.peg_monitoring = {
            'moet_usdc_prices': [],
            'moet_usdf_prices': [],
            'arbitrage_events': [],
            'peg_deviations': []
        }
    
    def _setup_uniswap_v3_pools(self):
        """Setup Uniswap V3 pools - new structure if advanced MOET enabled, legacy otherwise"""
        
        # Check if advanced MOET system is enabled
        if getattr(self.config, 'enable_advanced_moet_system', False):
            print("🏊 Setting up advanced pool structure: USDC:BTC, USDF:BTC, MOET:USDC, MOET:USDF")
            self._setup_advanced_pool_structure()
        else:
            print("🏊 Setting up legacy pool structure: MOET:BTC")
            self._setup_legacy_pool_structure()
    
    def _setup_advanced_pool_structure(self):
        """Setup new pool structure: USDC:BTC, USDF:BTC, MOET:USDC, MOET:USDF"""
        # Split original MOET:BTC liquidity 50/50 between USDC:BTC and USDF:BTC
        btc_pool_size = self.config.moet_btc_pool_size / 2  # Split in half
        
        # Create BTC pairs (for final swap in deleveraging chain)
        self.usdc_btc_pool = create_usdc_btc_pool(
            pool_size_usd=btc_pool_size,
            btc_price=self.config.btc_initial_price,
            concentration=self.config.moet_btc_concentration
        )
        
        self.usdf_btc_pool = create_usdf_btc_pool(
            pool_size_usd=btc_pool_size,
            btc_price=self.config.btc_initial_price,
            concentration=self.config.moet_btc_concentration
        )
        
        # Create MOET:stablecoin pairs (same size and conditions as MOET:YT pool)
        # Use MOET:YT pool size for consistency
        stablecoin_pool_size = getattr(self.config, 'moet_yield_pool_size', 500_000)  # $500K like MOET:YT
        
        self.moet_usdc_pool = create_moet_usdc_pool(
            pool_size_usd=stablecoin_pool_size,
            concentration=0.95,  # Same as MOET:YT (95% concentration)
            token0_ratio=0.5  # 50/50 split
        )
        
        self.moet_usdf_pool = create_moet_usdf_pool(
            pool_size_usd=stablecoin_pool_size,
            concentration=0.95,  # Same as MOET:YT (95% concentration)
            token0_ratio=0.5  # 50/50 split
        )
        
        # Create slippage calculators for all pools
        self.usdc_btc_calculator = UniswapV3SlippageCalculator(self.usdc_btc_pool)
        self.usdf_btc_calculator = UniswapV3SlippageCalculator(self.usdf_btc_pool)
        self.moet_usdc_calculator = UniswapV3SlippageCalculator(self.moet_usdc_pool)
        self.moet_usdf_calculator = UniswapV3SlippageCalculator(self.moet_usdf_pool)
        
        # Keep legacy reference for backward compatibility (use USDC:BTC as default)
        self.moet_btc_pool = self.usdc_btc_pool  # Legacy compatibility
        self.slippage_calculator = self.usdc_btc_calculator  # Legacy compatibility
        
        print(f"   ✅ Created USDC:BTC pool (${btc_pool_size:,.0f})")
        print(f"   ✅ Created USDF:BTC pool (${btc_pool_size:,.0f})")
        print(f"   ✅ Created MOET:USDC pool (${stablecoin_pool_size:,.0f}) - 95% concentration")
        print(f"   ✅ Created MOET:USDF pool (${stablecoin_pool_size:,.0f}) - 95% concentration")
    
    def _setup_legacy_pool_structure(self):
        """Setup legacy single MOET:BTC pool"""
        from ..core.uniswap_v3_math import create_moet_btc_pool, UniswapV3SlippageCalculator
        
        # Create single MOET:BTC pool (legacy behavior)
        self.moet_btc_pool = create_moet_btc_pool(
            pool_size_usd=self.config.moet_btc_pool_size,
            btc_price=self.config.btc_initial_price,
            concentration=self.config.moet_btc_concentration
        )
        
        # Create slippage calculator
        self.slippage_calculator = UniswapV3SlippageCalculator(self.moet_btc_pool)
        
        print(f"   ✅ Created legacy MOET:BTC pool (${self.config.moet_btc_pool_size:,.0f})")
    
    def _setup_arbitrage_agents(self):
        """Setup MOET arbitrage agents for peg maintenance"""
        print(f"🤖 Initializing {self.config.num_arbitrage_agents} MOET arbitrage agents...")
        
        # Create arbitrage agents
        self.arbitrage_agents = create_moet_arbitrage_agents(
            num_agents=self.config.num_arbitrage_agents,
            initial_balance=self.config.arbitrage_agent_balance
        )
        
        # Add arbitrage agents to main agents dict
        for agent in self.arbitrage_agents:
            self.agents[agent.agent_id] = agent
            # Set engine reference for protocol access (not pool access)
            agent.engine = self
        
        # Initialize arbitrage competition system
        self.arbitrage_opportunity_queue = []
        self.last_arbitrage_agent_index = 0  # For round-robin selection
        
        print(f"   ✅ Created {len(self.arbitrage_agents)} arbitrage agents with ${self.config.arbitrage_agent_balance:,.0f} each")
    
    def _process_arbitrage_agents_with_competition(self, current_minute: int) -> Dict:
        """Process arbitrage agents with competition mechanism - only one agent per opportunity"""
        if not hasattr(self, 'arbitrage_agents') or not self.arbitrage_agents:
            return {}
        
        # Check if advanced MOET system is available
        if not (hasattr(self.protocol, 'enable_advanced_moet') and self.protocol.enable_advanced_moet):
            return {}
        
        swap_data = {}
        
        # Step 1: Detect all opportunities (let all agents check)
        opportunities = []
        for agent in self.arbitrage_agents:
            if agent.active:
                opportunity = agent._detect_redeemer_arbitrage_opportunity(current_minute)
                if opportunity and opportunity.get('expected_profit', 0) > 0:
                    opportunity['agent_id'] = agent.agent_id
                    opportunities.append(opportunity)
        
        # Step 2: Execute opportunities with competition (round-robin)
        executed_count = 0
        for opportunity in opportunities:
            if executed_count >= len(opportunities):  # Limit to prevent over-execution
                break
                
            # Select agent using round-robin
            selected_agent = self._select_arbitrage_agent()
            if selected_agent and selected_agent.active:
                # Execute the arbitrage using the Redeemer system
                result = selected_agent._execute_redeemer_arbitrage(opportunity, current_minute)
                if result and result.get('success', False):
                    executed_count += 1
                    
                    # Track swap data for pool state updates
                    swap_key = f"arbitrage_{opportunity['type']}_{current_minute}_{selected_agent.agent_id}"
                    swap_data[swap_key] = {
                        'agent_id': selected_agent.agent_id,
                        'type': opportunity['type'],
                        'volume': opportunity['trade_size'],
                        'profit': result.get('actual_profit', 0),
                        'fees_generated': result.get('fees_generated', 0)
                    }
        
        return swap_data
    
    def _select_arbitrage_agent(self):
        """Select next arbitrage agent using round-robin"""
        if not self.arbitrage_agents:
            return None
            
        # Round-robin selection
        selected_agent = self.arbitrage_agents[self.last_arbitrage_agent_index]
        self.last_arbitrage_agent_index = (self.last_arbitrage_agent_index + 1) % len(self.arbitrage_agents)
        
        return selected_agent
        
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
            
            # Monitor MOET peg (if arbitrage agents exist)
            if hasattr(self, 'arbitrage_agents'):
                self._monitor_moet_peg(step)
            
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
    
    def _execute_arbitrage_mint(self, agent, params: dict) -> bool:
        """Execute mint arbitrage action"""
        if hasattr(agent, 'execute_mint_arbitrage'):
            current_minute = params.get('current_minute', 0)
            return agent.execute_mint_arbitrage(params, current_minute)
        return False
    
    def _execute_arbitrage_redeem(self, agent, params: dict) -> bool:
        """Execute redeem arbitrage action"""
        if hasattr(agent, 'execute_redeem_arbitrage'):
            current_minute = params.get('current_minute', 0)
            return agent.execute_redeem_arbitrage(params, current_minute)
        return False
    
    def _monitor_moet_peg(self, current_step: int):
        """Monitor MOET peg in USDC and USDF pools"""
        try:
            if current_step % 1440 == 0:  # Debug once per day
                print(f"🔍 DEBUG: _monitor_moet_peg called at step {current_step}")
                print(f"   Has moet_usdc_pool: {hasattr(self, 'moet_usdc_pool')}")
                print(f"   Has moet_usdf_pool: {hasattr(self, 'moet_usdf_pool')}")
            
            # Get current MOET prices from pools
            usdc_price = self.moet_usdc_pool.get_price() if hasattr(self, 'moet_usdc_pool') else 1.0
            usdf_price = self.moet_usdf_pool.get_price() if hasattr(self, 'moet_usdf_pool') else 1.0
            
            # Record prices
            self.peg_monitoring['moet_usdc_prices'].append({
                'step': current_step,
                'price': usdc_price,
                'deviation': abs(usdc_price - 1.0)
            })
            
            self.peg_monitoring['moet_usdf_prices'].append({
                'step': current_step,
                'price': usdf_price,
                'deviation': abs(usdf_price - 1.0)
            })
            
            # Calculate overall peg deviation
            avg_price = (usdc_price + usdf_price) / 2
            max_deviation = max(abs(usdc_price - 1.0), abs(usdf_price - 1.0))
            
            self.peg_monitoring['peg_deviations'].append({
                'step': current_step,
                'avg_price': avg_price,
                'max_deviation': max_deviation,
                'usdc_price': usdc_price,
                'usdf_price': usdf_price
            })
            
            # Log significant deviations
            if max_deviation > 0.01:  # >1% deviation
                print(f"📊 Step {current_step}: MOET peg deviation detected")
                print(f"   USDC pool: ${usdc_price:.4f} ({(usdc_price-1)*100:+.2f}%)")
                print(f"   USDF pool: ${usdf_price:.4f} ({(usdf_price-1)*100:+.2f}%)")
            
        except Exception as e:
            print(f"   ⚠️  Peg monitoring error: {e}")
    
    def get_peg_monitoring_summary(self) -> dict:
        """Get summary of peg monitoring data"""
        if not hasattr(self, 'peg_monitoring'):
            return {}
        
        if not self.peg_monitoring['peg_deviations']:
            return {}
        
        deviations = [d['max_deviation'] for d in self.peg_monitoring['peg_deviations']]
        
        return {
            'total_observations': len(self.peg_monitoring['peg_deviations']),
            'max_deviation': max(deviations) if deviations else 0,
            'avg_deviation': sum(deviations) / len(deviations) if deviations else 0,
            'deviations_over_1pct': len([d for d in deviations if d > 0.01]),
            'deviations_over_2pct': len([d for d in deviations if d > 0.02]),
            'arbitrage_events': len(self.peg_monitoring['arbitrage_events'])
        }
    
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
  