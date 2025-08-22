#!/usr/bin/env python3
"""
Simulation factory for creating configured simulation instances.

This module routes configurations to appropriate simulation engines and
handles the initialization of the complete Agent-Action-Market system.
"""

from typing import Dict, Any, List
import random
from ..agents.base import AgentPopulation, BaseAgent
from ..agents.policies.trader import TraderPolicy
from ..agents.policies.lender import LenderPolicy
from ..agents.policies.tidal_lender import TidalLenderPolicy
from ..markets.base import MarketRegistry
from ..markets.uniswap_v2 import UniswapV2Market
from ..markets.tidal_protocol import TidalProtocolMarket
from .engine import TokenomicsSimulation
from .primitives import AgentState, Asset
from ...config.schemas.tidal_config import TidalProtocolConfig, PolicyType, MarketType


class SimulationFactory:
    """Factory for creating simulation instances from configuration"""
    
    @staticmethod
    def create_simulation(config: TidalProtocolConfig) -> TokenomicsSimulation:
        """
        Create a complete simulation instance from configuration
        
        Args:
            config: Validated Tidal Protocol configuration
            
        Returns:
            Configured TokenomicsSimulation instance
        """
        # Set random seed if specified
        if config.simulation.random_seed is not None:
            random.seed(config.simulation.random_seed)
        
        # Create agent population
        agent_population = SimulationFactory._create_agent_population(config)
        
        # Create market registry
        market_registry = SimulationFactory._create_market_registry(config)
        
        # Extract initial prices
        initial_prices = {
            asset_type: asset_config.initial_price
            for asset_type, asset_config in config.simulation.assets.items()
        }
        
        # Create simulation engine
        simulation = TokenomicsSimulation(
            agent_population=agent_population,
            market_registry=market_registry,
            initial_prices=initial_prices
        )
        
        # Set protocol treasury
        simulation.state.protocol_treasury = config.simulation.protocol_treasury_initial
        
        return simulation
    
    @staticmethod
    def _create_agent_population(config: TidalProtocolConfig) -> AgentPopulation:
        """Create agent population from configuration"""
        population = AgentPopulation()
        agent_id_counter = 0
        
        # Create agents for each policy configuration
        for policy_config in config.simulation.agent_policies:
            for i in range(policy_config.count):
                agent_id = f"{policy_config.type.value}_{agent_id_counter:04d}"
                agent_id_counter += 1
                
                # Create policy instance
                policy = SimulationFactory._create_policy(policy_config.type, policy_config.params)
                
                # Create initial agent state
                initial_state = SimulationFactory._create_initial_agent_state(
                    policy_config.initial_balance_usd, config
                )
                
                # Create agent
                agent = BaseAgent(
                    agent_id=agent_id,
                    initial_state=initial_state,
                    policy=policy
                )
                
                population.add_agent(agent)
        
        return population
    
    @staticmethod
    def _create_policy(policy_type: PolicyType, params: Dict[str, Any]):
        """Create a policy instance from type and parameters"""
        if policy_type == PolicyType.TRADER:
            return TraderPolicy(
                trading_frequency=params.get('trading_frequency', 0.1),
                momentum_threshold=params.get('momentum_threshold', 0.05),
                max_trade_size_pct=params.get('max_trade_size_pct', 0.1),
                risk_tolerance=params.get('risk_tolerance', 0.5)
            )
        
        elif policy_type == PolicyType.LENDER:
            return LenderPolicy(
                action_frequency=params.get('action_frequency', 0.05),
                min_supply_apy=params.get('min_supply_apy', 0.02),
                max_borrow_rate=params.get('max_borrow_rate', 0.15),
                target_health_factor=params.get('target_health_factor', 2.0),
                supply_ratio=params.get('supply_ratio', 0.8)
            )
        
        elif policy_type == PolicyType.TIDAL_LENDER:
            return TidalLenderPolicy(
                action_frequency=params.get('action_frequency', 0.08),
                min_supply_apy=params.get('min_supply_apy', 0.025),
                target_health_factor=params.get('target_health_factor', 2.0),
                min_health_factor=params.get('min_health_factor', 1.3),
                max_utilization_rate=params.get('max_utilization_rate', 0.85),
                moet_borrowing_ratio=params.get('moet_borrowing_ratio', 0.6),
                collateral_diversification=params.get('collateral_diversification', True),
                risk_tolerance=params.get('risk_tolerance', 0.5)
            )
        
        elif policy_type == PolicyType.HOLD:
            from ..agents.base import HoldPolicy
            return HoldPolicy()
        
        else:
            # Default to hold policy for unsupported types
            from ..agents.base import HoldPolicy
            return HoldPolicy()
    
    @staticmethod
    def _create_initial_agent_state(balance_usd: float, config: TidalProtocolConfig) -> AgentState:
        """Create initial agent state with balanced portfolio"""
        # Use updated Tidal Protocol asset prices
        asset_prices = {
            Asset.ETH: 4400.0,
            Asset.BTC: 118000.0,
            Asset.FLOW: 0.40,
            Asset.USDC: 1.0,
            Asset.MOET: 1.0
        }
        
        # Simple portfolio allocation focused on collateral assets
        token_balances = {}
        
        # Allocate across collateral assets (not MOET initially)
        # 50% USDC (stable collateral)
        token_balances[Asset.USDC] = balance_usd * 0.5 / asset_prices[Asset.USDC]
        
        # 30% ETH (high-value collateral)
        token_balances[Asset.ETH] = balance_usd * 0.3 / asset_prices[Asset.ETH]
        
        # 15% BTC (high-value collateral)
        token_balances[Asset.BTC] = balance_usd * 0.15 / asset_prices[Asset.BTC]
        
        # 5% FLOW (lower-grade collateral)
        token_balances[Asset.FLOW] = balance_usd * 0.05 / asset_prices[Asset.FLOW]
        
        # Start with no MOET (will be minted through borrowing)
        token_balances[Asset.MOET] = 0.0
        
        # Ensure all assets have entries
        for asset in Asset:
            if asset not in token_balances:
                token_balances[asset] = 0.0
        
        return AgentState(
            token_balances=token_balances,
            cash_balance=0.0,  # Cash is represented in USDC balance
            staked_balance=0.0,
            lp_balance=0.0,
            ve_balance=0.0
        )
    
    @staticmethod
    def _create_market_registry(config: TidalProtocolConfig) -> MarketRegistry:
        """Create market registry from configuration"""
        registry = MarketRegistry()
        
        # Create markets based on configuration
        for market_config in config.simulation.markets:
            if not market_config.enabled:
                continue
            
            market = SimulationFactory._create_market(market_config, config)
            if market:
                # Register market with appropriate actions
                actions = SimulationFactory._get_market_actions(market_config.type)
                registry.register_market(market, actions)
        
        return registry
    
    @staticmethod
    def _create_market(market_config, config: TidalProtocolConfig):
        """Create a market instance from configuration"""
        if market_config.type == MarketType.UNISWAP_V2:
            # Extract UniswapV2-specific configuration
            initial_reserves = getattr(market_config, 'initial_reserves', {})
            fee_rate = getattr(market_config, 'fee_rate', 0.003)
            
            return UniswapV2Market(
                market_id=market_config.market_id,
                initial_reserves=initial_reserves,
                fee_rate=fee_rate
            )
        
        elif market_config.type == MarketType.COMPOUND_LENDING:
            # Placeholder for compound lending market
            # Would create CompoundLendingMarket here
            return None
        
        elif market_config.type == MarketType.STAKING:
            # Placeholder for staking market
            # Would create StakingMarket here
            return None
        
        elif market_config.type == MarketType.TIDAL_PROTOCOL:
            # Create Tidal Protocol market
            return TidalProtocolMarket(
                market_id=market_config.market_id
            )
        
        return None
    
    @staticmethod
    def _get_market_actions(market_type: MarketType) -> List:
        """Get the action types supported by a market"""
        from ..simulation.primitives import ActionKind
        
        if market_type == MarketType.UNISWAP_V2:
            return [
                ActionKind.SWAP_BUY,
                ActionKind.SWAP_SELL,
                ActionKind.ADD_LIQUIDITY,
                ActionKind.REMOVE_LIQUIDITY,
                ActionKind.COLLECT_FEES
            ]
        
        elif market_type == MarketType.COMPOUND_LENDING:
            return [
                ActionKind.SUPPLY,
                ActionKind.WITHDRAW,
                ActionKind.BORROW,
                ActionKind.REPAY,
                ActionKind.LIQUIDATE
            ]
        
        elif market_type == MarketType.STAKING:
            return [
                ActionKind.STAKE,
                ActionKind.UNSTAKE,
                ActionKind.CLAIM_REWARD
            ]
        
        elif market_type == MarketType.TIDAL_PROTOCOL:
            return [
                ActionKind.SUPPLY,
                ActionKind.WITHDRAW,
                ActionKind.BORROW,
                ActionKind.REPAY,
                ActionKind.LIQUIDATE,
                ActionKind.MINT,
                ActionKind.BURN,
                ActionKind.SWAP_BUY,
                ActionKind.SWAP_SELL,
                ActionKind.ADD_LIQUIDITY,
                ActionKind.REMOVE_LIQUIDITY
            ]
        
        return []


class MonteCarloSimulator:
    """Monte Carlo simulation wrapper for statistical analysis"""
    
    def __init__(self, config: TidalProtocolConfig, n_simulations: int = None):
        """
        Initialize Monte Carlo simulator
        
        Args:
            config: Base configuration to use for all runs
            n_simulations: Number of simulations to run (overrides config)
        """
        self.config = config
        self.n_simulations = n_simulations or config.simulation.monte_carlo_runs
        self.results: List[Dict[str, Any]] = []
    
    def run_monte_carlo(self, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Run Monte Carlo simulations
        
        Args:
            verbose: Whether to print progress updates
            
        Returns:
            List of simulation results
        """
        if verbose:
            print(f"Running {self.n_simulations} Monte Carlo simulations...")
        
        self.results = []
        
        for i in range(self.n_simulations):
            if verbose and (i + 1) % 10 == 0:
                print(f"Completed {i + 1}/{self.n_simulations} simulations")
            
            # Create simulation with potential variations
            simulation = self._create_simulation_with_variations(i)
            
            # Run simulation
            try:
                result = simulation.run_simulation(
                    max_days=self.config.simulation.max_days,
                    verbose=False
                )
                result['simulation_id'] = i
                self.results.append(result)
            
            except Exception as e:
                if verbose:
                    print(f"Simulation {i} failed: {e}")
                continue
        
        if verbose:
            print(f"Monte Carlo completed! {len(self.results)} successful simulations")
        
        return self.results
    
    def _create_simulation_with_variations(self, simulation_id: int) -> TokenomicsSimulation:
        """Create simulation with random variations for Monte Carlo"""
        # Create base simulation
        simulation = SimulationFactory.create_simulation(self.config)
        
        # Apply random price shocks if enabled
        if self.config.simulation.price_shock_enabled:
            self._apply_price_shocks(simulation)
        
        return simulation
    
    def _apply_price_shocks(self, simulation: TokenomicsSimulation):
        """Apply random price shocks for Monte Carlo variation"""
        for asset, asset_config in self.config.simulation.assets.items():
            if asset == Asset.MOET:
                continue  # Don't shock the primary token initially
            
            # Generate random shock based on asset volatility
            volatility = asset_config.volatility_std
            extreme_drop = asset_config.extreme_drop_percentage
            
            # Generate shock with bias toward the extreme drop scenario
            if random.random() < 0.1:  # 10% chance of extreme scenario
                shock = random.uniform(extreme_drop, extreme_drop * 0.5)
            else:
                # Normal volatility
                shock = random.normalvariate(0, volatility)
                # Clamp to reasonable bounds
                shock = max(-0.5, min(0.3, shock))
            
            simulation.add_price_shock(asset, shock)
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary statistics of Monte Carlo results"""
        if not self.results:
            return {"error": "No results available"}
        
        from ...analysis.metrics.calculator import TokenomicsMetricsCalculator
        
        calculator = TokenomicsMetricsCalculator(self.results)
        metrics = calculator.calculate_comprehensive_metrics()
        
        return {
            "monte_carlo_summary": {
                "total_simulations": len(self.results),
                "successful_simulations": len(self.results),
                "configuration": self.config.client_name
            },
            "metrics": metrics
        }
