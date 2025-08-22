#!/usr/bin/env python3
"""
Main simulation engine for the Tidal Protocol simulation.

This module orchestrates the Agent-Action-Market flow and manages the
simulation state across time periods.
"""

import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from .primitives import MarketSnapshot, Action, Event, Asset
from ..agents.base import AgentPopulation
from ..markets.base import MarketRegistry


@dataclass
class SimulationState:
    """Container for all simulation state"""
    current_day: int = 0
    current_block: int = 0
    agents: Dict[str, Any] = field(default_factory=dict)
    protocol_treasury: float = 0.0
    total_protocol_revenue: float = 0.0
    circulating_supply: Dict[Asset, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.circulating_supply:
            self.circulating_supply = {
                Asset.MOET: 1000000.0,  # 1M MOET initial supply
                Asset.ETH: 0.0,
                Asset.BTC: 0.0,
                Asset.FLOW: 0.0,
                Asset.USDC: 0.0
            }


class TokenomicsSimulation:
    """Main simulation engine implementing Agent-Action-Market pattern"""
    
    def __init__(self, 
                 agent_population: AgentPopulation,
                 market_registry: MarketRegistry,
                 initial_prices: Dict[Asset, float] = None):
        """
        Initialize the simulation engine
        
        Args:
            agent_population: Population of agents to simulate
            market_registry: Registry of markets for action execution
            initial_prices: Initial asset prices
        """
        self.agent_population = agent_population
        self.market_registry = market_registry
        
        # Initialize prices
        self.current_prices = initial_prices or {
            Asset.ETH: 3000.0,
            Asset.BTC: 45000.0,
            Asset.FLOW: 0.50,
            Asset.USDC: 1.0,
            Asset.MOET: 1.0
        }
        
        # Initialize simulation state
        self.state = SimulationState()
        self._populate_agent_lookup()
        
        # History tracking
        self.price_history: List[Dict[Asset, float]] = []
        self.event_history: List[List[Event]] = []
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.blocks_per_day = 7200  # Approximate blocks per day
        self.max_actions_per_agent_per_block = 3
    
    def _populate_agent_lookup(self):
        """Populate agent lookup in simulation state"""
        self.state.agents = {
            agent.agent_id: agent for agent in self.agent_population.agents
        }
    
    def run_simulation(self, max_days: int = 365, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the main simulation loop
        
        Args:
            max_days: Maximum number of days to simulate
            verbose: Whether to print progress updates
            
        Returns:
            Dictionary containing simulation results
        """
        if verbose:
            print(f"Starting simulation for {max_days} days with {len(self.agent_population)} agents")
        
        for day in range(max_days):
            self.state.current_day = day
            
            # Apply any scheduled price shocks
            self._apply_scheduled_shocks(day)
            
            # Run daily simulation
            daily_events = self._simulate_day()
            
            # Update price history
            self.price_history.append(self.current_prices.copy())
            self.event_history.append(daily_events)
            
            # Calculate and store metrics
            daily_metrics = self._calculate_daily_metrics()
            self.metrics_history.append(daily_metrics)
            
            # Progress updates
            if verbose and (day + 1) % 30 == 0:
                print(f"Completed day {day + 1}/{max_days}")
                self._print_daily_summary(daily_metrics)
        
        if verbose:
            print("Simulation completed!")
        
        return self._generate_final_results()
    
    def _simulate_day(self) -> List[Event]:
        """Simulate a single day"""
        daily_events = []
        
        # Simulate multiple blocks per day
        for block in range(self.blocks_per_day):
            self.state.current_block = self.state.current_day * self.blocks_per_day + block
            
            # 1. Create market snapshot
            snapshot = self._create_market_snapshot()
            
            # 2. Collect actions from all agents
            all_actions = self.agent_population.get_all_actions(snapshot)
            
            # 3. Shuffle actions to simulate random block ordering
            random.shuffle(all_actions)
            
            # 4. Execute actions through markets
            block_events = []
            for action in all_actions:
                events = self.market_registry.route_action(action, self.state.__dict__)
                block_events.extend(events)
            
            # 5. End-of-block operations for all markets
            eob_events = self.market_registry.end_of_block_all(self.state.__dict__)
            block_events.extend(eob_events)
            
            daily_events.extend(block_events)
            
            # Update prices based on market activity (simplified)
            self._update_prices(block_events)
        
        return daily_events
    
    def _create_market_snapshot(self) -> MarketSnapshot:
        """Create read-only market snapshot for agent decisions"""
        # Get market data from all registered markets
        markets_data = self.market_registry.get_all_market_data()
        
        # Calculate market cap (simplified)
        moet_price = self.current_prices.get(Asset.MOET, 1.0)
        moet_supply = self.state.circulating_supply.get(Asset.MOET, 1000000.0)
        market_cap = moet_price * moet_supply
        
        # Calculate daily volume from market data
        daily_volume = 0.0
        for market_data in markets_data.values():
            volume_data = market_data.get("total_volume_usd", 0.0)
            daily_volume += volume_data
        
        return MarketSnapshot(
            timestamp=self.state.current_day,
            token_prices=self.current_prices.copy(),
            market_cap=market_cap,
            daily_volume=daily_volume,
            protocol_treasury=self.state.protocol_treasury,
            markets=markets_data
        )
    
    def _update_prices(self, events: List[Event]):
        """Update asset prices based on market events"""
        # Simple price update based on trading volume and direction
        # In a real simulation, this would be more sophisticated
        
        for event in events:
            if not event.success:
                continue
                
            from .primitives import ActionKind
            if event.action_kind in [ActionKind.SWAP_BUY, ActionKind.SWAP_SELL]:
                # Extract price impact from swap events
                price_impact = event.result.get("price_impact", 0.0)
                asset_out = event.result.get("asset_out", "MOET")
                
                # Apply small price changes based on trading activity
                if asset_out == "MOET":
                    # Buying MOET increases price slightly
                    self.current_prices[Asset.MOET] *= (1 + price_impact * 0.01)
                else:
                    # Selling MOET decreases price slightly
                    self.current_prices[Asset.MOET] *= (1 - price_impact * 0.01)
                
                # Keep price within reasonable bounds
                self.current_prices[Asset.MOET] = max(0.1, min(10.0, self.current_prices[Asset.MOET]))
    
    def _calculate_daily_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive daily metrics"""
        # Get current snapshot for calculations
        snapshot = self._create_market_snapshot()
        
        # Population metrics
        population_summary = self.agent_population.get_population_summary(snapshot)
        
        # Market metrics
        total_liquidity = 0.0
        total_fees = 0.0
        
        for market_data in snapshot.markets.values():
            # Sum up liquidity across all markets
            reserves = market_data.get("reserves", {})
            for reserve in reserves.values():
                total_liquidity += reserve  # Simplified - assumes USD pricing
            
            # Sum up fees
            fees = market_data.get("total_fees_collected", {})
            for fee in fees.values():
                total_fees += fee
        
        # Price metrics
        price_changes = {}
        if len(self.price_history) > 0:
            previous_prices = self.price_history[-1]
            for asset, current_price in self.current_prices.items():
                previous_price = previous_prices.get(asset, current_price)
                if previous_price > 0:
                    price_changes[asset] = (current_price - previous_price) / previous_price
                else:
                    price_changes[asset] = 0.0
        
        return {
            "day": self.state.current_day,
            "prices": self.current_prices.copy(),
            "price_changes": price_changes,
            "market_cap": snapshot.market_cap,
            "daily_volume": snapshot.daily_volume,
            "total_liquidity": total_liquidity,
            "total_fees_collected": total_fees,
            "protocol_treasury": self.state.protocol_treasury,
            "agent_metrics": population_summary,
            "total_events": len(self.event_history[-1]) if self.event_history else 0
        }
    
    def _print_daily_summary(self, metrics: Dict[str, Any]):
        """Print daily summary for verbose mode"""
        print(f"Day {metrics['day']} Summary:")
        print(f"  MOET Price: ${metrics['prices'][Asset.MOET]:.4f}")
        print(f"  Market Cap: ${metrics['market_cap']:,.0f}")
        print(f"  Daily Volume: ${metrics['daily_volume']:,.0f}")
        print(f"  Total Agents: {metrics['agent_metrics']['total_agents']}")
        print(f"  Events: {metrics['total_events']}")
        print()
    
    def _generate_final_results(self) -> Dict[str, Any]:
        """Generate final simulation results"""
        if not self.metrics_history:
            return {"error": "No simulation data available"}
        
        final_metrics = self.metrics_history[-1]
        
        # Calculate summary statistics
        final_price = self.current_prices.get(Asset.MOET, 1.0)
        initial_price = 1.0  # Initial MOET price
        total_return = (final_price - initial_price) / initial_price
        
        # Price volatility
        price_changes = [
            metrics.get("price_changes", {}).get(Asset.MOET, 0.0)
            for metrics in self.metrics_history
        ]
        
        import numpy as np
        volatility = np.std(price_changes) if price_changes else 0.0
        
        return {
            "simulation_summary": {
                "days_simulated": len(self.metrics_history),
                "total_agents": len(self.agent_population),
                "total_markets": len(self.market_registry.markets),
                "total_events": sum(len(events) for events in self.event_history)
            },
            "final_state": {
                "moet_price": final_price,
                "market_cap": final_metrics["market_cap"],
                "protocol_treasury": final_metrics["protocol_treasury"],
                "total_liquidity": final_metrics["total_liquidity"]
            },
            "performance_metrics": {
                "total_return": total_return,
                "volatility": volatility,
                "max_price": max(metrics["prices"].get(Asset.MOET, 1.0) for metrics in self.metrics_history),
                "min_price": min(metrics["prices"].get(Asset.MOET, 1.0) for metrics in self.metrics_history)
            },
            "history": {
                "prices": self.price_history,
                "metrics": self.metrics_history,
                "events": self.event_history
            }
        }
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        return {
            "day": self.state.current_day,
            "block": self.state.current_block,
            "prices": self.current_prices.copy(),
            "treasury": self.state.protocol_treasury,
            "agents": len(self.agent_population),
            "markets": list(self.market_registry.markets.keys())
        }
    
    def add_price_shock(self, asset: Asset, shock_percentage: float):
        """Apply a price shock to an asset"""
        if asset in self.current_prices:
            self.current_prices[asset] *= (1 + shock_percentage)
            # Ensure price stays positive
            self.current_prices[asset] = max(0.01, self.current_prices[asset])
    
    def schedule_price_shocks(self, price_shocks: Dict[Asset, float], shock_day: int):
        """
        Schedule price shocks to occur on a specific day
        
        Args:
            price_shocks: Dictionary of Asset -> percentage change
            shock_day: Day when shocks should occur
        """
        if not hasattr(self, '_scheduled_shocks'):
            self._scheduled_shocks = {}
        
        self._scheduled_shocks[shock_day] = price_shocks
    
    def _apply_scheduled_shocks(self, current_day: int):
        """Apply any scheduled price shocks for the current day"""
        if not hasattr(self, '_scheduled_shocks'):
            return
        
        if current_day in self._scheduled_shocks:
            shocks = self._scheduled_shocks[current_day]
            
            print(f"  💥 Applying price shocks on day {current_day}:")
            for asset, shock_percentage in shocks.items():
                if asset in self.current_prices:
                    old_price = self.current_prices[asset]
                    new_price = old_price * (1 + shock_percentage)
                    self.current_prices[asset] = max(0.01, new_price)  # Prevent negative prices
                    
                    print(f"    {asset.value}: ${old_price:.4f} → ${new_price:.4f} ({shock_percentage:.1%})")
