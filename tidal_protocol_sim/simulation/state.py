#!/usr/bin/env python3
"""
Protocol and Agent State Management

Simplified state management without complex event systems.
"""

from typing import Dict, Set
from ..core.protocol import Asset


class SimulationState:
    """Global simulation state"""
    
    def __init__(self):
        # Asset prices
        self.current_prices = {
            Asset.ETH: 4400.0,
            Asset.BTC: 118_000.0,
            Asset.FLOW: 0.40,
            Asset.USDC: 1.0,
            Asset.MOET: 1.0
        }
        
        # Market state
        self.liquidatable_agents: Set[str] = set()
        self.market_stress_level = 0.0  # 0-1 scale
        
        # Protocol metrics
        self.total_liquidation_volume = 0.0
        self.total_trade_volume = 0.0
        self.protocol_revenue = 0.0
    
    def apply_price_shock(self, shocks: Dict[Asset, float]):
        """Apply price shocks to assets"""
        for asset, shock_pct in shocks.items():
            if asset in self.current_prices:
                self.current_prices[asset] *= (1 + shock_pct)
                # Ensure prices don't go negative
                self.current_prices[asset] = max(0.01, self.current_prices[asset])
    
    def get_market_stress_indicator(self) -> float:
        """Calculate market stress based on price movements"""
        initial_prices = {
            Asset.ETH: 4400.0,
            Asset.BTC: 118_000.0,
            Asset.FLOW: 0.40,
            Asset.USDC: 1.0
        }
        
        total_deviation = 0.0
        for asset, initial_price in initial_prices.items():
            current_price = self.current_prices[asset]
            deviation = abs(current_price - initial_price) / initial_price
            total_deviation += deviation
        
        # Normalize to 0-1 scale
        self.market_stress_level = min(total_deviation / 2.0, 1.0)
        return self.market_stress_level
    
    def get_state_summary(self) -> dict:
        """Get summary of current state"""
        return {
            "current_prices": dict(self.current_prices),
            "liquidatable_agents": len(self.liquidatable_agents),
            "market_stress_level": self.get_market_stress_indicator(),
            "total_liquidation_volume": self.total_liquidation_volume,
            "total_trade_volume": self.total_trade_volume
        }