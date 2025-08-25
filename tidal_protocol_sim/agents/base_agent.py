#!/usr/bin/env python3
"""
Minimal Agent Interface

Base class for all Tidal Protocol agents with simplified state management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
from enum import Enum
from ..core.protocol import Asset


class AgentAction(Enum):
    """Agent action types"""
    SUPPLY = "supply"
    WITHDRAW = "withdraw" 
    BORROW = "borrow"
    REPAY = "repay"
    LIQUIDATE = "liquidate"
    SWAP = "swap"
    HOLD = "hold"


class AgentState:
    """Simplified agent state"""
    
    def __init__(self, agent_id: str, initial_balance: float = 100_000.0, agent_type: str = "base"):
        self.agent_id = agent_id
        
        # Initialize realistic positions based on agent type
        if agent_type == "tidal_lender":
            # Lenders start with 70% of assets supplied as collateral, 30% borrowed
            self.token_balances = {
                Asset.ETH: initial_balance * 0.1 / 4400.0,    # Keep 10% liquid
                Asset.BTC: initial_balance * 0.05 / 118_000.0, # Keep 5% liquid
                Asset.FLOW: initial_balance * 0.1 / 0.40,     # Keep 10% liquid
                Asset.USDC: initial_balance * 0.05,           # Keep 5% liquid
                Asset.MOET: initial_balance * 0.3             # 30% borrowed MOET
            }
            
            # Most assets are supplied as collateral
            self.supplied_balances = {
                Asset.ETH: initial_balance * 0.2 / 4400.0,    # $20k in ETH collateral
                Asset.BTC: initial_balance * 0.15 / 118_000.0, # $15k in BTC collateral
                Asset.FLOW: initial_balance * 0.1 / 0.40,     # $10k in FLOW collateral
                Asset.USDC: initial_balance * 0.25            # $25k in USDC collateral
            }
            self.borrowed_balances = {Asset.MOET: initial_balance * 0.3}  # 30% borrowed
            
        elif agent_type == "basic_trader":
            # Traders start with balanced token holdings, minimal borrowing
            self.token_balances = {
                Asset.ETH: initial_balance * 0.25 / 4400.0,
                Asset.BTC: initial_balance * 0.2 / 118_000.0,
                Asset.FLOW: initial_balance * 0.2 / 0.40,
                Asset.USDC: initial_balance * 0.25,
                Asset.MOET: initial_balance * 0.1
            }
            self.supplied_balances = {
                Asset.ETH: initial_balance * 0.05 / 4400.0,   # Small collateral
                Asset.BTC: initial_balance * 0.05 / 118_000.0,
                Asset.FLOW: 0.0,
                Asset.USDC: 0.0
            }
            self.borrowed_balances = {Asset.MOET: initial_balance * 0.1}  # 10% borrowed
            
        elif agent_type == "liquidator":
            # Liquidators start with liquid assets ready for liquidations
            self.token_balances = {
                Asset.ETH: initial_balance * 0.1 / 4400.0,
                Asset.BTC: initial_balance * 0.1 / 118_000.0,
                Asset.FLOW: initial_balance * 0.1 / 0.40,
                Asset.USDC: initial_balance * 0.2,
                Asset.MOET: initial_balance * 0.5             # Lots of MOET for liquidations
            }
            self.supplied_balances = {asset: 0.0 for asset in Asset if asset != Asset.MOET}
            self.borrowed_balances = {Asset.MOET: 0.0}
            
        else:
            # Default balanced approach
            self.token_balances = {
                Asset.ETH: initial_balance * 0.3 / 4400.0,
                Asset.BTC: initial_balance * 0.2 / 118_000.0,
                Asset.FLOW: initial_balance * 0.2 / 0.40,
                Asset.USDC: initial_balance * 0.3,
                Asset.MOET: 0.0
            }
            self.supplied_balances = {asset: 0.0 for asset in Asset if asset != Asset.MOET}
            self.borrowed_balances = {Asset.MOET: 0.0}
        
        # Agent metrics
        self.total_value = initial_balance
        self.profit_loss = 0.0
        self.health_factor = float('inf')
    
    def get_total_collateral_value(self, asset_prices: Dict[Asset, float]) -> float:
        """Calculate total collateral value"""
        total_value = 0.0
        for asset, amount in self.supplied_balances.items():
            if asset != Asset.MOET:
                total_value += amount * asset_prices.get(asset, 0.0)
        return total_value
    
    def get_total_debt_value(self, asset_prices: Dict[Asset, float]) -> float:
        """Calculate total debt value"""
        moet_debt = self.borrowed_balances.get(Asset.MOET, 0.0)
        moet_price = asset_prices.get(Asset.MOET, 1.0)
        return moet_debt * moet_price
    
    def update_health_factor(self, asset_prices: Dict[Asset, float], collateral_factors: Dict[Asset, float]):
        """Update health factor based on current positions"""
        collateral_value = 0.0
        
        for asset, amount in self.supplied_balances.items():
            if asset != Asset.MOET:
                asset_price = asset_prices.get(asset, 0.0)
                cf = collateral_factors.get(asset, 0.0)
                collateral_value += amount * asset_price * cf
        
        debt_value = self.get_total_debt_value(asset_prices)
        
        if debt_value <= 0:
            self.health_factor = float('inf')
        else:
            self.health_factor = collateral_value / debt_value


class BaseAgent(ABC):
    """Minimal agent interface"""
    
    def __init__(self, agent_id: str, agent_type: str, initial_balance: float = 100_000.0):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = AgentState(agent_id, initial_balance, agent_type)
        
        # Agent parameters (to be set by subclasses)
        self.risk_tolerance = 0.5
        self.active = True
    
    @abstractmethod
    def decide_action(self, protocol_state: dict, asset_prices: Dict[Asset, float]) -> tuple:
        """
        Decide what action to take based on current protocol state
        
        Returns:
            Tuple of (action_type, params)
        """
        pass
    
    def is_healthy(self) -> bool:
        """Check if agent position is healthy (HF > 1.0)"""
        return self.state.health_factor > 1.0
    
    def needs_emergency_action(self) -> bool:
        """Check if agent needs emergency action (HF < 1.1)"""
        return self.state.health_factor < 1.1
    
    def can_borrow_safely(self, amount: float, asset_prices: Dict[Asset, float], target_hf: float = 1.5) -> bool:
        """Check if agent can safely borrow additional amount"""
        collateral_value = self.state.get_total_collateral_value(asset_prices)
        current_debt = self.state.get_total_debt_value(asset_prices)
        new_debt = current_debt + amount
        
        if new_debt <= 0:
            return True
        
        return (collateral_value / new_debt) >= target_hf
    
    def execute_action(self, action_type: AgentAction, params: dict) -> bool:
        """Execute an action and update agent state"""
        # This would interface with the TidalProtocol
        # For now, just return success
        return True
    
    def get_portfolio_summary(self, asset_prices: Dict[Asset, float]) -> dict:
        """Get summary of agent's portfolio"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "health_factor": self.state.health_factor,
            "total_collateral_value": self.state.get_total_collateral_value(asset_prices),
            "total_debt_value": self.state.get_total_debt_value(asset_prices),
            "token_balances": dict(self.state.token_balances),
            "supplied_balances": dict(self.state.supplied_balances),
            "borrowed_balances": dict(self.state.borrowed_balances),
            "profit_loss": self.state.profit_loss
        }