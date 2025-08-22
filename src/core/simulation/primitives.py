#!/usr/bin/env python3
"""
Core simulation primitives for the Tidal Protocol simulation.

This module defines the fundamental data structures and enums used throughout
the simulation, following the Agent-Action-Market pattern.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Any, Optional
import time


class ActionKind(Enum):
    """Enumeration of all possible DeFi actions"""
    
    # ── Trading
    SWAP_BUY            = auto()
    SWAP_SELL           = auto()
    LIMIT_ORDER         = auto()
    CANCEL_ORDER        = auto()

    # ── Liquidity & AMMs
    ADD_LIQUIDITY       = auto()
    REMOVE_LIQUIDITY    = auto()
    COLLECT_FEES        = auto()

    # ── Lending / Borrowing
    SUPPLY              = auto()  # Supply assets to lending protocol
    WITHDRAW            = auto()  # Withdraw supplied assets
    DEPOSIT_COLLATERAL  = auto()
    WITHDRAW_COLLATERAL = auto()
    BORROW              = auto()
    REPAY               = auto()
    LIQUIDATE           = auto()

    # ── Staking / Locking
    STAKE               = auto()
    UNSTAKE             = auto()
    LOCK_TOKENS         = auto()   # Lock tokens for vote-escrow
    EXTEND_LOCK         = auto()
    UNLOCK              = auto()

    # ── Yield / Rewards
    CLAIM_REWARD        = auto()
    COMPOUND_REWARD     = auto()
    DELEGATE_VOTE       = auto()
    VOTE_GAUGE_WEIGHT   = auto()

    # ── Treasury / Protocol Ops
    BUYBACK_BURN        = auto()
    MINT                = auto()
    BURN                = auto()
    ALLOCATE_INCENTIVE  = auto()

    # ── Governance
    CREATE_PROPOSAL     = auto()
    VOTE_PROPOSAL       = auto()
    VOTE_FOR_POOL       = auto()  # Velodrome-style pool voting

    # ── Hold (no action)
    HOLD                = auto()


class Asset(Enum):
    """Supported assets in the protocol"""
    ETH = "ETH"
    BTC = "BTC"
    FLOW = "FLOW"
    USDC = "USDC"
    MOET = "MOET"  # Protocol stablecoin


@dataclass
class Action:
    """Agent intention to perform a DeFi operation"""
    kind: ActionKind
    agent_id: str
    params: Dict[str, Any]  # market_id, amount, asset, etc.
    ts: int = None
    
    def __post_init__(self):
        if self.ts is None:
            self.ts = int(time.time())


@dataclass
class Event:
    """Market execution result from an action"""
    action_kind: ActionKind
    agent_id: str
    market_id: str
    result: Dict[str, Any]  # tokens_received, fees_paid, etc.
    ts: int = None
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.ts is None:
            self.ts = int(time.time())


@dataclass
class MarketSnapshot:
    """Read-only market state for agent decisions"""
    timestamp: int
    token_prices: Dict[Asset, float]
    market_cap: float
    daily_volume: float
    protocol_treasury: float
    markets: Dict[str, Dict[str, Any]]  # market-specific data
    
    # Convenience properties for common market data
    @property
    def token_price(self) -> float:
        """Primary token price (MOET)"""
        return self.token_prices.get(Asset.MOET, 1.0)
    
    @property
    def staking_apy(self) -> float:
        """Current staking APY"""
        staking_data = self.markets.get('staking', {})
        return staking_data.get('apy', 0.0)
    
    @property
    def pool_apy(self) -> float:
        """Current liquidity pool APY"""
        pool_data = self.markets.get('uniswap_v2', {})
        return pool_data.get('apy', 0.0)


@dataclass
class AgentState:
    """Clean state separation for agents"""
    # Asset balances
    token_balances: Dict[Asset, float]
    staked_balance: float = 0.0
    lp_balance: float = 0.0
    ve_balance: float = 0.0  # Vote-escrow balance
    cash_balance: float = 0.0
    
    # Lending positions
    supplied_balances: Dict[Asset, float] = None
    borrowed_balances: Dict[Asset, float] = None
    
    # History and metadata
    last_action_timestamp: int = 0
    total_fees_paid: float = 0.0
    total_rewards_earned: float = 0.0
    
    def __post_init__(self):
        if self.supplied_balances is None:
            self.supplied_balances = {asset: 0.0 for asset in Asset}
        if self.borrowed_balances is None:
            self.borrowed_balances = {asset: 0.0 for asset in Asset}
    
    def get_total_value(self, prices: Dict[Asset, float]) -> float:
        """Calculate total portfolio value in USD"""
        total = 0.0
        
        # Token balances
        for asset, balance in self.token_balances.items():
            total += balance * prices.get(asset, 0.0)
        
        # Supplied balances (positive value)
        for asset, balance in self.supplied_balances.items():
            total += balance * prices.get(asset, 0.0)
        
        # Borrowed balances (negative value)
        for asset, balance in self.borrowed_balances.items():
            total -= balance * prices.get(asset, 0.0)
        
        # Other balances
        total += self.cash_balance
        total += self.staked_balance * prices.get(Asset.MOET, 1.0)
        total += self.lp_balance  # Assume LP tokens are priced in USD
        
        return total
    
    def get_health_factor(self, prices: Dict[Asset, float], collateral_factors: Dict[Asset, float]) -> float:
        """Calculate health factor for lending positions"""
        collateral_value = 0.0
        borrowed_value = 0.0
        
        # Calculate effective collateral value
        for asset, balance in self.supplied_balances.items():
            if balance > 0:
                asset_value = balance * prices.get(asset, 0.0)
                collateral_factor = collateral_factors.get(asset, 0.0)
                collateral_value += asset_value * collateral_factor
        
        # Calculate total borrowed value
        for asset, balance in self.borrowed_balances.items():
            if balance > 0:
                borrowed_value += balance * prices.get(asset, 0.0)
        
        if borrowed_value == 0:
            return float('inf')
        
        return collateral_value / borrowed_value
