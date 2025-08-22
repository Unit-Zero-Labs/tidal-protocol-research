#!/usr/bin/env python3
"""
Tidal Protocol Market - Comprehensive implementation of all Tidal-specific mechanisms.

This market implements the complete Tidal Protocol lending system with all its nuanced
mechanics, including MOET stablecoin management, Ebisu-style debt caps, kinked interest
rates, and integrated liquidity pool management.
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import numpy as np
from dataclasses import dataclass, field
from .base import BaseMarket
from ..simulation.primitives import Action, Event, ActionKind, Asset
from ..math.tidal_math import TidalMath


@dataclass
class TidalAssetPool:
    """Individual asset pool within Tidal Protocol"""
    asset: Asset
    total_supplied: float = 0.0
    total_borrowed: float = 0.0
    reserve_balance: float = 0.0  # Protocol reserves for this asset
    
    # Interest accrual tracking
    supply_index: float = 1.0  # Compound interest index for suppliers
    borrow_index: float = 1.0  # Compound interest index for borrowers
    last_update_block: int = 0
    
    # Asset-specific parameters
    collateral_factor: float = 0.75
    liquidation_threshold: float = 0.80  # Slightly higher than collateral factor
    liquidation_penalty: float = 0.08  # 8% liquidation penalty
    reserve_factor: float = 0.15  # 15% of interest to protocol
    
    # Kinked interest rate model parameters
    base_rate_per_block: float = 0
    multiplier_per_block: float = 11415525114
    jump_per_block: float = 253678335870
    kink: float = 0.80  # 80% utilization
    
    def __post_init__(self):
        """Set asset-specific parameters"""
        if self.asset == Asset.ETH:
            self.collateral_factor = 0.9
            self.liquidation_threshold = 0.92
        elif self.asset == Asset.BTC:
            self.collateral_factor = 0.9
            self.liquidation_threshold = 0.92
        elif self.asset == Asset.FLOW:
            self.collateral_factor = 0.6
            self.liquidation_threshold = 0.65
        elif self.asset == Asset.USDC:
            self.collateral_factor = 1.0
            self.liquidation_threshold = 1.0
    
    @property
    def utilization_rate(self) -> float:
        """Calculate current utilization rate"""
        if self.total_supplied <= 0:
            return 0.0
        return self.total_borrowed / self.total_supplied
    
    def calculate_borrow_rate(self) -> float:
        """Calculate current borrow rate using kinked interest model"""
        utilization = self.utilization_rate
        
        if utilization <= self.kink:
            # Below kink: linear rate
            rate = self.base_rate_per_block + (utilization * self.multiplier_per_block / 1e18)
        else:
            # Above kink: jump rate
            base_rate = self.base_rate_per_block + (self.kink * self.multiplier_per_block / 1e18)
            jump_rate = (utilization - self.kink) * self.jump_per_block / 1e18
            rate = base_rate + jump_rate
        
        # Convert to annual rate
        blocks_per_year = 15768000  # Tidal-specific block timing
        return rate * blocks_per_year / 1e18
    
    def calculate_supply_rate(self) -> float:
        """Calculate supply rate based on borrow rate and utilization"""
        borrow_rate = self.calculate_borrow_rate()
        return borrow_rate * self.utilization_rate * (1 - self.reserve_factor)


@dataclass
class MoetStablecoin:
    """MOET stablecoin management system"""
    total_supply: float = 1000000.0  # Initial 1M MOET supply
    circulating_supply: float = 1000000.0
    target_price: float = 1.0  # $1.00 peg
    current_price: float = 1.0
    
    # Peg stability mechanism parameters
    mint_fee: float = 0.001  # 0.1% mint fee
    burn_fee: float = 0.001  # 0.1% burn fee
    stability_fund: float = 100000.0  # Stability fund for peg defense
    
    # Price stability bands
    upper_band: float = 1.02  # +2% from peg
    lower_band: float = 0.98  # -2% from peg
    
    def calculate_mint_amount(self, collateral_value_usd: float, collateral_factor: float) -> float:
        """Calculate how much MOET can be minted against collateral"""
        max_mint = collateral_value_usd * collateral_factor
        mint_fee_amount = max_mint * self.mint_fee
        return max_mint - mint_fee_amount
    
    def is_peg_stable(self) -> bool:
        """Check if MOET is within stability bands"""
        return self.lower_band <= self.current_price <= self.upper_band
    
    def calculate_stability_action(self) -> Optional[str]:
        """Determine if stability mechanism should activate"""
        if self.current_price > self.upper_band:
            return "mint_pressure"  # Too high, encourage minting
        elif self.current_price < self.lower_band:
            return "burn_pressure"  # Too low, encourage burning
        return None


@dataclass
class TidalLiquidityPool:
    """Tidal's integrated liquidity pools for MOET pairs"""
    pair_assets: Tuple[Asset, Asset]
    reserves: Dict[Asset, float] = field(default_factory=dict)
    fee_rate: float = 0.003  # 0.3% trading fee
    lp_token_supply: float = 0.0
    
    # Uniswap V3-style concentrated liquidity parameters
    liquidity_distribution: Dict[str, Any] = field(default_factory=dict)
    active_liquidity: float = 0.0
    price_range_factor: float = 0.1  # 10% price range concentration
    
    def __post_init__(self):
        """Initialize reserves"""
        for asset in self.pair_assets:
            if asset not in self.reserves:
                self.reserves[asset] = 0.0
    
    def get_liquidation_capacity(self, max_slippage: float = 0.04) -> float:
        """
        Calculate liquidation capacity using constant product formula
        
        For constant product AMM (x*y=k), the amount that can be sold
        before reaching max_slippage is calculated using:
        
        slippage = amount_in / (reserve_in + amount_in)
        Solving for amount_in: amount_in = (slippage * reserve_in) / (1 - slippage)
        """
        if not self.reserves or len(self.reserves) < 2:
            return 0.0
        
        # Find MOET reserve (the asset we're selling during liquidations)
        moet_reserve = self.reserves.get(Asset.MOET, 0.0)
        if moet_reserve <= 0:
            return 0.0
        
        # Calculate max amount of collateral asset that can be liquidated
        # before MOET price moves by max_slippage
        max_liquidation_amount = (max_slippage * moet_reserve) / (1 - max_slippage)
        
        # Apply safety factor for real-world conditions
        safety_factor = 0.8  # 20% buffer for market conditions
        
        return max_liquidation_amount * safety_factor
    
    def calculate_swap_output(self, amount_in: float, asset_in: Asset, asset_out: Asset) -> Tuple[float, float, float, float]:
        """Calculate swap output using Uniswap V3 concentrated liquidity model with proper slippage"""
        reserve_in = self.reserves.get(asset_in, 0.0)
        reserve_out = self.reserves.get(asset_out, 0.0)
        
        if reserve_in <= 0 or reserve_out <= 0:
            return 0.0, 0.0, 0.0, 0.0
        
        # Calculate current price and standard deviation for concentration
        current_price = reserve_out / reserve_in
        
        # Standard deviation for normal distribution (10% of current price)
        price_std_dev = 0.10
        
        # Use Uniswap V3 concentrated liquidity calculation
        from ..math.tidal_math import TidalMath
        amount_out, fee_amount, slippage_percent, actual_price = TidalMath.calculate_uniswap_v3_concentrated_liquidity(
            amount_in=amount_in,
            reserve_in=reserve_in,
            reserve_out=reserve_out,
            current_price=current_price,
            price_std_dev=price_std_dev,
            fee_rate=self.fee_rate
        )
        
        return amount_out, fee_amount, slippage_percent, actual_price


class TidalProtocolMarket(BaseMarket):
    """
    Comprehensive Tidal Protocol Market Implementation
    
    This market implements all Tidal-specific mechanisms:
    - Multi-asset lending with kinked interest rates
    - MOET stablecoin minting/burning
    - Ebisu-style debt cap calculations
    - Integrated liquidity pools
    - Protocol revenue distribution
    - Liquidation cascades
    """
    
    def __init__(self, market_id: str = "tidal_protocol"):
        super().__init__(market_id)
        
        # Set supported actions
        self.supported_actions.update([
            ActionKind.SUPPLY,
            ActionKind.WITHDRAW,
            ActionKind.BORROW,
            ActionKind.REPAY,
            ActionKind.LIQUIDATE,
            ActionKind.MINT,  # Mint MOET
            ActionKind.BURN,  # Burn MOET
            ActionKind.SWAP_BUY,
            ActionKind.SWAP_SELL,
            ActionKind.ADD_LIQUIDITY,
            ActionKind.REMOVE_LIQUIDITY
        ])
        
        # Initialize Tidal-specific components
        self.asset_pools = self._initialize_asset_pools()
        self.moet = MoetStablecoin()
        self.liquidity_pools = self._initialize_liquidity_pools()
        
        # Protocol state
        self.protocol_treasury = 0.0
        self.total_protocol_revenue = 0.0
        self.lp_rewards_distributed = 0.0
        self.current_block = 0
        
        # Tidal-specific parameters
        self.lp_rewards_factor = 0.50  # 50% of protocol revenue to LP rewards
        self.dex_liquidity_allocation = 0.35  # 35% for debt cap calculations
        
        # Risk parameters
        self.target_health_factor = 1.2
        self.liquidation_close_factor = 0.5  # Max 50% of debt can be liquidated
        
        # Extreme scenario parameters for debt cap
        self.extreme_price_drops = {
            Asset.ETH: 0.15,
            Asset.BTC: 0.15,
            Asset.FLOW: 0.35,
            Asset.USDC: 0.15
        }
    
    def _initialize_asset_pools(self) -> Dict[Asset, TidalAssetPool]:
        """Initialize asset pools with Tidal-specific parameters"""
        pools = {}
        
        # ETH pool
        pools[Asset.ETH] = TidalAssetPool(
            asset=Asset.ETH,
            total_supplied=7_000_000,  # $7M initial
            collateral_factor=0.75,
            liquidation_threshold=0.80
        )
        
        # BTC pool
        pools[Asset.BTC] = TidalAssetPool(
            asset=Asset.BTC,
            total_supplied=3_500_000,  # $3.5M initial
            collateral_factor=0.75,
            liquidation_threshold=0.80
        )
        
        # FLOW pool
        pools[Asset.FLOW] = TidalAssetPool(
            asset=Asset.FLOW,
            total_supplied=2_100_000,  # $2.1M initial
            collateral_factor=0.50,
            liquidation_threshold=0.60
        )
        
        # USDC pool
        pools[Asset.USDC] = TidalAssetPool(
            asset=Asset.USDC,
            total_supplied=1_400_000,  # $1.4M initial
            collateral_factor=0.90,
            liquidation_threshold=0.92
        )
        
        return pools
    
    def _initialize_liquidity_pools(self) -> Dict[str, TidalLiquidityPool]:
        """Initialize MOET trading pairs with $2.5M total liquidity each"""
        pools = {}
        
        # MOET/USDC pool - $2.5M total liquidity ($1.25M each side)
        pools["MOET_USDC"] = TidalLiquidityPool(
            pair_assets=(Asset.MOET, Asset.USDC),
            reserves={Asset.MOET: 1250000, Asset.USDC: 1250000},  # $1.25M each side
            lp_token_supply=1250000
        )
        
        # MOET/ETH pool - $2.5M total liquidity ($1.25M each side)
        eth_amount = 1250000 / 4400.0  # ~284.09 ETH at $4400/ETH
        pools["MOET_ETH"] = TidalLiquidityPool(
            pair_assets=(Asset.MOET, Asset.ETH),
            reserves={Asset.MOET: 1250000, Asset.ETH: eth_amount},
            lp_token_supply=1250000
        )
        
        # MOET/BTC pool - $2.5M total liquidity ($1.25M each side)
        btc_amount = 1250000 / 118000.0  # ~10.59 BTC at $118,000
        pools["MOET_BTC"] = TidalLiquidityPool(
            pair_assets=(Asset.MOET, Asset.BTC),
            reserves={Asset.MOET: 1250000, Asset.BTC: btc_amount},
            lp_token_supply=1250000
        )
        
        # MOET/FLOW pool - $2.5M total liquidity ($1.25M each side)
        flow_amount = 1250000 / 0.40  # 3.125M FLOW at $0.40
        pools["MOET_FLOW"] = TidalLiquidityPool(
            pair_assets=(Asset.MOET, Asset.FLOW),
            reserves={Asset.MOET: 1250000, Asset.FLOW: flow_amount},
            lp_token_supply=1250000
        )
        
        return pools
    
    def route(self, action: Action, simulation_state: Dict[str, Any]) -> List[Event]:
        """Route Tidal Protocol actions"""
        if action.kind == ActionKind.SUPPLY:
            return self._handle_supply(action, simulation_state)
        elif action.kind == ActionKind.WITHDRAW:
            return self._handle_withdraw(action, simulation_state)
        elif action.kind == ActionKind.BORROW:
            return self._handle_borrow(action, simulation_state)
        elif action.kind == ActionKind.REPAY:
            return self._handle_repay(action, simulation_state)
        elif action.kind == ActionKind.LIQUIDATE:
            return self._handle_liquidate(action, simulation_state)
        elif action.kind == ActionKind.MINT:
            return self._handle_mint_moet(action, simulation_state)
        elif action.kind == ActionKind.BURN:
            return self._handle_burn_moet(action, simulation_state)
        elif action.kind == ActionKind.SWAP_BUY:
            return self._handle_swap_buy(action, simulation_state)
        elif action.kind == ActionKind.SWAP_SELL:
            return self._handle_swap_sell(action, simulation_state)
        elif action.kind == ActionKind.ADD_LIQUIDITY:
            return self._handle_add_liquidity(action, simulation_state)
        elif action.kind == ActionKind.REMOVE_LIQUIDITY:
            return self._handle_remove_liquidity(action, simulation_state)
        
        return []
    
    def _handle_supply(self, action: Action, simulation_state: Dict[str, Any]) -> List[Event]:
        """Handle asset supply to Tidal Protocol"""
        asset = action.params.get("asset")
        amount = action.params.get("amount", 0.0)
        
        if asset not in self.asset_pools:
            return [self.create_event(action, {}, False, f"Asset {asset} not supported")]
        
        # Get agent
        agents = simulation_state.get("agents", {})
        agent = agents.get(action.agent_id)
        if not agent:
            return [self.create_event(action, {}, False, "Agent not found")]
        
        # Check balance
        available_balance = agent.state.token_balances.get(asset, 0.0)
        if available_balance < amount:
            return [self.create_event(action, {}, False, "Insufficient balance")]
        
        # Check supply cap (equal to liquidity in collateral asset to MOET pool)
        supply_cap = self._get_supply_cap(asset)
        current_supplied = self.asset_pools[asset].total_supplied
        
        if current_supplied + amount > supply_cap:
            max_additional_supply = supply_cap - current_supplied
            return [self.create_event(action, {}, False, 
                f"Supply cap exceeded. Max additional supply: {max_additional_supply:.2f}")]
        
        # Accrue interest before supply
        self._accrue_interest(asset)
        
        pool = self.asset_pools[asset]
        
        # Calculate supply shares (cTokens equivalent)
        if pool.total_supplied > 0:
            supply_shares = amount / pool.supply_index
        else:
            supply_shares = amount  # First supplier gets 1:1
        
        # Update pool state
        pool.total_supplied += amount
        
        # Update agent state
        agent.state.token_balances[asset] -= amount
        agent.state.supplied_balances[asset] = agent.state.supplied_balances.get(asset, 0.0) + supply_shares
        
        return [self.create_event(action, {
            "asset": asset.value,
            "amount_supplied": amount,
            "supply_shares": supply_shares,
            "new_supply_rate": pool.calculate_supply_rate(),
            "pool_utilization": pool.utilization_rate
        })]
    
    def _handle_borrow(self, action: Action, simulation_state: Dict[str, Any]) -> List[Event]:
        """Handle MOET borrowing against collateral"""
        asset = action.params.get("asset", Asset.MOET)  # Tidal only allows borrowing MOET
        amount = action.params.get("amount", 0.0)
        
        if asset != Asset.MOET:
            return [self.create_event(action, {}, False, "Tidal Protocol only allows borrowing MOET")]
        
        # Get agent
        agents = simulation_state.get("agents", {})
        agent = agents.get(action.agent_id)
        if not agent:
            return [self.create_event(action, {}, False, "Agent not found")]
        
        # Calculate borrowing power
        total_collateral_value = 0.0
        for collateral_asset, pool in self.asset_pools.items():
            supplied_shares = agent.state.supplied_balances.get(collateral_asset, 0.0)
            if supplied_shares > 0:
                # Convert shares to underlying amount
                underlying_amount = supplied_shares * pool.supply_index
                asset_price = simulation_state.get("current_prices", {}).get(collateral_asset, 1.0)
                collateral_value = underlying_amount * asset_price * pool.collateral_factor
                total_collateral_value += collateral_value
        
        # Check current debt
        current_moet_debt = agent.state.borrowed_balances.get(Asset.MOET, 0.0)
        
        # Calculate max additional borrow
        max_total_borrow = total_collateral_value / self.target_health_factor
        max_additional_borrow = max_total_borrow - current_moet_debt
        
        if amount > max_additional_borrow:
            return [self.create_event(action, {}, False, 
                f"Insufficient collateral. Max borrow: {max_additional_borrow:.2f}")]
        
        # Check protocol debt cap
        current_debt_cap = self.calculate_debt_cap(simulation_state)
        total_protocol_debt = sum(
            agent_state.borrowed_balances.get(Asset.MOET, 0.0) 
            for agent_state in [a.state for a in agents.values()]
        )
        
        if total_protocol_debt + amount > current_debt_cap:
            return [self.create_event(action, {}, False, "Protocol debt cap exceeded")]
        
        # Mint MOET and update state
        mint_amount = self.moet.calculate_mint_amount(amount, 1.0)  # No additional collateral factor for MOET mint
        
        # Update agent state
        agent.state.borrowed_balances[Asset.MOET] = current_moet_debt + amount
        agent.state.token_balances[Asset.MOET] = agent.state.token_balances.get(Asset.MOET, 0.0) + mint_amount
        
        # Update MOET supply
        self.moet.total_supply += mint_amount
        self.moet.circulating_supply += mint_amount
        
        # Calculate health factor
        new_health_factor = total_collateral_value / (current_moet_debt + amount) if (current_moet_debt + amount) > 0 else float('inf')
        
        return [self.create_event(action, {
            "asset": Asset.MOET.value,
            "amount_borrowed": amount,
            "moet_minted": mint_amount,
            "mint_fee": amount - mint_amount,
            "health_factor": new_health_factor,
            "collateral_value": total_collateral_value
        })]
    
    def _handle_liquidate(self, action: Action, simulation_state: Dict[str, Any]) -> List[Event]:
        """Handle liquidation of undercollateralized positions"""
        target_agent_id = action.params.get("target_agent_id")
        collateral_asset = action.params.get("collateral_asset")
        repay_amount = action.params.get("repay_amount", 0.0)
        
        # Get agents
        agents = simulation_state.get("agents", {})
        target_agent = agents.get(target_agent_id)
        liquidator_agent = agents.get(action.agent_id)
        
        if not target_agent or not liquidator_agent:
            return [self.create_event(action, {}, False, "Agent not found")]
        
        # Check if position is liquidatable
        health_factor = self._calculate_agent_health_factor(target_agent, simulation_state)
        
        if health_factor >= 1.0:
            return [self.create_event(action, {}, False, f"Position healthy (HF: {health_factor:.3f})")]
        
        # Validate collateral asset
        if collateral_asset not in self.asset_pools:
            return [self.create_event(action, {}, False, f"Invalid collateral asset: {collateral_asset}")]
        
        pool = self.asset_pools[collateral_asset]
        
        # Check liquidator has MOET to repay
        liquidator_moet = liquidator_agent.state.token_balances.get(Asset.MOET, 0.0)
        if liquidator_moet < repay_amount:
            return [self.create_event(action, {}, False, "Insufficient MOET for repayment")]
        
        # Calculate max liquidation amount (close factor)
        target_debt = target_agent.state.borrowed_balances.get(Asset.MOET, 0.0)
        max_liquidation = target_debt * self.liquidation_close_factor
        
        if repay_amount > max_liquidation:
            repay_amount = max_liquidation
        
        # Calculate collateral to seize (with liquidation penalty)
        asset_price = simulation_state.get("current_prices", {}).get(collateral_asset, 1.0)
        collateral_value = repay_amount * (1 + pool.liquidation_penalty)
        collateral_amount = collateral_value / asset_price
        
        # Check target has enough collateral
        target_collateral_shares = target_agent.state.supplied_balances.get(collateral_asset, 0.0)
        target_collateral_amount = target_collateral_shares * pool.supply_index
        
        if collateral_amount > target_collateral_amount:
            collateral_amount = target_collateral_amount
            repay_amount = (collateral_amount * asset_price) / (1 + pool.liquidation_penalty)
        
        # Execute liquidation
        # 1. Liquidator pays MOET debt
        liquidator_agent.state.token_balances[Asset.MOET] -= repay_amount
        target_agent.state.borrowed_balances[Asset.MOET] -= repay_amount
        
        # 2. Transfer collateral to liquidator
        collateral_shares_to_seize = collateral_amount / pool.supply_index
        target_agent.state.supplied_balances[collateral_asset] -= collateral_shares_to_seize
        liquidator_agent.state.supplied_balances[collateral_asset] = \
            liquidator_agent.state.supplied_balances.get(collateral_asset, 0.0) + collateral_shares_to_seize
        
        # 3. Burn repaid MOET
        self.moet.total_supply -= repay_amount
        self.moet.circulating_supply -= repay_amount
        
        # 4. Protocol revenue from liquidation penalty
        penalty_revenue = repay_amount * pool.liquidation_penalty
        self.protocol_treasury += penalty_revenue
        self.total_protocol_revenue += penalty_revenue
        
        # Calculate new health factor
        new_health_factor = self._calculate_agent_health_factor(target_agent, simulation_state)
        
        return [self.create_event(action, {
            "target_agent": target_agent_id,
            "collateral_asset": collateral_asset.value,
            "repay_amount": repay_amount,
            "collateral_seized": collateral_amount,
            "liquidation_penalty": penalty_revenue,
            "old_health_factor": health_factor,
            "new_health_factor": new_health_factor
        })]
    
    def _handle_mint_moet(self, action: Action, simulation_state: Dict[str, Any]) -> List[Event]:
        """Handle direct MOET minting (for stability mechanisms)"""
        amount = action.params.get("amount", 0.0)
        
        # Get agent
        agents = simulation_state.get("agents", {})
        agent = agents.get(action.agent_id)
        if not agent:
            return [self.create_event(action, {}, False, "Agent not found")]
        
        # Check if peg stability allows minting
        stability_action = self.moet.calculate_stability_action()
        if stability_action == "burn_pressure":
            return [self.create_event(action, {}, False, "MOET below peg, minting restricted")]
        
        # Calculate mint fee
        mint_fee = amount * self.moet.mint_fee
        net_mint = amount - mint_fee
        
        # Update MOET supply and agent balance
        self.moet.total_supply += net_mint
        self.moet.circulating_supply += net_mint
        agent.state.token_balances[Asset.MOET] = agent.state.token_balances.get(Asset.MOET, 0.0) + net_mint
        
        # Add fee to protocol treasury
        self.protocol_treasury += mint_fee
        self.total_protocol_revenue += mint_fee
        
        return [self.create_event(action, {
            "amount_requested": amount,
            "amount_minted": net_mint,
            "mint_fee": mint_fee,
            "moet_price": self.moet.current_price,
            "total_supply": self.moet.total_supply
        })]
    
    def _handle_swap_buy(self, action: Action, simulation_state: Dict[str, Any]) -> List[Event]:
        """Handle buying MOET through integrated liquidity pools"""
        asset_in = action.params.get("asset_in")
        amount_in = action.params.get("amount_in", 0.0)
        min_amount_out = action.params.get("min_amount_out", 0.0)
        
        # Find appropriate liquidity pool
        pool_key = f"MOET_{asset_in.value}" if asset_in != Asset.MOET else None
        if pool_key not in self.liquidity_pools:
            return [self.create_event(action, {}, False, f"No liquidity pool for {asset_in}/MOET")]
        
        pool = self.liquidity_pools[pool_key]
        
        # Get agent
        agents = simulation_state.get("agents", {})
        agent = agents.get(action.agent_id)
        if not agent:
            return [self.create_event(action, {}, False, "Agent not found")]
        
        # Check balance
        if agent.state.token_balances.get(asset_in, 0.0) < amount_in:
            return [self.create_event(action, {}, False, "Insufficient balance")]
        
        # Calculate swap with proper slippage
        amount_out, fee_amount, slippage_percent, actual_price = pool.calculate_swap_output(amount_in, asset_in, Asset.MOET)
        
        if amount_out < min_amount_out:
            return [self.create_event(action, {}, False, "Slippage too high")]
        
        # Execute swap
        agent.state.token_balances[asset_in] -= amount_in
        agent.state.token_balances[Asset.MOET] = agent.state.token_balances.get(Asset.MOET, 0.0) + amount_out
        
        # Update pool reserves
        pool.reserves[asset_in] += amount_in
        pool.reserves[Asset.MOET] -= amount_out
        
        # Distribute fees (LP rewards)
        lp_fee_reward = fee_amount * self.lp_rewards_factor
        self.lp_rewards_distributed += lp_fee_reward
        
        # Update MOET price based on trade
        self._update_moet_price_from_trade(pool, amount_in, amount_out)
        
        return [self.create_event(action, {
            "asset_in": asset_in.value,
            "amount_in": amount_in,
            "asset_out": Asset.MOET.value,
            "amount_out": amount_out,
            "fee_paid": fee_amount,
            "slippage_percent": slippage_percent,
            "actual_price": actual_price,
            "lp_rewards": lp_fee_reward,
            "new_moet_price": self.moet.current_price
        })]
    
    def calculate_debt_cap(self, simulation_state: Dict[str, Any]) -> float:
        """
        Calculate Tidal's debt cap using Ebisu-style methodology
        
        Formula: Debt Cap = A × B × C
        Where:
        A: Amount able to be profitably liquidated via DEX
        B: Allocation of DEX liquidity to other lending markets (35%)
        C: Percentage of collateral that is underwater in extreme drop
        """
        
        # A: Calculate total liquidation capacity across all pools
        total_liquidation_capacity = 0.0
        current_prices = simulation_state.get("current_prices", {
            Asset.ETH: 4400.0,
            Asset.BTC: 118000.0,
            Asset.FLOW: 0.40,
            Asset.USDC: 1.0,
            Asset.MOET: 1.0
        })
        
        for pool_key, lp_pool in self.liquidity_pools.items():
            liquidation_capacity = lp_pool.get_liquidation_capacity()
            # Convert to USD terms
            if "ETH" in pool_key:
                liquidation_capacity *= current_prices.get(Asset.ETH, 4400.0)
            elif "BTC" in pool_key:
                liquidation_capacity *= current_prices.get(Asset.BTC, 118000.0)
            elif "FLOW" in pool_key:
                liquidation_capacity *= current_prices.get(Asset.FLOW, 0.40)
            elif "USDC" in pool_key:
                liquidation_capacity *= current_prices.get(Asset.USDC, 1.0)
            
            total_liquidation_capacity += liquidation_capacity
        
        # B: DEX liquidity allocation factor
        dex_allocation = self.dex_liquidity_allocation
        
        # C: Calculate underwater collateral percentage in extreme scenarios
        total_collateral_value = 0.0
        weighted_underwater = 0.0
        
        for asset, pool in self.asset_pools.items():
            asset_price = current_prices.get(asset, 1.0)
            asset_collateral_value = pool.total_supplied * asset_price
            total_collateral_value += asset_collateral_value
        
        if total_collateral_value > 0:
            for asset, pool in self.asset_pools.items():
                asset_price = current_prices.get(asset, 1.0)
                asset_collateral_value = pool.total_supplied * asset_price
                weight = asset_collateral_value / total_collateral_value
                
                # Calculate underwater percentage for this asset
                drop_percentage = self.extreme_price_drops.get(asset, 0.15)  # Default 15% drop
                weighted_underwater += weight * drop_percentage
        
        # Calculate final debt cap
        debt_cap = total_liquidation_capacity * dex_allocation * weighted_underwater
        
        # Ensure minimum debt cap
        return max(debt_cap, 100000.0)  # Minimum $100k debt cap
    
    def _get_supply_cap(self, asset: Asset) -> float:
        """
        Get supply cap for an asset (equal to liquidity in collateral asset to MOET pool)
        """
        # Find the corresponding MOET pair pool
        pool_key = f"MOET_{asset.value}"
        if pool_key not in self.liquidity_pools:
            # Default high cap if no corresponding pool
            return 1000000000.0  # 1B cap
        
        lp_pool = self.liquidity_pools[pool_key]
        
        # Supply cap equals the amount of collateral asset in the MOET pair
        collateral_reserve = lp_pool.reserves.get(asset, 0.0)
        
        # Convert to USD terms for supply cap
        current_prices = getattr(self, '_current_prices', {})
        asset_price = current_prices.get(asset, 1.0)
        
        supply_cap_usd = collateral_reserve * asset_price
        
        return supply_cap_usd
    
    def _accrue_interest(self, asset: Asset):
        """Accrue interest for an asset pool"""
        if asset not in self.asset_pools:
            return
        
        pool = self.asset_pools[asset]
        blocks_elapsed = self.current_block - pool.last_update_block
        
        if blocks_elapsed <= 0:
            return
        
        # Calculate interest
        borrow_rate_per_block = pool.calculate_borrow_rate() / 15768000  # Annual to per-block
        interest_factor = (1 + borrow_rate_per_block) ** blocks_elapsed
        
        # Update borrow index
        pool.borrow_index *= interest_factor
        
        # Calculate supply interest (after reserves)
        if pool.total_supplied > 0:
            total_interest = pool.total_borrowed * (interest_factor - 1)
            reserve_amount = total_interest * pool.reserve_factor
            supply_interest = total_interest - reserve_amount
            
            # Update supply index
            pool.supply_index *= (1 + supply_interest / pool.total_supplied)
            
            # Add to protocol treasury
            self.protocol_treasury += reserve_amount
            self.total_protocol_revenue += reserve_amount
        
        pool.last_update_block = self.current_block
    
    def _calculate_agent_health_factor(self, agent, simulation_state: Dict[str, Any]) -> float:
        """Calculate agent's health factor"""
        total_collateral_value = 0.0
        total_debt_value = 0.0
        
        current_prices = simulation_state.get("current_prices", {})
        
        # Calculate collateral value
        for asset, pool in self.asset_pools.items():
            supplied_shares = agent.state.supplied_balances.get(asset, 0.0)
            if supplied_shares > 0:
                underlying_amount = supplied_shares * pool.supply_index
                asset_price = current_prices.get(asset, 1.0)
                collateral_value = underlying_amount * asset_price * pool.liquidation_threshold
                total_collateral_value += collateral_value
        
        # Calculate debt value
        moet_debt = agent.state.borrowed_balances.get(Asset.MOET, 0.0)
        if moet_debt > 0:
            moet_price = current_prices.get(Asset.MOET, 1.0)
            total_debt_value = moet_debt * moet_price
        
        if total_debt_value == 0:
            return float('inf')
        
        return total_collateral_value / total_debt_value
    
    def _update_moet_price_from_trade(self, pool: TidalLiquidityPool, amount_in: float, amount_out: float):
        """Update MOET price based on trading activity"""
        # Simple price impact calculation
        if Asset.MOET in pool.reserves and pool.reserves[Asset.MOET] > 0:
            price_impact = amount_out / pool.reserves[Asset.MOET]
            # Update MOET price (simplified)
            self.moet.current_price *= (1 + price_impact * 0.01)  # Small price impact
            
            # Keep price within reasonable bounds
            self.moet.current_price = max(0.5, min(2.0, self.moet.current_price))
    
    # ... (Additional handler methods would continue here)
    
    def end_of_block(self, simulation_state: Dict[str, Any]) -> List[Event]:
        """Handle end-of-block operations"""
        self.current_block += 1
        events = []
        
        # Accrue interest for all asset pools
        for asset in self.asset_pools:
            self._accrue_interest(asset)
        
        # Stability mechanism checks
        stability_action = self.moet.calculate_stability_action()
        if stability_action:
            # Could trigger automatic stability mechanisms
            pass
        
        # Distribute LP rewards
        if self.lp_rewards_distributed > 0:
            events.append(Event(
                action_kind=ActionKind.ALLOCATE_INCENTIVE,
                agent_id="protocol",
                market_id=self.market_id,
                result={
                    "lp_rewards_distributed": self.lp_rewards_distributed,
                    "total_protocol_revenue": self.total_protocol_revenue
                }
            ))
        
        return events
    
    def get_market_data(self) -> Dict[str, Any]:
        """Provide comprehensive Tidal Protocol market data"""
        # Asset pool data
        asset_data = {}
        for asset, pool in self.asset_pools.items():
            asset_data[asset.value] = {
                "total_supplied": pool.total_supplied,
                "total_borrowed": pool.total_borrowed,
                "utilization_rate": pool.utilization_rate,
                "borrow_rate": pool.calculate_borrow_rate(),
                "supply_rate": pool.calculate_supply_rate(),
                "collateral_factor": pool.collateral_factor,
                "liquidation_threshold": pool.liquidation_threshold,
                "reserve_balance": pool.reserve_balance
            }
        
        # MOET data
        moet_data = {
            "total_supply": self.moet.total_supply,
            "circulating_supply": self.moet.circulating_supply,
            "current_price": self.moet.current_price,
            "target_price": self.moet.target_price,
            "is_peg_stable": self.moet.is_peg_stable(),
            "stability_fund": self.moet.stability_fund
        }
        
        # Liquidity pool data
        lp_data = {}
        for pool_key, pool in self.liquidity_pools.items():
            lp_data[pool_key] = {
                "reserves": {asset.value: amount for asset, amount in pool.reserves.items()},
                "lp_token_supply": pool.lp_token_supply,
                "fee_rate": pool.fee_rate,
                "liquidation_capacity": pool.get_liquidation_capacity()
            }
        
        # Calculate current debt cap
        current_debt_cap = self.calculate_debt_cap({
            "current_prices": {
                Asset.ETH: 4400.0,
                Asset.BTC: 118000.0,
                Asset.FLOW: 0.40,
                Asset.USDC: 1.0,
                Asset.MOET: 1.0
            }
        })
        
        return {
            "protocol_type": "tidal_lending",
            "asset_pools": asset_data,
            "moet_stablecoin": moet_data,
            "liquidity_pools": lp_data,
            "protocol_treasury": self.protocol_treasury,
            "total_protocol_revenue": self.total_protocol_revenue,
            "lp_rewards_distributed": self.lp_rewards_distributed,
            "debt_cap": current_debt_cap,
            "target_health_factor": self.target_health_factor,
            "dex_liquidity_allocation": self.dex_liquidity_allocation
        }
