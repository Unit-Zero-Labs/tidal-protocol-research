#!/usr/bin/env python3
"""
Uniswap V2-style AMM market for the Tidal Protocol simulation.

This market implements constant product (x*y=k) automated market maker functionality.
"""

from typing import List, Dict, Any
import math
from .base import BaseMarket
from ..simulation.primitives import Action, Event, ActionKind, Asset


class UniswapV2Market(BaseMarket):
    """Constant product AMM market (x*y=k)"""
    
    def __init__(self, 
                 market_id: str = "uniswap_v2",
                 initial_reserves: Dict[Asset, float] = None,
                 fee_rate: float = 0.003):
        """
        Initialize Uniswap V2 market
        
        Args:
            market_id: Unique market identifier
            initial_reserves: Initial token reserves
            fee_rate: Trading fee rate (default 0.3%)
        """
        super().__init__(market_id)
        
        # Set up supported actions
        self.supported_actions.update([
            ActionKind.SWAP_BUY,
            ActionKind.SWAP_SELL,
            ActionKind.ADD_LIQUIDITY,
            ActionKind.REMOVE_LIQUIDITY,
            ActionKind.COLLECT_FEES
        ])
        
        # Initialize reserves
        self.reserves = initial_reserves or {
            Asset.MOET: 1000000.0,  # 1M MOET
            Asset.USDC: 1000000.0,  # 1M USDC
            Asset.ETH: 333.33,      # ~1M USD worth at $3000/ETH
            Asset.BTC: 22.22        # ~1M USD worth at $45000/BTC
        }
        
        self.fee_rate = fee_rate
        self.total_fees_collected = {asset: 0.0 for asset in Asset}
        self.lp_token_supply = 1000000.0  # Initial LP token supply
        self.daily_volume = {asset: 0.0 for asset in Asset}
        self.current_day = 0
    
    def route(self, action: Action, simulation_state: Dict[str, Any]) -> List[Event]:
        """Execute AMM actions"""
        if action.kind == ActionKind.SWAP_BUY:
            return self._handle_swap_buy(action, simulation_state)
        elif action.kind == ActionKind.SWAP_SELL:
            return self._handle_swap_sell(action, simulation_state)
        elif action.kind == ActionKind.ADD_LIQUIDITY:
            return self._handle_add_liquidity(action, simulation_state)
        elif action.kind == ActionKind.REMOVE_LIQUIDITY:
            return self._handle_remove_liquidity(action, simulation_state)
        elif action.kind == ActionKind.COLLECT_FEES:
            return self._handle_collect_fees(action, simulation_state)
        
        return []
    
    def _handle_swap_buy(self, action: Action, simulation_state: Dict[str, Any]) -> List[Event]:
        """Handle token buy (swap USDC/ETH/BTC for MOET)"""
        asset_in = action.params.get("asset_in", Asset.USDC)
        asset_out = action.params.get("asset_out", Asset.MOET)
        amount_in = action.params.get("amount_in", 0.0)
        min_amount_out = action.params.get("min_amount_out", 0.0)
        
        # Get agent
        agents = simulation_state.get("agents", {})
        agent = agents.get(action.agent_id)
        if not agent:
            return [self.create_event(action, {}, False, "Agent not found")]
        
        # Check agent balance
        if agent.state.token_balances.get(asset_in, 0.0) < amount_in:
            return [self.create_event(action, {}, False, "Insufficient balance")]
        
        # Calculate swap using constant product formula
        reserve_in = self.reserves.get(asset_in, 0.0)
        reserve_out = self.reserves.get(asset_out, 0.0)
        
        if reserve_in <= 0 or reserve_out <= 0:
            return [self.create_event(action, {}, False, "Invalid reserves")]
        
        # Apply fee
        amount_in_with_fee = amount_in * (1 - self.fee_rate)
        
        # Calculate amount out using constant product formula
        # amount_out = reserve_out - (reserve_in * reserve_out) / (reserve_in + amount_in_with_fee)
        amount_out = (amount_in_with_fee * reserve_out) / (reserve_in + amount_in_with_fee)
        
        # Check slippage
        if amount_out < min_amount_out:
            return [self.create_event(action, {}, False, "Slippage too high")]
        
        # Update reserves
        self.reserves[asset_in] += amount_in
        self.reserves[asset_out] -= amount_out
        
        # Update fees
        fee_amount = amount_in * self.fee_rate
        self.total_fees_collected[asset_in] += fee_amount
        
        # Update daily volume
        self.daily_volume[asset_in] += amount_in
        
        # Update agent balances
        agent.state.token_balances[asset_in] -= amount_in
        agent.state.token_balances[asset_out] = agent.state.token_balances.get(asset_out, 0.0) + amount_out
        agent.state.total_fees_paid += fee_amount
        
        return [self.create_event(action, {
            "amount_in": amount_in,
            "amount_out": amount_out,
            "fee_paid": fee_amount,
            "asset_in": asset_in.value,
            "asset_out": asset_out.value,
            "price_impact": self._calculate_price_impact(asset_in, asset_out, amount_in)
        })]
    
    def _handle_swap_sell(self, action: Action, simulation_state: Dict[str, Any]) -> List[Event]:
        """Handle token sell (swap MOET for USDC/ETH/BTC)"""
        asset_in = action.params.get("asset_in", Asset.MOET)
        asset_out = action.params.get("asset_out", Asset.USDC)
        amount_in = action.params.get("amount_in", 0.0)
        min_amount_out = action.params.get("min_amount_out", 0.0)
        
        # Get agent
        agents = simulation_state.get("agents", {})
        agent = agents.get(action.agent_id)
        if not agent:
            return [self.create_event(action, {}, False, "Agent not found")]
        
        # Check agent balance
        if agent.state.token_balances.get(asset_in, 0.0) < amount_in:
            return [self.create_event(action, {}, False, "Insufficient balance")]
        
        # Calculate swap using constant product formula
        reserve_in = self.reserves.get(asset_in, 0.0)
        reserve_out = self.reserves.get(asset_out, 0.0)
        
        if reserve_in <= 0 or reserve_out <= 0:
            return [self.create_event(action, {}, False, "Invalid reserves")]
        
        # Apply fee
        amount_in_with_fee = amount_in * (1 - self.fee_rate)
        
        # Calculate amount out
        amount_out = (amount_in_with_fee * reserve_out) / (reserve_in + amount_in_with_fee)
        
        # Check slippage
        if amount_out < min_amount_out:
            return [self.create_event(action, {}, False, "Slippage too high")]
        
        # Update reserves
        self.reserves[asset_in] += amount_in
        self.reserves[asset_out] -= amount_out
        
        # Update fees
        fee_amount = amount_in * self.fee_rate
        self.total_fees_collected[asset_in] += fee_amount
        
        # Update daily volume
        self.daily_volume[asset_in] += amount_in
        
        # Update agent balances
        agent.state.token_balances[asset_in] -= amount_in
        agent.state.token_balances[asset_out] = agent.state.token_balances.get(asset_out, 0.0) + amount_out
        agent.state.total_fees_paid += fee_amount
        
        return [self.create_event(action, {
            "amount_in": amount_in,
            "amount_out": amount_out,
            "fee_paid": fee_amount,
            "asset_in": asset_in.value,
            "asset_out": asset_out.value,
            "price_impact": self._calculate_price_impact(asset_in, asset_out, amount_in)
        })]
    
    def _handle_add_liquidity(self, action: Action, simulation_state: Dict[str, Any]) -> List[Event]:
        """Handle adding liquidity to the pool"""
        asset_a = action.params.get("asset_a", Asset.MOET)
        asset_b = action.params.get("asset_b", Asset.USDC)
        amount_a = action.params.get("amount_a", 0.0)
        amount_b = action.params.get("amount_b", 0.0)
        
        # Get agent
        agents = simulation_state.get("agents", {})
        agent = agents.get(action.agent_id)
        if not agent:
            return [self.create_event(action, {}, False, "Agent not found")]
        
        # Check agent balances
        if (agent.state.token_balances.get(asset_a, 0.0) < amount_a or
            agent.state.token_balances.get(asset_b, 0.0) < amount_b):
            return [self.create_event(action, {}, False, "Insufficient balance")]
        
        # Calculate LP tokens to mint
        reserve_a = self.reserves.get(asset_a, 0.0)
        reserve_b = self.reserves.get(asset_b, 0.0)
        
        if reserve_a > 0 and reserve_b > 0:
            # Existing pool - maintain ratio
            lp_tokens = min(
                (amount_a / reserve_a) * self.lp_token_supply,
                (amount_b / reserve_b) * self.lp_token_supply
            )
        else:
            # New pool
            lp_tokens = math.sqrt(amount_a * amount_b)
        
        # Update reserves
        self.reserves[asset_a] += amount_a
        self.reserves[asset_b] += amount_b
        self.lp_token_supply += lp_tokens
        
        # Update agent balances
        agent.state.token_balances[asset_a] -= amount_a
        agent.state.token_balances[asset_b] -= amount_b
        agent.state.lp_balance += lp_tokens
        
        return [self.create_event(action, {
            "amount_a": amount_a,
            "amount_b": amount_b,
            "lp_tokens_minted": lp_tokens,
            "asset_a": asset_a.value,
            "asset_b": asset_b.value
        })]
    
    def _handle_remove_liquidity(self, action: Action, simulation_state: Dict[str, Any]) -> List[Event]:
        """Handle removing liquidity from the pool"""
        lp_tokens = action.params.get("lp_tokens", 0.0)
        asset_a = action.params.get("asset_a", Asset.MOET)
        asset_b = action.params.get("asset_b", Asset.USDC)
        
        # Get agent
        agents = simulation_state.get("agents", {})
        agent = agents.get(action.agent_id)
        if not agent:
            return [self.create_event(action, {}, False, "Agent not found")]
        
        # Check LP token balance
        if agent.state.lp_balance < lp_tokens:
            return [self.create_event(action, {}, False, "Insufficient LP tokens")]
        
        # Calculate amounts to return
        share = lp_tokens / self.lp_token_supply
        amount_a = self.reserves.get(asset_a, 0.0) * share
        amount_b = self.reserves.get(asset_b, 0.0) * share
        
        # Update reserves
        self.reserves[asset_a] -= amount_a
        self.reserves[asset_b] -= amount_b
        self.lp_token_supply -= lp_tokens
        
        # Update agent balances
        agent.state.lp_balance -= lp_tokens
        agent.state.token_balances[asset_a] = agent.state.token_balances.get(asset_a, 0.0) + amount_a
        agent.state.token_balances[asset_b] = agent.state.token_balances.get(asset_b, 0.0) + amount_b
        
        return [self.create_event(action, {
            "lp_tokens_burned": lp_tokens,
            "amount_a": amount_a,
            "amount_b": amount_b,
            "asset_a": asset_a.value,
            "asset_b": asset_b.value
        })]
    
    def _handle_collect_fees(self, action: Action, simulation_state: Dict[str, Any]) -> List[Event]:
        """Handle fee collection for LP providers"""
        # Simplified fee distribution - could be more sophisticated
        return [self.create_event(action, {"message": "Fee collection not implemented yet"})]
    
    def _calculate_price_impact(self, asset_in: Asset, asset_out: Asset, amount_in: float) -> float:
        """Calculate price impact of a trade"""
        reserve_in = self.reserves.get(asset_in, 0.0)
        if reserve_in <= 0:
            return 0.0
        
        return amount_in / reserve_in
    
    def end_of_block(self, simulation_state: Dict[str, Any]) -> List[Event]:
        """Handle end-of-block operations"""
        # Reset daily volume tracking
        current_day = simulation_state.get("current_day", 0)
        if current_day != self.current_day:
            self.daily_volume = {asset: 0.0 for asset in Asset}
            self.current_day = current_day
        
        return []
    
    def get_market_data(self) -> Dict[str, Any]:
        """Provide market data for snapshots"""
        # Calculate current prices based on reserves
        moet_price = 1.0
        if self.reserves.get(Asset.MOET, 0) > 0 and self.reserves.get(Asset.USDC, 0) > 0:
            moet_price = self.reserves[Asset.USDC] / self.reserves[Asset.MOET]
        
        # Calculate total daily volume in USD
        total_volume_usd = sum(
            volume * 1.0  # Simplified - assume all volumes are in USD terms
            for volume in self.daily_volume.values()
        )
        
        # Calculate APY (simplified)
        total_fees_usd = sum(self.total_fees_collected.values())
        total_liquidity_usd = sum(
            reserve * 1.0  # Simplified pricing
            for reserve in self.reserves.values()
        )
        
        apy = (total_fees_usd * 365) / max(total_liquidity_usd, 1.0) if total_liquidity_usd > 0 else 0.0
        
        return {
            "reserves": {asset.value: reserve for asset, reserve in self.reserves.items()},
            "lp_token_supply": self.lp_token_supply,
            "total_fees_collected": {asset.value: fees for asset, fees in self.total_fees_collected.items()},
            "daily_volume": {asset.value: volume for asset, volume in self.daily_volume.items()},
            "total_volume_usd": total_volume_usd,
            "apy": apy,
            "moet_price": moet_price,
            "fee_rate": self.fee_rate
        }
