#!/usr/bin/env python3
"""
AAVE-Style Agent Implementation

Implements traditional liquidation mechanism where agents hold positions without
active rebalancing until health factor crosses 1.0, then face liquidation with
50% collateral seizure + 5% bonus.
"""

import random
from typing import Dict, Tuple, Optional
from .base_agent import BaseAgent, AgentAction, AgentState
from .high_tide_agent import HighTideAgentState  # Reuse the state structure
from ..core.protocol import Asset
from ..core.yield_tokens import YieldTokenManager
from ..core.uniswap_v3_math import calculate_liquidation_cost_with_slippage


class AaveAgentState(HighTideAgentState):
    """Extended agent state for AAVE-style scenario - same as High Tide but no rebalancing"""
    
    def __init__(self, agent_id: str, initial_balance: float, initial_hf: float, target_hf: float):
        # Initialize exactly like High Tide agent
        super().__init__(agent_id, initial_balance, initial_hf, target_hf)
        
        # Override to disable automatic rebalancing
        self.automatic_rebalancing = False
        
        # Track liquidation events
        self.liquidation_events = []
        self.total_liquidated_collateral = 0.0
        self.liquidation_penalties = 0.0


class AaveAgent(BaseAgent):
    """
    AAVE-style agent with traditional liquidation mechanism (no auto-rebalancing)
    """
    
    def __init__(self, agent_id: str, initial_hf: float, target_hf: float, initial_balance: float = 100_000.0):
        super().__init__(agent_id, "aave_agent", initial_balance)
        
        # Replace state with AaveAgentState (same parameters as High Tide)
        self.state = AaveAgentState(agent_id, initial_balance, initial_hf, target_hf)
        
        # Risk profile based on initial health factor (same as High Tide)
        if initial_hf >= 2.1:
            self.risk_profile = "conservative"
            self.color = "#2E8B57"  # Sea Green
        elif initial_hf >= 1.5:
            self.risk_profile = "moderate" 
            self.color = "#FF8C00"  # Dark Orange
        else:
            self.risk_profile = "aggressive"
            self.color = "#DC143C"  # Crimson
            
        self.risk_tolerance = 1.0 / initial_hf  # Inverse relationship
        
    def decide_action(self, protocol_state: dict, asset_prices: Dict[Asset, float]) -> tuple:
        """
        AAVE-style decision logic:
        1. Initially purchase yield tokens with borrowed MOET (same as High Tide)
        2. NO rebalancing - hold position until liquidation
        3. Track health factor but take no action
        """
        current_minute = protocol_state.get("current_step", 0)
        
        # Update health factor
        self._update_health_factor(asset_prices)
        
        # Check if we need to purchase yield tokens initially (same as High Tide)
        if (self.state.moet_debt > 0 and 
            len(self.state.yield_token_manager.yield_tokens) == 0):
            return self._initial_yield_token_purchase(current_minute)
        
        # NO REBALANCING - this is the key difference from High Tide
        # AAVE agents hold their position until liquidation
        
        # Default action - hold position (no matter what the health factor is)
        return (AgentAction.HOLD, {})
    
    def _initial_yield_token_purchase(self, current_minute: int) -> tuple:
        """Purchase yield tokens with initially borrowed MOET (same as High Tide)"""
        moet_available = self.state.borrowed_balances.get(Asset.MOET, 0.0)
        
        if moet_available > 0:
            # Use all borrowed MOET to purchase yield tokens
            return (AgentAction.SWAP, {
                "action_type": "buy_yield_tokens",
                "moet_amount": moet_available,
                "current_minute": current_minute
            })
        
        return (AgentAction.HOLD, {})
    
    def _update_health_factor(self, asset_prices: Dict[Asset, float]):
        """Update health factor for AAVE agent (same calculation as High Tide)"""
        collateral_value = self._calculate_effective_collateral_value(asset_prices)
        debt_value = self.state.moet_debt * asset_prices.get(Asset.MOET, 1.0)
        
        if debt_value <= 0:
            self.state.health_factor = float('inf')
        else:
            self.state.health_factor = collateral_value / debt_value
    
    def _calculate_effective_collateral_value(self, asset_prices: Dict[Asset, float]) -> float:
        """Calculate effective collateral value using BTC collateral factor (same as High Tide)"""
        btc_price = asset_prices.get(Asset.BTC, 100_000.0)
        btc_amount = self.state.supplied_balances.get(Asset.BTC, 0.0)
        # Use BTC collateral factor (should be 0.80)
        btc_collateral_factor = 0.80  # This matches what we set in protocol.py
        return btc_amount * btc_price * btc_collateral_factor
    
    def update_debt_interest(self, current_minute: int, btc_pool_borrow_rate: float):
        """Update debt with accrued interest (same as High Tide)"""
        if current_minute <= self.state.last_interest_update_minute:
            return
            
        minutes_elapsed = current_minute - self.state.last_interest_update_minute
        if minutes_elapsed <= 0:
            return
            
        # Convert annual rate to per-minute rate
        minute_rate = btc_pool_borrow_rate / (365 * 24 * 60)
        
        # Compound interest over elapsed minutes
        interest_factor = (1 + minute_rate) ** minutes_elapsed
        
        # Calculate interest accrued
        old_debt = self.state.moet_debt
        self.state.moet_debt *= interest_factor
        interest_accrued = self.state.moet_debt - old_debt
        
        self.state.total_interest_accrued += interest_accrued
        self.state.last_interest_update_minute = current_minute
    
    def execute_yield_token_purchase(self, moet_amount: float, current_minute: int) -> bool:
        """Execute yield token purchase (same as High Tide)"""
        if moet_amount <= 0:
            return False
            
        # Purchase yield tokens
        new_tokens = self.state.yield_token_manager.mint_yield_tokens(moet_amount, current_minute)
        
        # Update MOET debt (it's already borrowed, now used for yield tokens)
        # Debt remains the same, but MOET is now in yield tokens
        
        return len(new_tokens) > 0
    
    def execute_aave_liquidation(self, current_minute: int, asset_prices: Dict[Asset, float]) -> dict:
        """
        Execute AAVE-style liquidation:
        - Reduce debt by 50%
        - Liquidator receives: Debt Repaid * (1 + Liquidation Bonus)
        - Agent continues with reduced position
        """
        if self.state.health_factor >= 1.0:
            return {}  # No liquidation needed
        
        # Calculate liquidation amounts
        btc_price = asset_prices.get(Asset.BTC, 100_000.0)
        current_btc_collateral = self.state.supplied_balances.get(Asset.BTC, 0.0)
        current_debt = self.state.moet_debt
        
        # AAVE liquidation mechanics
        # 1. Debt reduction: 50% of current debt
        debt_reduction = current_debt * 0.50
        
        # 2. Liquidator receives: Debt Repaid * (1 + Liquidation Bonus) worth of collateral
        liquidation_bonus_rate = 0.05  # 5% bonus
        debt_repaid_value = debt_reduction  # MOET debt value (assuming 1:1 with USD)
        liquidator_receives_value = debt_repaid_value * (1 + liquidation_bonus_rate)
        
        # 3. Convert to BTC collateral seized: Liquidator gets liquidator_receives_value worth of BTC
        total_btc_seized = liquidator_receives_value / btc_price
        
        # 4. Ensure we don't seize more than available collateral
        total_btc_seized = min(total_btc_seized, current_btc_collateral)
        
        # Execute liquidation
        self.state.supplied_balances[Asset.BTC] -= total_btc_seized
        self.state.moet_debt -= debt_reduction
        
        # Calculate liquidation bonus in USD
        liquidation_bonus_value = debt_repaid_value * liquidation_bonus_rate
        
        # Track liquidation event
        liquidation_event = {
            "minute": current_minute,
            "btc_seized": total_btc_seized,
            "btc_value_seized": total_btc_seized * btc_price,
            "debt_reduced": debt_reduction,
            "debt_repaid_value": debt_repaid_value,
            "liquidation_bonus_rate": liquidation_bonus_rate,
            "liquidation_bonus_value": liquidation_bonus_value,
            "liquidator_receives_value": liquidator_receives_value,
            "health_factor_before": self.state.health_factor,
            "remaining_collateral": self.state.supplied_balances.get(Asset.BTC, 0.0),
            "remaining_debt": self.state.moet_debt
        }
        
        self.state.liquidation_events.append(liquidation_event)
        self.state.total_liquidated_collateral += total_btc_seized * btc_price
        self.state.liquidation_penalties += liquidation_bonus_value
        
        # Update health factor after liquidation
        self._update_health_factor(asset_prices)
        liquidation_event["health_factor_after"] = self.state.health_factor
        
        # Agent becomes inactive if all collateral is seized
        if self.state.supplied_balances.get(Asset.BTC, 0.0) <= 0.001:  # Practically zero
            self.active = False
        
        return liquidation_event
    
    def calculate_cost_of_liquidation(self, final_btc_price: float, current_minute: int) -> float:
        """
        Calculate cost of liquidation for AAVE strategy including Uniswap v3 slippage
        
        For AAVE, the cost includes liquidation penalties plus slippage costs from liquidator swaps.
        """
        # Base liquidation penalties (5% bonus to liquidators)
        base_penalties = self.state.liquidation_penalties
        
        # Calculate slippage costs for all liquidation events
        total_slippage_cost = 0.0
        for event in self.state.liquidation_events:
            btc_seized = event["btc_seized"]
            btc_price_at_liquidation = event["btc_value_seized"] / btc_seized if btc_seized > 0 else final_btc_price
            
            if btc_seized > 0:
                slippage_result = calculate_liquidation_cost_with_slippage(
                    btc_seized,
                    btc_price_at_liquidation,
                    liquidation_percentage=1.0,  # Using full amount seized
                    liquidation_bonus=0.05,      # 5% bonus
                    pool_size_usd=500_000,       # $500K total pool
                    concentrated_range=0.2       # 20% concentration range
                )
                total_slippage_cost += slippage_result["slippage_cost"] + slippage_result["trading_fees"]
        
        # Total cost includes base penalties plus all slippage costs
        total_cost = base_penalties + total_slippage_cost
        
        return total_cost
    
    def get_detailed_portfolio_summary(self, asset_prices: Dict[Asset, float], current_minute: int) -> dict:
        """Get comprehensive portfolio summary for AAVE agent"""
        base_summary = super().get_portfolio_summary(asset_prices)
        
        # Add AAVE specific metrics
        yield_summary = self.state.yield_token_manager.get_portfolio_summary(current_minute)
        cost_of_liquidation = self.calculate_cost_of_liquidation(
            asset_prices.get(Asset.BTC, 100_000.0), 
            current_minute
        )
        
        aave_metrics = {
            "risk_profile": self.risk_profile,
            "color": self.color,
            "initial_health_factor": self.state.initial_health_factor,
            "target_health_factor": self.state.target_health_factor,
            "btc_amount": self.state.supplied_balances.get(Asset.BTC, 0.0),  # Current amount (may be reduced)
            "initial_btc_amount": 1.0,  # Always started with 1 BTC
            "initial_moet_debt": self.state.initial_moet_debt,
            "current_moet_debt": self.state.moet_debt,
            "total_interest_accrued": self.state.total_interest_accrued,
            "yield_token_portfolio": yield_summary,
            "total_yield_sold": 0.0,  # AAVE agents don't sell yield tokens
            "liquidation_events_count": len(self.state.liquidation_events),
            "total_liquidated_collateral": self.state.total_liquidated_collateral,
            "liquidation_penalties": self.state.liquidation_penalties,
            "cost_of_liquidation": cost_of_liquidation,
            "net_position_value": 100_000.0 - cost_of_liquidation,
            "automatic_rebalancing": False
        }
        
        # Merge with base summary
        base_summary.update(aave_metrics)
        return base_summary
        
    def get_liquidation_history(self) -> list:
        """Get history of liquidation events"""
        return self.state.liquidation_events.copy()


def create_aave_agents(num_agents: int, monte_carlo_variation: bool = True) -> list:
    """
    Create AAVE agents with SAME risk profile distribution as High Tide agents
    
    This ensures fair comparison between High Tide and AAVE strategies
    """
    if monte_carlo_variation:
        # Randomize agent count between 10-50 (same as High Tide)
        num_agents = random.randint(10, 50)
    
    agents = []
    
    # Calculate distribution (same as High Tide)
    conservative_count = int(num_agents * 0.3)
    moderate_count = int(num_agents * 0.4)
    aggressive_count = num_agents - conservative_count - moderate_count
    
    agent_id = 0
    
    # Create conservative agents (same parameters as High Tide)
    for i in range(conservative_count):
        initial_hf = random.uniform(2.1, 2.4)
        # Conservative: Small buffer (0.05-0.15 below initial)
        target_hf = initial_hf - random.uniform(0.05, 0.15)
        target_hf = max(target_hf, 1.1)  # Minimum target HF is 1.1
        
        agent = AaveAgent(
            f"aave_conservative_{agent_id}",
            initial_hf,
            target_hf
        )
        agents.append(agent)
        agent_id += 1
    
    # Create moderate agents (same parameters as High Tide)
    for i in range(moderate_count):
        initial_hf = random.uniform(1.5, 1.8)
        # Moderate: Medium buffer (0.15-0.25 below initial)
        target_hf = initial_hf - random.uniform(0.15, 0.25)
        target_hf = max(target_hf, 1.1)  # Minimum target HF is 1.1
        
        agent = AaveAgent(
            f"aave_moderate_{agent_id}",
            initial_hf,
            target_hf
        )
        agents.append(agent)
        agent_id += 1
    
    # Create aggressive agents (same parameters as High Tide)
    for i in range(aggressive_count):
        initial_hf = random.uniform(1.3, 1.5)
        # Aggressive: Larger buffer (0.15-0.4 below initial)
        target_hf = initial_hf - random.uniform(0.15, 0.4)
        target_hf = max(target_hf, 1.1)  # Minimum target HF is 1.1
        
        agent = AaveAgent(
            f"aave_aggressive_{agent_id}",
            initial_hf,
            target_hf
        )
        agents.append(agent)
        agent_id += 1
    
    return agents
