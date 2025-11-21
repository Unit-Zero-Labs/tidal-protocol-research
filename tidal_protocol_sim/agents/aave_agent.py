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
    """Extended agent state for AAVE-style scenario with tri-health factor system (no rebalancing)"""
    
    def __init__(self, agent_id: str, initial_balance: float, initial_hf: float, rebalancing_hf: float, target_hf: float):
        # Initialize exactly like High Tide agent with tri-health factor system
        super().__init__(agent_id, initial_balance, initial_hf, rebalancing_hf, target_hf)
        
        # Override to disable automatic rebalancing (health factors kept for comparison only)
        self.automatic_rebalancing = False
        
        # Track liquidation events
        self.liquidation_events = []
        self.total_liquidated_collateral = 0.0
        self.liquidation_penalties = 0.0
        
        # Track weekly harvest for incremental yield calculation
        self.last_harvest_yt_value = 0.0  # Total YT value at last harvest


class AaveAgent(BaseAgent):
    """
    AAVE-style agent with traditional liquidation mechanism (no auto-rebalancing)
    """
    
    def __init__(self, agent_id: str, initial_hf: float, rebalancing_hf: float, target_hf: float = None, initial_balance: float = 100_000.0):
        super().__init__(agent_id, "aave_agent", initial_balance)
        
        # Handle backward compatibility: if target_hf is None, use rebalancing_hf as target (old 2-factor system)
        if target_hf is None:
            target_hf = rebalancing_hf
            print(f"⚠️  Warning: {agent_id} using 2-factor compatibility mode. Consider updating to tri-health factor system.")
        
        # Replace state with AaveAgentState (tri-health factor system but no rebalancing)
        self.state = AaveAgentState(agent_id, initial_balance, initial_hf, rebalancing_hf, target_hf)
        
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
        """Calculate effective collateral value using Aave's liquidation threshold (not collateral factor)"""
        btc_price = asset_prices.get(Asset.BTC, 100_000.0)
        btc_amount = self.state.supplied_balances.get(Asset.BTC, 0.0)
        # Use Aave's BTC liquidation threshold (85%), not collateral factor (80%)
        # This must match the liquidation threshold used in debt calculation
        btc_liquidation_threshold = 0.85  # Aave's BTC liquidation threshold
        return btc_amount * btc_price * btc_liquidation_threshold
    
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
            
        # Purchase yield tokens (use direct minting at minute 0)
        new_tokens = self.state.yield_token_manager.mint_yield_tokens(moet_amount, current_minute, use_direct_minting=(current_minute == 0))
        
        # Update MOET debt (it's already borrowed, now used for yield tokens)
        # Debt remains the same, but MOET is now in yield tokens
        
        return len(new_tokens) > 0
    
    def execute_aave_liquidation(self, current_minute: int, asset_prices: Dict[Asset, float], 
                                pool_size_usd: float = 500_000) -> dict:
        """
        Execute AAVE-style liquidation with proper Uniswap V3 math:
        1. Seize 50% of collateral (BTC)
        2. Swap BTC -> MOET through Uniswap V3 pool
        3. Use MOET to pay down debt
        4. Liquidator receives 5% bonus on debt repaid (in BTC value)
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
        
        # 2. Calculate how much BTC to seize to get enough MOET to repay debt
        # We need to account for the 5% liquidation bonus
        liquidation_bonus_rate = 0.05  # 5% bonus
        debt_repaid_value = debt_reduction  # MOET debt value (assuming 1:1 with USD)
        
        # 3. Calculate BTC needed: debt_repaid_value * (1 + bonus) / btc_price
        # This gives us the BTC value needed to get enough MOET after the bonus
        btc_value_needed = debt_repaid_value * (1 + liquidation_bonus_rate)
        btc_to_seize = btc_value_needed / btc_price
        
        # 4. Ensure we don't seize more than available collateral
        btc_to_seize = min(btc_to_seize, current_btc_collateral)
        
        if btc_to_seize <= 0:
            return {}  # No collateral to liquidate
        
        # 5. Use Uniswap V3 math to calculate actual MOET received from BTC swap
        from ..core.uniswap_v3_math import create_moet_btc_pool, UniswapV3SlippageCalculator
        
        # Create pool and calculator
        pool = create_moet_btc_pool(pool_size_usd, btc_price)
        calculator = UniswapV3SlippageCalculator(pool)
        
        # Calculate BTC -> MOET swap with slippage
        btc_value_to_swap = btc_to_seize * btc_price
        swap_result = calculator.calculate_swap_slippage(btc_value_to_swap, "BTC")
        
        # Actual MOET received from swap (after slippage and fees)
        actual_moet_received = swap_result["amount_out"]
        
        # 6. Calculate actual debt that can be repaid (limited by MOET received)
        actual_debt_repaid = min(debt_reduction, actual_moet_received)
        
        # 7. Calculate liquidation bonus (5% of debt repaid, in BTC value)
        liquidation_bonus_value = actual_debt_repaid * liquidation_bonus_rate
        liquidation_bonus_btc = liquidation_bonus_value / btc_price
        
        # 8. Execute liquidation
        self.state.supplied_balances[Asset.BTC] -= btc_to_seize
        self.state.moet_debt -= actual_debt_repaid
        
        # Track liquidation event with Uniswap V3 details
        liquidation_event = {
            "minute": current_minute,
            "btc_seized": btc_to_seize,
            "btc_value_seized": btc_to_seize * btc_price,
            "debt_reduced": actual_debt_repaid,
            "debt_repaid_value": actual_debt_repaid,
            "liquidation_bonus_rate": liquidation_bonus_rate,
            "liquidation_bonus_value": liquidation_bonus_value,
            "liquidation_bonus_btc": liquidation_bonus_btc,
            "moet_received_from_swap": actual_moet_received,
            "swap_slippage": swap_result["slippage_amount"],
            "swap_fees": swap_result["trading_fees"],
            "price_impact": swap_result["price_impact_percentage"],
            "health_factor_before": self.state.health_factor,
            "remaining_collateral": self.state.supplied_balances.get(Asset.BTC, 0.0),
            "remaining_debt": self.state.moet_debt
        }
        
        self.state.liquidation_events.append(liquidation_event)
        self.state.total_liquidated_collateral += btc_to_seize * btc_price
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
        Calculate cost of liquidation for AAVE strategy
        
        The cost is simply the total BTC value seized (including bonus) across all liquidation events.
        This represents the actual cost to the agent from liquidations.
        """
        total_cost = 0.0
        
        for event in self.state.liquidation_events:
            # Cost is the BTC value seized (this includes the liquidation bonus)
            btc_value_seized = event["btc_value_seized"]
            total_cost += btc_value_seized
        
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
        
        # Calculate net position value properly (matches High Tide methodology)
        # Net position = BTC value + YT value - Debt
        btc_price = asset_prices.get(Asset.BTC, 100_000.0)
        btc_amount = self.state.supplied_balances.get(Asset.BTC, 0.0)
        btc_value = btc_amount * btc_price
        yt_value = yield_summary.get("total_current_value", 0.0)
        debt_value = self.state.moet_debt
        net_position_value = btc_value + yt_value - debt_value
        
        aave_metrics = {
            "risk_profile": self.risk_profile,
            "color": self.color,
            "initial_health_factor": self.state.initial_health_factor,
            "target_health_factor": self.state.target_health_factor,
            "btc_amount": btc_amount,  # Current amount (may be reduced)
            "initial_btc_amount": 1.0,  # Always started with 1 BTC
            "initial_moet_debt": self.state.initial_moet_debt,
            "current_moet_debt": debt_value,
            "total_interest_accrued": self.state.total_interest_accrued,
            "yield_token_portfolio": yield_summary,
            "total_yield_sold": 0.0,  # AAVE agents don't sell yield tokens
            "liquidation_events_count": len(self.state.liquidation_events),
            "total_liquidated_collateral": self.state.total_liquidated_collateral,
            "liquidation_penalties": self.state.liquidation_penalties,
            "cost_of_liquidation": cost_of_liquidation,
            "net_position_value": net_position_value,  # Properly calculated net position
            "automatic_rebalancing": False
        }
        
        # Merge with base summary
        base_summary.update(aave_metrics)
        return base_summary
        
    def get_liquidation_history(self) -> list:
        """Get history of liquidation events"""
        return self.state.liquidation_events.copy()
    
    def execute_weekly_rebalancing(self, current_minute: int, asset_prices: Dict[Asset, float], engine=None) -> dict:
        """
        Execute weekly manual rebalancing for AAVE agents:
        
        1. Check current HF vs initial HF
        2. If HF < initial HF: Sell YT → MOET → Pay down debt (deleverage for safety)
        3. If HF > initial HF: Sell rebased YT → BTC (harvest profits, only sell accrued yield)
        
        Returns dict with rebalancing details
        """
        # Update current health factor
        self._update_health_factor(asset_prices)
        
        current_hf = self.state.health_factor
        initial_hf = self.state.initial_health_factor
        
        rebalance_event = {
            "minute": current_minute,
            "day": current_minute / 1440,
            "week": current_minute / 10080,
            "hf_before": current_hf,
            "initial_hf": initial_hf,
            "action_taken": "none",
            "yt_sold": 0.0,
            "moet_received": 0.0,
            "btc_received": 0.0,
            "debt_repaid": 0.0,
            "hf_after": current_hf
        }
        
        # Check if we have any yield tokens
        yt_manager = self.state.yield_token_manager
        total_yt_value = yt_manager.calculate_total_value(current_minute)
        
        if total_yt_value <= 0:
            return rebalance_event
        
        # CASE 1: HF < Initial HF → Deleverage (sell YT → MOET → pay down debt)
        if current_hf < initial_hf:
            # Calculate how much YT to sell (sell enough to get back to initial HF)
            # We want to reduce debt to increase HF
            btc_price = asset_prices.get(Asset.BTC, 100_000.0)
            btc_amount = self.state.supplied_balances.get(Asset.BTC, 0.0)
            collateral_value = btc_amount * btc_price * 0.80  # 80% collateral factor
            
            # Target debt to reach initial HF: debt = collateral_value / initial_hf
            target_debt = collateral_value / initial_hf
            debt_to_repay = max(0, self.state.moet_debt - target_debt)
            
            # Limit to available YT (sell at most 50% of YT holdings per week)
            max_yt_to_sell = total_yt_value * 0.5
            yt_to_sell = min(debt_to_repay, max_yt_to_sell)
            
            if yt_to_sell > 10:  # Only if meaningful amount ($10 threshold)
                # Sell YT for MOET
                moet_received, actual_yt_sold = yt_manager.sell_yield_tokens(yt_to_sell, current_minute)
                
                if moet_received > 0:
                    # Pay down debt
                    debt_repaid = min(moet_received, self.state.moet_debt)
                    self.state.moet_debt -= debt_repaid
                    
                    # Update event
                    rebalance_event["action_taken"] = "deleverage"
                    rebalance_event["yt_sold"] = actual_yt_sold
                    rebalance_event["moet_received"] = moet_received
                    rebalance_event["debt_repaid"] = debt_repaid
                    
                    # Update HF after deleveraging
                    self._update_health_factor(asset_prices)
                    rebalance_event["hf_after"] = self.state.health_factor
        
        # CASE 2: HF > Initial HF → Harvest profits (sell rebased YT → BTC)
        elif current_hf > initial_hf:
            # Only sell the rebasing yield (accrued interest), not the principal
            # Calculate NEW yield since last harvest (not cumulative total)
            current_yt_value = yt_manager.calculate_total_value(current_minute)
            
            # First time harvesting? Set baseline
            if self.state.last_harvest_yt_value == 0:
                self.state.last_harvest_yt_value = yt_manager.total_initial_value_invested
            
            # New yield = current value - last harvest value
            incremental_yield = max(0, current_yt_value - self.state.last_harvest_yt_value)
            
            # Sell the incremental yield (100% of new yield this week)
            yt_to_sell = incremental_yield
            
            if yt_to_sell > 10:  # Only if meaningful amount ($10 threshold)
                # Sell YT for MOET first
                moet_received, actual_yt_sold = yt_manager.sell_yield_tokens(yt_to_sell, current_minute)
                
                if moet_received > 0 and engine:
                    # Swap MOET for BTC using engine's pools
                    # This requires access to the MOET:BTC pool
                    # For simplicity, approximate: BTC received = MOET / BTC price
                    btc_price = asset_prices.get(Asset.BTC, 100_000.0)
                    btc_received = moet_received / btc_price
                    
                    # Add to collateral
                    self.state.supplied_balances[Asset.BTC] = self.state.supplied_balances.get(Asset.BTC, 0.0) + btc_received
                    self.state.btc_amount += btc_received
                    
                    # Update last harvest value to current (after selling the yield)
                    self.state.last_harvest_yt_value = yt_manager.calculate_total_value(current_minute)
                    
                    # Update event
                    rebalance_event["action_taken"] = "harvest_profits"
                    rebalance_event["yt_sold"] = actual_yt_sold
                    rebalance_event["moet_received"] = moet_received
                    rebalance_event["btc_received"] = btc_received
                    rebalance_event["incremental_yield_this_week"] = incremental_yield
                    
                    # Update HF after adding BTC
                    self._update_health_factor(asset_prices)
                    rebalance_event["hf_after"] = self.state.health_factor
        
        return rebalance_event


def create_aave_agents(num_agents: int, monte_carlo_variation: bool = True) -> list:
    """
    Create AAVE agents with SAME risk profile distribution as High Tide agents using tri-health factor system
    
    Tri-Health Factor System (for comparison only - AAVE doesn't rebalance):
    - Initial HF: Starting position health
    - Rebalancing HF: Not used by AAVE (kept for comparison)
    - Target HF: Not used by AAVE (kept for comparison)
    
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
        # Conservative: Small rebalancing buffer (0.05-0.15 below initial)
        rebalancing_hf = initial_hf - random.uniform(0.05, 0.15)
        rebalancing_hf = max(rebalancing_hf, 1.1)  # Minimum rebalancing HF is 1.1
        # Target HF: Small safety buffer above rebalancing HF
        target_hf = rebalancing_hf + random.uniform(0.01, 0.05)
        target_hf = max(target_hf, 1.1)  # Minimum target HF is 1.1
        
        agent = AaveAgent(
            f"aave_conservative_{agent_id}",
            initial_hf,
            rebalancing_hf,
            target_hf
        )
        agents.append(agent)
        agent_id += 1
    
    # Create moderate agents (same parameters as High Tide)
    for i in range(moderate_count):
        initial_hf = random.uniform(1.5, 1.8)
        # Moderate: Medium rebalancing buffer (0.15-0.25 below initial)
        rebalancing_hf = initial_hf - random.uniform(0.15, 0.25)
        rebalancing_hf = max(rebalancing_hf, 1.1)  # Minimum rebalancing HF is 1.1
        # Target HF: Small safety buffer above rebalancing HF
        target_hf = rebalancing_hf + random.uniform(0.01, 0.05)
        target_hf = max(target_hf, 1.1)  # Minimum target HF is 1.1
        
        agent = AaveAgent(
            f"aave_moderate_{agent_id}",
            initial_hf,
            rebalancing_hf,
            target_hf
        )
        agents.append(agent)
        agent_id += 1
    
    # Create aggressive agents (same parameters as High Tide)
    for i in range(aggressive_count):
        initial_hf = random.uniform(1.3, 1.5)
        # Aggressive: Larger rebalancing buffer (0.15-0.4 below initial)
        rebalancing_hf = initial_hf - random.uniform(0.15, 0.4)
        rebalancing_hf = max(rebalancing_hf, 1.1)  # Minimum rebalancing HF is 1.1
        # Target HF: Small safety buffer above rebalancing HF
        target_hf = rebalancing_hf + random.uniform(0.01, 0.05)
        target_hf = max(target_hf, 1.1)  # Minimum target HF is 1.1
        
        agent = AaveAgent(
            f"aave_aggressive_{agent_id}",
            initial_hf,
            rebalancing_hf,
            target_hf
        )
        agents.append(agent)
        agent_id += 1
    
    return agents
