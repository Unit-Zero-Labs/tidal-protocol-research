#!/usr/bin/env python3
"""
High Tide Agent Implementation

Implements actively managed lending positions where agents automatically
purchase yield tokens and rebalance when health factors decline.
"""

import random
from typing import Dict, Tuple, Optional
from .base_agent import BaseAgent, AgentAction, AgentState
from ..core.protocol import Asset
from ..core.yield_tokens import YieldTokenManager
from ..core.uniswap_v3_math import calculate_rebalancing_cost_with_slippage


class HighTideAgentState(AgentState):
    """Extended agent state for High Tide scenario with tri-health factor system"""
    
    def __init__(self, agent_id: str, initial_balance: float, initial_hf: float, rebalancing_hf: float, target_hf: float, yield_token_pool=None):
        # Initialize with BTC collateral focus
        super().__init__(agent_id, initial_balance, "high_tide_agent")
        
        # Tri-Health Factor System Parameters
        self.initial_health_factor = initial_hf        # The HF when position was first opened
        self.rebalancing_health_factor = rebalancing_hf # Threshold that triggers automated rebalancing
        self.target_health_factor = target_hf          # Post-rebalancing health target (safety buffer)
        self.automatic_rebalancing = True
        
        # Override default initialization for High Tide scenario
        # Each agent deposits exactly 1 BTC (use actual starting BTC price)
        btc_price = initial_balance  # Use the provided initial_balance as BTC price
        btc_amount = 1.0  # Exactly 1 BTC
        
        # Calculate borrowing capacity using BTC collateral factor
        btc_collateral_factor = 0.80  # BTC collateral factor
        effective_collateral_value = btc_amount * btc_price * btc_collateral_factor  # 1 BTC × price × 0.80
        moet_to_borrow = effective_collateral_value / initial_hf  # Borrow based on initial HF
        
        # Reset balances for High Tide scenario
        self.token_balances = {
            Asset.ETH: 0.0,
            Asset.BTC: 0.0,  # All BTC will be supplied as collateral
            Asset.FLOW: 0.0,
            Asset.USDC: 0.0,
            Asset.MOET: moet_to_borrow  # Initial borrowed MOET available for YT purchase
        }
        
        # BTC supplied as collateral
        self.supplied_balances = {
            Asset.ETH: 0.0,
            Asset.BTC: btc_amount,  # 1 BTC supplied
            Asset.FLOW: 0.0,
            Asset.USDC: 0.0
        }
        
        # MOET borrowed based on initial health factor
        self.borrowed_balances = {Asset.MOET: moet_to_borrow}
        
        # Initialize yield token manager with pool
        self.yield_token_manager = YieldTokenManager(yield_token_pool)
        
        # Rebalancing tracking
        self.rebalancing_events = []
        self.total_yield_sold = 0.0  # Total YT sold (all purposes)
        self.total_yield_sold_for_rebalancing = 0.0  # YT sold for health factor rebalancing only
        self.total_rebalancing_slippage = 0.0  # Slippage from rebalancing YT sales
        self.emergency_liquidations = 0
        
        # Calculate initial health factor
        self.btc_amount = btc_amount
        self.moet_debt = moet_to_borrow
        self.initial_moet_debt = moet_to_borrow  # Track original debt
        self.initial_yield_token_value = 0.0  # Will be set when yield tokens are first purchased
        
        # Interest tracking
        self.total_interest_accrued = 0.0
        self.last_interest_update_minute = 0
        
        # Deleveraging tracking
        self.last_weekly_delever_minute = 0  # Track last weekly deleveraging
        self.deleveraging_events = []  # Track deleveraging history
        self.total_deleveraging_sales = 0.0  # Total YT sold for deleveraging
        self.total_deleveraging_slippage = 0.0  # Total slippage from deleveraging chain
        

class HighTideAgent(BaseAgent):
    """
    High Tide agent with automatic yield token purchase and rebalancing
    """
    
    def __init__(self, agent_id: str, initial_hf: float, rebalancing_hf: float, target_hf: float = None, initial_balance: float = 100_000.0, yield_token_pool=None):
        super().__init__(agent_id, "high_tide_agent", initial_balance)
        
        # Handle backward compatibility: if target_hf is None, use rebalancing_hf as target (old 2-factor system)
        if target_hf is None:
            target_hf = rebalancing_hf
            print(f"⚠️  Warning: {agent_id} using 2-factor compatibility mode. Consider updating to tri-health factor system.")
        
        # Replace state with HighTideAgentState (tri-health factor system)
        self.state = HighTideAgentState(agent_id, initial_balance, initial_hf, rebalancing_hf, target_hf, yield_token_pool)
        
        # CRITICAL FIX: Add reference to engine for real swap recording
        self.engine = None  # Will be set by engine during initialization
        
        # Risk profile based on initial health factor
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
        Decide action based on High Tide strategy:
        1. Initially purchase yield tokens with borrowed MOET
        2. Monitor health factor and rebalance if needed
        3. Emergency actions if health factor critical
        """
        current_minute = protocol_state.get("current_step", 0)
        
        # Update health factor
        self._update_health_factor(asset_prices)
        
        # Debug health factor tracking for agent2
        
        # Check if we need to purchase yield tokens initially (only at minute 0)
        if (current_minute == 0 and 
            self.state.moet_debt > 0 and 
            len(self.state.yield_token_manager.yield_tokens) == 0):
            return self._initial_yield_token_purchase(current_minute)
        
        # Check if agent is trying to purchase yield tokens after minute 0
        if (current_minute > 0 and 
            self.state.moet_debt > 0 and 
            len(self.state.yield_token_manager.yield_tokens) == 0):
            return ("no_action", {})
        
        # PERFORMANCE OPTIMIZATION: Check leverage opportunity every 10 minutes when HF > initial HF
        # This allows agents to take advantage of opportunities much faster than weekly checks
        if current_minute % 10 == 0:  # Every 10 minutes
            if self._check_leverage_opportunity(asset_prices):
                return self._execute_leverage_increase(asset_prices, current_minute)
        
        # Check if rebalancing is needed (HF below initial threshold)
        if self._needs_rebalancing():
            action = self._execute_rebalancing(asset_prices, current_minute)
            # Update health factor after potential rebalancing decision
            self._update_health_factor(asset_prices)
            return action
        
        # Check for deleveraging opportunities (NEW)
        deleveraging_action = self._check_deleveraging(asset_prices, current_minute)
        if deleveraging_action[0] != "no_action":
            return deleveraging_action
        
        # Check if emergency action needed (HF at or below 1.0)
        # Try to sell ALL remaining yield tokens before liquidation
        if self.state.health_factor <= 1.0:
            if self.state.yield_token_manager.yield_tokens:
                # Sell ALL remaining yield tokens in emergency
                return self._execute_emergency_yield_sale(current_minute)
            else:
                # No yield tokens left, must liquidate
                return self._emergency_liquidation_action()
        
        # Default action - hold position
        return (AgentAction.HOLD, {})
    
    def _initial_yield_token_purchase(self, current_minute: int) -> tuple:
        """Purchase yield tokens with initially borrowed MOET"""
        moet_available = self.state.borrowed_balances.get(Asset.MOET, 0.0)
        
        if moet_available > 0:
            # Use all borrowed MOET to purchase yield tokens
            return (AgentAction.SWAP, {
                "action_type": "buy_yield_tokens",
                "moet_amount": moet_available,
                "current_minute": current_minute
            })
        
        return (AgentAction.HOLD, {})
    
    def _needs_rebalancing(self) -> bool:
        """Check if position needs rebalancing using tri-health factor system"""
        if not self.state.automatic_rebalancing:
            return False
            
        # TRI-HEALTH FACTOR: Rebalance when current HF falls below the REBALANCING HF (trigger threshold)
        needs_rebalancing = self.state.health_factor < self.state.rebalancing_health_factor
        
        # Debug logging for rebalancing decisions
        if needs_rebalancing:
            print(f"        🔄 {self.agent_id}: HF {self.state.health_factor:.3f} < Rebalancing HF {self.state.rebalancing_health_factor:.3f} - REBALANCING TRIGGERED")
            print(f"           Target: Rebalance until HF >= {self.state.target_health_factor:.3f}")
        
        return needs_rebalancing
    
    def _check_leverage_opportunity(self, asset_prices: Dict[Asset, float]) -> bool:
        """Check if agent can increase leverage when HF > initial HF"""
        if self.state.health_factor > self.state.initial_health_factor:
            return True
        return False
    
    def _execute_leverage_increase(self, asset_prices: Dict[Asset, float], current_minute: int) -> tuple:
        """Increase leverage by borrowing more MOET to restore initial HF"""
        collateral_value = self._calculate_effective_collateral_value(asset_prices)
        current_debt = self.state.moet_debt
        
        # Calculate target debt for initial HF
        target_debt = collateral_value / self.state.initial_health_factor
        additional_moet_needed = target_debt - current_debt
        
        if additional_moet_needed <= 0:
            return (AgentAction.HOLD, {})
        
        return (AgentAction.BORROW, {
            "amount": additional_moet_needed,
            "current_minute": current_minute,
            "leverage_increase": True
        })
    
    def _execute_rebalancing(self, asset_prices: Dict[Asset, float], current_minute: int) -> tuple:
        """Execute iterative rebalancing by selling yield tokens until HF target is reached"""
        if not self.state.yield_token_manager.yield_tokens:
            # No yield tokens to sell, position cannot be saved
            return (AgentAction.HOLD, {})
        
        # Calculate how much debt reduction is needed using the specified formula:
        # Debt Reduction Needed = Current Debt - (Effective Collateral Value / Target Health Factor)
        collateral_value = self._calculate_effective_collateral_value(asset_prices)
        current_debt = self.state.moet_debt
        target_debt = collateral_value / self.state.target_health_factor  # FIXED: Use target HF, not initial HF
        debt_reduction_needed = current_debt - target_debt
        
        if debt_reduction_needed <= 0:
            return (AgentAction.HOLD, {})
        
        # Start iterative rebalancing loop
        return self._execute_iterative_rebalancing(debt_reduction_needed, current_minute, asset_prices)
    
    def _execute_iterative_rebalancing(self, initial_moet_needed: float, current_minute: int, asset_prices: Dict[Asset, float]) -> tuple:
        """Execute iterative rebalancing with slippage monitoring"""
        moet_needed = initial_moet_needed
        total_moet_raised = 0.0
        total_yield_tokens_sold = 0.0
        rebalance_cycle = 0
        
        print(f"        🔄 {self.agent_id}: Starting iterative rebalancing - need ${moet_needed:,.2f} MOET")
        print(f"           Current HF: {self.state.health_factor:.3f}, Target HF: {self.state.target_health_factor:.3f}")
        
        # FIXED: Stop when above rebalancing threshold, not when reaching exact target
        # Agent should AIM for target HF but STOP when safe (above rebalancing HF)
        while (self.state.health_factor < self.state.rebalancing_health_factor and 
               self.state.yield_token_manager.yield_tokens and
               rebalance_cycle < 3):  # Max 3 cycles - should only need 1-2 in practice
            
            rebalance_cycle += 1
            print(f"        🔄 Rebalance Cycle {rebalance_cycle}: Need ${moet_needed:,.2f} MOET")
            
            # Calculate yield tokens to sell (1:1 assumption)
            yield_tokens_to_sell = moet_needed
            
            # CRITICAL FIX: Use engine's real swap execution instead of YieldTokenManager quotes
            if self.engine:
                # Let the engine execute the REAL swap with pool state mutations
                success, swap_data = self.engine._execute_yield_token_sale(
                    self, 
                    {"moet_needed": moet_needed, "swap_type": "rebalancing"}, 
                    current_minute
                )
                
                if success and swap_data:
                    moet_received = swap_data.get("moet_received", 0.0)
                    actual_yield_tokens_sold_value = swap_data.get("yt_swapped", 0.0)
                else:
                    moet_received = 0.0
                    actual_yield_tokens_sold_value = 0.0
            else:
                # WARNING: This fallback should not happen in production! Engine reference missing.
                print(f"⚠️  WARNING: Agent {self.agent_id} using YieldTokenManager fallback - engine reference missing!")
                moet_received, actual_yield_tokens_sold_value = self.state.yield_token_manager.sell_yield_tokens(yield_tokens_to_sell, current_minute)
            
            if moet_received <= 0:
                print(f"        ❌ No MOET received from yield token sale - liquidity exhausted")
                break
            
            # Check slippage threshold (>5% slippage)
            if moet_received < 0.95 * actual_yield_tokens_sold_value:
                slippage_percent = (1 - moet_received / actual_yield_tokens_sold_value) * 100
                print(f"        ⚠️  HIGH SLIPPAGE: {actual_yield_tokens_sold_value:,.2f} yield tokens → ${moet_received:,.2f} MOET ({slippage_percent:.1f}% slippage)")
            
            # Pay down debt using MOET from agent's balance
            available_moet = self.state.token_balances.get(Asset.MOET, 0.0)
            debt_repayment = min(available_moet, self.state.moet_debt)
            self.state.moet_debt -= debt_repayment
            self.state.token_balances[Asset.MOET] -= debt_repayment
            total_moet_raised += moet_received
            total_yield_tokens_sold += actual_yield_tokens_sold_value
            
            # Update health factor with actual prices
            self._update_health_factor(asset_prices)
            
            print(f"        📊 Cycle {rebalance_cycle}: Received ${moet_received:,.2f} MOET, repaid ${debt_repayment:,.2f} debt, new HF: {self.state.health_factor:.3f}")
            
            # Check if we're back above rebalancing threshold (safe zone)
            if self.state.health_factor >= self.state.rebalancing_health_factor:
                print(f"        ✅ Rebalancing successful: HF {self.state.health_factor:.3f} > threshold {self.state.rebalancing_health_factor:.3f}")
                break
            
            # Calculate remaining MOET needed for next cycle
            collateral_value = self._calculate_effective_collateral_value(asset_prices)
            target_debt = collateral_value / self.state.target_health_factor  # FIXED: Use target HF, not initial HF
            moet_needed = self.state.moet_debt - target_debt
            
            if moet_needed <= 0:
                break
        
        # Update the agent's total yield sold counter
        self.state.total_yield_sold += total_moet_raised
        
        # Record the rebalancing event
        if total_moet_raised > 0:
            slippage_cost = total_yield_tokens_sold - total_moet_raised
            
            # CRITICAL FIX: Record in engine for real swap data
            if self.engine:
                self.engine.record_agent_rebalancing_event(
                    self.agent_id, current_minute, total_moet_raised, 
                    total_moet_raised, slippage_cost, self.state.health_factor
                )
            
            # Also keep agent-level record for backward compatibility
            self.state.rebalancing_events.append({
                "minute": current_minute,
                "moet_raised": total_moet_raised,
                "debt_repaid": total_moet_raised,
                "yield_tokens_sold_value": total_yield_tokens_sold,
                "slippage_cost": slippage_cost,
                "slippage_percentage": ((total_yield_tokens_sold - total_moet_raised) / total_yield_tokens_sold * 100) if total_yield_tokens_sold > 0 else 0.0,
                "health_factor_before": self.state.health_factor,
                "rebalance_cycles": rebalance_cycle
            })
        
        # TRI-HEALTH FACTOR: Check if we need to continue rebalancing to reach TARGET HF
        if (self.state.health_factor < self.state.target_health_factor and 
            not self.state.yield_token_manager.yield_tokens):
            print(f"        ❌ All yield tokens sold but HF still below TARGET HF: {self.state.health_factor:.3f} < {self.state.target_health_factor:.3f}")
            print(f"           Rebalancing HF was: {self.state.rebalancing_health_factor:.3f} (trigger)")
            return (AgentAction.HOLD, {"emergency": True})
        
        return (AgentAction.HOLD, {})
    
    def _execute_emergency_yield_sale(self, current_minute: int) -> tuple:
        """Emergency sale of ALL remaining yield tokens"""
        # Calculate total value of all remaining yield tokens
        total_yield_value = self.state.yield_token_manager.calculate_total_value(current_minute)
        
        return (AgentAction.SWAP, {
            "action_type": "emergency_sell_all_yield",
            "amount_needed": total_yield_value,  # Sell everything
            "current_minute": current_minute
        })
    
    def _emergency_liquidation_action(self) -> tuple:
        """Handle emergency liquidation scenario"""
        self.state.emergency_liquidations += 1
        return (AgentAction.HOLD, {"emergency": True})
    
    def execute_high_tide_liquidation(self, current_minute: int, asset_prices: Dict[Asset, float], simulation_engine) -> Optional[Dict]:
        """Execute High Tide liquidation with Uniswap V3 BTC→MOET swap"""
        
        # Ensure we have BTC price from simulation engine
        btc_price = asset_prices.get(Asset.BTC)
        if btc_price is None:
            raise ValueError(f"BTC price not provided in asset_prices for liquidation at minute {current_minute}")
        
        # Calculate how much debt to repay to bring HF back to 1.1
        collateral_value = self._calculate_effective_collateral_value(asset_prices)
        target_debt = collateral_value / 1.1  # Target HF of 1.1
        current_debt = self.state.moet_debt
        debt_to_repay = current_debt - target_debt
        
        if debt_to_repay <= 0:
            return None
        
        # Step 1: Calculate BTC needed for debt repayment
        btc_to_repay_debt = debt_to_repay / btc_price
        available_btc = self.state.supplied_balances.get(Asset.BTC, 0.0)
        
        if btc_to_repay_debt > available_btc:
            btc_to_repay_debt = available_btc
        
        # Step 2: Swap BTC → MOET through Uniswap V3 pool
        swap_result = simulation_engine.slippage_calculator.calculate_swap_slippage(
            btc_to_repay_debt, "BTC"
        )
        actual_moet_received = swap_result["amount_out"]
        slippage_amount = swap_result["slippage_amount"]
        slippage_percent = swap_result.get("slippage_percent", swap_result.get("slippage_percentage", 0.0))
        
        # Step 3: Repay debt with actual MOET received
        actual_debt_repaid = min(actual_moet_received, self.state.moet_debt)
        self.state.moet_debt -= actual_debt_repaid
        
        # Step 4: Calculate and seize bonus (5% of actual debt repaid)
        liquidation_bonus = actual_debt_repaid * 0.05
        btc_bonus = liquidation_bonus / btc_price
        total_btc_seized = btc_to_repay_debt + btc_bonus
        
        # Step 5: Update BTC collateral
        self.state.supplied_balances[Asset.BTC] -= total_btc_seized
        
        # Update health factor
        self._update_health_factor(asset_prices)
        
        # Record liquidation event
        liquidation_event = {
            "minute": current_minute,
            "agent_id": self.agent_id,
            "health_factor_before": self.state.health_factor,
            "health_factor_after": self.state.health_factor,
            "debt_repaid_value": actual_debt_repaid,
            "btc_seized_for_debt": btc_to_repay_debt,
            "btc_seized_for_bonus": btc_bonus,
            "total_btc_seized": total_btc_seized,
            "btc_value_seized": total_btc_seized * btc_price,
            "liquidation_bonus_value": liquidation_bonus,
            "swap_slippage_amount": slippage_amount,
            "swap_slippage_percent": slippage_percent,
            "liquidation_type": "high_tide_emergency"
        }
        
        return liquidation_event
    
    def _update_health_factor(self, asset_prices: Dict[Asset, float]):
        """Update health factor for High Tide agent"""
        collateral_value = self._calculate_effective_collateral_value(asset_prices)
        debt_value = self.state.moet_debt * asset_prices.get(Asset.MOET, 1.0)
        
        if debt_value <= 0:
            # If no debt, health factor is infinite (perfect health)
            self.state.health_factor = float('inf')
        else:
            # Normal health factor calculation
            self.state.health_factor = collateral_value / debt_value
            
        # Ensure health factor is never negative or zero (unless debt is zero)
        if self.state.health_factor <= 0 and debt_value > 0:
            self.state.health_factor = 0.001  # Small positive value to indicate critical state
    
    def _calculate_effective_collateral_value(self, asset_prices: Dict[Asset, float]) -> float:
        """Calculate effective collateral value using BTC collateral factor"""
        btc_price = asset_prices.get(Asset.BTC)
        if btc_price is None:
            raise ValueError("BTC price not provided in asset_prices for collateral value calculation")
        
        btc_amount = self.state.supplied_balances.get(Asset.BTC, 0.0)
        # Use BTC collateral factor (should be 0.80)
        btc_collateral_factor = 0.80  # This matches what we set in protocol.py
        return btc_amount * btc_price * btc_collateral_factor
    
    def update_debt_interest(self, current_minute: int, btc_pool_borrow_rate: float):
        """Update debt with accrued interest based on BTC pool utilization"""
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
    
    def execute_yield_token_purchase(self, moet_amount: float, current_minute: int, use_direct_minting: bool = False) -> bool:
        """Execute yield token purchase"""
        if moet_amount <= 0:
            return False
        
        # Check if we have enough MOET balance
        moet_balance = self.state.token_balances.get(Asset.MOET, 0.0)
        if moet_balance < moet_amount:
            print(f"           ❌ Insufficient MOET balance: need ${moet_amount:,.2f}, have ${moet_balance:,.2f}")
            return False
            
        # Purchase yield tokens
        new_tokens = self.state.yield_token_manager.mint_yield_tokens(moet_amount, current_minute, use_direct_minting)
        
        if new_tokens:
            # CRITICAL FIX: Subtract the MOET spent on yield tokens
            self.state.token_balances[Asset.MOET] -= moet_amount
            
            # Set initial yield token value if this is the first purchase
            if self.state.initial_yield_token_value == 0.0:
                self.state.initial_yield_token_value = self.state.yield_token_manager.calculate_total_value(current_minute)
            
            # Calculate actual YT received for logging
            yt_received = sum(token.initial_value for token in new_tokens)
            print(f"           💰 YT Purchase: ${moet_amount:,.2f} MOET → ${yt_received:,.2f} YT")
            print(f"           💳 MOET Balance: ${moet_balance:,.2f} → ${self.state.token_balances[Asset.MOET]:,.2f}")
            
            return True
        
        return False
    
    def execute_yield_token_sale(self, moet_amount_needed: float, current_minute: int, purpose: str = "rebalancing") -> float:
        """Execute yield token sale using REAL pool execution
        
        Args:
            moet_amount_needed: Amount of MOET needed
            current_minute: Current simulation minute
            purpose: "rebalancing" or "deleveraging" for tracking
        """
        
        # CRITICAL FIX: Use YieldTokenManager to determine how much to sell, but YieldTokenPool for execution
        yield_tokens_to_sell = self.state.yield_token_manager._calculate_yield_tokens_needed(moet_amount_needed)
        
        if yield_tokens_to_sell <= 0:
            return 0.0
            
        # Check if we have enough yield tokens
        total_yield_value = sum(token.get_current_value(current_minute) for token in self.state.yield_token_manager.yield_tokens)
        
        if total_yield_value < yield_tokens_to_sell:
            yield_tokens_to_sell = total_yield_value
            
        if yield_tokens_to_sell <= 0:
            return 0.0
        
        # CRITICAL FIX: Use the pool's REAL execution instead of manager's quotes
        moet_raised = self.state.yield_token_manager.yield_token_pool.execute_yield_token_sale(yield_tokens_to_sell)
        
        if moet_raised > 0:
            # Remove the sold tokens from the manager's inventory
            self.state.yield_token_manager._remove_yield_tokens(yield_tokens_to_sell, current_minute)
            
            # CRITICAL FIX: Don't repay debt here! Let the rebalancing loop handle debt repayment
            # to avoid double repayment. Just add the MOET to the agent's balance.
            self.state.token_balances[Asset.MOET] += moet_raised
            self.state.total_yield_sold += moet_raised
            
            # Track slippage and purpose-specific metrics
            slippage_loss = max(0, yield_tokens_to_sell - moet_raised)  # YT should be ~1:1 with MOET
            
            if purpose == "rebalancing":
                self.state.total_yield_sold_for_rebalancing += yield_tokens_to_sell
                self.state.total_rebalancing_slippage += slippage_loss
                print(f"        💸 {self.agent_id}: Rebalancing YT sale: ${yield_tokens_to_sell:,.0f} YT → ${moet_raised:,.0f} MOET (slippage: ${slippage_loss:.2f})")
            elif purpose == "deleveraging":
                # Deleveraging tracking will be handled in execute_deleveraging method
                print(f"        💸 {self.agent_id}: Deleveraging YT sale: ${yield_tokens_to_sell:,.0f} YT → ${moet_raised:,.0f} MOET (slippage: ${slippage_loss:.2f})")
            else:
                print(f"        💸 {self.agent_id}: YT sale ({purpose}): ${yield_tokens_to_sell:,.0f} YT → ${moet_raised:,.0f} MOET (slippage: ${slippage_loss:.2f})")
        else:
            print(f"    ❌ {self.agent_id}: No MOET raised from yield token sale")
            
        return moet_raised
    
    def calculate_cost_of_rebalancing(self, final_btc_price: float, current_minute: int, 
                                     pool_size_usd: float = 500_000, 
                                     concentrated_range: float = 0.2) -> float:
        """
        Calculate Cost of Rebalancing = Final BTC Price - Net Position Value
        
        Where:
        - Current Value of Collateral = Users Collateral Deposited * Current Market Price
        - Value of Debt = MOET taken as DEBT
        - Value of Yield Tokens = Value of Yield Tokens
        - Net Position Value = Current Collateral + (Value of Yield Tokens - Value of Debt)
        
        This represents the opportunity cost of the rebalancing strategy vs. just holding BTC.
        """
        # Current Value of Collateral = Users Collateral Deposited * Current Market Price
        current_collateral = self.state.btc_amount * final_btc_price
        
        # Value of Yield Tokens
        current_yield_token_value = self.state.yield_token_manager.calculate_total_value(current_minute)
        
        # Value of Debt = MOET taken as DEBT
        current_debt = self.state.moet_debt
        
        # Net Position Value = Current Collateral + Token Balance BTC + (Value of Yield Tokens - Value of Debt)
        token_balance_btc_value = self.state.token_balances.get(Asset.BTC, 0.0) * final_btc_price
        net_position_value = current_collateral + token_balance_btc_value + (current_yield_token_value - current_debt)
        
        # Cost of Rebalancing = Final BTC Price - Net Position Value
        cost_of_rebalancing = final_btc_price - net_position_value
        
        return cost_of_rebalancing
    
    def calculate_total_transaction_costs(self) -> float:
        """
        Calculate total transaction costs from all rebalancing events
        
        Returns:
            Total slippage costs + trading fees from all rebalancing events
        """
        total_costs = 0.0
        
        for event in self.state.rebalancing_events:
            total_costs += event.get("slippage_cost", 0.0)
            
        return total_costs
    
    def get_detailed_portfolio_summary(self, asset_prices: Dict[Asset, float], current_minute: int,
                                      pool_size_usd: float = 500_000, 
                                      concentrated_range: float = 0.2) -> dict:
        """Get comprehensive portfolio summary for High Tide agent"""
        base_summary = super().get_portfolio_summary(asset_prices)
        
        # Add High Tide specific metrics
        yield_summary = self.state.yield_token_manager.get_portfolio_summary(current_minute)
        btc_price = asset_prices.get(Asset.BTC)
        if btc_price is None:
            raise ValueError("BTC price not provided in asset_prices for portfolio summary")
            
        pnl_from_rebalancing = self.calculate_cost_of_rebalancing(
            btc_price, 
            current_minute,
            pool_size_usd,
            concentrated_range
        )
        
        total_transaction_costs = self.calculate_total_transaction_costs()
        
        # Calculate current yield token value
        current_yield_token_value = self.state.yield_token_manager.calculate_total_value(current_minute)
        
        high_tide_metrics = {
            "risk_profile": self.risk_profile,
            "color": self.color,
            "initial_health_factor": self.state.initial_health_factor,      # Starting position health
            "rebalancing_health_factor": self.state.rebalancing_health_factor,  # Trigger threshold
            "target_health_factor": self.state.target_health_factor,        # Post-rebalancing target
            "btc_amount": self.state.btc_amount,
            "initial_moet_debt": self.state.initial_moet_debt,
            "current_moet_debt": self.state.moet_debt,
            "total_interest_accrued": self.state.total_interest_accrued,
            "yield_token_portfolio": yield_summary,
            "total_yield_sold": self.state.total_yield_sold,
            "total_yield_sold_for_rebalancing": self.state.total_yield_sold_for_rebalancing,
            "total_rebalancing_slippage": self.state.total_rebalancing_slippage,
            "rebalancing_events_count": len(self.state.rebalancing_events),
            "emergency_liquidations": self.state.emergency_liquidations,
            # Deleveraging tracking (NEW)
            "deleveraging_events": self.state.deleveraging_events.copy(),
            "deleveraging_events_count": len(self.state.deleveraging_events),
            "total_deleveraging_sales": self.state.total_deleveraging_sales,
            "total_deleveraging_slippage": self.state.total_deleveraging_slippage,
            "cost_of_rebalancing": pnl_from_rebalancing,  # PnL from rebalancing strategy
            "total_slippage_costs": total_transaction_costs,  # Transaction costs (slippage + fees)
            "net_position_value": (self.state.btc_amount * btc_price) + (current_yield_token_value - self.state.moet_debt) + (self.state.token_balances.get(Asset.BTC, 0.0) * btc_price),
            "automatic_rebalancing": self.state.automatic_rebalancing
        }
        
        # Merge with base summary
        base_summary.update(high_tide_metrics)
        return base_summary
        
    def get_rebalancing_history(self) -> list:
        """Get history of rebalancing events"""
        return self.state.rebalancing_events.copy()
    
    def _check_deleveraging(self, asset_prices: Dict[Asset, float], current_minute: int) -> tuple:
        """
        Check if agent should delever based on:
        1. Health factor > initial HF + 5% (crazy price moves)
        2. Weekly 1% YT position sales
        """
        if not self.state.yield_token_manager.yield_tokens:
            return ("no_action", {})
        
        # Check 1: Health factor deleveraging (5% threshold)
        hf_threshold = self.state.initial_health_factor * 1.05  # 5% above initial HF
        if self.state.health_factor > hf_threshold:
            return self._execute_hf_deleveraging(asset_prices, current_minute)
        
        # Check 2: Weekly deleveraging (1% of YT position)
        minutes_per_week = 7 * 24 * 60  # 10,080 minutes
        if current_minute - self.state.last_weekly_delever_minute >= minutes_per_week:
            return self._execute_weekly_deleveraging(asset_prices, current_minute)
        
        return ("no_action", {})
    
    def _execute_hf_deleveraging(self, asset_prices: Dict[Asset, float], current_minute: int) -> tuple:
        """Execute deleveraging when HF is >5% above initial HF"""
        # Calculate total YT value
        total_yt_value = sum(token.get_current_value(current_minute) 
                           for token in self.state.yield_token_manager.yield_tokens)
        
        if total_yt_value <= 0:
            return ("no_action", {})
        
        # Sell enough YT to bring HF back to initial HF + 2% (safety buffer)
        target_hf = self.state.initial_health_factor * 1.02
        collateral_value = self._calculate_effective_collateral_value(asset_prices)
        target_debt = collateral_value / target_hf
        debt_reduction_needed = self.state.moet_debt - target_debt
        
        if debt_reduction_needed <= 0:
            return ("no_action", {})
        
        # Limit to available YT value
        yt_to_sell = min(debt_reduction_needed, total_yt_value)
        
        print(f"🔻 {self.agent_id}: HF deleveraging - HF {self.state.health_factor:.3f} > threshold {self.state.initial_health_factor * 1.05:.3f}")
        print(f"   Selling ${yt_to_sell:,.0f} YT to reduce debt by ${debt_reduction_needed:,.0f}")
        
        return ("delever_hf", {
            "yt_amount": yt_to_sell,
            "reason": "health_factor_threshold",
            "current_minute": current_minute,
            "target_hf": target_hf
        })
    
    def _execute_weekly_deleveraging(self, asset_prices: Dict[Asset, float], current_minute: int) -> tuple:
        """Execute weekly 1% YT position deleveraging"""
        # Calculate 1% of current YT position
        total_yt_value = sum(token.get_current_value(current_minute) 
                           for token in self.state.yield_token_manager.yield_tokens)
        
        if total_yt_value <= 0:
            return ("no_action", {})
        
        yt_to_sell = total_yt_value * 0.01  # 1% of position
        
        # Update last weekly delever time
        self.state.last_weekly_delever_minute = current_minute
        
        print(f"📅 {self.agent_id}: Weekly deleveraging - selling 1% of YT position (${yt_to_sell:,.0f})")
        
        return ("delever_weekly", {
            "yt_amount": yt_to_sell,
            "reason": "weekly_deleveraging",
            "current_minute": current_minute,
            "percentage": 0.01
        })
    
    def execute_deleveraging(self, params: dict, current_minute: int) -> bool:
        """
        Execute deleveraging by selling YT and following the swap chain:
        YT → MOET → USDC/USDF → BTC
        """
        yt_amount = params.get("yt_amount", 0)
        reason = params.get("reason", "unknown")
        
        if yt_amount <= 0:
            return False
        
        print(f"🔻 {self.agent_id}: Executing deleveraging - ${yt_amount:,.0f} YT ({reason})")
        
        # Step 1: Sell YT for MOET (using existing mechanism with deleveraging tracking)
        moet_received = self.execute_yield_token_sale(yt_amount, current_minute, purpose="deleveraging")
        
        if moet_received <= 0:
            print(f"   ❌ Failed to sell YT for MOET")
            return False
        
        print(f"   ✅ Step 1: Sold ${yt_amount:,.0f} YT → ${moet_received:,.0f} MOET")
        
        # Step 2-4: Execute the full swap chain: MOET → USDC/USDF → BTC
        btc_received = self._execute_deleveraging_swap_chain(moet_received, current_minute)
        
        if btc_received <= 0:
            print(f"   ❌ Failed to complete swap chain")
            return False
        
        print(f"   ✅ Swap chain complete: ${moet_received:,.0f} MOET → ${btc_received:.6f} BTC")
        
        # Calculate total deleveraging slippage (YT→MOET + MOET→USDC/USDF→BTC)
        yt_to_moet_slippage = max(0, yt_amount - moet_received)  # Step 1 slippage
        swap_chain_slippage = getattr(self, '_last_swap_chain_slippage', 0.0)  # Steps 2-3 slippage
        total_deleveraging_slippage = yt_to_moet_slippage + swap_chain_slippage
        
        # Record deleveraging event with detailed tracking
        deleveraging_event = {
            "minute": current_minute,
            "yt_sold": yt_amount,
            "moet_received": moet_received,
            "btc_received": btc_received,
            "reason": reason,
            "health_factor_before": self.state.health_factor,
            "swap_chain_details": getattr(self, '_last_swap_chain_details', {}),
            "yt_to_moet_slippage": yt_to_moet_slippage,
            "swap_chain_slippage": swap_chain_slippage,
            "total_slippage_cost": total_deleveraging_slippage
        }
        self.state.deleveraging_events.append(deleveraging_event)
        
        # Update cumulative tracking
        self.state.total_deleveraging_sales += yt_amount
        self.state.total_deleveraging_slippage += total_deleveraging_slippage
        
        print(f"   📊 Deleveraging complete: Total deleveraging sales: ${self.state.total_deleveraging_sales:,.0f}")
        
        return True
    
    def _execute_deleveraging_swap_chain(self, moet_amount: float, current_minute: int) -> float:
        """
        Execute simplified deleveraging: MOET → BTC at market price with fees
        Much simpler and more realistic than complex pool routing
        """
        import random
        
        if not self.engine:
            print(f"   ⚠️  No engine reference - cannot execute swap chain")
            return 0.0
        
        # Initialize detailed tracking
        swap_details = {
            "path_chosen": "",
            "step2_slippage": 0.0,
            "step3_slippage": 0.0,
            "total_slippage": 0.0,
            "step2_amount_in": moet_amount,
            "step2_amount_out": 0.0,
            "step3_amount_in": 0.0,
            "step3_amount_out": 0.0
        }
        
        # Step 2: MOET → USDC or USDF (randomly choose)
        use_usdc = random.choice([True, False])
        stablecoin = "USDC" if use_usdc else "USDF"
        swap_details["path_chosen"] = stablecoin
        
        print(f"   🔄 Step 2: Swapping ${moet_amount:,.0f} MOET → {stablecoin}")
        
        # Execute MOET → stablecoin swap through engine
        if use_usdc:
            stablecoin_received = self._swap_moet_to_usdc(moet_amount)
        else:
            stablecoin_received = self._swap_moet_to_usdf(moet_amount)
        
        if stablecoin_received <= 0:
            print(f"   ❌ Failed MOET → {stablecoin} swap")
            return 0.0
        
        # Track step 2 details with proper Uniswap V3 slippage calculation
        swap_details["step2_amount_out"] = stablecoin_received
        
        # Get the actual slippage from the swap result (if available)
        if use_usdc and hasattr(self.engine, 'moet_usdc_calculator'):
            # Use the same slippage tracking as MOET:YT pool
            try:
                swap_result = self.engine.moet_usdc_calculator.calculate_swap_slippage(moet_amount, "MOET")
                step2_slippage = swap_result.get("slippage_amount", 0.0)
                swap_details["step2_slippage_pct"] = swap_result.get("slippage_percentage", 0.0)
                swap_details["step2_price_impact"] = swap_result.get("price_impact_percentage", 0.0)
                pool_price = self.engine.moet_usdc_pool.get_price()
                print(f"   📊 MOET:USDC pool price: ${pool_price:.4f}, slippage: {swap_details['step2_slippage_pct']:.2f}%")
            except Exception as e:
                step2_slippage = max(0, moet_amount - stablecoin_received)  # Fallback
                print(f"   ⚠️  Using fallback slippage calculation: {e}")
        elif not use_usdc and hasattr(self.engine, 'moet_usdf_calculator'):
            try:
                swap_result = self.engine.moet_usdf_calculator.calculate_swap_slippage(moet_amount, "MOET")
                step2_slippage = swap_result.get("slippage_amount", 0.0)
                swap_details["step2_slippage_pct"] = swap_result.get("slippage_percentage", 0.0)
                swap_details["step2_price_impact"] = swap_result.get("price_impact_percentage", 0.0)
                pool_price = self.engine.moet_usdf_pool.get_price()
                print(f"   📊 MOET:USDF pool price: ${pool_price:.4f}, slippage: {swap_details['step2_slippage_pct']:.2f}%")
            except Exception as e:
                step2_slippage = max(0, moet_amount - stablecoin_received)  # Fallback
                print(f"   ⚠️  Using fallback slippage calculation: {e}")
        else:
            step2_slippage = max(0, moet_amount - stablecoin_received)  # Fallback
        
        swap_details["step2_slippage"] = step2_slippage
        
        print(f"   ✅ Step 2: ${moet_amount:,.0f} MOET → ${stablecoin_received:,.0f} {stablecoin} (slippage: ${step2_slippage:.2f})")
        
        # Step 3: USDC/USDF → BTC (Simplified market execution)
        swap_details["step3_amount_in"] = stablecoin_received
        print(f"   🔄 Step 3: Converting ${stablecoin_received:,.0f} {stablecoin} → BTC at market price")
        
        # Use simplified market execution instead of complex pool routing
        btc_received = self._execute_stablecoin_to_btc_market_order(stablecoin_received, stablecoin)
        
        if btc_received <= 0:
            print(f"   ❌ Failed {stablecoin} → BTC swap")
            return 0.0
        
        # Track step 3 details with proper Uniswap V3 slippage calculation
        btc_price = self.engine.state.current_prices.get(Asset.BTC, 50000)
        btc_value_usd = btc_received * btc_price
        swap_details["step3_amount_out"] = btc_value_usd
        
        # Calculate realistic slippage and fees for market execution
        trading_fee_rate = 0.001  # 0.1% trading fee (realistic for CEX)
        slippage_rate = 0.0005    # 0.05% slippage (realistic for large orders)
        
        trading_fees = stablecoin_received * trading_fee_rate
        slippage_cost = stablecoin_received * slippage_rate
        step3_slippage = trading_fees + slippage_cost
        
        swap_details["step3_slippage_pct"] = (step3_slippage / stablecoin_received) * 100
        swap_details["step3_price_impact"] = slippage_rate * 100
        swap_details["step3_trading_fees"] = trading_fees
        
        print(f"   📊 Market execution: Trading fees: ${trading_fees:.2f}, Slippage: ${slippage_cost:.2f}, Total: ${step3_slippage:.2f}")
        
        swap_details["step3_slippage"] = step3_slippage
        swap_details["total_slippage"] = step2_slippage + step3_slippage
        
        # Store total slippage and details for deleveraging tracking
        self._last_swap_chain_slippage = step2_slippage + step3_slippage
        self._last_swap_chain_details = swap_details
        
        print(f"   ✅ Step 3: ${stablecoin_received:,.0f} {stablecoin} → {btc_received:.6f} BTC (${btc_value_usd:,.0f}, slippage: ${step3_slippage:.2f})")
        
        # CRITICAL FIX: Deposit the BTC back into Tidal as collateral
        # Instead of just holding it in token_balances where it doesn't count toward performance
        print(f"   🏦 Step 4: Depositing {btc_received:.6f} BTC back into Tidal as collateral")
        
        try:
            # Deposit BTC back into the protocol as collateral using the correct method
            success = self.engine.protocol.supply(self.agent_id, Asset.BTC, btc_received)
            
            if success:
                # Update agent's BTC collateral amount
                self.state.btc_amount += btc_received
                
                # Remove BTC from token balance since it's now collateral
                if Asset.BTC in self.state.token_balances:
                    self.state.token_balances[Asset.BTC] = max(0, self.state.token_balances[Asset.BTC] - btc_received)
                
                print(f"   ✅ Step 4: Successfully deposited {btc_received:.6f} BTC as collateral")
                print(f"   📊 New total BTC collateral: {self.state.btc_amount:.6f} BTC")
            else:
                print(f"   ❌ Failed to deposit BTC as collateral - supply() returned False")
            
        except Exception as e:
            print(f"   ❌ Failed to deposit BTC as collateral: {e}")
            # If deposit fails, at least keep it in token balance
            pass
        
        # Store detailed tracking for deleveraging event
        self._last_swap_chain_details = swap_details
        self._last_swap_chain_slippage = swap_details["total_slippage"]
        
        print(f"   📊 Swap chain summary: Total slippage ${swap_details['total_slippage']:.2f} via {swap_details['path_chosen']} path")
        print(f"   🔍 Pool verification: YT→MOET→{swap_details['path_chosen']}→BTC chain completed successfully")
        
        return btc_received
    
    def _execute_stablecoin_to_btc_market_order(self, stablecoin_amount: float, stablecoin: str) -> float:
        """
        Execute stablecoin to BTC conversion at market price with realistic fees
        Much simpler and more predictable than complex pool routing
        """
        if not self.engine or stablecoin_amount <= 0:
            return 0.0
            
        try:
            # Get current BTC price
            btc_price = self.engine.state.current_prices.get(Asset.BTC, 50000)
            
            # Apply realistic trading costs
            trading_fee_rate = 0.001  # 0.1% trading fee
            slippage_rate = 0.0005    # 0.05% slippage
            
            # Calculate net amount after fees
            trading_fees = stablecoin_amount * trading_fee_rate
            slippage_cost = stablecoin_amount * slippage_rate
            net_stablecoin = stablecoin_amount - trading_fees - slippage_cost
            
            # Convert to BTC at market price
            btc_received = net_stablecoin / btc_price
            
            # Store slippage for deleveraging tracking
            total_cost = trading_fees + slippage_cost
            self._last_swap_chain_slippage = total_cost
            
            print(f"     💱 Market execution: ${stablecoin_amount:,.0f} {stablecoin} → {btc_received:.6f} BTC")
            print(f"     💸 Costs: Trading fee ${trading_fees:.2f}, Slippage ${slippage_cost:.2f}, Total: ${total_cost:.2f}")
            
            return btc_received
            
        except Exception as e:
            print(f"     ❌ Market execution failed: {e}")
            return 0.0
    
    def _swap_moet_to_usdc(self, moet_amount: float) -> float:
        """Swap MOET to USDC using engine's MOET:USDC pool"""
        if not hasattr(self.engine, 'moet_usdc_calculator'):
            print(f"   ⚠️  Engine missing MOET:USDC pool")
            return 0.0
        
        try:
            # Use the engine's MOET:USDC calculator
            swap_result = self.engine.moet_usdc_calculator.calculate_swap_slippage(
                moet_amount, "MOET"
            )
            usdc_received = swap_result.get("amount_out", 0.0)
            
            # Update agent balances
            self.state.token_balances[Asset.MOET] -= moet_amount
            self.state.token_balances[Asset.USDC] += usdc_received
            
            # Update pool state
            self.engine.moet_usdc_calculator.update_pool_state(swap_result)
            
            return usdc_received
            
        except Exception as e:
            print(f"   ❌ MOET→USDC swap error: {e}")
            return 0.0
    
    def _swap_moet_to_usdf(self, moet_amount: float) -> float:
        """Swap MOET to USDF using engine's MOET:USDF pool"""
        if not hasattr(self.engine, 'moet_usdf_calculator'):
            print(f"   ⚠️  Engine missing MOET:USDF pool")
            return 0.0
        
        try:
            # Use the engine's MOET:USDF calculator
            swap_result = self.engine.moet_usdf_calculator.calculate_swap_slippage(
                moet_amount, "MOET"
            )
            usdf_received = swap_result.get("amount_out", 0.0)
            
            # Update agent balances
            self.state.token_balances[Asset.MOET] -= moet_amount
            # Note: USDF not in Asset enum, so we'll track it separately or use USDC as proxy
            self.state.token_balances[Asset.USDC] += usdf_received  # Using USDC as proxy for now
            
            # Update pool state
            self.engine.moet_usdf_calculator.update_pool_state(swap_result)
            
            return usdf_received
            
        except Exception as e:
            print(f"   ❌ MOET→USDF swap error: {e}")
            return 0.0
    
    def _swap_usdc_to_btc(self, usdc_amount: float) -> float:
        """Swap USDC to BTC using engine's USDC:BTC pool"""
        if not hasattr(self.engine, 'usdc_btc_calculator'):
            print(f"   ⚠️  Engine missing USDC:BTC pool")
            return 0.0
        
        try:
            # Use the engine's USDC:BTC calculator
            swap_result = self.engine.usdc_btc_calculator.calculate_swap_slippage(
                usdc_amount, "USDC"
            )
            btc_received = swap_result.get("amount_out", 0.0)
            
            # Update agent balances
            self.state.token_balances[Asset.USDC] -= usdc_amount
            self.state.token_balances[Asset.BTC] += btc_received
            
            # Update pool state
            self.engine.usdc_btc_calculator.update_pool_state(swap_result)
            
            return btc_received
            
        except Exception as e:
            print(f"   ❌ USDC→BTC swap error: {e}")
            return 0.0
    
    def _swap_usdf_to_btc(self, usdf_amount: float) -> float:
        """Swap USDF to BTC using engine's USDF:BTC pool"""
        if not hasattr(self.engine, 'usdf_btc_calculator'):
            print(f"   ⚠️  Engine missing USDF:BTC pool")
            return 0.0
        
        try:
            # Use the engine's USDF:BTC calculator
            swap_result = self.engine.usdf_btc_calculator.calculate_swap_slippage(
                usdf_amount, "USDF"
            )
            btc_received = swap_result.get("amount_out", 0.0)
            
            # Update agent balances (using USDC as proxy for USDF)
            self.state.token_balances[Asset.USDC] -= usdf_amount
            self.state.token_balances[Asset.BTC] += btc_received
            
            # Update pool state
            self.engine.usdf_btc_calculator.update_pool_state(swap_result)
            
            return btc_received
            
        except Exception as e:
            print(f"   ❌ USDF→BTC swap error: {e}")
            return 0.0


def create_high_tide_agents(num_agents: int, monte_carlo_variation: bool = True, yield_token_pool = None) -> list:
    """
    Create High Tide agents with varied risk profiles using tri-health factor system
    
    Tri-Health Factor System:
    - Initial HF: Starting position health
    - Rebalancing HF: Trigger threshold for rebalancing
    - Target HF: Post-rebalancing safety buffer
    
    Risk Profile Distribution (backward compatibility with 2-factor system):
    - Conservative (30%): Initial HF = 2.1-2.4, Rebalancing HF = Initial - 0.05-0.15, Target HF = Rebalancing HF + 0.01-0.05
    - Moderate (40%): Initial HF = 1.5-1.8, Rebalancing HF = Initial - 0.15-0.25, Target HF = Rebalancing HF + 0.01-0.05
    - Aggressive (30%): Initial HF = 1.3-1.5, Rebalancing HF = Initial - 0.15-0.4, Target HF = Rebalancing HF + 0.01-0.05
    
    Minimum Target HF = 1.1 for all agents
    """
    if monte_carlo_variation:
        # Randomize agent count between 10-50
        num_agents = random.randint(10, 50)
    
    agents = []
    
    # Calculate distribution
    conservative_count = int(num_agents * 0.3)
    moderate_count = int(num_agents * 0.4)
    aggressive_count = num_agents - conservative_count - moderate_count
    
    agent_id = 0
    
    # Create conservative agents
    for i in range(conservative_count):
        initial_hf = random.uniform(2.1, 2.4)
        # Conservative: Small rebalancing buffer (0.05-0.15 below initial)
        rebalancing_hf = initial_hf - random.uniform(0.05, 0.15)
        rebalancing_hf = max(rebalancing_hf, 1.1)  # Minimum rebalancing HF is 1.1
        # Target HF: Small safety buffer above rebalancing HF
        target_hf = rebalancing_hf + random.uniform(0.01, 0.05)
        target_hf = max(target_hf, 1.1)  # Minimum target HF is 1.1
        
        agent = HighTideAgent(
            f"high_tide_conservative_{agent_id}",
            initial_hf,
            rebalancing_hf,
            target_hf,
            yield_token_pool=yield_token_pool
        )
        agents.append(agent)
        agent_id += 1
    
    # Create moderate agents
    for i in range(moderate_count):
        initial_hf = random.uniform(1.5, 1.8)
        # Moderate: Medium rebalancing buffer (0.15-0.25 below initial)
        rebalancing_hf = initial_hf - random.uniform(0.15, 0.25)
        rebalancing_hf = max(rebalancing_hf, 1.1)  # Minimum rebalancing HF is 1.1
        # Target HF: Small safety buffer above rebalancing HF
        target_hf = rebalancing_hf + random.uniform(0.01, 0.05)
        target_hf = max(target_hf, 1.1)  # Minimum target HF is 1.1
        
        agent = HighTideAgent(
            f"high_tide_moderate_{agent_id}",
            initial_hf,
            rebalancing_hf,
            target_hf,
            yield_token_pool=yield_token_pool
        )
        agents.append(agent)
        agent_id += 1
    
    # Create aggressive agents
    for i in range(aggressive_count):
        initial_hf = random.uniform(1.3, 1.5)
        # Aggressive: Larger rebalancing buffer (0.15-0.4 below initial)
        rebalancing_hf = initial_hf - random.uniform(0.15, 0.4)
        rebalancing_hf = max(rebalancing_hf, 1.1)  # Minimum rebalancing HF is 1.1
        # Target HF: Small safety buffer above rebalancing HF
        target_hf = rebalancing_hf + random.uniform(0.01, 0.05)
        target_hf = max(target_hf, 1.1)  # Minimum target HF is 1.1
        
        agent = HighTideAgent(
            f"high_tide_aggressive_{agent_id}",
            initial_hf,
            rebalancing_hf,
            target_hf,
            yield_token_pool=yield_token_pool
        )
        agents.append(agent)
        agent_id += 1
    
    return agents
