#!/usr/bin/env python3
"""
Pool Rebalancer Agents for MOET:YT Liquidity Pool Management

Two specialized agents that maintain the MOET:YT pool's price accuracy by arbitraging 
it against the "true" yield token price:

1. ALM Rebalancer - Time-based rebalancing at fixed intervals
2. Algo Rebalancer - Threshold-based rebalancing when price deviates by 50bps+

Both agents have $500,000 total liquidity and immediately sell yield tokens 
externally at true price to replenish their MOET reserves.
"""

from typing import Dict, Optional, Tuple, List
import time
from dataclasses import dataclass
from .base_agent import BaseAgent, AgentAction, AgentState
from ..core.protocol import Asset
from ..core.yield_tokens import YieldTokenPool, YieldToken, calculate_true_yield_token_price


class PoolRebalancerState(AgentState):
    """Enhanced state for pool rebalancer agents"""
    
    def __init__(self, agent_id: str, initial_liquidity: float = 500_000.0, agent_type: str = "pool_rebalancer"):
        # Initialize with MOET and YT balances for rebalancing
        super().__init__(agent_id, initial_liquidity, agent_type)
        
        # Pool rebalancer specific balances
        # Start with all liquidity in MOET for purchasing underpriced YT
        self.moet_balance = initial_liquidity  # $500k MOET
        self.yield_token_balance = 0.0  # No initial YT holdings
        
        # Track rebalancing activity
        self.total_rebalances = 0
        self.total_volume_rebalanced = 0.0
        self.total_profit_from_arbing = 0.0
        self.last_rebalance_minute = 0
        
        # Agent-specific parameters
        self.enabled = True
        self.min_rebalance_amount = 0.0  # No minimum - rebalance any amount needed
        self.max_single_rebalance = 500_000.0  # Maximum $500k single rebalance (full liquidity)
        
        # Arbitrage delay functionality
        self.arb_delay_enabled = False  # Default FALSE for backward compatibility
        self.arb_delay_time_units = 60  # Default 1-hour delay (will be auto-converted based on simulation time scale)
        self.pending_yt_sales = []  # List of pending YT sales: [(time_available, yt_amount, true_price)]
        self._simulation_time_scale = None  # Will be auto-detected: "minutes" or "hours"
    
    def _detect_simulation_time_scale(self, current_time: int, simulation_duration: int) -> str:
        """
        Auto-detect whether simulation is running in minutes or hours based on typical patterns
        
        Args:
            current_time: Current simulation time step
            simulation_duration: Total expected simulation duration
            
        Returns:
            "minutes" or "hours" based on detected time scale
        """
        if self._simulation_time_scale is not None:
            return self._simulation_time_scale
        
        # Heuristic: if simulation_duration > 1000, likely using minutes
        # If simulation_duration < 100, likely using hours
        if simulation_duration > 500:
            self._simulation_time_scale = "minutes"
        elif simulation_duration < 100:
            self._simulation_time_scale = "hours"  
        else:
            # Ambiguous case - use current_time as additional hint
            # If we're seeing time steps > 100, probably minutes
            if current_time > 100:
                self._simulation_time_scale = "minutes"
            else:
                self._simulation_time_scale = "hours"
        
        print(f"🔍 Auto-detected simulation time scale: {self._simulation_time_scale}")
        return self._simulation_time_scale
    
    def _get_arb_delay_in_simulation_units(self, current_time: int, simulation_duration: int = 10000) -> int:
        """
        Convert the 1-hour arbitrage delay to appropriate simulation time units
        
        Args:
            current_time: Current simulation time step
            simulation_duration: Total expected simulation duration (for scale detection)
            
        Returns:
            Delay in simulation time units (1 hour converted appropriately)
        """
        time_scale = self._detect_simulation_time_scale(current_time, simulation_duration)
        
        if time_scale == "minutes":
            # 1 hour = 60 minutes
            return 60
        else:  # time_scale == "hours"
            # 1 hour = 1 hour
            return 1
    
    def _process_pending_yt_sales(self, current_time: int, simulation_duration: int = 10000):
        """
        Process any pending YT sales that are now available due to arbitrage delay
        
        Args:
            current_time: Current simulation time step
            simulation_duration: Total expected simulation duration (for scale detection)
        """
        if not self.arb_delay_enabled or not self.pending_yt_sales:
            return
        
        # Check for sales that are now available
        available_sales = []
        remaining_sales = []
        
        for sale_time, yt_amount, true_price in self.pending_yt_sales:
            if current_time >= sale_time:
                available_sales.append((yt_amount, true_price))
            else:
                remaining_sales.append((sale_time, yt_amount, true_price))
        
        # Process available sales - convert YT back to MOET
        total_moet_recovered = 0.0
        for yt_amount, true_price in available_sales:
            moet_recovered = yt_amount * true_price
            self.moet_balance += moet_recovered
            total_moet_recovered += moet_recovered
            
            print(f"🔄 Arbitrage delay completed: Converted {yt_amount:.2f} YT → ${moet_recovered:.2f} MOET")
        
        # Update pending sales list
        self.pending_yt_sales = remaining_sales
        
        if total_moet_recovered > 0:
            print(f"💰 Total MOET recovered from delayed arbitrage: ${total_moet_recovered:.2f}")
            print(f"💰 Updated MOET balance: ${self.moet_balance:,.2f}")
    
    def _add_pending_yt_sale(self, yt_amount: float, true_price: float, current_time: int, simulation_duration: int = 10000):
        """
        Add a YT sale to the pending queue with arbitrage delay
        
        Args:
            yt_amount: Amount of YT to be sold
            true_price: True price of YT for conversion
            current_time: Current simulation time step
            simulation_duration: Total expected simulation duration (for scale detection)
        """
        if not self.arb_delay_enabled:
            # No delay - immediate conversion
            moet_recovered = yt_amount * true_price
            self.moet_balance += moet_recovered
            print(f"🔄 Immediate arbitrage: Converted {yt_amount:.2f} YT → ${moet_recovered:.2f} MOET")
            return
        
        # Calculate delay in appropriate time units
        delay = self._get_arb_delay_in_simulation_units(current_time, simulation_duration)
        available_time = current_time + delay
        
        self.pending_yt_sales.append((available_time, yt_amount, true_price))
        
        time_scale = self._detect_simulation_time_scale(current_time, simulation_duration)
        delay_description = f"{delay} {'minutes' if time_scale == 'minutes' else 'hours'}"
        
        print(f"⏳ Arbitrage delayed: {yt_amount:.2f} YT will be available in {delay_description} (at time {available_time})")
        print(f"⏳ Current pending YT sales: {len(self.pending_yt_sales)}")
    
    def set_arb_delay_enabled(self, enabled: bool):
        """Enable or disable arbitrage delay"""
        self.arb_delay_enabled = enabled
        if not enabled:
            # If disabling, immediately process all pending sales
            for sale_time, yt_amount, true_price in self.pending_yt_sales:
                moet_recovered = yt_amount * true_price
                self.moet_balance += moet_recovered
                print(f"🔄 Processing delayed sale immediately: {yt_amount:.2f} YT → ${moet_recovered:.2f} MOET")
            self.pending_yt_sales.clear()


class ALMRebalancer(BaseAgent):
    """
    Asset Liability Management (ALM) Rebalancer
    
    Rebalances the MOET:YT pool at fixed time intervals to maintain price accuracy
    with the true yield token price. Operates on a time cadence regardless of 
    price deviation magnitude.
    """
    
    def __init__(self, agent_id: str = "alm_rebalancer", rebalance_interval_minutes: int = 720):  # 12 hours default
        super().__init__(agent_id, "alm_rebalancer", 500_000.0)
        self.state = PoolRebalancerState(agent_id, 500_000.0, "alm_rebalancer")
        
        # ALM-specific parameters
        self.rebalance_interval_minutes = rebalance_interval_minutes
        self.next_rebalance_minute = rebalance_interval_minutes
        
        # Pool reference (set by engine)
        self.yield_token_pool: Optional[YieldTokenPool] = None
        
    def set_yield_token_pool(self, pool: YieldTokenPool):
        """Set the yield token pool reference"""
        self.yield_token_pool = pool
        
    def decide_action(self, protocol_state: dict, asset_prices: Dict[Asset, float]) -> tuple:
        """
        Decide whether to rebalance based on time interval
        
        Returns:
            Tuple of (action_type, params)
        """
        if not self.state.enabled or not self.yield_token_pool:
            return (AgentAction.HOLD, {})
            
        current_minute = protocol_state.get("current_minute", 0)
        
        # Check if it's time for scheduled rebalancing
        if current_minute >= self.next_rebalance_minute:
            # Calculate true yield token price and pool price
            true_yt_price = self._calculate_true_yield_token_price(current_minute)
            pool_yt_price = self._get_pool_yield_token_price()
            
            if true_yt_price is None:
                print(f"ALM Rebalancer: true_yt_price is None at minute {current_minute}")
                return (AgentAction.HOLD, {})
            if pool_yt_price is None:
                print(f"ALM Rebalancer: pool_yt_price is None at minute {current_minute}")
                return (AgentAction.HOLD, {})
            
            # Calculate price deviation
            price_deviation = abs(pool_yt_price - true_yt_price) / true_yt_price
            
            # Always rebalance on schedule (ALM characteristic)
            rebalance_params = self._calculate_rebalance_amount(true_yt_price, pool_yt_price, current_minute)
            
            print(f"ALM Rebalancer at minute {current_minute}:")
            print(f"   Calculated amount: ${rebalance_params.get('amount', 0):,.2f}")
            print(f"   Direction: {rebalance_params.get('direction', 'unknown')}")
            print(f"   True price: ${rebalance_params.get('true_price', 0):.6f}")
            print(f"   Pool price: ${rebalance_params.get('pool_price', 0):.6f}")
            print(f"   Min required: ${self.state.min_rebalance_amount}")
            
            if rebalance_params["amount"] >= self.state.min_rebalance_amount:
                # Schedule next rebalance
                self.next_rebalance_minute = current_minute + self.rebalance_interval_minutes
                
                return (AgentAction.SWAP, rebalance_params)
            else:
                print(f"ALM Rebalancer: amount {rebalance_params['amount']:.2f} below minimum {self.state.min_rebalance_amount}")
        
        return (AgentAction.HOLD, {})
    
    def _calculate_true_yield_token_price(self, current_minute: int) -> Optional[float]:
        """Calculate the true yield token price based on 10% APR"""
        return calculate_true_yield_token_price(current_minute, 0.10, 1.0)
    
    def _get_pool_yield_token_price(self) -> Optional[float]:
        """Get current YT price from the MOET:YT pool"""
        if not self.yield_token_pool:
            return None
            
        try:
            # Get pool state to determine current price
            pool_state = self.yield_token_pool.get_pool_state()
            current_price = pool_state.get("current_price", 1.0)
            
            # In our pool, price represents MOET per YT, so YT price = 1/price
            # But if price is already YT per MOET, use it directly
            return current_price
        except Exception as e:
            print(f"Error getting pool YT price: {e}")
            return None
    
    def _calculate_rebalance_amount(self, true_price: float, pool_price: float, current_minute: int) -> Dict:
        """Calculate exact rebalance amount needed to bring pool to target price"""
        
        if not self.yield_token_pool:
            return {"direction": "hold", "amount": 0, "true_price": true_price, 
                   "pool_price": pool_price, "price_deviation_pct": 0, "rebalancer_type": "ALM", "minute": current_minute}
        
        # Use simple percentage-based calculation for now to avoid complex Uniswap V3 math issues
        try:
            price_diff_pct = abs(pool_price - true_price) / true_price
            
            # FIXED: Correct arbitrage direction logic
            if pool_price > true_price:
                # Pool YT overpriced -> buy YT from pool (remove supply) to lower pool price
                direction = "buy_yt_with_moet"
                print(f"   ALM: Pool price ${pool_price:.6f} > True price ${true_price:.6f} → BUY YT from pool")
            else:
                # Pool YT underpriced -> mint YT externally + sell to pool (add supply) to raise pool price  
                direction = "sell_yt_for_moet"
                print(f"   ALM: Pool price ${pool_price:.6f} < True price ${true_price:.6f} → MINT & SELL YT to pool")
            
            # Calculate exact rebalance amount using proper Uniswap V3 math
            rebalance_amount = self._calculate_exact_swap_amount_for_target_price(
                pool_price, true_price, direction
            )
            
            # Cap at maximum single rebalance and ensure minimum
            rebalance_amount = min(rebalance_amount, self.state.max_single_rebalance)
            rebalance_amount = max(rebalance_amount, 100.0)  # Minimum $100 rebalance
            
            # Check available balances - now considering external YT minting capability
            if direction == "sell_yt_for_moet":
                available_yt_value = self.state.yield_token_balance
                yt_shortfall = max(0, rebalance_amount - available_yt_value)
                yt_minting_cost = yt_shortfall * true_price
                
                if yt_shortfall > 0:
                    if self.state.moet_balance >= yt_minting_cost:
                        print(f"   ALM: Can mint ${yt_shortfall:.2f} YT externally (cost: ${yt_minting_cost:.2f} MOET)")
                        print(f"   ALM: Using ${rebalance_amount:.2f} YT total (${available_yt_value:.2f} existing + ${yt_shortfall:.2f} minted)")
                    else:
                        # Limit rebalance to what we can afford to mint
                        max_mintable_yt = self.state.moet_balance / true_price
                        rebalance_amount = min(rebalance_amount, available_yt_value + max_mintable_yt)
                        print(f"   ALM: Limited by MOET balance - can only rebalance ${rebalance_amount:.2f} YT")
                else:
                    print(f"   ALM: Using ${rebalance_amount:.2f} YT from existing balance")
            else:
                available_moet = self.state.moet_balance
                rebalance_amount = min(rebalance_amount, available_moet)
                print(f"   ALM: Using ${rebalance_amount:.2f} MOET (available: ${available_moet:.2f})")
            
            return {
                "direction": direction,
                "amount": rebalance_amount,
                "true_price": true_price,
                "pool_price": pool_price,
                "price_deviation_pct": price_diff_pct * 100,
                "rebalancer_type": "ALM",
                "minute": current_minute
            }
            
        except Exception as e:
            print(f"ALM rebalance calculation error: {e}")
            # Fallback to simple calculation
            price_diff_pct = abs(pool_price - true_price) / true_price
            direction = "sell_yt_for_moet" if pool_price > true_price else "buy_yt_with_moet"
            fallback_amount = min(10_000.0, price_diff_pct * 50_000.0)  # Conservative fallback
            
            return {
                "direction": direction,
                "amount": fallback_amount,
                "true_price": true_price,
                "pool_price": pool_price,
                "price_deviation_pct": price_diff_pct * 100,
                "rebalancer_type": "ALM",
                "minute": current_minute
            }
    
    def execute_rebalance(self, params: Dict) -> bool:
        """Execute the rebalancing operation"""
        if not self.yield_token_pool:
            return False
            
        try:
            direction = params["direction"]
            amount = params["amount"]
            true_price = params["true_price"]
            
            if direction == "sell_yt_for_moet":
                # Check if we need to mint YT externally first
                if self.state.yield_token_balance < amount:
                    yt_to_mint = amount - self.state.yield_token_balance
                    yt_minting_cost = yt_to_mint * true_price  # Cost to mint YT externally at true price
                    
                    if self.state.moet_balance >= yt_minting_cost:
                        # Step 1: Mint YT externally using MOET
                        print(f"   Step 1 - Minting ${yt_to_mint:.2f} YT externally at ${true_price:.6f} (cost: ${yt_minting_cost:.2f})")
                        self.state.moet_balance -= yt_minting_cost
                        self.state.yield_token_balance += yt_to_mint
                        print(f"   After minting: MOET=${self.state.moet_balance:,.2f}, YT=${self.state.yield_token_balance:.2f}")
                    else:
                        print(f"   Insufficient MOET to mint YT: need ${yt_minting_cost:.2f}, have ${self.state.moet_balance:.2f}")
                        return False
                
                # Store pool price before the swap
                pool_price_before = self.yield_token_pool.uniswap_pool.get_price()
                print(f"   Pool price BEFORE selling YT: ${pool_price_before:.6f}")
                
                # Step 2: Sell YT to pool, receive MOET
                moet_received = self.yield_token_pool.execute_yield_token_sale(amount)
                
                # Check pool price after the swap
                pool_price_after = self.yield_token_pool.uniswap_pool.get_price()
                print(f"   Pool price AFTER selling YT: ${pool_price_after:.6f}")
                print(f"   Price change: ${pool_price_after - pool_price_before:+.6f}")
                print(f"   Expected: Price should INCREASE when we sell YT (add YT supply to pool)")
                
                if moet_received > 0:
                    # Step 3: Update balances
                    self.state.yield_token_balance -= amount
                    self.state.moet_balance += moet_received
                    
                    print(f"   Step 3 - After pool sale: MOET=${self.state.moet_balance:,.2f}, YT=${self.state.yield_token_balance:.2f}")
                    
                    # Calculate arbitrage profit: pool MOET received vs external YT minting cost
                    yt_minting_cost = amount * true_price
                    profit = moet_received - yt_minting_cost
                    print(f"   Arbitrage profit: ${moet_received:.2f} (pool sale) - ${yt_minting_cost:.2f} (minting cost) = ${profit:.2f}")
                    
                    self._record_rebalance(params, profit, amount)
                    return True
                    
            else:  # buy_yt_with_moet
                # Store pool price before the swap
                pool_price_before = self.yield_token_pool.uniswap_pool.get_price()
                print(f"   Pool price BEFORE buying YT: ${pool_price_before:.6f}")
                
                # Buy YT from pool with MOET
                yt_received = self.yield_token_pool.execute_yield_token_purchase(amount)
                
                # Check pool price after the swap
                pool_price_after = self.yield_token_pool.uniswap_pool.get_price()
                print(f"   Pool price AFTER buying YT: ${pool_price_after:.6f}")
                print(f"   Price change: ${pool_price_after - pool_price_before:+.6f}")
                print(f"   Expected: Price should INCREASE when we buy YT (remove YT from pool)")
                
                if yt_received > 0:
                    # Step 1: Show initial state
                    print(f"   Step 1 - Initial state: MOET=${self.state.moet_balance:,.0f}, YT=${self.state.yield_token_balance:.0f}")
                    
                    # Step 2: Update balances after buying YT from pool
                    self.state.moet_balance -= amount
                    self.state.yield_token_balance += yt_received
                    
                    print(f"   Step 2 - After buying ${yt_received:.2f} YT from pool: MOET=${self.state.moet_balance:,.0f}, YT=${self.state.yield_token_balance:.2f}")
                    
                    # Step 3: Handle external YT sale with optional arbitrage delay
                    current_time = params.get("minute", 0)
                    simulation_duration = params.get("simulation_duration", 10000)
                    
                    if self.state.arb_delay_enabled:
                        # Add YT to pending sales queue
                        self.state._add_pending_yt_sale(yt_received, true_price, current_time, simulation_duration)
                        # Keep YT balance as is (already added yt_received above)
                        
                        # Calculate expected profit (for logging, not actual balance change)
                        expected_profit = (yt_received * true_price) - amount
                        print(f"   Step 3 - YT sale delayed: ${yt_received:.2f} YT queued for future sale")
                        print(f"   Expected arbitrage profit: ${yt_received * true_price:.2f} (future sale) - ${amount:.2f} (pool cost) = ${expected_profit:.2f}")
                        profit = 0.0  # No immediate profit due to delay
                    else:
                        # Immediate external YT sale (original behavior)
                        external_moet_from_yt_sale = yt_received * true_price
                        self.state.moet_balance += external_moet_from_yt_sale
                        self.state.yield_token_balance -= yt_received  # Remove YT that was sold externally
                        
                        print(f"   Step 3 - After immediate external YT sale at ${true_price:.6f}: MOET=${self.state.moet_balance:,.2f}, YT=${self.state.yield_token_balance:.0f}")
                        
                        # Calculate arbitrage profit
                        profit = external_moet_from_yt_sale - amount
                        print(f"   Arbitrage profit: ${external_moet_from_yt_sale:.2f} (external sale) - ${amount:.2f} (pool cost) = ${profit:.2f}")
                    
                    self._record_rebalance(params, profit, amount)
                    return True
                    
        except Exception as e:
            print(f"ALM Rebalancer execution error: {e}")
            return False
            
        return False
    
    def _record_rebalance(self, params: Dict, profit: float, volume: float):
        """Record rebalancing activity"""
        self.state.total_rebalances += 1
        self.state.total_volume_rebalanced += volume
        self.state.total_profit_from_arbing += profit
        self.state.last_rebalance_minute = params.get("minute", 0)
        
        print(f"🔄 ALM Rebalancer: {params['direction']} ${volume:,.0f} at minute {params.get('minute', 0)}")
        print(f"   Price deviation: {params['price_deviation_pct']:.2f}%, Profit: ${profit:.2f}")
    
    def _calculate_exact_swap_amount_for_target_price(self, current_price: float, target_price: float, direction: str) -> float:
        """Calculate exact swap amount needed to move pool price to target using Uniswap V3 math"""
        if not self.yield_token_pool or not hasattr(self.yield_token_pool, 'uniswap_pool'):
            # Fallback to simple estimation if no pool access
            price_gap = abs(target_price - current_price)
            return min(price_gap * 100_000.0, 50_000.0)
        
        try:
            import math
            from ..core.uniswap_v3_math import get_amount0_delta, get_amount1_delta, Q96, MIN_SQRT_RATIO, MAX_SQRT_RATIO
            
            # Get pool state
            pool = self.yield_token_pool.uniswap_pool
            current_sqrt_price = pool.sqrt_price_x96
            # Convert target price to sqrt_price_x96 format
            sqrt_price = math.sqrt(target_price)
            target_sqrt_price = int(sqrt_price * Q96)
            target_sqrt_price = max(MIN_SQRT_RATIO, min(MAX_SQRT_RATIO, target_sqrt_price))
            liquidity = pool.liquidity
            
            print(f"   🔍 DEBUG: Pool liquidity: {liquidity:,}")
            print(f"   🔍 DEBUG: Current price: ${current_price:.6f}, Target price: ${target_price:.6f}")
            print(f"   🔍 DEBUG: Current sqrt_price: {current_sqrt_price:,}, Target sqrt_price: {target_sqrt_price:,}")
            
            if liquidity == 0:
                print(f"   ❌ WARNING: Pool liquidity is zero!")
                return 10_000.0  # Use larger fallback amount
            
            # Calculate required amount based on swap direction
            if direction == "buy_yt_with_moet":
                # MOET -> YT swap (zero_for_one = False, we want token1 = YT)
                # We need to provide MOET (token0) to get YT (token1)
                amount_moet_needed_scaled = get_amount0_delta(
                    current_sqrt_price, target_sqrt_price, liquidity, True
                )
                amount_moet_needed = amount_moet_needed_scaled / 1e6  # Convert from scaled amount
                print(f"   🔍 DEBUG: MOET needed (scaled): {amount_moet_needed_scaled:,}")
                print(f"   🔍 DEBUG: MOET needed (USD): ${amount_moet_needed:,.2f}")
                return max(100.0, amount_moet_needed)
                
            elif direction == "sell_yt_for_moet":
                # YT -> MOET swap (zero_for_one = True, we provide token1 = YT)
                # We need to provide YT (token1) to get MOET (token0)
                amount_yt_needed_scaled = get_amount1_delta(
                    target_sqrt_price, current_sqrt_price, liquidity, True
                )
                amount_yt_needed = amount_yt_needed_scaled / 1e6  # Convert from scaled amount
                print(f"   🔍 DEBUG: YT needed (scaled): {amount_yt_needed_scaled:,}")
                print(f"   🔍 DEBUG: YT needed (USD): ${amount_yt_needed:,.2f}")
                return max(100.0, amount_yt_needed)
            
        except Exception as e:
            print(f"   ❌ ERROR calculating exact swap amount: {e}")
            # Fallback to price-based estimation
            price_gap = abs(target_price - current_price)
            fallback_amount = min(price_gap * 100_000.0, 50_000.0)
            print(f"   🔄 Using fallback amount: ${fallback_amount:,.2f}")
            return fallback_amount
        
        return 10_000.0  # Final fallback


class AlgoRebalancer(BaseAgent):
    """
    Algorithmic Rebalancer
    
    Rebalances the MOET:YT pool only when price deviation exceeds 50bps threshold.
    Operates continuously monitoring price but only acts on significant deviations.
    """
    
    def __init__(self, agent_id: str = "algo_rebalancer", deviation_threshold_bps: float = 50.0):
        super().__init__(agent_id, "algo_rebalancer", 500_000.0)
        self.state = PoolRebalancerState(agent_id, 500_000.0, "algo_rebalancer")
        
        # Algo-specific parameters
        self.deviation_threshold_bps = deviation_threshold_bps  # 50 basis points = 0.5%
        self.deviation_threshold_decimal = deviation_threshold_bps / 10000.0  # Convert to decimal
        
        # Pool reference (set by engine)
        self.yield_token_pool: Optional[YieldTokenPool] = None
        
    def set_yield_token_pool(self, pool: YieldTokenPool):
        """Set the yield token pool reference"""
        self.yield_token_pool = pool
        
    def decide_action(self, protocol_state: dict, asset_prices: Dict[Asset, float]) -> tuple:
        """
        Decide whether to rebalance based on price deviation threshold
        
        Returns:
            Tuple of (action_type, params)
        """
        if not self.state.enabled or not self.yield_token_pool:
            return (AgentAction.HOLD, {})
            
        current_minute = protocol_state.get("current_minute", 0)
        
        # Calculate true yield token price and pool price
        true_yt_price = self._calculate_true_yield_token_price(current_minute)
        pool_yt_price = self._get_pool_yield_token_price()
        
        if true_yt_price is None or pool_yt_price is None:
            return (AgentAction.HOLD, {})
        
        # Calculate price deviation
        price_deviation = abs(pool_yt_price - true_yt_price) / true_yt_price
        
        # Only rebalance if deviation exceeds threshold
        if price_deviation >= self.deviation_threshold_decimal:
            rebalance_params = self._calculate_rebalance_amount(true_yt_price, pool_yt_price, current_minute)
            
            if rebalance_params["amount"] >= self.state.min_rebalance_amount:
                return (AgentAction.SWAP, rebalance_params)
        
        return (AgentAction.HOLD, {})
    
    def _calculate_true_yield_token_price(self, current_minute: int) -> Optional[float]:
        """Calculate the true yield token price based on 10% APR"""
        return calculate_true_yield_token_price(current_minute, 0.10, 1.0)
    
    def _get_pool_yield_token_price(self) -> Optional[float]:
        """Get current YT price from the MOET:YT pool"""
        if not self.yield_token_pool:
            return None
            
        try:
            # Get pool state to determine current price
            pool_state = self.yield_token_pool.get_pool_state()
            current_price = pool_state.get("current_price", 1.0)
            
            return current_price
        except Exception as e:
            print(f"Error getting pool YT price: {e}")
            return None
    
    def _calculate_rebalance_amount(self, true_price: float, pool_price: float, current_minute: int) -> Dict:
        """Calculate optimal rebalance amount and direction"""
        
        # FIXED: Correct arbitrage direction logic for Algo rebalancer
        if pool_price > true_price:
            # Pool YT is overpriced -> buy YT from pool (remove supply) to lower pool price
            direction = "buy_yt_with_moet"
            price_diff_pct = (pool_price - true_price) / true_price
        else:
            # Pool YT is underpriced -> mint YT externally + sell to pool (add supply) to raise pool price
            direction = "sell_yt_for_moet" 
            price_diff_pct = (true_price - pool_price) / true_price
        
        # Calculate exact rebalance amount using proper Uniswap V3 math
        rebalance_amount = self._calculate_exact_swap_amount_for_target_price(
            pool_price, true_price, direction
        )
        
        # Cap at maximum single rebalance
        rebalance_amount = min(rebalance_amount, self.state.max_single_rebalance)
        
        # Ensure we have sufficient balance - now considering external YT minting capability
        if direction == "sell_yt_for_moet":
            available_yt_value = self.state.yield_token_balance
            yt_shortfall = max(0, rebalance_amount - available_yt_value)
            yt_minting_cost = yt_shortfall * true_price
            
            if yt_shortfall > 0:
                if self.state.moet_balance >= yt_minting_cost:
                    # Can mint the needed YT externally
                    pass  # Keep rebalance_amount as calculated
                else:
                    # Limit rebalance to what we can afford to mint
                    max_mintable_yt = self.state.moet_balance / true_price
                    rebalance_amount = min(rebalance_amount, available_yt_value + max_mintable_yt)
        else:
            available_moet = self.state.moet_balance
            rebalance_amount = min(rebalance_amount, available_moet)
        
        return {
            "direction": direction,
            "amount": rebalance_amount,
            "true_price": true_price,
            "pool_price": pool_price,
            "price_deviation_pct": price_diff_pct * 100,
            "rebalancer_type": "Algo",
            "minute": current_minute
        }
    
    def execute_rebalance(self, params: Dict) -> bool:
        """Execute the rebalancing operation"""
        if not self.yield_token_pool:
            return False
            
        try:
            direction = params["direction"]
            amount = params["amount"]
            true_price = params["true_price"]
            
            if direction == "sell_yt_for_moet":
                # Check if we need to mint YT externally first
                if self.state.yield_token_balance < amount:
                    yt_to_mint = amount - self.state.yield_token_balance
                    yt_minting_cost = yt_to_mint * true_price  # Cost to mint YT externally at true price
                    
                    if self.state.moet_balance >= yt_minting_cost:
                        # Step 1: Mint YT externally using MOET
                        print(f"   Algo: Minting ${yt_to_mint:.2f} YT externally at ${true_price:.6f} (cost: ${yt_minting_cost:.2f})")
                        self.state.moet_balance -= yt_minting_cost
                        self.state.yield_token_balance += yt_to_mint
                    else:
                        print(f"   Algo: Insufficient MOET to mint YT: need ${yt_minting_cost:.2f}, have ${self.state.moet_balance:.2f}")
                        return False
                
                # Sell YT to pool, receive MOET
                moet_received = self.yield_token_pool.execute_yield_token_sale(amount)
                
                if moet_received > 0:
                    # Update balances
                    self.state.yield_token_balance -= amount
                    self.state.moet_balance += moet_received
                    
                    # Calculate arbitrage profit: pool MOET received vs external YT minting cost
                    yt_minting_cost = amount * true_price
                    profit = moet_received - yt_minting_cost
                    print(f"   Algo arbitrage profit: ${moet_received:.2f} (pool sale) - ${yt_minting_cost:.2f} (minting cost) = ${profit:.2f}")
                    
                    self._record_rebalance(params, profit, amount)
                    return True
                    
            else:  # buy_yt_with_moet
                # Buy YT from pool with MOET
                yt_received = self.yield_token_pool.execute_yield_token_purchase(amount)
                
                if yt_received > 0:
                    # Step 1: Show initial state
                    print(f"   Step 1 - Initial state: MOET=${self.state.moet_balance:,.0f}, YT=${self.state.yield_token_balance:.0f}")
                    
                    # Step 2: Update balances after buying YT from pool
                    self.state.moet_balance -= amount
                    self.state.yield_token_balance += yt_received
                    
                    print(f"   Step 2 - After buying ${yt_received:.2f} YT from pool: MOET=${self.state.moet_balance:,.0f}, YT=${self.state.yield_token_balance:.2f}")
                    
                    # Step 3: Handle external YT sale with optional arbitrage delay
                    current_time = params.get("minute", 0)
                    simulation_duration = params.get("simulation_duration", 10000)
                    
                    if self.state.arb_delay_enabled:
                        # Add YT to pending sales queue
                        self.state._add_pending_yt_sale(yt_received, true_price, current_time, simulation_duration)
                        # Keep YT balance as is (already added yt_received above)
                        
                        # Calculate expected profit (for logging, not actual balance change)
                        expected_profit = (yt_received * true_price) - amount
                        print(f"   Step 3 - Algo YT sale delayed: ${yt_received:.2f} YT queued for future sale")
                        print(f"   Expected arbitrage profit: ${yt_received * true_price:.2f} (future sale) - ${amount:.2f} (pool cost) = ${expected_profit:.2f}")
                        profit = 0.0  # No immediate profit due to delay
                    else:
                        # Immediate external YT sale (original behavior)
                        external_moet_from_yt_sale = yt_received * true_price
                        self.state.moet_balance += external_moet_from_yt_sale
                        self.state.yield_token_balance -= yt_received  # Remove YT that was sold externally
                        
                        print(f"   Step 3 - After immediate external YT sale at ${true_price:.6f}: MOET=${self.state.moet_balance:,.2f}, YT=${self.state.yield_token_balance:.0f}")
                        
                        # Calculate arbitrage profit
                        profit = external_moet_from_yt_sale - amount
                        print(f"   Arbitrage profit: ${external_moet_from_yt_sale:.2f} (external sale) - ${amount:.2f} (pool cost) = ${profit:.2f}")
                    
                    self._record_rebalance(params, profit, amount)
                    return True
                    
        except Exception as e:
            print(f"Algo Rebalancer execution error: {e}")
            return False
            
        return False
    
    def _record_rebalance(self, params: Dict, profit: float, volume: float):
        """Record rebalancing activity"""
        self.state.total_rebalances += 1
        self.state.total_volume_rebalanced += volume
        self.state.total_profit_from_arbing += profit
        self.state.last_rebalance_minute = params.get("minute", 0)
        
        print(f"⚡ Algo Rebalancer: {params['direction']} ${volume:,.0f} at minute {params.get('minute', 0)}")
        print(f"   Price deviation: {params['price_deviation_pct']:.2f}%, Profit: ${profit:,.2f}")
    
    def _calculate_exact_swap_amount_for_target_price(self, current_price: float, target_price: float, direction: str) -> float:
        """Calculate exact swap amount needed to move pool price to target using Uniswap V3 math"""
        if not self.yield_token_pool or not hasattr(self.yield_token_pool, 'uniswap_pool'):
            # Fallback to simple estimation if no pool access
            price_gap = abs(target_price - current_price)
            return min(price_gap * 100_000.0, 50_000.0)
        
        try:
            import math
            from ..core.uniswap_v3_math import get_amount0_delta, get_amount1_delta, Q96, MIN_SQRT_RATIO, MAX_SQRT_RATIO
            
            # Get pool state
            pool = self.yield_token_pool.uniswap_pool
            current_sqrt_price = pool.sqrt_price_x96
            # Convert target price to sqrt_price_x96 format
            sqrt_price = math.sqrt(target_price)
            target_sqrt_price = int(sqrt_price * Q96)
            target_sqrt_price = max(MIN_SQRT_RATIO, min(MAX_SQRT_RATIO, target_sqrt_price))
            liquidity = pool.liquidity
            
            print(f"   🔍 DEBUG: Pool liquidity: {liquidity:,}")
            print(f"   🔍 DEBUG: Current price: ${current_price:.6f}, Target price: ${target_price:.6f}")
            print(f"   🔍 DEBUG: Current sqrt_price: {current_sqrt_price:,}, Target sqrt_price: {target_sqrt_price:,}")
            
            if liquidity == 0:
                print(f"   ❌ WARNING: Pool liquidity is zero!")
                return 10_000.0  # Use larger fallback amount
            
            # Calculate required amount based on swap direction
            if direction == "buy_yt_with_moet":
                # MOET -> YT swap (zero_for_one = False, we want token1 = YT)
                # We need to provide MOET (token0) to get YT (token1)
                amount_moet_needed_scaled = get_amount0_delta(
                    current_sqrt_price, target_sqrt_price, liquidity, True
                )
                amount_moet_needed = amount_moet_needed_scaled / 1e6  # Convert from scaled amount
                print(f"   🔍 DEBUG: MOET needed (scaled): {amount_moet_needed_scaled:,}")
                print(f"   🔍 DEBUG: MOET needed (USD): ${amount_moet_needed:,.2f}")
                return max(100.0, amount_moet_needed)
                
            elif direction == "sell_yt_for_moet":
                # YT -> MOET swap (zero_for_one = True, we provide token1 = YT)
                # We need to provide YT (token1) to get MOET (token0)
                amount_yt_needed_scaled = get_amount1_delta(
                    target_sqrt_price, current_sqrt_price, liquidity, True
                )
                amount_yt_needed = amount_yt_needed_scaled / 1e6  # Convert from scaled amount
                print(f"   🔍 DEBUG: YT needed (scaled): {amount_yt_needed_scaled:,}")
                print(f"   🔍 DEBUG: YT needed (USD): ${amount_yt_needed:,.2f}")
                return max(100.0, amount_yt_needed)
            
        except Exception as e:
            print(f"   ❌ ERROR calculating exact swap amount: {e}")
            # Fallback to price-based estimation
            price_gap = abs(target_price - current_price)
            fallback_amount = min(price_gap * 100_000.0, 50_000.0)
            print(f"   🔄 Using fallback amount: ${fallback_amount:,.2f}")
            return fallback_amount
        
        return 10_000.0  # Final fallback


class LiquidityRangeManager:
    """
    Manages dynamic liquidity range adjustments for MOET:YT pool
    
    Tracks ALM rebalance count and periodically updates the concentrated
    liquidity range to stay centered around the true yield token price,
    maintaining the asymmetric token ratio while preserving total liquidity.
    """
    
    def __init__(self, range_update_interval: int = 6, range_width: float = 0.01, 
                 enable_pool_replenishment: bool = True, target_pool_size: float = 500_000):
        """
        Initialize the Liquidity Range Manager
        
        Args:
            range_update_interval: Update range every N ALM rebalances (default: 6 = 3 days)
            range_width: Range width as percentage around center (default: 1% = 0.01)
            enable_pool_replenishment: Whether to automatically replenish pool reserves (default: True)
            target_pool_size: Target pool size in USD (default: $500,000)
        """
        self.range_update_interval = range_update_interval
        self.range_width = range_width
        self.enable_pool_replenishment = enable_pool_replenishment
        self.target_pool_size = target_pool_size
        self.alm_rebalance_count = 0
        self.last_range_update_minute = 0
        self.range_update_history = []
        self.replenishment_history = []
        
    def should_update_range(self) -> bool:
        """Check if it's time to update the liquidity range"""
        return self.alm_rebalance_count > 0 and self.alm_rebalance_count % self.range_update_interval == 0
    
    def on_alm_rebalance(self):
        """Call this after each ALM rebalance to track count"""
        self.alm_rebalance_count += 1
    
    def update_liquidity_range(self, pool, true_yt_price: float, current_minute: int) -> dict:
        """
        Update the pool's liquidity range around the true YT price
        
        Args:
            pool: YieldTokenPool with uniswap_pool to update
            true_yt_price: Current true yield token price (center of new range)
            current_minute: Current simulation minute for logging
            
        Returns:
            dict: Results of the range update operation
        """
        try:
            # Get the Uniswap V3 pool
            uniswap_pool = pool.uniswap_pool
            
            # ENHANCEMENT: Pool replenishment to maintain $500k size with 75/25 skew
            current_moet_reserve = pool.moet_reserve
            current_yt_reserve = pool.yield_token_reserve
            current_total_value = current_moet_reserve + (current_yt_reserve * true_yt_price)
            
            if self.enable_pool_replenishment:
                target_token0_ratio = 0.75   # 75% MOET, 25% YT
                target_moet_value = self.target_pool_size * target_token0_ratio  # $375k MOET
                target_yt_amount = (self.target_pool_size * (1 - target_token0_ratio)) / true_yt_price  # $125k worth of YT
                
                # Always adjust to exact target (can be positive or negative)
                pool.moet_reserve = target_moet_value
                pool.yield_token_reserve = target_yt_amount
                
                # CRITICAL: Update the underlying Uniswap V3 pool's total_liquidity to match new reserves
                # This ensures the range update works with the correct liquidity amount
                pool.uniswap_pool.total_liquidity = self.target_pool_size
                
                # CRITICAL: Use optimal range bounds from pre-computed lookup table
                # This replaces the hardcoded asymmetric formula with mathematically optimal bounds
                from ..analysis.optimal_range_lookup import OptimalRangeLookup
                
                # Initialize lookup table (cached after first use)
                if not hasattr(self, '_optimal_lookup'):
                    self._optimal_lookup = OptimalRangeLookup()
                
                # Get optimal bounds for current time
                P_lower_optimal, P_upper_optimal = self._optimal_lookup.get_optimal_bounds(current_minute)
                
                # Set the optimal bounds on the Uniswap pool before reinitializing
                # This ensures the positioning uses our mathematically optimal ranges
                pool.uniswap_pool._optimal_bounds = (P_lower_optimal, P_upper_optimal)
                
                # CRITICAL: Use the same initialization process as original pool creation
                # This ensures consistent liquidity density calculation
                pool.uniswap_pool._initialize_concentrated_positions()
                
                # CRITICAL: Complete the full initialization sequence like __post_init__
                pool.uniswap_pool._validate_position_coverage()
                pool.uniswap_pool._update_legacy_fields()
                
                # CRITICAL: Update YieldTokenPool's legacy properties to match Uniswap pool
                pool._update_legacy_properties()
                
                moet_adjustment = target_moet_value - current_moet_reserve
                yt_adjustment = target_yt_amount - current_yt_reserve
                
                # Record replenishment
                replenishment_event = {
                    "minute": current_minute,
                    "pool_value_before": current_total_value,
                    "pool_value_after": self.target_pool_size,
                    "moet_adjustment": moet_adjustment,
                    "yt_adjustment": yt_adjustment,
                    "target_moet": target_moet_value,
                    "target_yt": target_yt_amount
                }
                self.replenishment_history.append(replenishment_event)
                
                print(f"💰 POOL REPLENISHMENT at minute {current_minute}:")
                print(f"   📊 Pool value: ${current_total_value:,.0f} → ${self.target_pool_size:,.0f}")
                print(f"   🔄 MOET: ${current_moet_reserve:,.0f} → ${target_moet_value:,.0f} ({moet_adjustment:+,.0f})")
                print(f"   🔄 YT: {current_yt_reserve:,.2f} → {target_yt_amount:,.2f} ({yt_adjustment:+,.2f})")
                print(f"   🎯 Skew: 75/25 MOET/YT | Total adjustments: {len(self.replenishment_history)}")

            # Use existing token ratio to maintain asymmetric position
            token0_ratio = uniswap_pool.token0_ratio
            
            # OPTIMAL BOUNDS: Skip update_liquidity_range since we already set optimal bounds
            # The _initialize_asymmetric_yield_token_positions() call above uses the optimal bounds
            if hasattr(self, '_optimal_lookup'):
                # Get the optimal bounds we just used
                P_lower_optimal, P_upper_optimal = self._optimal_lookup.get_optimal_bounds(current_minute)
                
                # Calculate actual range width from optimal bounds
                optimal_range_width = (P_upper_optimal - P_lower_optimal) / true_yt_price
                
                # Create result dictionary manually since we bypassed update_liquidity_range
                result = {
                    "success": True,
                    "center_price": true_yt_price,
                    "actual_range": f"${P_lower_optimal:.6f} - ${P_upper_optimal:.6f}",
                    "tick_range": "optimal_ticks",  # We could calculate exact ticks if needed
                    "liquidity_preserved": 500_000_000_000,  # Placeholder
                    "new_liquidity": sum(pos.liquidity for pos in uniswap_pool.positions) if hasattr(uniswap_pool, 'positions') else 500_000_000_000,
                    "token0_ratio": token0_ratio,
                    "optimal_range_width": optimal_range_width,
                    "range_source": "optimal_lookup_table"
                }
                
                print(f"🎯 OPTIMAL RANGE APPLIED from lookup table:")
                print(f"   📏 Optimal Range: [{P_lower_optimal:.6f}, {P_upper_optimal:.6f}]")
                print(f"   📐 Optimal Width: {optimal_range_width*100:.2f}% (vs hardcoded {self.range_width*100:.1f}%)")
                
            else:
                # Fallback to original method if optimal lookup not available
                print("⚠️  No optimal lookup available, using hardcoded range_width")
                result = uniswap_pool.update_liquidity_range(
                    center_price=true_yt_price,
                    range_width=self.range_width,
                    token0_ratio=token0_ratio
                )
            
            # Add timing information
            result["minute"] = current_minute
            result["alm_rebalance_count"] = self.alm_rebalance_count
            result["days_since_start"] = current_minute / (24 * 60)
            
            # Log the range update
            if result["success"]:
                print(f"🔄 RANGE UPDATE at minute {current_minute} (Day {result['days_since_start']:.1f}):")
                print(f"   📊 ALM Rebalance #{self.alm_rebalance_count} → Range update triggered")
                print(f"   🎯 Center Price: ${true_yt_price:.6f}")
                
                # Show optimal range width if available, otherwise hardcoded width
                if "optimal_range_width" in result:
                    print(f"   📏 New Range: {result['actual_range']} ({result['optimal_range_width']*100:.2f}% optimal width)")
                    print(f"   🎯 Range Source: {result.get('range_source', 'optimal_lookup_table')}")
                else:
                    print(f"   📏 New Range: {result['actual_range']} (±{self.range_width*100:.1f}%)")
                    print(f"   ⚠️  Range Source: hardcoded_fallback")
                
                print(f"   🔢 Ticks: {result['tick_range']}")
                print(f"   💧 Liquidity: {result['liquidity_preserved']:,} → {result['new_liquidity']:,}")
                print(f"   ⚖️  Token Ratio: {token0_ratio*100:.1f}% MOET / {(1-token0_ratio)*100:.1f}% YT")
                
                # Store in history
                self.range_update_history.append(result)
                self.last_range_update_minute = current_minute
            else:
                print(f"❌ RANGE UPDATE FAILED at minute {current_minute}: {result.get('reason', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "reason": f"Exception during range update: {str(e)}",
                "minute": current_minute,
                "alm_rebalance_count": self.alm_rebalance_count
            }
            print(f"❌ RANGE UPDATE ERROR at minute {current_minute}: {str(e)}")
            return error_result
    
    def get_range_update_summary(self) -> dict:
        """Get summary of all range updates performed"""
        return {
            "total_updates": len(self.range_update_history),
            "alm_rebalances_tracked": self.alm_rebalance_count,
            "update_interval": self.range_update_interval,
            "range_width": self.range_width,
            "last_update_minute": self.last_range_update_minute,
            "update_history": self.range_update_history
        }


class PoolRebalancerManager:
    """
    Manager class to coordinate both rebalancer agents and provide unified interface
    """
    
    def __init__(self, alm_interval_minutes: int = 720, algo_threshold_bps: float = 50.0, 
                 enable_pool_replenishment: bool = True, target_pool_size: float = 500_000):
        self.alm_rebalancer = ALMRebalancer("alm_rebalancer", alm_interval_minutes)
        self.algo_rebalancer = AlgoRebalancer("algo_rebalancer", algo_threshold_bps)
        
        # Add Liquidity Range Manager with pool replenishment
        self.range_manager = LiquidityRangeManager(
            range_update_interval=6,  # Every 6 ALM rebalances = 3 days
            range_width=0.01,  # ±1% range around true YT price
            enable_pool_replenishment=enable_pool_replenishment,
            target_pool_size=target_pool_size
        )
        
        self.enabled = False  # Default to disabled for backward compatibility
        
    def set_enabled(self, enabled: bool):
        """Enable or disable pool rebalancing"""
        self.enabled = enabled
        self.alm_rebalancer.state.enabled = enabled
        self.algo_rebalancer.state.enabled = enabled
        
    def set_yield_token_pool(self, pool: YieldTokenPool):
        """Set the yield token pool for both rebalancers"""
        self.alm_rebalancer.set_yield_token_pool(pool)
        self.algo_rebalancer.set_yield_token_pool(pool)
    
    def set_arb_delay_enabled(self, enabled: bool):
        """Enable or disable arbitrage delay for both rebalancers"""
        self.alm_rebalancer.state.set_arb_delay_enabled(enabled)
        self.algo_rebalancer.state.set_arb_delay_enabled(enabled)
        
        delay_status = "ENABLED" if enabled else "DISABLED"
        print(f"🔄 Arbitrage delay {delay_status} for both ALM and Algo rebalancers")
    
    def get_arb_delay_status(self) -> Dict:
        """Get arbitrage delay status for both rebalancers"""
        return {
            "alm_arb_delay_enabled": self.alm_rebalancer.state.arb_delay_enabled,
            "algo_arb_delay_enabled": self.algo_rebalancer.state.arb_delay_enabled,
            "alm_pending_sales": len(self.alm_rebalancer.state.pending_yt_sales),
            "algo_pending_sales": len(self.algo_rebalancer.state.pending_yt_sales),
            "alm_time_scale": self.alm_rebalancer.state._simulation_time_scale,
            "algo_time_scale": self.algo_rebalancer.state._simulation_time_scale
        }
    
    def get_range_management_status(self) -> Dict:
        """Get liquidity range management status and history"""
        return self.range_manager.get_range_update_summary()
        
    def process_rebalancing(self, protocol_state: dict, asset_prices: Dict[Asset, float]) -> List[Dict]:
        """Process both rebalancers and execute any needed rebalancing"""
        if not self.enabled:
            return []
        
        current_time = protocol_state.get("current_minute", 0)
        simulation_duration = protocol_state.get("simulation_duration", 10000)
        
        # First, process any pending YT sales from arbitrage delay
        self.alm_rebalancer.state._process_pending_yt_sales(current_time, simulation_duration)
        self.algo_rebalancer.state._process_pending_yt_sales(current_time, simulation_duration)
        
        rebalancing_events = []
        
        # Add simulation_duration to params for time scale detection
        enhanced_protocol_state = protocol_state.copy()
        enhanced_protocol_state["simulation_duration"] = simulation_duration
        
        # Process ALM rebalancer
        alm_action, alm_params = self.alm_rebalancer.decide_action(enhanced_protocol_state, asset_prices)
        if alm_action == AgentAction.SWAP and alm_params.get("amount", 0) > 0:
            # Add simulation_duration to params for arbitrage delay processing
            alm_params["simulation_duration"] = simulation_duration
            success = self.alm_rebalancer.execute_rebalance(alm_params)
            if success:
                rebalancing_events.append({
                    "rebalancer": "ALM",
                    "minute": current_time,
                    "params": alm_params,
                    "success": True
                })
                
                # Track ALM rebalance for range management
                self.range_manager.on_alm_rebalance()
                
                # Check if we should update liquidity range
                if self.range_manager.should_update_range():
                    # Calculate true YT price for range centering
                    from ..core.yield_tokens import calculate_true_yield_token_price
                    true_yt_price = calculate_true_yield_token_price(current_time, 0.10, 1.0)
                    
                    # Update the range
                    range_result = self.range_manager.update_liquidity_range(
                        pool=self.alm_rebalancer.yield_token_pool,
                        true_yt_price=true_yt_price,
                        current_minute=current_time
                    )
                    
                    # Add range update to rebalancing events
                    rebalancing_events.append({
                        "rebalancer": "RANGE_MANAGER",
                        "minute": current_time,
                        "params": range_result,
                        "success": range_result["success"]
                    })
        
        # Process Algo rebalancer
        algo_action, algo_params = self.algo_rebalancer.decide_action(enhanced_protocol_state, asset_prices)
        if algo_action == AgentAction.SWAP and algo_params.get("amount", 0) > 0:
            # Add simulation_duration to params for arbitrage delay processing
            algo_params["simulation_duration"] = simulation_duration
            success = self.algo_rebalancer.execute_rebalance(algo_params)
            if success:
                rebalancing_events.append({
                    "rebalancer": "Algo",
                    "minute": current_time,
                    "params": algo_params,
                    "success": True
                })
        
        return rebalancing_events
    
    def get_rebalancer_summary(self) -> Dict:
        """Get summary of both rebalancers' activity"""
        return {
            "enabled": self.enabled,
            "alm_rebalancer": {
                "total_rebalances": self.alm_rebalancer.state.total_rebalances,
                "total_volume": self.alm_rebalancer.state.total_volume_rebalanced,
                "total_profit": self.alm_rebalancer.state.total_profit_from_arbing,
                "moet_balance": self.alm_rebalancer.state.moet_balance,
                "yt_balance": self.alm_rebalancer.state.yield_token_balance,
                "next_rebalance_minute": self.alm_rebalancer.next_rebalance_minute
            },
            "algo_rebalancer": {
                "total_rebalances": self.algo_rebalancer.state.total_rebalances,
                "total_volume": self.algo_rebalancer.state.total_volume_rebalanced,
                "total_profit": self.algo_rebalancer.state.total_profit_from_arbing,
                "moet_balance": self.algo_rebalancer.state.moet_balance,
                "yt_balance": self.algo_rebalancer.state.yield_token_balance,
                "deviation_threshold_bps": self.algo_rebalancer.deviation_threshold_bps
            }
        }
