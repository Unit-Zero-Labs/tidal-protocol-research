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
        self.min_rebalance_amount = 1000.0  # Minimum $1k rebalance
        self.max_single_rebalance = 50_000.0  # Maximum $50k single rebalance


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
            
            if true_yt_price is None or pool_yt_price is None:
                return (AgentAction.HOLD, {})
            
            # Calculate price deviation
            price_deviation = abs(pool_yt_price - true_yt_price) / true_yt_price
            
            # Always rebalance on schedule (ALM characteristic)
            rebalance_params = self._calculate_rebalance_amount(true_yt_price, pool_yt_price, current_minute)
            
            if rebalance_params["amount"] >= self.state.min_rebalance_amount:
                # Schedule next rebalance
                self.next_rebalance_minute = current_minute + self.rebalance_interval_minutes
                
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
            
            # In our pool, price represents MOET per YT, so YT price = 1/price
            # But if price is already YT per MOET, use it directly
            return current_price
        except Exception as e:
            print(f"Error getting pool YT price: {e}")
            return None
    
    def _calculate_rebalance_amount(self, true_price: float, pool_price: float, current_minute: int) -> Dict:
        """Calculate optimal rebalance amount and direction"""
        
        # Determine if pool price is too high or too low
        if pool_price > true_price:
            # Pool YT is overpriced -> sell YT to pool, buy MOET
            direction = "sell_yt_for_moet"
            price_diff_pct = (pool_price - true_price) / true_price
        else:
            # Pool YT is underpriced -> buy YT from pool with MOET
            direction = "buy_yt_with_moet" 
            price_diff_pct = (true_price - pool_price) / true_price
        
        # Calculate rebalance amount based on price deviation
        # Larger deviations warrant larger rebalances
        base_amount = min(25_000.0, price_diff_pct * 100_000.0)  # Scale with deviation
        rebalance_amount = min(base_amount, self.state.max_single_rebalance)
        
        # Ensure we have sufficient balance
        if direction == "sell_yt_for_moet":
            available_yt_value = self.state.yield_token_balance
            rebalance_amount = min(rebalance_amount, available_yt_value)
            # If we don't have YT to sell, skip this rebalance
            if available_yt_value <= 0:
                rebalance_amount = 0
        else:
            available_moet = self.state.moet_balance
            rebalance_amount = min(rebalance_amount, available_moet)
        
        return {
            "direction": direction,
            "amount": rebalance_amount,
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
                # Sell YT to pool, receive MOET
                moet_received = self.yield_token_pool.execute_yield_token_sale(amount)
                
                if moet_received > 0:
                    # Update balances
                    self.state.yield_token_balance -= amount
                    self.state.moet_balance += moet_received
                    
                    # Immediately "sell" YT externally at true price to replenish YT balance
                    # This simulates selling YT outside our pool system
                    external_moet_from_yt_sale = amount * true_price
                    self.state.moet_balance += external_moet_from_yt_sale
                    
                    # Calculate arbitrage profit
                    total_moet_gained = moet_received + external_moet_from_yt_sale
                    cost_of_yt = amount  # Original cost basis
                    profit = total_moet_gained - cost_of_yt
                    
                    self._record_rebalance(params, profit, amount)
                    return True
                    
            else:  # buy_yt_with_moet
                # Buy YT from pool with MOET
                yt_received = self.yield_token_pool.execute_yield_token_purchase(amount)
                
                if yt_received > 0:
                    # Update balances
                    self.state.moet_balance -= amount
                    self.state.yield_token_balance += yt_received
                    
                    # Immediately "sell" YT externally at true price
                    external_moet_from_yt_sale = yt_received * true_price
                    self.state.moet_balance += external_moet_from_yt_sale
                    
                    # Calculate arbitrage profit
                    profit = external_moet_from_yt_sale - amount
                    
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
        print(f"   Price deviation: {params['price_deviation_pct']:.2f}%, Profit: ${profit:,.2f}")


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
        
        # Determine if pool price is too high or too low
        if pool_price > true_price:
            # Pool YT is overpriced -> sell YT to pool, buy MOET
            direction = "sell_yt_for_moet"
            price_diff_pct = (pool_price - true_price) / true_price
        else:
            # Pool YT is underpriced -> buy YT from pool with MOET
            direction = "buy_yt_with_moet" 
            price_diff_pct = (true_price - pool_price) / true_price
        
        # Calculate rebalance amount based on price deviation magnitude
        # More aggressive rebalancing for larger deviations
        base_amount = min(50_000.0, price_diff_pct * 200_000.0)  # Scale aggressively with deviation
        rebalance_amount = min(base_amount, self.state.max_single_rebalance)
        
        # Ensure we have sufficient balance
        if direction == "sell_yt_for_moet":
            available_yt_value = self.state.yield_token_balance
            rebalance_amount = min(rebalance_amount, available_yt_value)
            # If we don't have YT to sell, skip this rebalance
            if available_yt_value <= 0:
                rebalance_amount = 0
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
                # Sell YT to pool, receive MOET
                moet_received = self.yield_token_pool.execute_yield_token_sale(amount)
                
                if moet_received > 0:
                    # Update balances
                    self.state.yield_token_balance -= amount
                    self.state.moet_balance += moet_received
                    
                    # Immediately "sell" YT externally at true price to replenish YT balance
                    external_moet_from_yt_sale = amount * true_price
                    self.state.moet_balance += external_moet_from_yt_sale
                    
                    # Calculate arbitrage profit
                    total_moet_gained = moet_received + external_moet_from_yt_sale
                    cost_of_yt = amount
                    profit = total_moet_gained - cost_of_yt
                    
                    self._record_rebalance(params, profit, amount)
                    return True
                    
            else:  # buy_yt_with_moet
                # Buy YT from pool with MOET
                yt_received = self.yield_token_pool.execute_yield_token_purchase(amount)
                
                if yt_received > 0:
                    # Update balances
                    self.state.moet_balance -= amount
                    self.state.yield_token_balance += yt_received
                    
                    # Immediately "sell" YT externally at true price
                    external_moet_from_yt_sale = yt_received * true_price
                    self.state.moet_balance += external_moet_from_yt_sale
                    
                    # Calculate arbitrage profit
                    profit = external_moet_from_yt_sale - amount
                    
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


class PoolRebalancerManager:
    """
    Manager class to coordinate both rebalancer agents and provide unified interface
    """
    
    def __init__(self, alm_interval_minutes: int = 720, algo_threshold_bps: float = 50.0):
        self.alm_rebalancer = ALMRebalancer("alm_rebalancer", alm_interval_minutes)
        self.algo_rebalancer = AlgoRebalancer("algo_rebalancer", algo_threshold_bps)
        
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
        
    def process_rebalancing(self, protocol_state: dict, asset_prices: Dict[Asset, float]) -> List[Dict]:
        """Process both rebalancers and execute any needed rebalancing"""
        if not self.enabled:
            return []
            
        rebalancing_events = []
        
        # Process ALM rebalancer
        alm_action, alm_params = self.alm_rebalancer.decide_action(protocol_state, asset_prices)
        if alm_action == AgentAction.SWAP and alm_params.get("amount", 0) > 0:
            success = self.alm_rebalancer.execute_rebalance(alm_params)
            if success:
                rebalancing_events.append({
                    "rebalancer": "ALM",
                    "minute": protocol_state.get("current_minute", 0),
                    "params": alm_params,
                    "success": True
                })
        
        # Process Algo rebalancer
        algo_action, algo_params = self.algo_rebalancer.decide_action(protocol_state, asset_prices)
        if algo_action == AgentAction.SWAP and algo_params.get("amount", 0) > 0:
            success = self.algo_rebalancer.execute_rebalance(algo_params)
            if success:
                rebalancing_events.append({
                    "rebalancer": "Algo",
                    "minute": protocol_state.get("current_minute", 0),
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
