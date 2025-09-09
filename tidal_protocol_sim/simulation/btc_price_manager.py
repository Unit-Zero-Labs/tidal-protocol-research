#!/usr/bin/env python3
"""
BTC Price Decline Manager

Manages gradual BTC price decline with historical volatility patterns.
Used for stress testing scenarios across different protocol implementations.
"""

import random
from typing import Dict


class BTCPriceDeclineManager:
    """Manages gradual BTC price decline with realistic volatility patterns"""
    
    def __init__(self, initial_price: float = 100_000.0, duration: int = 60, 
                 final_price_range: tuple = (75_000.0, 85_000.0)):
        """
        Initialize BTC price decline manager
        
        Args:
            initial_price: Starting BTC price
            duration: Duration of decline in minutes
            final_price_range: Tuple of (min_price, max_price) for final price
        """
        self.initial_price = initial_price
        self.duration = duration
        
        # Historical decline rates based on real market data
        self.decline_rates = [-0.0054, -0.0053, -0.0046, -0.0043, -0.0040]
        self.max_decline = -0.0095
        
        # Calculate target final price
        final_price_min, final_price_max = final_price_range
        self.target_final_price = random.uniform(final_price_min, final_price_max)
        
        # Track price history
        self.price_history = [self.initial_price]
        self.current_price = self.initial_price
        
    def update_btc_price(self, minute: int) -> float:
        """
        Update BTC price for current minute
        
        Args:
            minute: Current minute in the simulation
            
        Returns:
            Updated BTC price
        """
        if minute == 0:
            return self.initial_price
            
        # Use historical volatility pattern
        base_decline = random.choice(self.decline_rates)
        variation = random.uniform(-0.0005, 0.0005)
        
        # Occasionally use maximum decline (5% probability)
        if random.random() < 0.05:
            decline_rate = self.max_decline
        else:
            decline_rate = base_decline + variation
            
        # Adjust to meet target final price
        progress = minute / self.duration
        if progress > 0.8:  # In final 20% of decline, converge to target
            current_decline = (self.current_price - self.initial_price) / self.initial_price
            remaining_decline_needed = (self.target_final_price - self.initial_price) / self.initial_price - current_decline
            remaining_minutes = self.duration - minute
            if remaining_minutes > 0:
                target_decline = remaining_decline_needed / remaining_minutes
                decline_rate = (decline_rate + target_decline) / 2
        
        self.current_price *= (1 + decline_rate)
        self.current_price = max(self.current_price, self.initial_price * 0.5)
        
        self.price_history.append(self.current_price)
        return self.current_price
        
    def get_decline_statistics(self) -> Dict[str, float]:
        """
        Get statistics about the price decline
        
        Returns:
            Dictionary containing decline statistics
        """
        if len(self.price_history) < 2:
            return {}
            
        final_price = self.price_history[-1]
        total_decline = (final_price - self.initial_price) / self.initial_price
        
        return {
            "initial_price": self.initial_price,
            "final_price": final_price,
            "total_decline_percent": total_decline * 100,
            "target_final_price": self.target_final_price,
            "duration_minutes": len(self.price_history) - 1
        }
    
    def reset(self, initial_price: float = None, final_price_range: tuple = None):
        """
        Reset the price manager for a new simulation
        
        Args:
            initial_price: New starting price (optional)
            final_price_range: New final price range (optional)
        """
        if initial_price is not None:
            self.initial_price = initial_price
        
        if final_price_range is not None:
            final_price_min, final_price_max = final_price_range
            self.target_final_price = random.uniform(final_price_min, final_price_max)
        
        self.current_price = self.initial_price
        self.price_history = [self.initial_price]
