#!/usr/bin/env python3
"""
MOET Stablecoin Mechanics (Simplified - No Fees)

This module implements the MOET stablecoin system without mint/burn fees
as specified in the refactoring requirements.
"""

from typing import Optional, Tuple


class MoetStablecoin:
    """Simplified MOET stablecoin without mint/burn fees"""
    
    def __init__(self, initial_supply: float = 1_000_000.0):
        self.total_supply = initial_supply
        self.target_price = 1.0  # $1.00 peg
        self.current_price = 1.0
        
        # Price stability bands (±2% as specified)
        self.stability_bands = (0.98, 1.02)
    
    def mint(self, amount: float) -> float:
        """1:1 minting, no fees"""
        if amount <= 0:
            return 0.0
        
        self.total_supply += amount
        return amount  # Returns full amount minted
    
    def burn(self, amount: float) -> float:
        """1:1 burning, no fees"""
        if amount <= 0:
            return 0.0
        
        # Cannot burn more than total supply
        actual_burn = min(amount, self.total_supply)
        self.total_supply -= actual_burn
        return actual_burn  # Returns actual amount burned
    
    def is_peg_stable(self) -> bool:
        """Check if MOET is within stability bands"""
        return self.stability_bands[0] <= self.current_price <= self.stability_bands[1]
    
    def calculate_stability_action(self) -> Optional[str]:
        """Determine if stability mechanism should activate"""
        if self.current_price > self.stability_bands[1]:
            return "mint_pressure"  # Price too high, encourage minting
        elif self.current_price < self.stability_bands[0]:
            return "burn_pressure"  # Price too low, encourage burning
        return None
    
    def get_peg_deviation(self) -> float:
        """Get percentage deviation from peg"""
        return (self.current_price - self.target_price) / self.target_price
    
    def update_price(self, new_price: float):
        """Update current MOET price"""
        self.current_price = max(0.01, new_price)  # Prevent negative prices
    
    def get_stability_pressure(self) -> Tuple[str, float]:
        """Get stability mechanism pressure and magnitude"""
        if self.current_price > self.stability_bands[1]:
            pressure = (self.current_price - self.stability_bands[1]) / self.target_price
            return "mint_pressure", pressure
        elif self.current_price < self.stability_bands[0]:
            pressure = (self.stability_bands[0] - self.current_price) / self.target_price
            return "burn_pressure", pressure
        else:
            return "stable", 0.0
    
    def get_state(self) -> dict:
        """Get current MOET system state"""
        return {
            "total_supply": self.total_supply,
            "current_price": self.current_price,
            "target_price": self.target_price,
            "is_peg_stable": self.is_peg_stable(),
            "peg_deviation": self.get_peg_deviation(),
            "stability_bands": self.stability_bands,
            "stability_action": self.calculate_stability_action()
        }