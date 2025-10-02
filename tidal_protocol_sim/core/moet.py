#!/usr/bin/env python3
"""
MOET Stablecoin Mechanics with Bonder System

This module implements the sophisticated MOET stablecoin system including:
- Bonder reserve management with dynamic bond auctions
- EMA-based interest rate calculations
- Redeemer contract for backing reserves
- Governance-controlled parameters
"""

from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
from collections import deque
import math


@dataclass
class BondAuction:
    """Represents a bond auction event"""
    timestamp: int  # minute
    target_amount: float  # USD amount needed
    starting_apr: float  # Initial APR offered
    final_apr: float  # Final APR when filled
    amount_filled: float  # Actual amount raised
    bond_price: float  # Final bond price
    auction_duration_minutes: int  # How long auction took
    filled_completely: bool  # Whether auction filled target amount


@dataclass
class ReserveState:
    """Current state of MOET backing reserves"""
    usdc_balance: float
    usdf_balance: float
    target_reserves_ratio: float  # e.g., 0.30 for 30%
    
    @property
    def total_reserves(self) -> float:
        return self.usdc_balance + self.usdf_balance
    
    def get_reserve_ratio(self, total_moet_supply: float) -> float:
        """Calculate current reserve ratio"""
        if total_moet_supply <= 0:
            return 1.0
        return self.total_reserves / total_moet_supply
    
    def get_reserve_deficit(self, total_moet_supply: float) -> float:
        """Calculate how much reserves are below target"""
        target_amount = total_moet_supply * self.target_reserves_ratio
        return max(0, target_amount - self.total_reserves)


class BonderSystem:
    """Manages bond auctions and reserve maintenance"""
    
    def __init__(self, governance_params: Dict):
        self.governance = governance_params
        self.auction_history: List[BondAuction] = []
        self.pending_auction: Optional[BondAuction] = None
        
        # EMA tracking for bond costs
        self.bond_apr_history = deque(maxlen=1000)  # Store recent APRs
        self.current_bond_cost_ema = 0.0
        
        # Auction parameters
        self.base_auction_duration = 60  # 1 hour default
        self.max_auction_duration = 360  # 6 hours max
        self.starting_premium = 0.005  # 0.5% starting premium
        self.max_premium = 0.10  # 10% max premium
        
    def calculate_bond_apr(self, reserve_state: ReserveState, total_moet_supply: float) -> float:
        """Calculate instantaneous bond APR based on reserve deficit"""
        if total_moet_supply <= 0:
            return 0.0
            
        target_reserves = total_moet_supply * reserve_state.target_reserves_ratio
        actual_reserves = reserve_state.total_reserves
        
        if actual_reserves >= target_reserves:
            return 0.0  # No bonds needed
            
        # BondAPR_t = max(0, (TargetReserves - ActualReserves) / TargetReserves)
        deficit_ratio = (target_reserves - actual_reserves) / target_reserves
        return max(0.0, deficit_ratio)
    
    def should_trigger_auction(self, reserve_state: ReserveState, total_moet_supply: float) -> bool:
        """Determine if bond auction should be triggered"""
        if self.pending_auction is not None:
            return False  # Auction already in progress
            
        deficit = reserve_state.get_reserve_deficit(total_moet_supply)
        return deficit > 1000.0  # Trigger if deficit > $1000
    
    def start_bond_auction(self, reserve_state: ReserveState, total_moet_supply: float, 
                          current_minute: int, benchmark_rate: float = 0.05) -> BondAuction:
        """Start a new bond auction"""
        deficit = reserve_state.get_reserve_deficit(total_moet_supply)
        base_apr = self.calculate_bond_apr(reserve_state, total_moet_supply)
        
        # Starting yield = Benchmark + Starting Premium + Base APR
        starting_apr = benchmark_rate + self.starting_premium + base_apr
        
        auction = BondAuction(
            timestamp=current_minute,
            target_amount=deficit,
            starting_apr=starting_apr,
            final_apr=starting_apr,  # Will be updated as auction progresses
            amount_filled=0.0,
            bond_price=1.0 / (1.0 + starting_apr / 365),  # Overnight pricing
            auction_duration_minutes=0,
            filled_completely=False
        )
        
        self.pending_auction = auction
        return auction
    
    def process_auction(self, current_minute: int, market_conditions: Dict) -> Optional[BondAuction]:
        """Process ongoing auction and determine if it fills"""
        if self.pending_auction is None:
            return None
            
        auction = self.pending_auction
        elapsed = current_minute - auction.timestamp
        
        # Dynamic pricing - increase premium over time
        time_factor = min(1.0, elapsed / self.base_auction_duration)
        additional_premium = time_factor * self.max_premium
        auction.final_apr = auction.starting_apr + additional_premium
        auction.bond_price = 1.0 / (1.0 + auction.final_apr / 365)
        auction.auction_duration_minutes = elapsed
        
        # Determine fill probability based on yield attractiveness
        fill_probability = self._calculate_fill_probability(auction, market_conditions)
        
        # Simulate auction outcome
        import random
        if random.random() < fill_probability or elapsed >= self.max_auction_duration:
            # Auction fills (or times out)
            if elapsed >= self.max_auction_duration:
                # Partial fill on timeout
                auction.amount_filled = auction.target_amount * 0.7  # 70% fill
                auction.filled_completely = False
            else:
                # Complete fill
                auction.amount_filled = auction.target_amount
                auction.filled_completely = True
            
            # Record the auction
            self.auction_history.append(auction)
            self.bond_apr_history.append(auction.final_apr)
            
            # Update EMA
            self._update_bond_cost_ema(auction.final_apr)
            
            self.pending_auction = None
            return auction
            
        return None  # Auction continues
    
    def _calculate_fill_probability(self, auction: BondAuction, market_conditions: Dict) -> float:
        """Calculate probability that auction fills this minute"""
        base_probability = 0.05  # 5% base chance per minute
        
        # Higher yield = higher probability
        yield_factor = min(2.0, auction.final_apr / 0.05)  # Cap at 2x for 5%+ yield
        
        # Market stress increases fill probability
        stress_factor = market_conditions.get('stress_level', 1.0)
        
        return min(0.95, base_probability * yield_factor * stress_factor)
    
    def _update_bond_cost_ema(self, new_apr: float):
        """Update EMA of bond costs with 7-day half-life"""
        if not self.bond_apr_history:
            self.current_bond_cost_ema = new_apr
            return
            
        # 7-day half-life = 7 * 24 * 60 = 10,080 minutes
        # EMA alpha = 1 - exp(-ln(2) / half_life_periods)
        half_life_minutes = self.governance.get('ema_half_life_days', 7) * 24 * 60
        alpha = 1 - math.exp(-math.log(2) / half_life_minutes)
        
        self.current_bond_cost_ema = alpha * new_apr + (1 - alpha) * self.current_bond_cost_ema
    
    def get_current_bond_cost(self) -> float:
        """Get current EMA of bond costs for interest calculation"""
        return self.current_bond_cost_ema


class RedeemerContract:
    """Manages MOET backing reserves and redemptions"""
    
    def __init__(self, initial_usdc: float = 0.0, initial_usdf: float = 0.0, 
                 target_reserves_ratio: float = 0.30):
        self.reserve_state = ReserveState(
            usdc_balance=initial_usdc,
            usdf_balance=initial_usdf,
            target_reserves_ratio=target_reserves_ratio
        )
        
    def add_reserves(self, usdc_amount: float = 0.0, usdf_amount: float = 0.0):
        """Add reserves from bond auction proceeds"""
        self.reserve_state.usdc_balance += usdc_amount
        self.reserve_state.usdf_balance += usdf_amount
    
    def process_redemption(self, moet_amount: float) -> bool:
        """Process MOET redemption at $1.00"""
        if self.reserve_state.total_reserves >= moet_amount:
            # Redeem proportionally from USDC/USDF
            total_reserves = self.reserve_state.total_reserves
            usdc_ratio = self.reserve_state.usdc_balance / total_reserves
            usdf_ratio = self.reserve_state.usdf_balance / total_reserves
            
            self.reserve_state.usdc_balance -= moet_amount * usdc_ratio
            self.reserve_state.usdf_balance -= moet_amount * usdf_ratio
            return True
        return False
    
    def get_state(self) -> Dict:
        """Get current redeemer state"""
        return {
            'usdc_balance': self.reserve_state.usdc_balance,
            'usdf_balance': self.reserve_state.usdf_balance,
            'total_reserves': self.reserve_state.total_reserves,
            'target_reserves_ratio': self.reserve_state.target_reserves_ratio
        }


class MoetStablecoin:
    """MOET stablecoin with sophisticated Bonder system and interest mechanics"""
    
    def __init__(self, initial_supply: float = 1_000_000.0, enable_advanced_system: bool = False):
        self.total_supply = initial_supply
        self.target_price = 1.0  # $1.00 peg
        self.current_price = 1.0
        
        # Price stability bands (±2% as specified)
        self.stability_bands = (0.98, 1.02)
        
        # Advanced system toggle
        self.enable_advanced_system = enable_advanced_system
        
        if self.enable_advanced_system:
            # Governance parameters
            self.governance_params = {
                'r_floor': 0.02,  # 2% governance profit margin
                'target_reserves_ratio': 0.30,  # 30% target reserves
                'ema_half_life_days': 7,  # 7-day EMA half-life
                'benchmark_rate': 0.05  # 5% benchmark rate
            }
            
            # Initialize sophisticated components
            self.bonder_system = BonderSystem(self.governance_params)
            self.redeemer = RedeemerContract(
                target_reserves_ratio=self.governance_params['target_reserves_ratio']
            )
            
            # Interest rate tracking
            self.current_moet_interest_rate = self.governance_params['r_floor']  # Start with floor rate
            self.interest_rate_history = []
            
        else:
            # Legacy simple system
            self.bonder_system = None
            self.redeemer = None
            self.current_moet_interest_rate = 0.0
    
    def initialize_reserves(self, initial_moet_debt: float):
        """Initialize reserves at 50% of initial MOET debt (50/50 USDC/USDF)"""
        if self.enable_advanced_system and self.redeemer:
            initial_reserves = initial_moet_debt * 0.50
            usdc_amount = initial_reserves * 0.50
            usdf_amount = initial_reserves * 0.50
            self.redeemer.add_reserves(usdc_amount, usdf_amount)
    
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
    
    def process_minute_update(self, current_minute: int, market_conditions: Dict = None) -> Dict:
        """Process sophisticated MOET system updates each minute"""
        if not self.enable_advanced_system:
            return {'advanced_system_enabled': False}
        
        if market_conditions is None:
            market_conditions = {'stress_level': 1.0}
        
        results = {
            'advanced_system_enabled': True,
            'bond_auction_triggered': False,
            'bond_auction_completed': False,
            'interest_rate_updated': False,
            'reserve_state': self.redeemer.get_state()
        }
        
        # Check if bond auction should be triggered
        if self.bonder_system.should_trigger_auction(self.redeemer.reserve_state, self.total_supply):
            auction = self.bonder_system.start_bond_auction(
                self.redeemer.reserve_state, 
                self.total_supply, 
                current_minute,
                self.governance_params['benchmark_rate']
            )
            results['bond_auction_triggered'] = True
            results['new_auction'] = {
                'target_amount': auction.target_amount,
                'starting_apr': auction.starting_apr,
                'timestamp': auction.timestamp
            }
        
        # Process ongoing auction
        completed_auction = self.bonder_system.process_auction(current_minute, market_conditions)
        if completed_auction:
            results['bond_auction_completed'] = True
            results['completed_auction'] = {
                'amount_filled': completed_auction.amount_filled,
                'final_apr': completed_auction.final_apr,
                'duration_minutes': completed_auction.auction_duration_minutes,
                'filled_completely': completed_auction.filled_completely
            }
            
            # Add proceeds to reserves (50/50 USDC/USDF)
            usdc_amount = completed_auction.amount_filled * 0.50
            usdf_amount = completed_auction.amount_filled * 0.50
            self.redeemer.add_reserves(usdc_amount, usdf_amount)
        
        # Update MOET interest rate: r_MOET = r_floor + r_bond-cost
        new_rate = self._calculate_moet_interest_rate()
        if abs(new_rate - self.current_moet_interest_rate) > 0.0001:  # 1 bps threshold
            self.current_moet_interest_rate = new_rate
            results['interest_rate_updated'] = True
            results['new_interest_rate'] = new_rate
            
            self.interest_rate_history.append({
                'minute': current_minute,
                'rate': new_rate,
                'r_floor': self.governance_params['r_floor'],
                'r_bond_cost': self.bonder_system.get_current_bond_cost()
            })
        
        results['current_interest_rate'] = self.current_moet_interest_rate
        results['reserve_ratio'] = self.redeemer.reserve_state.get_reserve_ratio(self.total_supply)
        
        return results
    
    def _calculate_moet_interest_rate(self) -> float:
        """Calculate MOET interest rate: r_MOET = r_floor + r_bond-cost"""
        if not self.enable_advanced_system:
            return 0.0
            
        r_floor = self.governance_params['r_floor']
        r_bond_cost = self.bonder_system.get_current_bond_cost()
        
        # Add peg stability adjustments if needed (future enhancement)
        peg_adjustment = 0.0  # For now, keep at peg
        
        return r_floor + r_bond_cost + peg_adjustment
    
    def get_current_interest_rate(self) -> float:
        """Get current MOET borrowing interest rate"""
        return self.current_moet_interest_rate
    
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
        base_state = {
            "total_supply": self.total_supply,
            "current_price": self.current_price,
            "target_price": self.target_price,
            "is_peg_stable": self.is_peg_stable(),
            "peg_deviation": self.get_peg_deviation(),
            "stability_bands": self.stability_bands,
            "stability_action": self.calculate_stability_action(),
            "advanced_system_enabled": self.enable_advanced_system,
            "current_interest_rate": self.current_moet_interest_rate
        }
        
        if self.enable_advanced_system:
            base_state.update({
                "governance_params": self.governance_params,
                "reserve_state": self.redeemer.get_state() if self.redeemer else None,
                "bonder_system": {
                    "current_bond_cost_ema": self.bonder_system.get_current_bond_cost(),
                    "auction_history_count": len(self.bonder_system.auction_history),
                    "pending_auction": self.bonder_system.pending_auction is not None,
                    "recent_auctions": [
                        {
                            'timestamp': auction.timestamp,
                            'final_apr': auction.final_apr,
                            'amount_filled': auction.amount_filled,
                            'filled_completely': auction.filled_completely
                        }
                        for auction in self.bonder_system.auction_history[-5:]  # Last 5 auctions
                    ]
                } if self.bonder_system else None,
                "interest_rate_components": {
                    "r_floor": self.governance_params['r_floor'],
                    "r_bond_cost": self.bonder_system.get_current_bond_cost() if self.bonder_system else 0.0,
                    "total_rate": self.current_moet_interest_rate
                }
            })
        
        return base_state
    
    def get_auction_summary(self) -> Dict:
        """Get summary of bond auction activity"""
        if not self.enable_advanced_system or not self.bonder_system:
            return {'advanced_system_enabled': False}
        
        auctions = self.bonder_system.auction_history
        if not auctions:
            return {
                'advanced_system_enabled': True,
                'total_auctions': 0,
                'total_raised': 0.0,
                'avg_apr': 0.0,
                'avg_duration': 0.0,
                'fill_rate': 0.0,
                'current_ema_cost': self.bonder_system.get_current_bond_cost()
            }
        
        return {
            'advanced_system_enabled': True,
            'total_auctions': len(auctions),
            'total_raised': sum(a.amount_filled for a in auctions),
            'avg_apr': sum(a.final_apr for a in auctions) / len(auctions),
            'avg_duration': sum(a.auction_duration_minutes for a in auctions) / len(auctions),
            'fill_rate': sum(1 for a in auctions if a.filled_completely) / len(auctions),
            'current_ema_cost': self.bonder_system.get_current_bond_cost()
        }