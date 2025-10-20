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
    
    @property
    def ideal_usdc_ratio(self) -> float:
        """Ideal USDC weight (50%)"""
        return 0.50
    
    @property
    def ideal_usdf_ratio(self) -> float:
        """Ideal USDF weight (50%)"""
        return 0.50
    
    @property
    def current_usdc_ratio(self) -> float:
        """Current USDC weight in reserves"""
        if self.total_reserves <= 0:
            return 0.50  # Default to ideal if no reserves
        return self.usdc_balance / self.total_reserves
    
    @property
    def current_usdf_ratio(self) -> float:
        """Current USDF weight in reserves"""
        if self.total_reserves <= 0:
            return 0.50  # Default to ideal if no reserves
        return self.usdf_balance / self.total_reserves
    
    def get_weight_deviation(self) -> float:
        """Calculate current deviation from ideal 50/50 weights"""
        usdc_deviation = abs(self.current_usdc_ratio - self.ideal_usdc_ratio)
        usdf_deviation = abs(self.current_usdf_ratio - self.ideal_usdf_ratio)
        return max(usdc_deviation, usdf_deviation)  # Maximum deviation
    
    def calculate_post_deposit_deviation(self, usdc_deposit: float, usdf_deposit: float) -> float:
        """Calculate weight deviation after a potential deposit"""
        new_usdc_balance = self.usdc_balance + usdc_deposit
        new_usdf_balance = self.usdf_balance + usdf_deposit
        new_total = new_usdc_balance + new_usdf_balance
        
        if new_total <= 0:
            return 0.0
            
        new_usdc_ratio = new_usdc_balance / new_total
        new_usdf_ratio = new_usdf_balance / new_total
        
        usdc_deviation = abs(new_usdc_ratio - self.ideal_usdc_ratio)
        usdf_deviation = abs(new_usdf_ratio - self.ideal_usdf_ratio)
        return max(usdc_deviation, usdf_deviation)
    
    def calculate_post_withdrawal_deviation(self, usdc_withdrawal: float, usdf_withdrawal: float) -> float:
        """Calculate weight deviation after a potential withdrawal"""
        new_usdc_balance = max(0, self.usdc_balance - usdc_withdrawal)
        new_usdf_balance = max(0, self.usdf_balance - usdf_withdrawal)
        new_total = new_usdc_balance + new_usdf_balance
        
        if new_total <= 0:
            return 0.0
            
        new_usdc_ratio = new_usdc_balance / new_total
        new_usdf_ratio = new_usdf_balance / new_total
        
        usdc_deviation = abs(new_usdc_ratio - self.ideal_usdc_ratio)
        usdf_deviation = abs(new_usdf_ratio - self.ideal_usdf_ratio)
        return max(usdc_deviation, usdf_deviation)
    
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
        
        # Hourly auction parameters
        self.last_auction_hour = -1  # Track which hour we last ran an auction
        
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
    
    def should_run_hourly_auction(self, reserve_state: ReserveState, total_moet_supply: float, current_minute: int) -> bool:
        """Determine if we should run an hourly bond auction (deficit-based trigger)"""
        current_hour = current_minute // 60  # Convert minutes to hours
        
        # Check if it's a new hour and we haven't run this hour
        is_new_hour = current_hour > self.last_auction_hour
        
        # Check if there's a deficit that needs funding
        deficit = reserve_state.get_reserve_deficit(total_moet_supply)
        has_deficit = deficit > 100.0  # Minimum $100 deficit to trigger auction
        
        return is_new_hour and has_deficit
    
    def start_bond_auction(self, reserve_state: ReserveState, total_moet_supply: float, 
                          current_minute: int) -> BondAuction:
        """Start a new bond auction with pure deficit-based pricing"""
        deficit = reserve_state.get_reserve_deficit(total_moet_supply)
        bond_apr = self.calculate_bond_apr(reserve_state, total_moet_supply)
        
        # Pure deficit-based APR (no benchmark or premium)
        starting_apr = bond_apr
        
        auction = BondAuction(
            timestamp=current_minute,
            target_amount=deficit,
            starting_apr=starting_apr,
            final_apr=starting_apr,  # Will be updated each minute based on deficit
            amount_filled=0.0,
            bond_price=1.0 / (1.0 + starting_apr / 365),  # Overnight pricing
            auction_duration_minutes=0,
            filled_completely=False
        )
        
        self.pending_auction = auction
        return auction
    
    def process_hourly_auction(self, current_minute: int, market_conditions: Dict, 
                              reserve_state: ReserveState, total_moet_supply: float) -> Optional[BondAuction]:
        """Process hourly bond auction"""
        current_hour = current_minute // 60
        
        # Check if we should start a new auction
        if self.should_run_hourly_auction(reserve_state, total_moet_supply, current_minute):
            # Start new hourly auction
            auction = self.start_bond_auction(reserve_state, total_moet_supply, current_minute)
            self.last_auction_hour = current_hour
            
            # Calculate fill percentage based on APR attractiveness
            fill_percentage = self._calculate_hourly_fill_percentage(auction, market_conditions)
            
            # Apply the fill
            auction.amount_filled = auction.target_amount * fill_percentage
            auction.filled_completely = (fill_percentage >= 0.99)
            auction.auction_duration_minutes = 60  # One hour duration
            auction.bond_price = 1.0 / (1.0 + auction.final_apr / 365)
            
            # Record the auction
            self.auction_history.append(auction)
            self.bond_apr_history.append(auction.final_apr)
            
            # Update EMA with the realized APR
            self._update_bond_cost_ema(auction.final_apr)
            
            return auction
            
        return None  # No auction this hour
    
    def _calculate_hourly_fill_percentage(self, auction: BondAuction, market_conditions: Dict) -> float:
        """Calculate what percentage of hourly bond auction gets filled based on APR"""
        
        # Base fill rates based on APR attractiveness (lower than daily since more frequent)
        apr = auction.final_apr
        
        if apr >= 0.15:      # 15%+ APR
            base_fill = 0.60  # 60% fill - very attractive
        elif apr >= 0.10:    # 10-15% APR  
            base_fill = 0.45  # 45% fill - attractive
        elif apr >= 0.05:    # 5-10% APR
            base_fill = 0.25  # 25% fill - moderate
        elif apr >= 0.025:   # 2.5-5% APR
            base_fill = 0.10  # 10% fill - low interest
        else:                # <2.5% APR
            base_fill = 0.03  # 3% fill - minimal interest
        
        # Market stress can increase fill rates
        stress_factor = market_conditions.get('stress_level', 1.0)
        
        # Add some randomness (±20% variation)
        import random
        randomness = random.uniform(0.8, 1.2)
        
        final_fill = min(1.0, base_fill * stress_factor * randomness)
        return final_fill
    
    def _update_bond_cost_ema(self, new_apr: float):
        """Update EMA of bond costs with 7-day half-life (hourly updates)"""
        if not self.bond_apr_history:
            self.current_bond_cost_ema = new_apr
            return
            
        # 7-day half-life = 7 * 24 = 168 hours (since we update hourly now)
        # EMA alpha = 1 - exp(-ln(2) / half_life_periods)
        half_life_hours = self.governance.get('ema_half_life_days', 7) * 24
        alpha = 1 - math.exp(-math.log(2) / half_life_hours)
        
        self.current_bond_cost_ema = alpha * new_apr + (1 - alpha) * self.current_bond_cost_ema
    
    def get_current_bond_cost(self) -> float:
        """Get current EMA of bond costs for interest calculation"""
        return self.current_bond_cost_ema


@dataclass
class DepositResult:
    """Result of a deposit operation"""
    moet_minted: float
    base_fee: float
    imbalance_fee: float
    total_fee: float
    fee_percentage: float
    is_balanced: bool
    post_deviation: float

@dataclass
class RedemptionResult:
    """Result of a redemption operation"""
    usdc_received: float
    usdf_received: float
    base_fee: float
    imbalance_fee: float
    total_fee: float
    fee_percentage: float
    is_proportional: bool
    post_deviation: float

class RedeemerContract:
    """Enhanced Redeemer managing MOET backing reserves with dynamic fee structure"""
    
    def __init__(self, initial_usdc: float = 0.0, initial_usdf: float = 0.0, 
                 target_reserves_ratio: float = 0.30):
        self.reserve_state = ReserveState(
            usdc_balance=initial_usdc,
            usdf_balance=initial_usdf,
            target_reserves_ratio=target_reserves_ratio
        )
        
        # Fee structure parameters (governance controlled)
        self.fee_params = {
            'balanced_deposit_fee': 0.0001,    # 0.01% for balanced deposits
            'imbalanced_deposit_fee': 0.0002,  # 0.02% base for imbalanced deposits
            'proportional_redemption_fee': 0.0,  # 0% for proportional redemptions
            'single_asset_redemption_fee': 0.0002,  # 0.02% base for single-asset redemptions
            'imbalance_scale_k': 0.005,        # K = 50 bps scale factor
            'imbalance_convexity_gamma': 2.0,  # γ = 2.0 for quadratic scaling
            'tolerance_band': 0.02             # 2% tolerance before imbalance fees kick in
        }
        
        # Fee revenue tracking
        self.total_fees_collected = 0.0
        self.fee_history = []
        
    def calculate_imbalance_fee(self, post_deviation: float, deposit_amount: float) -> float:
        """Calculate imbalance fee: K * max(0, Δw(post) - Δw(tol))^γ"""
        K = self.fee_params['imbalance_scale_k']
        gamma = self.fee_params['imbalance_convexity_gamma']
        tolerance = self.fee_params['tolerance_band']
        
        # Only charge imbalance fee if deviation exceeds tolerance
        excess_deviation = max(0.0, post_deviation - tolerance)
        
        if excess_deviation <= 0:
            return 0.0
            
        # fee(imb) = K * max(0, Δw(post) - Δw(tol))^γ
        fee_rate = K * (excess_deviation ** gamma)
        return deposit_amount * fee_rate
    
    def estimate_deposit_fee(self, usdc_amount: float, usdf_amount: float) -> Dict:
        """Estimate fees for a potential deposit without executing it"""
        total_deposit = usdc_amount + usdf_amount
        
        if total_deposit <= 0:
            return {
                'total_fee': 0.0,
                'base_fee': 0.0,
                'imbalance_fee': 0.0,
                'fee_percentage': 0.0,
                'is_balanced': True,
                'post_deviation': 0.0
            }
        
        # Check if deposit is balanced (maintains 50/50 ratio)
        deposit_usdc_ratio = usdc_amount / total_deposit
        deposit_usdf_ratio = usdf_amount / total_deposit
        
        # Consider balanced if within 1% of ideal ratios
        is_balanced = (abs(deposit_usdc_ratio - 0.50) <= 0.01 and 
                      abs(deposit_usdf_ratio - 0.50) <= 0.01)
        
        # Calculate post-transaction deviation
        post_deviation = self.reserve_state.calculate_post_deposit_deviation(usdc_amount, usdf_amount)
        
        if is_balanced:
            base_fee = total_deposit * self.fee_params['balanced_deposit_fee']
            imbalance_fee = 0.0
        else:
            base_fee = total_deposit * self.fee_params['imbalanced_deposit_fee']
            imbalance_fee = self.calculate_imbalance_fee(post_deviation, total_deposit)
        
        total_fee = base_fee + imbalance_fee
        fee_percentage = total_fee / total_deposit if total_deposit > 0 else 0.0
        
        return {
            'total_fee': total_fee,
            'base_fee': base_fee,
            'imbalance_fee': imbalance_fee,
            'fee_percentage': fee_percentage,
            'is_balanced': is_balanced,
            'post_deviation': post_deviation
        }
    
    def deposit_assets_for_moet(self, usdc_amount: float, usdf_amount: float) -> DepositResult:
        """Deposit USDC/USDF to mint MOET with dynamic fee structure"""
        total_deposit = usdc_amount + usdf_amount
        
        if total_deposit <= 0:
            return DepositResult(0.0, 0.0, 0.0, 0.0, 0.0, True, 0.0)
        
        # Calculate fees
        fee_estimate = self.estimate_deposit_fee(usdc_amount, usdf_amount)
        
        # Execute the deposit
        self.reserve_state.usdc_balance += usdc_amount
        self.reserve_state.usdf_balance += usdf_amount
        
        # MOET minted = deposit amount - fees (1:1 backing minus fees)
        moet_minted = total_deposit - fee_estimate['total_fee']
        
        # Track fee revenue
        self.total_fees_collected += fee_estimate['total_fee']
        self.fee_history.append({
            'type': 'deposit',
            'amount': total_deposit,
            'fee': fee_estimate['total_fee'],
            'is_balanced': fee_estimate['is_balanced']
        })
        
        return DepositResult(
            moet_minted=moet_minted,
            base_fee=fee_estimate['base_fee'],
            imbalance_fee=fee_estimate['imbalance_fee'],
            total_fee=fee_estimate['total_fee'],
            fee_percentage=fee_estimate['fee_percentage'],
            is_balanced=fee_estimate['is_balanced'],
            post_deviation=fee_estimate['post_deviation']
        )
    
    def estimate_redemption_fee(self, moet_amount: float, desired_asset: str = "proportional") -> Dict:
        """Estimate fees for a potential redemption without executing it"""
        if self.reserve_state.total_reserves < moet_amount:
            return {
                'error': 'Insufficient reserves',
                'total_fee': 0.0,
                'base_fee': 0.0,
                'imbalance_fee': 0.0,
                'fee_percentage': 0.0
            }
        
        if desired_asset == "proportional":
            # Proportional redemption has no fees
            return {
                'total_fee': 0.0,
                'base_fee': 0.0,
                'imbalance_fee': 0.0,
                'fee_percentage': 0.0,
                'is_proportional': True,
                'post_deviation': self.reserve_state.get_weight_deviation()
            }
        
        # Single-asset redemption
        base_fee = moet_amount * self.fee_params['single_asset_redemption_fee']
        
        # Calculate post-withdrawal deviation
        if desired_asset.upper() == "USDC":
            post_deviation = self.reserve_state.calculate_post_withdrawal_deviation(moet_amount, 0.0)
        elif desired_asset.upper() == "USDF":
            post_deviation = self.reserve_state.calculate_post_withdrawal_deviation(0.0, moet_amount)
        else:
            return {'error': 'Invalid asset. Must be USDC, USDF, or proportional'}
        
        # Calculate imbalance fee
        imbalance_fee = self.calculate_imbalance_fee(post_deviation, moet_amount)
        
        total_fee = base_fee + imbalance_fee
        fee_percentage = total_fee / moet_amount if moet_amount > 0 else 0.0
        
        return {
            'total_fee': total_fee,
            'base_fee': base_fee,
            'imbalance_fee': imbalance_fee,
            'fee_percentage': fee_percentage,
            'is_proportional': False,
            'post_deviation': post_deviation
        }
    
    def redeem_moet_for_assets(self, moet_amount: float, desired_asset: str = "proportional") -> RedemptionResult:
        """Redeem MOET for underlying assets with dynamic fee structure"""
        if self.reserve_state.total_reserves < moet_amount:
            return RedemptionResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0.0)
        
        # Calculate fees
        fee_estimate = self.estimate_redemption_fee(moet_amount, desired_asset)
        
        if 'error' in fee_estimate:
            return RedemptionResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0.0)
        
        # Net amount after fees
        net_redemption = moet_amount - fee_estimate['total_fee']
        
        if desired_asset == "proportional":
            # Proportional redemption
            total_reserves = self.reserve_state.total_reserves
            usdc_ratio = self.reserve_state.usdc_balance / total_reserves
            usdf_ratio = self.reserve_state.usdf_balance / total_reserves
            
            usdc_received = net_redemption * usdc_ratio
            usdf_received = net_redemption * usdf_ratio
            
            # Update reserves
            self.reserve_state.usdc_balance -= usdc_received
            self.reserve_state.usdf_balance -= usdf_received
            
        elif desired_asset.upper() == "USDC":
            # Single-asset USDC redemption
            usdc_received = net_redemption
            usdf_received = 0.0
            
            # Update reserves (remove USDC, keep USDF)
            self.reserve_state.usdc_balance -= usdc_received
            
        elif desired_asset.upper() == "USDF":
            # Single-asset USDF redemption
            usdc_received = 0.0
            usdf_received = net_redemption
            
            # Update reserves (keep USDC, remove USDF)
            self.reserve_state.usdf_balance -= usdf_received
            
        else:
            return RedemptionResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0.0)
        
        # Track fee revenue
        self.total_fees_collected += fee_estimate['total_fee']
        self.fee_history.append({
            'type': 'redemption',
            'amount': moet_amount,
            'fee': fee_estimate['total_fee'],
            'desired_asset': desired_asset
        })
        
        return RedemptionResult(
            usdc_received=usdc_received,
            usdf_received=usdf_received,
            base_fee=fee_estimate['base_fee'],
            imbalance_fee=fee_estimate['imbalance_fee'],
            total_fee=fee_estimate['total_fee'],
            fee_percentage=fee_estimate['fee_percentage'],
            is_proportional=(desired_asset == "proportional"),
            post_deviation=fee_estimate['post_deviation']
        )
    
    def add_reserves(self, usdc_amount: float = 0.0, usdf_amount: float = 0.0):
        """Add reserves from bond auction proceeds (maintains existing interface)"""
        self.reserve_state.usdc_balance += usdc_amount
        self.reserve_state.usdf_balance += usdf_amount
    
    def process_redemption(self, moet_amount: float) -> bool:
        """Legacy proportional redemption method (maintains existing interface)"""
        result = self.redeem_moet_for_assets(moet_amount, "proportional")
        return result.usdc_received > 0 or result.usdf_received > 0
    
    def get_current_pool_weights(self) -> Dict:
        """Get current USDC/USDF pool composition"""
        return {
            'usdc_ratio': self.reserve_state.current_usdc_ratio,
            'usdf_ratio': self.reserve_state.current_usdf_ratio,
            'ideal_usdc_ratio': self.reserve_state.ideal_usdc_ratio,
            'ideal_usdf_ratio': self.reserve_state.ideal_usdf_ratio,
            'weight_deviation': self.reserve_state.get_weight_deviation()
        }
    
    def get_optimal_deposit_ratio(self, total_amount: float) -> Dict:
        """Calculate optimal deposit composition to minimize fees"""
        return {
            'usdc_amount': total_amount * 0.50,
            'usdf_amount': total_amount * 0.50,
            'estimated_fee': total_amount * self.fee_params['balanced_deposit_fee'],
            'fee_percentage': self.fee_params['balanced_deposit_fee']
        }
    
    def get_state(self) -> Dict:
        """Get comprehensive redeemer state"""
        return {
            'usdc_balance': self.reserve_state.usdc_balance,
            'usdf_balance': self.reserve_state.usdf_balance,
            'total_reserves': self.reserve_state.total_reserves,
            'target_reserves_ratio': self.reserve_state.target_reserves_ratio,
            'current_weights': self.get_current_pool_weights(),
            'fee_params': self.fee_params,
            'total_fees_collected': self.total_fees_collected,
            'fee_history_count': len(self.fee_history)
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
                'target_reserves_ratio': 0.10,  # 10% target reserves (updated)
                'ema_half_life_days': 0.5,  # 12-hour EMA half-life (much more responsive)
            }
            
            # Initialize sophisticated components
            self.bonder_system = BonderSystem(self.governance_params)
            self.redeemer = RedeemerContract(
                target_reserves_ratio=self.governance_params['target_reserves_ratio']
            )
            
            # Interest rate tracking
            self.current_moet_interest_rate = self.governance_params['r_floor']  # Start with floor rate
            self.interest_rate_history = []
            
            # Enhanced tracking for JSON results
            self.moet_rate_history = []      # Minute-by-minute MOET interest rates
            self.bond_apr_history_detailed = []  # Minute-by-minute bond APRs
            self.reserve_history = []        # Target vs actual reserves over time
            self.deficit_history = []        # Reserve deficit over time
            
        else:
            # Legacy simple system
            self.bonder_system = None
            self.redeemer = None
            self.current_moet_interest_rate = 0.0
    
    def initialize_reserves(self, initial_moet_debt: float):
        """Initialize reserves at 8% of initial MOET debt (50/50 USDC/USDF)"""
        if self.enable_advanced_system and self.redeemer:
            initial_reserves = initial_moet_debt * 0.08  # 8% backing (creates immediate deficit)
            usdc_amount = initial_reserves * 0.50        # 4% USDC
            usdf_amount = initial_reserves * 0.50        # 4% USDF
            
            # Reset reserves to exact amounts (don't add to existing)
            self.redeemer.reserve_state.usdc_balance = usdc_amount
            self.redeemer.reserve_state.usdf_balance = usdf_amount
    
    def mint_from_deposit(self, usdc_amount: float, usdf_amount: float) -> DepositResult:
        """Mint MOET from USDC/USDF deposit through enhanced Redeemer"""
        if not self.enable_advanced_system or not self.redeemer:
            # Fallback to simple 1:1 minting for legacy system
            total_deposit = usdc_amount + usdf_amount
            self.total_supply += total_deposit
            return DepositResult(total_deposit, 0.0, 0.0, 0.0, 0.0, True, 0.0)
        
        # Use enhanced Redeemer for deposit processing
        deposit_result = self.redeemer.deposit_assets_for_moet(usdc_amount, usdf_amount)
        
        # Update total supply with minted amount
        self.total_supply += deposit_result.moet_minted
        
        return deposit_result
    
    def redeem_for_assets(self, moet_amount: float, desired_asset: str = "proportional") -> RedemptionResult:
        """Redeem MOET for underlying assets through enhanced Redeemer"""
        if not self.enable_advanced_system or not self.redeemer:
            # Fallback to simple 1:1 burning for legacy system
            actual_burn = min(moet_amount, self.total_supply)
            self.total_supply -= actual_burn
            return RedemptionResult(actual_burn * 0.5, actual_burn * 0.5, 0.0, 0.0, 0.0, 0.0, True, 0.0)
        
        # Use enhanced Redeemer for redemption processing
        redemption_result = self.redeemer.redeem_moet_for_assets(moet_amount, desired_asset)
        
        # Update total supply (burn the redeemed MOET)
        if redemption_result.usdc_received > 0 or redemption_result.usdf_received > 0:
            self.total_supply -= moet_amount
        
        return redemption_result
    
    def mint(self, amount: float) -> float:
        """Legacy 1:1 minting method (maintains existing interface)"""
        if amount <= 0:
            return 0.0
        
        self.total_supply += amount
        return amount  # Returns full amount minted
    
    def burn(self, amount: float) -> float:
        """Legacy 1:1 burning method (maintains existing interface)"""
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
        
        # Process hourly bond auctions (deficit-based trigger)
        completed_auction = self.bonder_system.process_hourly_auction(
            current_minute, market_conditions, self.redeemer.reserve_state, self.total_supply
        )
        
        if completed_auction:
            results['bond_auction_triggered'] = True
            results['new_auction'] = {
                'target_amount': completed_auction.target_amount,
                'starting_apr': completed_auction.starting_apr,
                'timestamp': completed_auction.timestamp
            }
        if completed_auction:
            results['bond_auction_completed'] = True
            results['completed_auction'] = {
                'amount_filled': completed_auction.amount_filled,
                'target_amount': completed_auction.target_amount,
                'final_apr': completed_auction.final_apr,
                'duration_minutes': completed_auction.auction_duration_minutes,
                'fill_percentage': completed_auction.amount_filled / completed_auction.target_amount if completed_auction.target_amount > 0 else 0,
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
        
        # Enhanced tracking for JSON results
        current_bond_apr = self.bonder_system.calculate_bond_apr(self.redeemer.reserve_state, self.total_supply)
        target_reserves = self.total_supply * self.governance_params['target_reserves_ratio']
        actual_reserves = self.redeemer.reserve_state.total_reserves
        deficit = self.redeemer.reserve_state.get_reserve_deficit(self.total_supply)
        
        # Track minute-by-minute data
        self.moet_rate_history.append({
            'minute': current_minute,
            'moet_interest_rate': self.current_moet_interest_rate,
            'r_floor': self.governance_params['r_floor'],
            'r_bond_cost': self.bonder_system.get_current_bond_cost()
        })
        
        self.bond_apr_history_detailed.append({
            'minute': current_minute,
            'bond_apr': current_bond_apr,
            'deficit_ratio': current_bond_apr  # Same as bond APR in our system
        })
        
        self.reserve_history.append({
            'minute': current_minute,
            'target_reserves': target_reserves,
            'actual_reserves': actual_reserves,
            'reserve_ratio': actual_reserves / self.total_supply if self.total_supply > 0 else 0,
            'target_ratio': self.governance_params['target_reserves_ratio']
        })
        
        self.deficit_history.append({
            'minute': current_minute,
            'deficit': deficit,
            'deficit_ratio': deficit / target_reserves if target_reserves > 0 else 0
        })
        
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
                "redeemer_system": {
                    "total_fees_collected": self.redeemer.total_fees_collected if self.redeemer else 0.0,
                    "fee_history_count": len(self.redeemer.fee_history) if self.redeemer else 0,
                    "current_pool_weights": self.redeemer.get_current_pool_weights() if self.redeemer else None,
                    "fee_parameters": self.redeemer.fee_params if self.redeemer else None
                } if self.redeemer else None,
                "interest_rate_components": {
                    "r_floor": self.governance_params['r_floor'],
                    "r_bond_cost": self.bonder_system.get_current_bond_cost() if self.bonder_system else 0.0,
                    "total_rate": self.current_moet_interest_rate
                },
                # Enhanced tracking data for JSON results
                "tracking_data": {
                    "moet_rate_history": self.moet_rate_history,
                    "bond_apr_history": self.bond_apr_history_detailed,
                    "reserve_history": self.reserve_history,
                    "deficit_history": self.deficit_history
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