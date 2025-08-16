#!/usr/bin/env python3
"""
Tidal Protocol Lending Simulation with Uniswap V3-Style Liquidity Pools

This script simulates the Tidal lending protocol, including:
- Multi-asset lending with collateral factors
- Interest rate model with kinked rates
- Uniswap V3-style liquidity pool math
- Monte Carlo simulation for stress testing
- Debt cap calculations using Ebisu-style methodology

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PROTOCOL CONFIGURATION AND CONSTANTS
# =============================================================================

class Asset(Enum):
    """Supported collateral assets"""
    ETH = "ETH"
    BTC = "BTC"
    FLOW = "FLOW"
    USDC = "USDC"
    MOET = "MOET"  # Protocol stablecoin

# Protocol Parameters
RESERVE_FACTOR = 0.15  # 15% of interest goes to protocol revenue
LP_REWARDS_FACTOR = 0.50  # 50% of protocol revenue distributed to LP pools

# Interest Rate Model Parameters (per-block)
BASE_RATE_PER_BLOCK = 0
MULTIPLIER_PER_BLOCK = 11415525114
JUMP_PER_BLOCK = 253678335870
KINK = 8e17  # 0.80 utilization, scaled 1e18
BLOCKS_PER_MINUTE = 30
BLOCKS_PER_DAY = 43200
BLOCKS_PER_YEAR = 15768000

# Starting Liquidity Values (Total: $14,000,000)
INITIAL_LIQUIDITY = {
    Asset.ETH: 7_000_000,    # $7M
    Asset.BTC: 3_500_000,    # $3.5M
    Asset.FLOW: 2_100_000,   # $2.1M
    Asset.USDC: 1_400_000    # $1.4M
}

# Initial Asset Prices (USD)
INITIAL_PRICES = {
    Asset.ETH: 3000.0,
    Asset.BTC: 45000.0,
    Asset.FLOW: 0.50,
    Asset.USDC: 1.0,
    Asset.MOET: 1.0
}

# Collateral Factors (how much of collateral value can be borrowed against)
COLLATERAL_FACTORS = {
    Asset.ETH: 0.75,
    Asset.BTC: 0.75,
    Asset.FLOW: 0.50,
    Asset.USDC: 0.90
}

# Target Health Factors
TARGET_HEALTH_FACTOR = 1.5

# Extreme Price Drop Percentages
EXTREME_PRICE_DROPS = {
    Asset.ETH: 0.15,    # -15%
    Asset.BTC: 0.15,    # -15%
    Asset.FLOW: 0.35,   # -35%
    Asset.USDC: 0.15    # -15%
}

# DEX Liquidity Allocation to Other Markets
DEX_LIQUIDITY_ALLOCATION = 0.35  # 35%

# Uniswap V3 Parameters
UNISWAP_V3_FEE_TIER = 0.003  # 0.3%
MAX_SLIPPAGE_THRESHOLD = 0.04  # 4%

@dataclass
class ProtocolState:
    """Current state of the protocol"""
    asset_prices: Dict[Asset, float]
    total_supplied: Dict[Asset, float]
    total_borrowed: Dict[Asset, float]
    utilization: Dict[Asset, float]
    borrow_rates: Dict[Asset, float]
    supply_rates: Dict[Asset, float]
    health_factors: Dict[Asset, float]
    protocol_revenue: float
    lp_rewards: float
    debt_cap: float

class UniswapV3Pool:
    """Simplified Uniswap V3 pool simulation"""
    
    def __init__(self, asset: Asset, total_liquidity: float, price: float):
        self.asset = asset
        self.total_liquidity = total_liquidity
        self.current_price = price
        self.liquidity_distribution = self._create_liquidity_distribution()
    
    def _create_liquidity_distribution(self) -> Dict[str, float]:
        """Create normal distribution of liquidity around current price"""
        # Assume liquidity is concentrated within 1 standard deviation
        std_dev = self.current_price * 0.1  # 10% standard deviation
        
        # Create price buckets with normal distribution
        price_range = np.linspace(
            self.current_price - 2*std_dev,
            self.current_price + 2*std_dev,
            21
        )
        
        # Normal distribution weights
        weights = stats.norm.pdf(price_range, self.current_price, std_dev)
        weights = weights / weights.sum()  # Normalize
        
        return {
            'prices': price_range,
            'weights': weights,
            'liquidity': weights * self.total_liquidity
        }
    
    def calculate_slippage(self, amount_in: float, is_buy: bool = True) -> Tuple[float, float]:
        """
        Calculate slippage for a given trade amount
        
        Args:
            amount_in: Amount of asset to trade
            is_buy: True if buying the asset, False if selling
            
        Returns:
            Tuple of (amount_out, slippage_percentage)
        """
        if amount_in <= 0:
            return 0.0, 0.0
        
        # Simplified slippage calculation based on liquidity depth
        # In a real Uniswap V3 pool, this would use the constant product formula
        # with concentrated liquidity positions
        
        # Estimate available liquidity at current price
        current_liquidity = self.liquidity_distribution['liquidity'][10]  # Middle bucket
        
        # Calculate slippage based on trade size vs liquidity
        slippage = min(amount_in / (current_liquidity * 0.1), 0.20)  # Cap at 20%
        
        if is_buy:
            # Price impact for buying (price goes up)
            amount_out = amount_in * (1 - slippage)
        else:
            # Price impact for selling (price goes down)
            amount_out = amount_in * (1 + slippage)
        
        return amount_out, slippage
    
    def get_liquidation_capacity(self, max_slippage: float = MAX_SLIPPAGE_THRESHOLD) -> float:
        """
        Calculate how much of the asset can be liquidated before exceeding max slippage
        
        Args:
            max_slippage: Maximum acceptable slippage (default 4%)
            
        Returns:
            Maximum amount that can be liquidated
        """
        # Find the amount that would cause max_slippage
        current_liquidity = self.liquidity_distribution['liquidity'][10]
        
        # Reverse engineer the trade size from slippage
        max_trade_size = current_liquidity * 0.1 * max_slippage
        
        return max_trade_size

class TidalProtocol:
    """Main Tidal Protocol simulation class"""
    
    def __init__(self):
        self.asset_prices = INITIAL_PRICES.copy()
        self.total_supplied = INITIAL_LIQUIDITY.copy()
        self.total_borrowed = {asset: 0.0 for asset in Asset if asset != Asset.MOET}
        self.pools = self._initialize_pools()
        self.protocol_revenue = 0.0
        self.lp_rewards = 0.0
        
    def _initialize_pools(self) -> Dict[Asset, UniswapV3Pool]:
        """Initialize Uniswap V3-style pools for each asset"""
        pools = {}
        for asset in Asset:
            if asset != Asset.MOET:
                # Create MOET/Asset pool
                pool_liquidity = INITIAL_LIQUIDITY[asset]
                pools[asset] = UniswapV3Pool(asset, pool_liquidity, self.asset_prices[asset])
        return pools
    
    def calculate_utilization(self, asset: Asset) -> float:
        """Calculate utilization rate for an asset"""
        if self.total_supplied[asset] == 0:
            return 0.0
        return self.total_borrowed[asset] / self.total_supplied[asset]
    
    def calculate_borrow_rate(self, asset: Asset) -> float:
        """Calculate borrow rate using kinked interest rate model"""
        utilization = self.calculate_utilization(asset)
        
        if utilization <= KINK / 1e18:
            # Below kink: linear rate
            rate = BASE_RATE_PER_BLOCK + (utilization * MULTIPLIER_PER_BLOCK / 1e18)
        else:
            # Above kink: jump rate
            base_rate = BASE_RATE_PER_BLOCK + (KINK * MULTIPLIER_PER_BLOCK / 1e18**2)
            jump_rate = (utilization - KINK / 1e18) * JUMP_PER_BLOCK / 1e18
            rate = base_rate + jump_rate
        
        # Convert to annual rate
        annual_rate = rate * BLOCKS_PER_YEAR / 1e18
        return annual_rate
    
    def calculate_supply_rate(self, asset: Asset) -> float:
        """Calculate supply rate based on borrow rate and utilization"""
        borrow_rate = self.calculate_borrow_rate(asset)
        utilization = self.calculate_utilization(asset)
        
        # Supply rate = borrow rate * utilization * (1 - reserve factor)
        supply_rate = borrow_rate * utilization * (1 - RESERVE_FACTOR)
        return supply_rate
    
    def calculate_effective_collateral_value(self, asset: Asset, balance: float) -> float:
        """Calculate effective collateral value"""
        return balance * self.asset_prices[asset] * COLLATERAL_FACTORS[asset]
    
    def calculate_borrow_limit(self, effective_collateral_value: float) -> float:
        """Calculate borrow limit based on collateral and health factor"""
        return effective_collateral_value / TARGET_HEALTH_FACTOR
    
    def calculate_health_factor(self, effective_collateral_value: float, borrowed_value: float) -> float:
        """Calculate health factor for a position"""
        if borrowed_value == 0:
            return float('inf')
        return effective_collateral_value / borrowed_value
    
    def calculate_protocol_revenue(self) -> float:
        """Calculate monthly protocol revenue"""
        total_borrowed_value = sum(
            self.total_borrowed[asset] * self.asset_prices[asset]
            for asset in self.total_borrowed
        )
        
        # Use average borrow rate across all assets
        avg_borrow_rate = np.mean([
            self.calculate_borrow_rate(asset) 
            for asset in self.total_borrowed
        ])
        
        monthly_revenue = total_borrowed_value * avg_borrow_rate * RESERVE_FACTOR / 12
        return monthly_revenue
    
    def calculate_lp_rewards(self) -> float:
        """Calculate LP rewards (50% of protocol revenue)"""
        protocol_revenue = self.calculate_protocol_revenue()
        return protocol_revenue * LP_REWARDS_FACTOR
    
    def calculate_debt_cap(self) -> float:
        """
        Calculate debt cap using Ebisu-style methodology
        
        Formula: Debt Cap = A * B * C
        Where:
        A: Amount able to be profitably liquidated via DEX
        B: Allocation of DEX liquidity to other lending markets (35%)
        C: Percentage of collateral that is underwater in extreme drop
        """
        
        # Calculate A: Liquidation capacity across all pools
        total_liquidation_capacity = 0
        for asset, pool in self.pools.items():
            liquidation_capacity = pool.get_liquidation_capacity()
            total_liquidation_capacity += liquidation_capacity * self.asset_prices[asset]
        
        # B: DEX liquidity allocation (35%)
        dex_allocation = DEX_LIQUIDITY_ALLOCATION
        
        # C: Underwater collateral percentage (weighted average)
        total_collateral_value = sum(
            self.total_supplied[asset] * self.asset_prices[asset]
            for asset in self.total_supplied
        )
        
        underwater_percentage = 0
        for asset in self.total_supplied:
            if asset in EXTREME_PRICE_DROPS:
                asset_value = self.total_supplied[asset] * self.asset_prices[asset]
                weight = asset_value / total_collateral_value
                underwater_percentage += weight * EXTREME_PRICE_DROPS[asset]
        
        # Calculate debt cap
        debt_cap = total_liquidation_capacity * dex_allocation * underwater_percentage
        
        return debt_cap
    
    def get_protocol_state(self) -> ProtocolState:
        """Get current protocol state"""
        utilization = {asset: self.calculate_utilization(asset) for asset in self.total_supplied}
        borrow_rates = {asset: self.calculate_borrow_rate(asset) for asset in self.total_supplied}
        supply_rates = {asset: self.calculate_supply_rate(asset) for asset in self.total_supplied}
        
        # Calculate health factors (simplified - using total values)
        health_factors = {}
        for asset in self.total_supplied:
            if asset != Asset.MOET:
                effective_collateral = self.calculate_effective_collateral_value(
                    asset, self.total_supplied[asset]
                )
                borrowed_value = self.total_borrowed[asset] * self.asset_prices[asset]
                health_factors[asset] = self.calculate_health_factor(
                    effective_collateral, borrowed_value
                )
        
        protocol_revenue = self.calculate_protocol_revenue()
        lp_rewards = self.calculate_lp_rewards()
        debt_cap = self.calculate_debt_cap()
        
        return ProtocolState(
            asset_prices=self.asset_prices.copy(),
            total_supplied=self.total_supplied.copy(),
            total_borrowed=self.total_borrowed.copy(),
            utilization=utilization,
            borrow_rates=borrow_rates,
            supply_rates=supply_rates,
            health_factors=health_factors,
            protocol_revenue=protocol_revenue,
            lp_rewards=lp_rewards,
            debt_cap=debt_cap
        )
    
    def apply_price_shock(self, asset: Asset, shock_percentage: float):
        """Apply a price shock to an asset"""
        self.asset_prices[asset] *= (1 + shock_percentage)
        
        # Update pool price
        if asset in self.pools:
            self.pools[asset].current_price = self.asset_prices[asset]
    
    def simulate_borrowing(self, asset: Asset, amount: float):
        """Simulate borrowing of an asset"""
        if asset in self.total_borrowed:
            self.total_borrowed[asset] += amount
    
    def simulate_supply(self, asset: Asset, amount: float):
        """Simulate supplying an asset"""
        if asset in self.total_supplied:
            self.total_supplied[asset] += amount

class MonteCarloSimulator:
    """Monte Carlo simulation for stress testing"""
    
    def __init__(self, protocol: TidalProtocol, n_simulations: int = 1000):
        self.protocol = protocol
        self.n_simulations = n_simulations
        self.results = []
    
    def run_simulation(self) -> List[ProtocolState]:
        """Run Monte Carlo simulation"""
        print(f"Running {self.n_simulations} Monte Carlo simulations...")
        
        for i in range(self.n_simulations):
            # Create a copy of the protocol for this simulation
            sim_protocol = TidalProtocol()
            
            # Apply random price shocks
            for asset in [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]:
                # Generate random shock based on extreme drop scenarios
                if asset == Asset.FLOW:
                    # FLOW has higher volatility (-35% max)
                    shock = np.random.normal(-0.175, 0.1)  # Mean -17.5%, std 10%
                else:
                    # Other assets have lower volatility (-15% max)
                    shock = np.random.normal(-0.075, 0.05)  # Mean -7.5%, std 5%
                
                # Clamp shock to reasonable bounds
                shock = max(-0.50, min(0.20, shock))  # Between -50% and +20%
                sim_protocol.apply_price_shock(asset, shock)
            
            # Simulate random changes in borrowing demand
            for asset in [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]:
                # Random utilization change
                utilization_change = np.random.normal(0, 0.1)  # Mean 0, std 10%
                utilization_change = max(-0.3, min(0.3, utilization_change))  # Clamp
                
                # Adjust borrowed amount to achieve target utilization
                target_utilization = sim_protocol.calculate_utilization(asset) + utilization_change
                target_utilization = max(0, min(0.95, target_utilization))  # Clamp
                
                if target_utilization > 0:
                    target_borrowed = target_utilization * sim_protocol.total_supplied[asset]
                    sim_protocol.total_borrowed[asset] = target_borrowed
            
            # Get final state
            final_state = sim_protocol.get_protocol_state()
            self.results.append(final_state)
            
            if (i + 1) % 100 == 0:
                print(f"Completed {i + 1}/{self.n_simulations} simulations")
        
        print("Simulation completed!")
        return self.results
    
    def generate_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics across all simulation runs"""
        if not self.results:
            raise ValueError("No simulation results available. Run simulation first.")
        
        # Extract key metrics
        metrics = {
            'protocol_revenue': [],
            'lp_rewards': [],
            'debt_cap': [],
            'avg_utilization': [],
            'avg_health_factor': [],
            'total_collateral_value': []
        }
        
        for result in self.results:
            metrics['protocol_revenue'].append(result.protocol_revenue)
            metrics['lp_rewards'].append(result.lp_rewards)
            metrics['debt_cap'].append(result.debt_cap)
            
            # Average utilization across assets
            avg_util = np.mean(list(result.utilization.values()))
            metrics['avg_utilization'].append(avg_util)
            
            # Average health factor across assets
            avg_hf = np.mean(list(result.health_factors.values()))
            metrics['avg_health_factor'].append(avg_hf)
            
            # Total collateral value
            total_collateral = sum(
                result.total_supplied[asset] * result.asset_prices[asset]
                for asset in result.total_supplied
            )
            metrics['total_collateral_value'].append(total_collateral)
        
        # Create DataFrame
        df = pd.DataFrame(metrics)
        
        # Calculate summary statistics
        summary = df.describe(percentiles=[0.05, 0.25, 0.75, 0.95])
        
        return summary
    
    def plot_results(self):
        """Create visualization plots of simulation results"""
        if not self.results:
            print("No results to plot. Run simulation first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Tidal Protocol Monte Carlo Simulation Results', fontsize=16)
        
        # Extract data
        protocol_revenue = [r.protocol_revenue for r in self.results]
        lp_rewards = [r.lp_rewards for r in self.results]
        debt_cap = [r.debt_cap for r in self.results]
        avg_utilization = [np.mean(list(r.utilization.values())) for r in self.results]
        avg_health_factor = [np.mean(list(r.health_factors.values())) for r in self.results]
        total_collateral = [
            sum(r.total_supplied[asset] * r.asset_prices[asset] for asset in r.total_supplied)
            for r in self.results
        ]
        
        # Plot 1: Protocol Revenue Distribution
        axes[0, 0].hist(protocol_revenue, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Protocol Revenue Distribution')
        axes[0, 0].set_xlabel('Monthly Revenue ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Plot 2: LP Rewards Distribution
        axes[0, 1].hist(lp_rewards, bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('LP Rewards Distribution')
        axes[0, 1].set_xlabel('Monthly LP Rewards ($)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: Debt Cap Distribution
        axes[0, 2].hist(debt_cap, bins=50, alpha=0.7, color='red')
        axes[0, 2].set_title('Debt Cap Distribution')
        axes[0, 2].set_xlabel('Debt Cap ($)')
        axes[0, 2].set_ylabel('Frequency')
        
        # Plot 4: Utilization vs Health Factor
        axes[1, 0].scatter(avg_utilization, avg_health_factor, alpha=0.6, color='purple')
        axes[1, 0].set_title('Utilization vs Health Factor')
        axes[1, 0].set_xlabel('Average Utilization')
        axes[1, 0].set_ylabel('Average Health Factor')
        
        # Plot 5: Total Collateral Value Distribution
        axes[1, 1].hist(total_collateral, bins=50, alpha=0.7, color='orange')
        axes[1, 1].set_title('Total Collateral Value Distribution')
        axes[1, 1].set_xlabel('Total Collateral Value ($)')
        axes[1, 1].set_ylabel('Frequency')
        
        # Plot 6: Price Impact Analysis
        price_changes = []
        for result in self.results:
            for asset in [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]:
                initial_price = INITIAL_PRICES[asset]
                final_price = result.asset_prices[asset]
                price_change = (final_price - initial_price) / initial_price
                price_changes.append(price_change)
        
        axes[1, 2].hist(price_changes, bins=50, alpha=0.7, color='brown')
        axes[1, 2].set_title('Asset Price Change Distribution')
        axes[1, 2].set_xlabel('Price Change (%)')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()

def print_protocol_state(state: ProtocolState):
    """Print formatted protocol state"""
    print("\n" + "="*80)
    print("TIDAL PROTOCOL STATE")
    print("="*80)
    
    print(f"\nASSET PRICES:")
    for asset, price in state.asset_prices.items():
        print(f"  {asset.value}: ${price:,.2f}")
    
    print(f"\nSUPPLY AND BORROWING:")
    for asset in [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]:
        supplied = state.total_supplied[asset]
        borrowed = state.total_borrowed[asset]
        utilization = state.utilization[asset]
        print(f"  {asset.value}:")
        print(f"    Supplied: ${supplied:,.2f}")
        print(f"    Borrowed: ${borrowed:,.2f}")
        print(f"    Utilization: {utilization:.2%}")
    
    print(f"\nINTEREST RATES:")
    for asset in [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]:
        borrow_rate = state.borrow_rates[asset]
        supply_rate = state.supply_rates[asset]
        print(f"  {asset.value}:")
        print(f"    Borrow APR: {borrow_rate:.2%}")
        print(f"    Supply APR: {supply_rate:.2%}")
    
    print(f"\nHEALTH FACTORS:")
    for asset, hf in state.health_factors.items():
        print(f"  {asset.value}: {hf:.2f}")
    
    print(f"\nPROTOCOL METRICS:")
    print(f"  Monthly Protocol Revenue: ${state.protocol_revenue:,.2f}")
    print(f"  Monthly LP Rewards: ${state.lp_rewards:,.2f}")
    print(f"  Debt Cap: ${state.debt_cap:,.2f}")
    
    print("="*80)

def main():
    """Main function to run the simulation"""
    print("Tidal Protocol Lending Simulation")
    print("="*50)
    
    # Initialize protocol
    protocol = TidalProtocol()
    
    # Get initial state
    initial_state = protocol.get_protocol_state()
    print("\nInitial Protocol State:")
    print_protocol_state(initial_state)
    
    # Run Monte Carlo simulation
    simulator = MonteCarloSimulator(protocol, n_simulations=1000)
    results = simulator.run_simulation()
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("MONTE CARLO SIMULATION SUMMARY STATISTICS")
    print("="*80)
    
    summary_stats = simulator.generate_summary_statistics()
    print(summary_stats)
    
    # Show some key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Protocol Revenue Analysis
    revenue_values = [r.protocol_revenue for r in results]
    print(f"Protocol Revenue:")
    print(f"  Mean: ${np.mean(revenue_values):,.2f}")
    print(f"  95th Percentile: ${np.percentile(revenue_values, 95):,.2f}")
    print(f"  5th Percentile: ${np.percentile(revenue_values, 5):,.2f}")
    
    # Debt Cap Analysis
    debt_cap_values = [r.debt_cap for r in results]
    print(f"\nDebt Cap:")
    print(f"  Mean: ${np.mean(debt_cap_values):,.2f}")
    print(f"  95th Percentile: ${np.percentile(debt_cap_values, 95):,.2f}")
    print(f"  5th Percentile: ${np.percentile(debt_cap_values, 5):,.2f}")
    
    # Risk Analysis
    low_health_factor_count = sum(
        1 for r in results 
        if any(hf < 1.1 for hf in r.health_factors.values())
    )
    risk_percentage = (low_health_factor_count / len(results)) * 100
    print(f"\nRisk Analysis:")
    print(f"  Simulations with health factors < 1.1: {risk_percentage:.1f}%")
    
    # Generate plots
    print("\nGenerating visualization plots...")
    simulator.plot_results()
    
    print("\nSimulation completed successfully!")

if __name__ == "__main__":
    main()
