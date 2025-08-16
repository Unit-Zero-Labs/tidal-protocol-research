#!/usr/bin/env python3
"""
Configuration file for Tidal Protocol Simulation

This file contains all configurable parameters for the simulation.
Modify these values to test different scenarios and configurations.
"""

# =============================================================================
# PROTOCOL CONFIGURATION
# =============================================================================

# Reserve Factor: Percentage of interest that goes to protocol revenue
RESERVE_FACTOR = 0.15  # 15%

# LP Rewards Factor: Percentage of protocol revenue distributed to LP pools
LP_REWARDS_FACTOR = 0.50  # 50%

# Target Health Factor: Minimum health factor for safe borrowing
TARGET_HEALTH_FACTOR = 1.5

# DEX Liquidity Allocation: Percentage allocated to other lending markets
DEX_LIQUIDITY_ALLOCATION = 0.35  # 35%

# =============================================================================
# INTEREST RATE MODEL PARAMETERS (per-block)
# =============================================================================

# Base rate per block (usually 0)
BASE_RATE_PER_BLOCK = 0

# Multiplier per block for linear rate calculation
MULTIPLIER_PER_BLOCK = 11415525114

# Jump rate per block for rates above kink
JUMP_PER_BLOCK = 253678335870

# Kink point (utilization threshold for rate jump)
KINK = 8e17  # 0.80 utilization, scaled 1e18

# Block timing constants
BLOCKS_PER_MINUTE = 30
BLOCKS_PER_DAY = 43200
BLOCKS_PER_YEAR = 15768000

# =============================================================================
# INITIAL LIQUIDITY ALLOCATION (Total: $14,000,000)
# =============================================================================

INITIAL_LIQUIDITY = {
    'ETH': 7_000_000,    # $7M (50%)
    'BTC': 3_500_000,    # $3.5M (25%)
    'FLOW': 2_100_000,   # $2.1M (15%)
    'USDC': 1_400_000    # $1.4M (10%)
}

# =============================================================================
# INITIAL ASSET PRICES (USD)
# =============================================================================

INITIAL_PRICES = {
    'ETH': 3000.0,
    'BTC': 45000.0,
    'FLOW': 0.50,
    'USDC': 1.0,
    'MOET': 1.0
}

# =============================================================================
# COLLATERAL FACTORS
# =============================================================================

COLLATERAL_FACTORS = {
    'ETH': 0.75,    # 75% of value can be borrowed against
    'BTC': 0.75,    # 75% of value can be borrowed against
    'FLOW': 0.50,   # 50% of value can be borrowed against (higher risk)
    'USDC': 0.90    # 90% of value can be borrowed against (stablecoin)
}

# =============================================================================
# EXTREME PRICE DROP SCENARIOS
# =============================================================================

EXTREME_PRICE_DROPS = {
    'ETH': 0.15,    # -15% in extreme scenarios
    'BTC': 0.15,    # -15% in extreme scenarios
    'FLOW': 0.35,   # -35% in extreme scenarios (higher volatility)
    'USDC': 0.15    # -15% in extreme scenarios
}

# =============================================================================
# UNISWAP V3 PARAMETERS
# =============================================================================

# Fee tier for pools
UNISWAP_V3_FEE_TIER = 0.003  # 0.3%

# Maximum acceptable slippage for liquidation calculations
MAX_SLIPPAGE_THRESHOLD = 0.04  # 4%

# Liquidity distribution standard deviation (as percentage of price)
LIQUIDITY_STD_DEV = 0.10  # 10%

# =============================================================================
# MONTE CARLO SIMULATION PARAMETERS
# =============================================================================

# Default number of simulation runs
DEFAULT_N_SIMULATIONS = 1000

# Price shock parameters for Monte Carlo
PRICE_SHOCK_PARAMS = {
    'ETH': {
        'mean': -0.075,      # Mean -7.5%
        'std': 0.05,         # Standard deviation 5%
        'min': -0.50,        # Minimum -50%
        'max': 0.20          # Maximum +20%
    },
    'BTC': {
        'mean': -0.075,      # Mean -7.5%
        'std': 0.05,         # Standard deviation 5%
        'min': -0.50,        # Minimum -50%
        'max': 0.20          # Maximum +20%
    },
    'FLOW': {
        'mean': -0.175,      # Mean -17.5% (higher volatility)
        'std': 0.10,         # Standard deviation 10%
        'min': -0.50,        # Minimum -50%
        'max': 0.20          # Maximum +20%
    },
    'USDC': {
        'mean': -0.075,      # Mean -7.5%
        'std': 0.05,         # Standard deviation 5%
        'min': -0.50,        # Minimum -50%
        'max': 0.20          # Maximum +20%
    }
}

# Utilization change parameters for Monte Carlo
UTILIZATION_CHANGE_PARAMS = {
    'mean': 0.0,             # Mean change 0%
    'std': 0.1,              # Standard deviation 10%
    'min': -0.3,             # Minimum -30%
    'max': 0.3               # Maximum +30%
}

# =============================================================================
# RISK THRESHOLDS
# =============================================================================

# Health factor thresholds for risk classification
RISK_THRESHOLDS = {
    'LOW': 1.5,      # Health factor >= 1.5
    'MEDIUM': 1.2,   # Health factor >= 1.2
    'HIGH': 1.1      # Health factor >= 1.1
}

# Utilization thresholds for risk classification
UTILIZATION_THRESHOLDS = {
    'SAFE': 0.6,     # Utilization < 60%
    'MODERATE': 0.8, # Utilization < 80%
    'HIGH': 0.9      # Utilization < 90%
    # Above 90% is considered dangerous
}

# =============================================================================
# REPORTING AND OUTPUT PARAMETERS
# =============================================================================

# Percentiles for statistical summaries
SUMMARY_PERCENTILES = [0.05, 0.25, 0.75, 0.95]

# Number of bins for histograms
HISTOGRAM_BINS = 50

# Figure size for plots
PLOT_FIGURE_SIZE = (18, 12)

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config():
    """Validate that all configuration parameters are within reasonable bounds"""
    errors = []
    
    # Check reserve factor
    if not 0 <= RESERVE_FACTOR <= 1:
        errors.append(f"RESERVE_FACTOR must be between 0 and 1, got {RESERVE_FACTOR}")
    
    # Check LP rewards factor
    if not 0 <= LP_REWARDS_FACTOR <= 1:
        errors.append(f"LP_REWARDS_FACTOR must be between 0 and 1, got {LP_REWARDS_FACTOR}")
    
    # Check total liquidity
    total_liquidity = sum(INITIAL_LIQUIDITY.values())
    if total_liquidity <= 0:
        errors.append(f"Total liquidity must be positive, got {total_liquidity}")
    
    # Check collateral factors
    for asset, cf in COLLATERAL_FACTORS.items():
        if not 0 <= cf <= 1:
            errors.append(f"Collateral factor for {asset} must be between 0 and 1, got {cf}")
    
    # Check price drops are negative
    for asset, drop in EXTREME_PRICE_DROPS.items():
        if drop >= 0:
            errors.append(f"Price drop for {asset} must be negative, got {drop}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    print("✓ Configuration validation passed")
    return True

def print_config_summary():
    """Print a summary of the current configuration"""
    print("Tidal Protocol Simulation Configuration")
    print("="*50)
    
    print(f"Reserve Factor: {RESERVE_FACTOR:.1%}")
    print(f"LP Rewards Factor: {LP_REWARDS_FACTOR:.1%}")
    print(f"Target Health Factor: {TARGET_HEALTH_FACTOR:.1f}")
    print(f"DEX Liquidity Allocation: {DEX_LIQUIDITY_ALLOCATION:.1%}")
    
    print(f"\nTotal Protocol Liquidity: ${sum(INITIAL_LIQUIDITY.values()):,}")
    print("\nAsset Allocation:")
    for asset, liquidity in INITIAL_LIQUIDITY.items():
        percentage = (liquidity / sum(INITIAL_LIQUIDITY.values())) * 100
        print(f"  {asset}: ${liquidity:,} ({percentage:.1f}%)")
    
    print("\nCollateral Factors:")
    for asset, cf in COLLATERAL_FACTORS.items():
        print(f"  {asset}: {cf:.1%}")
    
    print("\nExtreme Price Drops:")
    for asset, drop in EXTREME_PRICE_DROPS.items():
        print(f"  {asset}: {drop:.1%}")
    
    print(f"\nMonte Carlo Simulations: {DEFAULT_N_SIMULATIONS:,}")
    print("="*50)

if __name__ == "__main__":
    # Validate and display configuration
    try:
        validate_config()
        print_config_summary()
    except Exception as e:
        print(f"Configuration error: {e}")
