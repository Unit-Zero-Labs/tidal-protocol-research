# Base Case Scenario: High Tide vs AAVE Comparison

## Overview
This simulation provides a controlled comparison between High Tide's active risk management system and AAVE's traditional liquidation mechanism, isolating the impact of rebalancing from interest rate dynamics.

## Test Parameters
- **Duration**: 3 months (90 days / 129,600 minutes)
- **BTC Price Range**: ±15% ($85k - $115k, starting at $100k)
- **Fixed Borrow Rate**: 5.6234% APR (AAVE's historical 90-day average)
- **Agents**: 100 identical agents per system
- **Initial Deposit**: 1 BTC per agent @ $100k
- **Tri-Health Factor**: 1.05 (initial) / 1.015 (rebalancing trigger) / 1.03 (target)

## Architecture Compliance

### ✅ Uses Existing Tidal Protocol Architecture

1. **Engine Inheritance**
   - `HighTideVaultEngine` → inherits from `TidalProtocolEngine`
   - `AaveProtocolEngine` → inherits from `TidalProtocolEngine`
   - Both use **Uniswap V3 pools** for all swaps (MOET:BTC, MOET:YT)

2. **Agent Implementation**
   - High Tide agents: `HighTideAgent` with full rebalancing
   - AAVE agents: `AaveAgent` with liquidation only
   - **Both purchase initial YT tokens** (key for fair comparison)
   - Both inherit yield token management from `HighTideAgentState`

3. **Pool System**
   - MOET:BTC pool: $10M liquidity, 80% concentration
   - MOET:YT pool: $500K liquidity, 95% concentration, 75/25 ratio
   - ALM rebalancer: 12-hour intervals (High Tide only)
   - Algo rebalancer: 50 bps deviation threshold (High Tide only)

4. **Interest Rate Handling**
   - **Fixed rate override**: 5.6234% APR (no Bonder system)
   - Advanced MOET system disabled (`enable_advanced_moet_system = False`)
   - Interest accrued minute-by-minute using compound interest formula

## Key Differences from Full Year Sim

1. **No Bonder System**: Fixed interest rate instead of dynamic bond auctions
2. **Shorter Duration**: 3 months vs 12 months
3. **Controlled Price Path**: ±15% range vs real 2024 data (+119%)
4. **Parallel Comparison**: Runs both systems simultaneously with identical agents
5. **Focus on Risk Management**: Isolates rebalancing impact from interest dynamics

## System Behaviors

### High Tide System
- ✅ Initial YT purchase (MOET → YT at t=0)
- ✅ Delevering when HF drops (YT → MOET → Stable → BTC → redeposit)
- ✅ ALM rebalancing (time-based, 12-hour intervals)
- ✅ Algo rebalancing (deviation-based, 50 bps threshold)
- ✅ Emergency rebalancing (if pool health deteriorates)
- ❌ Bonder system (disabled for this test)

### AAVE System
- ✅ Initial YT purchase (MOET → YT at t=0)
- ✅ Hold YT forever (earn passive yield)
- ✅ Liquidation when HF < 1.0 (50% collateral + 5% bonus)
- ❌ No rebalancing
- ❌ No active risk management
- ❌ Bonder system (disabled for this test)

## Data Collection

### Per-Minute Tracking
- BTC price updates
- Agent health factors
- Pool states (reserves, prices)
- Rebalancing events (High Tide)
- Liquidation events (AAVE)

### Daily Snapshots
- Agent portfolios
- Pool balances
- Yield earned
- Interest accrued

## Output Structure

```
tidal_protocol_sim/results/Base_Case_HT_vs_AAVE_3mo/
├── Base_Case_HT_vs_AAVE_3mo_results.json  # Full simulation data
└── charts/
    ├── btc_price_evolution.png            # 3-month BTC price path
    ├── survival_rate_comparison.png       # Active vs liquidated agents
    ├── net_apy_comparison.png             # Average net APY comparison
    └── health_factor_evolution.png        # HF tracking (both systems)
```

## Usage

```bash
# Run the comparison
python sim_tests/base_case_ht_vs_aave_comparison.py

# Expected runtime: ~10-15 minutes
# Expected output: JSON results + 4 comparison charts
```

## Metrics Calculated

### Survival Metrics
- Active agents vs liquidated agents
- Survival rate percentage
- Time-to-liquidation distribution

### Performance Metrics
- Average net APY (yield earned - interest paid)
- Total interest paid
- Total yield earned
- Final portfolio values

### Comparative Analysis
- Survival rate delta (HT - AAVE)
- Net APY delta (HT - AAVE)
- Interest paid delta
- Yield earned delta

## Technical Whitepaper Use Case

This simulation is designed to generate data for the **Technical Whitepaper** showing:

1. **Capital Efficiency**: How much more efficient is active rebalancing?
2. **Risk Management**: What is the survival rate improvement?
3. **Net Returns**: Does rebalancing justify the gas costs?
4. **Liquidation Costs**: What is the true cost of AAVE liquidations?

## Next Steps

### Immediate
- ✅ Basic comparison charts (survival, APY, HF evolution)
- 🔄 Full chart suite (matching full_year_sim)

### Future Enhancements
- Add full year comparison with Bonder system enabled
- Multiple scenario testing (bear market, flash crash, sideways)
- Gas cost analysis (rebalancing tx costs)
- Slippage impact analysis
- Pool utilization metrics

## Architecture Validation

This simulation **fully respects the existing architecture**:
- Uses real Uniswap V3 math for all swaps
- Uses real yield token accrual
- Uses real rebalancing logic
- Uses real liquidation mechanics
- No shortcuts, no approximations, no novel mechanics

Everything flows through the pools exactly as designed!

