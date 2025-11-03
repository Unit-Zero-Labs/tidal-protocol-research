# Asymmetric Comparison: High Tide (Dynamic MOET) vs AAVE (Historical Rates)

## Overview

This configuration enables an **asymmetric but realistic comparison** where:
- **High Tide**: Uses the native Advanced MOET System with dynamic, market-driven rates
- **AAVE**: Uses historical AAVE borrow rates from real market data

## What Changes

### High Tide (Advanced MOET System Enabled)

**Interest Rate Mechanism:**
```
MOET Rate = r_floor + r_bond_cost + peg_adjustment
           = 2% + (EMA of bond auction yields) + 0%
```

**Dynamic Components:**
- `r_floor`: 2% fixed governance profit margin
- `r_bond_cost`: Variable, depends on bond auction outcomes
  - Initial: ~20% (starting with 8% reserves, 10% target)
  - Converges toward market-clearing rate as reserves fill
  - Updates every minute based on 7-day EMA of auction yields
- `peg_adjustment`: 0% (MOET maintained at $1.00)

**Bond Auction System:**
- **Trigger**: Continuous auctions whenever reserves < 10% of MOET supply
- **APR Calculation**: `(Target - Actual) / Target`
  - Example: 8% actual, 10% target → 20% bond APR
- **Fill Probability**: 5% base per minute, increases with higher yields
- **Reserve Building**: As auctions fill, r_bond_cost decreases

**Reserve Management:**
- Target: 10% of total MOET supply in USDC/USDF backing
- Automatic bond issuance when below target
- Redeemer contract for USDC/USDF ↔ MOET conversions

**Stablecoin Pools:**
- No MOET:USDC or MOET:USDF Uniswap pools
- Redeemer contract handles all stablecoin conversions
- Deleveraging uses bond auctions instead of stablecoin swaps

### AAVE (Historical Rates)

**Interest Rate Mechanism:**
```
AAVE Rate = Historical daily rate from CSV
```

**Static Components:**
- Fixed daily rates from historical data
- No dynamic response to simulation events
- Example (2024): 3.2% - 8.7% APR range

**No Changes:**
- AAVE operates exactly as in symmetric comparison
- Uses historical rates that real AAVE users faced
- No bond auctions or reserve mechanics

## Key Differences in Simulation Behavior

### Week 1-2: High Initial Rates
- **High Tide**: ~20% MOET borrow rate (bond auctions filling reserves)
- **AAVE**: ~5-7% historical rate (typical market conditions)
- **Impact**: High Tide agents face higher initial costs, slower leverage buildup

### Week 3-8: Rate Convergence
- **High Tide**: r_bond_cost decreases as reserves approach 10% target
- **AAVE**: Continues with historical rates (may fluctuate)
- **Impact**: High Tide rates potentially converge toward AAVE levels

### Week 9-52: Steady State
- **High Tide**: r_floor (2%) + stable r_bond_cost (varies by market)
- **AAVE**: Historical rate pattern continues
- **Impact**: Rate differential depends on bond market dynamics

## Expected Outcomes

### Scenario 1: Bull Market (2024)
- **High Tide Advantage**: Better in later months as rates stabilize
- **AAVE Advantage**: Better in early months with lower rates
- **Net Result**: Likely closer performance than symmetric comparison

### Scenario 2: Bear Market (2022)
- **High Tide Advantage**: Active deleveraging prevents liquidation
- **AAVE Disadvantage**: Static positioning → liquidation cascades
- **Net Result**: High Tide still wins, but possibly at higher cost

### Scenario 3: Low Vol Market (2025)
- **High Tide Disadvantage**: Higher average rates + transaction costs
- **AAVE Advantage**: Lower historical rates + no rebalancing costs
- **Net Result**: AAVE likely outperforms even more

## Intellectual Honesty Considerations

### Pros of Asymmetric Comparison
✅ **Realistic**: Shows how Tidal would actually operate with its native economics
✅ **Complete Picture**: Demonstrates full advanced MOET system functionality
✅ **Market Test**: Tests whether dynamic rates are competitive vs fixed historical rates
✅ **Educational**: Shows trade-offs between sophisticated vs simple systems

### Cons of Asymmetric Comparison
❌ **Not Apples-to-Apples**: Protocols face different economic environments
❌ **Attribution Unclear**: Is outperformance from automation or favorable rates?
❌ **Potentially Unfair**: Early high rates may handicap High Tide unfairly
❌ **Complex Interpretation**: Readers must understand both rate mechanisms

## Recommendation for Whitepaper

**Option A: Replace Symmetric Studies**
- Run all 5 studies with asymmetric setup
- Explicitly note rate mechanism differences
- Focus narrative on "real-world Tidal vs historical AAVE"

**Option B: Add as Study 6**
- Keep Studies 1-5 as symmetric (current approach)
- Add new "Study 6: Advanced MOET System" 
- Compare asymmetric results to symmetric baseline

**Option C: Dual Reporting**
- Show both symmetric and asymmetric results side-by-side
- Let readers see impact of rate mechanism choice
- Most intellectually honest but complex presentation

## How to Run

The configuration is already updated in `full_year_sim.py`:

```python
# Line 127
self.enable_advanced_moet_system = True  # Always enable for High Tide
```

Just run the simulation as normal:
```bash
python3 sim_tests/full_year_sim.py
```

**Expected Outputs:**
- All standard comparison charts
- Additional advanced MOET charts:
  - MOET system analysis
  - Reserve management timeline
  - Bond auction history
  - Interest rate decomposition

## Switching Between Modes

**For Symmetric Comparison (Current Whitepaper):**
```python
# Line 127 - uncomment option 1
self.enable_advanced_moet_system = not self.use_historical_rates
```

**For Asymmetric Comparison (Native Tidal Economics):**
```python
# Line 127 - use option 2 (currently active)
self.enable_advanced_moet_system = True
```

## Summary

The asymmetric comparison is **more realistic** but **less clean** for attribution. It answers the question: "How does Tidal's full economic model perform against AAVE's real historical rates?" rather than "How does automation alone improve outcomes?"

Both comparisons have value - symmetric isolates the automation benefit, asymmetric shows the complete system in action.

