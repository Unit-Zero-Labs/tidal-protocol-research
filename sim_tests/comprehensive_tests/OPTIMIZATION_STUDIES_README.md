# Optimization Studies

This directory contains **optimization studies** that are distinct from the standard comparison studies (Studies 1-10).

## Purpose

While Studies 1-10 compare Aave vs High Tide under various market conditions with fixed parameters, optimization studies use advanced techniques to find **optimal thresholds and configurations** that answer questions like:

- What is the **minimum viable health factor** that could survive a market?
- What leverage frequency provides the best risk/reward?
- What pool parameters optimize capital efficiency?

## Study 11: Minimum Viable Health Factor (Weekly Rebalancing)

### Overview

**Objective**: Find the absolute minimum health factor that could have survived the 2022 bear market on Aave with weekly rebalancing.

**Why This Matters**: This study reframes liquidations from "average outcomes" to "critical failures that must be avoided." By finding the minimum viable HF, we can:
1. Understand the true risk floor for Aave during bear markets
2. Compare this "optimal" Aave strategy against High Tide's standard 1.2 HF
3. Demonstrate that even at its theoretical best, Aave requires significantly higher safety margins

### How It Works

**Phase 1: Binary Search Optimization**
- Tests different health factors (starting range: 1.05 to 2.0)
- Runs fail-fast simulations that exit immediately on liquidation
- Uses binary search to converge to minimum viable HF (±0.01 precision)
- Memory-efficient: cleans up after each test run

**Phase 2: Detailed Comparison**
- Runs full simulation with optimal Aave HF vs High Tide at 1.2 HF
- Both protocols use weekly rebalancing
- Generates comprehensive charts and analysis

### Running the Study

```bash
# From the sim_tests directory
python study11_2022_bear_minimum_hf_weekly.py
```

**Expected Runtime**: 20-30 minutes
- Phase 1 (optimization): 10-15 minutes (5-8 iterations)
- Phase 2 (comparison): 10-15 minutes

### Key Features

1. **Fail-Fast Architecture**: Simulations exit immediately on liquidation to save time
2. **Memory Management**: Aggressive cleanup between tests to prevent memory bloat
3. **Binary Search**: Efficient convergence (O(log n) vs linear search)
4. **Standalone**: Doesn't depend on run_all_studies.py

### Output

**Phase 1 Output**:
```
BINARY SEARCH: Finding Minimum Viable Health Factor
========================================================================
Iteration 1: Testing HF = 1.525
  ✅ SURVIVED - Agent maintained position for full year
  → Trying lower HF

Iteration 2: Testing HF = 1.263
  ❌ LIQUIDATED at minute 245,280 (day 170.3)
  → Trying higher HF

...

BINARY SEARCH COMPLETE
Minimum Viable HF: 1.387
```

**Phase 2 Output**:
- Full comparison charts (survival, health factors, net position value)
- Detailed CSV data
- JSON summary with optimization history

**Results Location**:
```
tidal_protocol_sim/results/Study_11_2022_Bear_Minimum_HF_Weekly_Aave_[HF]_vs_HT_1.20/
├── charts/
│   ├── survival_comparison.png
│   ├── health_factor_comparison.png
│   ├── performance_metrics.png
│   └── ...
├── summary.json (includes optimization_summary)
└── detailed_metrics.csv
```

## Future Optimization Studies

### Study 12: Minimum Viable HF (Daily Rebalancing) [Planned]
Same as Study 11 but with daily rebalancing frequency.

```bash
python study12_2022_bear_minimum_hf_daily.py
```

### Study 13: Minimum Viable HF (Monthly Rebalancing) [Planned]
Same as Study 11 but with monthly rebalancing frequency.

```bash
python study13_2022_bear_minimum_hf_monthly.py
```

### Study 14: Optimal Rebalancing Frequency [Planned]
Find the optimal rebalancing frequency (daily/weekly/monthly) for different market conditions.

## Technical Details

### Binary Search Algorithm

```python
lower = 1.05  # Must be above liquidation threshold
upper = 2.0   # Conservative starting point

while (upper - lower) > precision:
    test_hf = (lower + upper) / 2.0
    
    if simulation_survives(test_hf):
        upper = test_hf  # Can try lower
    else:
        lower = test_hf  # Need higher HF
        
minimum_viable_hf = upper
```

### Fail-Fast Implementation

```python
for minute in simulation:
    update_prices_and_state()
    
    if agent.health_factor < 1.0:
        # EXIT IMMEDIATELY - DON'T CONTINUE
        return {"survived": False, "liquidation_minute": minute}
        
    execute_weekly_rebalancing_if_needed()

return {"survived": True, "liquidation_minute": None}
```

### Memory Management

```python
# After each test:
1. Delete engine and agent objects
2. Remove temporary result files
3. Call gc.collect() to force garbage collection
```

## Design Philosophy

1. **Precision over Speed**: Takes time to find exact minimum (±0.01 HF)
2. **Transparency**: Shows all iterations and decisions
3. **Reproducibility**: Uses same market data as standard studies
4. **Integration**: Seamlessly uses existing engine architecture

## Narrative Value

These optimization studies strengthen the High Tide narrative by:

1. **Reframing Risk**: Liquidations aren't "average outcomes" - they're failures
2. **Fair Comparison**: Compare Aave at its theoretical best vs High Tide's standard
3. **Concrete Numbers**: "Even with perfect HF tuning, Aave needed 1.38x to survive 2022"
4. **Capital Efficiency**: Highlight how much extra margin Aave requires

## Running Multiple Studies

To run all three rebalancing frequency studies (when available):

```bash
# Run sequentially
python study11_2022_bear_minimum_hf_weekly.py
python study12_2022_bear_minimum_hf_daily.py
python study13_2022_bear_minimum_hf_monthly.py

# Compare results
python compare_optimization_studies.py  # [Future]
```

## Notes

- These studies are **not** included in `run_all_studies.py` by default
- They serve a different purpose than the symmetric/asymmetric comparisons
- Run them separately when you need optimization analysis
- Results are still saved to the standard results directory


