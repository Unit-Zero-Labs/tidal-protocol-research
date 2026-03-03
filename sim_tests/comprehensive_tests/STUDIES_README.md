# High Tide vs AAVE: 10-Study Simulation Suite

This directory contains 10 individual study scripts that replicate the comprehensive analysis from the whitepaper.

## Study Overview

### Symmetric Studies (1-5): Historical Rates for Both Protocols
Both High Tide and AAVE use identical historical AAVE rates to isolate the automation advantage.

| Study | Script | Market | Duration | BTC Change | HF Config |
|-------|--------|--------|----------|------------|-----------|
| **1** | `run_study_1_2021_mixed_symmetric.py` | 2021 Mixed | 365 days | +59.6% | Equal (1.3) |
| **2** | `run_study_2_2024_bull_symmetric.py` | 2024 Bull | 365 days | +119% | Equal (1.3) |
| **3** | `run_study_3_2024_capital_efficiency_symmetric.py` | 2024 Bull | 365 days | +119% | HT 1.1 vs AAVE 1.95 |
| **4** | `run_study_4_2022_bear_symmetric.py` | 2022 Bear | 365 days | -64.2% | Equal (1.3) |
| **5** | `run_study_5_2025_lowvol_symmetric.py` | 2025 Low Vol | 268 days | +21.2% | Equal (1.3) |

### Asymmetric Studies (6-10): Advanced MOET for High Tide
High Tide uses the Advanced MOET system (dynamic market-driven rates) while AAVE uses historical rates.

| Study | Script | Market | Duration | BTC Change | HF Config |
|-------|--------|--------|----------|------------|-----------|
| **6** | `run_study_6_2021_mixed_asymmetric.py` | 2021 Mixed | 365 days | +59.6% | Equal (1.3) |
| **7** | `run_study_7_2024_bull_asymmetric.py` | 2024 Bull | 365 days | +119% | Equal (1.3) |
| **8** | `run_study_8_2024_capital_efficiency_asymmetric.py` | 2024 Bull | 365 days | +119% | HT 1.1 vs AAVE 1.95 |
| **9** | `run_study_9_2022_bear_asymmetric.py` | 2022 Bear | 365 days | -64.2% | Equal (1.3) |
| **10** | `run_study_10_2025_lowvol_asymmetric.py` | 2025 Low Vol | 268 days | +21.2% | Equal (1.3) |

## Running Individual Studies

### Option 1: Run Individual Study

```bash
# From project root directory
python sim_tests/run_study_1_2021_mixed_symmetric.py
```

Each study will:
- Print configuration details
- Run the full simulation
- Save results to `tidal_protocol_sim/results/<study_name>/`
- Generate comparison charts automatically

### Option 2: Run All Studies Sequentially

```bash
# Run all 10 studies in order (takes ~80-120 minutes total)
python sim_tests/run_all_studies.py
```

This will execute all studies in sequence and provide a summary at the end.

## Study Configuration

All studies use:
- **1 agent per protocol** (High Tide + AAVE)
- **No ecosystem growth** (clean comparison)
- **Weekly yield harvesting** (enabled)
- **Historical BTC price data** from CSV files
- **Historical AAVE rates** from CSV files

## Expected Runtimes

| Study Type | Runtime | Notes |
|------------|---------|-------|
| Symmetric (Studies 1-5) | ~5-10 min | Simpler rate calculations |
| Asymmetric (Studies 6-10) | ~8-15 min | Advanced MOET adds complexity |
| Study 5 & 10 | Slightly less | Only 268 days vs 365 |

**Total runtime for all 10 studies: ~80-120 minutes**

## Results Output

Each study creates a unique results directory:

```
tidal_protocol_sim/results/
├── Full_Year_2021_BTC_Mixed_Market_Equal_HF_Weekly_Yield_Harvest_HT_vs_AAVE_Comparison/
│   ├── charts/
│   │   ├── net_position_apy_comparison.png
│   │   ├── btc_capital_preservation_comparison.png
│   │   └── ...
│   ├── simulation_results.json
│   └── detailed_metrics.csv
├── Full_Year_2024_BTC_Bull_Market_Equal_HF_Weekly_Yield_Harvest_HT_vs_AAVE_Comparison/
│   └── ...
└── ...
```

## Key Metrics to Watch

### For Each Study, Track:
1. **Survival Rate**: High Tide vs AAVE
2. **Total Return**: USD performance
3. **BTC Accumulation**: Quantity of BTC held
4. **Final Health Factor**: Capital efficiency
5. **Liquidation Events**: Critical for bear markets

### Expected Key Findings:
- **Studies 1-3**: High Tide outperforms in USD and BTC
- **Study 4 & 9**: High Tide survives, AAVE liquidates (bear market)
- **Study 5 & 10**: AAVE may win USD, High Tide wins BTC accumulation

## Data Requirements

Ensure these files exist in the project root:
- `btc-usd-max.csv` (BTC historical prices)
- `rates_compute.csv` (AAVE historical rates)

## Troubleshooting

### If a study fails:
1. Check that CSV data files are present
2. Verify the market year has data available
3. Check console output for specific errors
4. Ensure sufficient disk space for results

### If results differ from whitepaper:
- Whitepaper used 20 agents per protocol
- These scripts use 1 agent for efficiency
- Results should be qualitatively similar (same trends, survival rates)
- Absolute values may differ slightly due to single-agent execution

## Customization

To modify a study's parameters, edit the corresponding script's configuration section:

```python
# Example: Change number of agents
config.num_agents = 20  # Default is 1

# Example: Enable ecosystem growth
config.ecosystem_growth_enabled = True

# Example: Change health factor
config.agent_initial_hf = 1.5
```

## Next Steps After Running Studies

1. Review results in each study's `results/` directory
2. Compare charts across symmetric vs asymmetric studies
3. Analyze survival rates in bear market studies (4 & 9)
4. Compare BTC accumulation across all studies
5. Update whitepaper with actual results

## Questions?

See the main simulation file: `sim_tests/full_year_sim.py`
Or review the whitepaper: `reports/High_Tide_vs_AAVE_Comparative_Analysis_Whitepaper.md`

