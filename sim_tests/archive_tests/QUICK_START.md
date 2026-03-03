# Quick Start Guide: Running All 10 Studies

## Prerequisites

Ensure you have:
- ✅ `btc-usd-max.csv` in project root
- ✅ `rates_compute.csv` in project root
- ✅ Python environment configured
- ✅ All dependencies installed

## Option 1: Run All Studies at Once (Recommended for overnight runs)

```bash
cd /Users/connorflanagan/Desktop/UnitZeroLabs/tidal-protocol-research
python sim_tests/run_all_studies.py
```

**Runtime**: ~80-120 minutes total
**Output**: All 10 study results in separate folders

---

## Option 2: Run Studies Individually

### Symmetric Studies (Historical Rates for Both)

```bash
# Study 1: 2021 Mixed Market
python sim_tests/run_study_1_2021_mixed_symmetric.py

# Study 2: 2024 Bull Market (Equal HF)
python sim_tests/run_study_2_2024_bull_symmetric.py

# Study 3: 2024 Capital Efficiency (HT 1.1 vs AAVE 1.95)
python sim_tests/run_study_3_2024_capital_efficiency_symmetric.py

# Study 4: 2022 Bear Market ⚠️ CRITICAL TEST
python sim_tests/run_study_4_2022_bear_symmetric.py

# Study 5: 2025 Low Volatility
python sim_tests/run_study_5_2025_lowvol_symmetric.py
```

### Asymmetric Studies (Advanced MOET for High Tide)

```bash
# Study 6: 2021 Mixed Market (Advanced MOET)
python sim_tests/run_study_6_2021_mixed_asymmetric.py

# Study 7: 2024 Bull Market (Advanced MOET)
python sim_tests/run_study_7_2024_bull_asymmetric.py

# Study 8: 2024 Capital Efficiency (Advanced MOET)
python sim_tests/run_study_8_2024_capital_efficiency_asymmetric.py

# Study 9: 2022 Bear Market (Advanced MOET) ⚠️ CRITICAL TEST
python sim_tests/run_study_9_2022_bear_asymmetric.py

# Study 10: 2025 Low Volatility (Advanced MOET)
python sim_tests/run_study_10_2025_lowvol_asymmetric.py
```

---

## Priority Studies (If Time-Limited)

If you can only run a few studies, prioritize these:

### Top Priority (Core Findings):
1. **Study 4** - 2022 Bear Market (Symmetric) - Proves survival vs liquidation
2. **Study 3** - 2024 Capital Efficiency - Proves safe operation at 1.1 HF
3. **Study 6** - 2021 Mixed (Advanced MOET) - Shows MOET system performance

### Secondary Priority:
4. **Study 2** - 2024 Bull Market - Shows automation advantage in bull runs
5. **Study 9** - 2022 Bear (Advanced MOET) - Validates MOET in stress conditions

---

## What to Expect

### During Execution:
- Console output showing minute-by-minute progress
- Regular status updates (every 10,000 minutes)
- Memory usage tracking
- Event logging (rebalancing, liquidations, etc.)

### After Completion:
Each study creates its own results directory:

```
tidal_protocol_sim/results/
└── [Study_Name]/
    ├── charts/
    │   ├── net_position_apy_comparison.png
    │   ├── btc_capital_preservation_comparison.png
    │   └── ... (other charts)
    ├── simulation_results.json
    ├── detailed_metrics.csv
    └── comparison_summary.txt
```

---

## Key Results to Check

After each study, check:

1. **Survival Rate**
   - Study 4 & 9: High Tide should survive, AAVE liquidates
   
2. **Total Return**
   - Studies 1-3, 6-8: High Tide should outperform
   - Studies 5, 10: AAVE may win in USD, HT wins in BTC
   
3. **BTC Accumulation**
   - All studies: High Tide should accumulate 4-7% more BTC
   
4. **Final Charts**
   - Compare net position growth
   - Verify BTC accumulation trends
   - Check health factor evolution

---

## Estimated Runtimes

| Study | Type | Duration | Runtime |
|-------|------|----------|---------|
| 1 | Symmetric | 365 days | ~8 min |
| 2 | Symmetric | 365 days | ~8 min |
| 3 | Symmetric | 365 days | ~8 min |
| 4 | Symmetric | 365 days | ~8 min |
| 5 | Symmetric | 268 days | ~6 min |
| 6 | Asymmetric | 365 days | ~12 min |
| 7 | Asymmetric | 365 days | ~12 min |
| 8 | Asymmetric | 365 days | ~12 min |
| 9 | Asymmetric | 365 days | ~12 min |
| 10 | Asymmetric | 268 days | ~10 min |

**Total**: ~96 minutes

---

## Troubleshooting

### Study fails with "File not found":
```bash
# Check that data files exist
ls -la btc-usd-max.csv
ls -la rates_compute.csv
```

### Study fails with "No data for year":
Check that `btc-usd-max.csv` and `rates_compute.csv` contain data for the requested year (2021, 2022, 2024, or 2025).

### Out of memory:
Studies use 1 agent to minimize memory. If still encountering issues, run studies individually rather than using `run_all_studies.py`.

### Results differ from whitepaper:
- Whitepaper used 20 agents (these use 1 agent)
- Results should show same trends (survival, relative performance)
- Absolute values may differ slightly

---

## Next Steps

After running all studies:

1. ✅ Review all comparison charts
2. ✅ Compile survival rates (especially Studies 4 & 9)
3. ✅ Compare BTC accumulation across all studies
4. ✅ Analyze symmetric vs asymmetric performance deltas
5. ✅ Update whitepaper with actual results from your machine
6. ✅ Generate summary tables comparing all 10 studies

---

## Questions?

- See: `sim_tests/STUDIES_README.md` for detailed documentation
- See: `sim_tests/full_year_sim.py` for simulation implementation
- See: `reports/High_Tide_vs_AAVE_Comparative_Analysis_Whitepaper.md` for expected results

