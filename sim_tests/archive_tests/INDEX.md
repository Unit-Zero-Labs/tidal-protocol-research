# Study Scripts Index

## 📁 File Organization

```
sim_tests/
│
├── 📚 DOCUMENTATION
│   ├── INDEX.md                    ← You are here
│   ├── SETUP_COMPLETE.md          ← Setup verification & next steps
│   ├── QUICK_START.md             ← Fast reference for running studies
│   └── STUDIES_README.md          ← Detailed documentation
│
├── 🎯 MASTER RUNNER
│   └── run_all_studies.py         ← Run all 10 studies sequentially
│
├── 📊 SYMMETRIC STUDIES (1-5)
│   ├── run_study_1_2021_mixed_symmetric.py
│   ├── run_study_2_2024_bull_symmetric.py
│   ├── run_study_3_2024_capital_efficiency_symmetric.py
│   ├── run_study_4_2022_bear_symmetric.py
│   └── run_study_5_2025_lowvol_symmetric.py
│
├── 📈 ASYMMETRIC STUDIES (6-10)
│   ├── run_study_6_2021_mixed_asymmetric.py
│   ├── run_study_7_2024_bull_asymmetric.py
│   ├── run_study_8_2024_capital_efficiency_asymmetric.py
│   ├── run_study_9_2022_bear_asymmetric.py
│   └── run_study_10_2025_lowvol_asymmetric.py
│
└── 🔧 CORE ENGINE
    └── full_year_sim.py            ← Simulation engine (already exists)
```

## 🚀 Quick Access

### First Time? Start Here:
1. Read: [`SETUP_COMPLETE.md`](SETUP_COMPLETE.md) - Verify setup
2. Read: [`QUICK_START.md`](QUICK_START.md) - Learn how to run
3. Run: `python sim_tests/run_study_1_2021_mixed_symmetric.py` - Test

### Want to Run Everything?
```bash
python sim_tests/run_all_studies.py
```

### Need Details?
Read: [`STUDIES_README.md`](STUDIES_README.md) - Full documentation

## 📊 Study Matrix

| Study | Market | Type | HF | Duration | Advanced MOET | Script |
|-------|--------|------|-----|----------|---------------|--------|
| **1** | 2021 Mixed (+60%) | Symmetric | 1.3 | 365d | ❌ | `run_study_1_2021_mixed_symmetric.py` |
| **2** | 2024 Bull (+119%) | Symmetric | 1.3 | 365d | ❌ | `run_study_2_2024_bull_symmetric.py` |
| **3** | 2024 Bull (+119%) | Symmetric | 1.1 vs 1.95 | 365d | ❌ | `run_study_3_2024_capital_efficiency_symmetric.py` |
| **4** | 2022 Bear (-64%) | Symmetric | 1.3 | 365d | ❌ | `run_study_4_2022_bear_symmetric.py` |
| **5** | 2025 Low Vol (+21%) | Symmetric | 1.3 | 268d | ❌ | `run_study_5_2025_lowvol_symmetric.py` |
| **6** | 2021 Mixed (+60%) | Asymmetric | 1.3 | 365d | ✅ | `run_study_6_2021_mixed_asymmetric.py` |
| **7** | 2024 Bull (+119%) | Asymmetric | 1.3 | 365d | ✅ | `run_study_7_2024_bull_asymmetric.py` |
| **8** | 2024 Bull (+119%) | Asymmetric | 1.1 vs 1.95 | 365d | ✅ | `run_study_8_2024_capital_efficiency_asymmetric.py` |
| **9** | 2022 Bear (-64%) | Asymmetric | 1.3 | 365d | ✅ | `run_study_9_2022_bear_asymmetric.py` |
| **10** | 2025 Low Vol (+21%) | Asymmetric | 1.3 | 268d | ✅ | `run_study_10_2025_lowvol_asymmetric.py` |

## ⚡ Priority Studies

### Must Run (Core Findings):
- **Study 4** 🔴 - Bear market survival test (AAVE liquidates)
- **Study 3** 🟡 - Capital efficiency proof (1.1 vs 1.95 HF)
- **Study 6** 🟢 - Advanced MOET validation

### Should Run (Supporting Evidence):
- **Study 2** - Bull market automation advantage
- **Study 9** - Bear market with Advanced MOET

## 📈 Expected Outcomes

### ✅ High Tide Wins:
- Studies 1, 2, 3, 6, 7, 8 (USD returns)
- Studies 4, 9 (Survival in bear markets)
- ALL STUDIES (BTC accumulation)

### ❌ AAVE Wins:
- Studies 5, 10 (USD returns in low volatility)

### 🎯 Key Finding:
**Study 4 & 9**: High Tide survives with +6% BTC, AAVE liquidates 100%

## ⏱️ Time Estimates

| Scenario | Studies | Runtime |
|----------|---------|---------|
| **Quick Test** | Study 1 only | ~8 min |
| **Priority** | Studies 3, 4, 6 | ~28 min |
| **Symmetric** | Studies 1-5 | ~38 min |
| **Asymmetric** | Studies 6-10 | ~58 min |
| **Everything** | All 10 studies | ~96 min |

## 📂 Results Location

After running, results will be saved to:
```
tidal_protocol_sim/results/
└── [Study_Name]/
    ├── charts/
    ├── simulation_results.json
    └── detailed_metrics.csv
```

## 🔗 Related Files

- 📄 Whitepaper: `reports/High_Tide_vs_AAVE_Comparative_Analysis_Whitepaper.md`
- 📊 BTC Data: `btc-usd-max.csv` (root directory)
- 💰 Rate Data: `rates_compute.csv` (root directory)
- 🔧 Engine: `sim_tests/full_year_sim.py`

## 🆘 Getting Help

1. **Setup questions**: See [`SETUP_COMPLETE.md`](SETUP_COMPLETE.md)
2. **How to run**: See [`QUICK_START.md`](QUICK_START.md)
3. **Study details**: See [`STUDIES_README.md`](STUDIES_README.md)
4. **Expected results**: See whitepaper in `reports/`

## ✅ Pre-Flight Checklist

Before running:
- [ ] `btc-usd-max.csv` exists in root
- [ ] `rates_compute.csv` exists in root
- [ ] Python 3.8+ installed
- [ ] All dependencies installed
- [ ] Sufficient disk space (~500 MB for all results)

## 🚀 Let's Go!

Start with Study 1:
```bash
python sim_tests/run_study_1_2021_mixed_symmetric.py
```

Or run everything:
```bash
python sim_tests/run_all_studies.py
```

---

*Last updated: November 2025*
*Questions? Check QUICK_START.md or STUDIES_README.md*

