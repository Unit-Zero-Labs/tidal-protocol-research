
## **System Prompt: Implement Comprehensive Results Storage and Visualization System**

### **Current State Analysis:**
- Results are currently only printed to terminal output
- There's a basic `export_results()` function that can save to JSON when `--output` flag is used
- The system has existing charting capabilities in `analysis/liquidation_charts.py` with matplotlib/seaborn
- No automatic results storage or run versioning system exists

### **Required Implementation:**

**1. Results Directory Structure:**
```
/results/
├── ETH_Flash_Crash/
│   ├── run_001_20241201_143022/
│   │   ├── results.json
│   │   ├── charts/
│   │   │   ├── price_evolution.png
│   │   │   ├── liquidation_events.png
│   │   │   ├── protocol_health.png
│   │   │   └── risk_metrics.png
│   │   └── summary.md
│   └── run_002_20241201_150315/
│       └── ...
├── MOET_Depeg/
│   ├── run_001_20241201_144530/
│   └── run_002_20241201_151200/
└── Cascading_Liquidations/
    └── run_001_20241201_145045/
```

**2. Automatic Results Management:**
- **Auto-create results directory structure** on first run
- **Sequential run numbering** with timestamp format: `run_{number:03d}_{YYYYMMDD}_{HHMMSS}`
- **Automatic detection** of next run number by scanning existing directories
- **Metadata tracking** in each run folder (scenario name, parameters, execution time)

**3. Comprehensive Visualization Suite:**

**Chart Types to Generate:**
- **Price Evolution Charts**: Asset price movements over simulation time
- **Liquidation Events Timeline**: When and how many liquidations occurred
- **Protocol Health Dashboard**: Treasury, debt cap utilization, MOET stability
- **Risk Metrics Heatmap**: Health factors, utilization rates across assets
- **Monte Carlo Distribution Plots**: Statistical distributions of key metrics
- **Agent Behavior Analysis**: Agent actions and portfolio evolution

**4. Enhanced Export Functionality:**
- **Structured JSON export** with nested data for easy analysis
- **CSV exports** for spreadsheet analysis
- **Summary markdown reports** with key insights
- **Chart generation** using existing matplotlib/seaborn infrastructure
- **Comparison utilities** to analyze differences between runs

**5. Integration Points:**
- **Modify `TidalSimulationEngine`** to capture detailed time-series data
- **Enhance `StressTestRunner`** to automatically save results after each scenario
- **Update CLI interface** to show results location and provide comparison options
- **Add results browsing commands**: `--list-results`, `--compare-runs`, `--view-charts`

**6. Implementation Requirements:**

**Core Classes to Create:**
```python
class ResultsManager:
    """Handles automatic results storage and versioning"""
    
class ChartGenerator:
    """Generates comprehensive visualization suite"""
    
class RunComparator:
    """Compares results between different runs"""
```

**Key Features:**
- **Thread-safe run numbering** to prevent conflicts
- **Configurable chart themes** and output formats
- **Memory-efficient** data storage for large Monte Carlo runs
- **Error handling** for file system operations
- **Progress indicators** for chart generation

**7. Expected User Experience:**
```bash
# Run scenario - automatically saves to results/
python3 run_tidal_simulation.py --scenario MOET_Depeg

# Output shows:
# "Results saved to: results/MOET_Depeg/run_003_20241201_143022/"
# "Charts generated: 6 charts saved to charts/ subfolder"

# Browse results
python3 run_tidal_simulation.py --list-results MOET_Depeg
python3 run_tidal_simulation.py --compare-runs MOET_Depeg run_002 run_003
```

**8. Technical Specifications:**
- **Use existing matplotlib/seaborn** infrastructure from `analysis/liquidation_charts.py`
- **Extend JSON serialization** to handle all simulation data types
- **Implement proper error handling** for file operations
- **Add configuration options** for chart styling and output formats
- **Ensure backward compatibility** with existing CLI interface

**9. Priority Implementation Order:**
1. **Results directory management** and auto-numbering
2. **Enhanced data capture** in simulation engine
3. **Basic chart generation** for key metrics
4. **CLI integration** and user experience
5. **Advanced comparison and analysis tools**

**10. Success Criteria:**
- Every simulation run automatically saves complete results
- Charts are generated without user intervention
- Easy comparison between different runs of same scenario
- Clear file organization that scales with usage
- Comprehensive visualization covering all important metrics
- Seamless integration with existing CLI workflow

This system should transform the current terminal-only output into a comprehensive, persistent, and visual analysis platform for the Tidal Protocol tokenomics simulation.

BUT the charts are inadequate because **the core issue is that the simulations aren't generating any actual data**. Here's what's happening and how to fix it:

## **Root Cause: Simulation Data is All Zeros**

The fundamental problem is **not the chart generation** - it's that the stress test scenarios aren't actually running properly. All the simulation data shows zeros:
- No price changes
- No liquidations  
- No agent activity
- No meaningful metrics

## **Why This Is Happening:**

1. **Scenario Execution Issue**: The stress scenarios (ETH_Flash_Crash, MOET_Depeg, etc.) are defined correctly but aren't actually being applied to the simulation engine properly.

2. **Missing Agent Activity**: The agents aren't taking actions because either:
   - They don't have initial positions to be stressed
   - The market shocks aren't being applied at the right time
   - The simulation engine isn't processing agent decisions correctly

3. **Timing Problem**: The stress events might be applied but then immediately overwritten, or applied after the simulation has already completed.

## **What You Need to Fix:**

1. **Verify Scenario Application**: Check that `_apply_price_shock()`, `_apply_moet_depeg_scenario()`, etc. are actually modifying the simulation state.

2. **Check Agent Initialization**: Ensure agents start with meaningful positions (collateral, debt) that can be stressed.

3. **Verify Simulation Flow**: The stress scenarios should apply their effects AND then run the simulation, not the other way around.

4. **Add Debug Output**: Temporarily add logging to see what's happening:
   - Are prices actually changing?
   - Do agents have positions to liquidate?
   - Are the scenario setup functions being called?

## **Quick Diagnostic Test:**

Run this to see what's actually in the simulation data:
```bash
cat tidal_protocol_sim/results/ETH_Flash_Crash/run_004_20250824_170532/results.json | head -50
```

If you see all zeros in the statistics, the problem is definitely in the simulation execution, not the charts.

**The charts can only be as good as the data they're given.** Fix the simulation data generation first, then the charts will automatically become meaningful.