#!/usr/bin/env python3
"""
Run Liquidation Analysis

Simple script to run the liquidation analysis and generate charts
for the Tidal Protocol $2.5M liquidity setup.
"""

import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from tidal_protocol_sim.analysis.liquidation_charts import main as run_liquidation_analysis
    
    print("Starting Liquidation Analysis for Tidal Protocol")
    print("Analyzing $2.5M liquidity pool setup...")
    print("=" * 60)
    
    # Run the analysis
    run_liquidation_analysis()
    
    print("\n" + "=" * 60)
    print("✓ Liquidation analysis completed successfully!")
    print("✓ Charts have been generated and saved as PNG files")
    print("=" * 60)
    
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure you're running this from the correct directory")
    print("and that all dependencies are installed:")
    print("  pip install matplotlib seaborn pandas numpy")
    
except Exception as e:
    print(f"Error running liquidation analysis: {e}")
    import traceback
    traceback.print_exc()
