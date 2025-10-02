#!/usr/bin/env python3
"""
Chart Rebuilding Script

Uses the exact same chart generation functions from full_year_sim.py to rebuild
charts from existing JSON simulation results.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import glob

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the exact same simulation class to use its chart functions
from sim_tests.full_year_sim import FullYearSimulation, FullYearSimConfig
from tidal_protocol_sim.core.protocol import Asset
from tidal_protocol_sim.core.yield_tokens import calculate_true_yield_token_price


class ChartRebuilder(FullYearSimulation):
    """Chart rebuilder that inherits from FullYearSimulation to use exact same chart methods"""
    
    def __init__(self, json_file_path: str):
        """Initialize with path to JSON results file"""
        self.json_file_path = Path(json_file_path)
        
        # Create a minimal config (we won't run simulation, just use chart methods)
        config = FullYearSimConfig()
        
        # Initialize parent class
        super().__init__(config)
        
        # Load the JSON data into self.results (same format as FullYearSimulation)
        self.load_data()
        
    def load_data(self):
        """Load simulation data from JSON file into self.results"""
        print(f"📂 Loading data from: {self.json_file_path}")
        
        with open(self.json_file_path, 'r') as f:
            self.results = json.load(f)
            
        # Check what data structure we have
        detailed_logs = self.results.get('detailed_logs', [])
        pool_snapshots = self.results.get('pool_state_snapshots', [])
        simulation_results = self.results.get('simulation_results', {})
        
        print(f"✅ Loaded {len(detailed_logs)} detailed log entries")
        print(f"✅ Loaded {len(pool_snapshots)} pool state snapshots")
        print(f"✅ Loaded simulation results with keys: {list(simulation_results.keys())}")
        
    def rebuild_all_charts(self):
        """Rebuild all charts using the exact same methods from FullYearSimulation"""
        
        # Determine output directory from JSON file path
        results_dir = self.json_file_path.parent
        output_dir = results_dir / "charts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("📊 Rebuilding all charts using original FullYearSimulation methods...")
        
        # Use the exact same chart generation method from FullYearSimulation
        self._generate_test_charts()
        
        print(f"📊 Charts rebuilt and saved to: {output_dir}")


def find_latest_json_file():
    """Find the most recent JSON file in the Full_Year_2024_BTC_Simulation directory"""
    
    results_dir = Path("tidal_protocol_sim/results/Full_Year_2024_BTC_Simulation")
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return None
    
    # Find all JSON files
    json_files = list(results_dir.glob("*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in: {results_dir}")
        return None
    
    # Sort by modification time and get the latest
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    
    print(f"📁 Found latest JSON file: {latest_file.name}")
    return latest_file


def main():
    """Main function to rebuild charts from latest JSON results"""
    
    print("🔧 Chart Rebuilding Script")
    print("=" * 50)
    
    # Find the latest JSON file
    json_file = find_latest_json_file()
    
    if not json_file:
        print("❌ No JSON file found to rebuild charts from")
        return
    
    # Create chart rebuilder and rebuild all charts
    rebuilder = ChartRebuilder(json_file)
    rebuilder.rebuild_all_charts()
    
    print("\n✅ Chart rebuilding complete!")


if __name__ == "__main__":
    main()
