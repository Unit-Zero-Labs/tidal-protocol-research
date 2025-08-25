#!/usr/bin/env python3
"""
Tidal Protocol Simulation Runner

This is a convenience script to run the Tidal Protocol simulation from the root directory.
It properly handles the import paths and forwards all arguments to the main simulation.
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Run the tidal protocol simulation with proper path handling"""
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    tidal_sim_dir = script_dir / "tidal_protocol_sim"
    
    # Check if tidal_protocol_sim directory exists
    if not tidal_sim_dir.exists():
        print("Error: tidal_protocol_sim directory not found!")
        print(f"Expected location: {tidal_sim_dir}")
        return 1
    
    # Change to the tidal_protocol_sim directory and run main.py
    main_py = tidal_sim_dir / "main.py"
    
    if not main_py.exists():
        print("Error: main.py not found in tidal_protocol_sim directory!")
        return 1
    
    # Forward all command-line arguments to the main script
    cmd = [sys.executable, str(main_py)] + sys.argv[1:]
    
    try:
        # Run the command from within the tidal_protocol_sim directory
        result = subprocess.run(cmd, cwd=str(tidal_sim_dir))
        return result.returncode
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error running simulation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
