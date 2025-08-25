#!/usr/bin/env python3
"""
Results Management System

Handles automatic results storage, versioning, and directory management.
"""

import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class RunMetadata:
    """Metadata for a single simulation run"""
    run_id: str
    scenario_name: str
    timestamp: str
    parameters: Dict[str, Any]
    execution_time: float
    status: str = "completed"


class ResultsManager:
    """Handles automatic results storage and versioning"""
    
    def __init__(self, base_results_dir: str = "results"):
        self.base_results_dir = Path(base_results_dir)
        self._lock = threading.Lock()
        self._ensure_results_directory()
    
    def _ensure_results_directory(self):
        """Create results directory structure if it doesn't exist"""
        self.base_results_dir.mkdir(exist_ok=True)
    
    def create_run_directory(self, scenario_name: str) -> Path:
        """
        Create a new run directory with sequential numbering
        
        Args:
            scenario_name: Name of the stress test scenario
            
        Returns:
            Path to the created run directory
        """
        with self._lock:
            scenario_dir = self.base_results_dir / scenario_name
            scenario_dir.mkdir(exist_ok=True)
            
            # Find next run number
            run_number = self._get_next_run_number(scenario_dir)
            
            # Create timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create run directory
            run_dir_name = f"run_{run_number:03d}_{timestamp}"
            run_dir = scenario_dir / run_dir_name
            run_dir.mkdir(exist_ok=True)
            
            # Create subdirectories
            (run_dir / "charts").mkdir(exist_ok=True)
            
            return run_dir
    
    def _get_next_run_number(self, scenario_dir: Path) -> int:
        """Get the next sequential run number for a scenario"""
        if not scenario_dir.exists():
            return 1
        
        run_dirs = [d for d in scenario_dir.iterdir() 
                   if d.is_dir() and d.name.startswith("run_")]
        
        if not run_dirs:
            return 1
        
        # Extract run numbers
        run_numbers = []
        for run_dir in run_dirs:
            try:
                # Parse run_XXX_timestamp format
                parts = run_dir.name.split("_")
                if len(parts) >= 2 and parts[0] == "run":
                    run_numbers.append(int(parts[1]))
            except (ValueError, IndexError):
                continue
        
        return max(run_numbers) + 1 if run_numbers else 1
    
    def save_results(
        self, 
        run_dir: Path, 
        results: Dict[str, Any], 
        metadata: RunMetadata
    ) -> Path:
        """
        Save simulation results to the run directory
        
        Args:
            run_dir: Directory to save results in
            results: Simulation results data
            metadata: Run metadata
            
        Returns:
            Path to the saved results file
        """
        # Save main results
        results_file = run_dir / "results.json"
        
        # Prepare data for serialization
        serializable_results = self._make_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save metadata
        metadata_file = run_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)
        
        return results_file
    
    def save_summary_report(self, run_dir: Path, summary: Dict[str, Any]) -> Path:
        """Save a markdown summary report"""
        summary_file = run_dir / "summary.md"
        
        with open(summary_file, 'w') as f:
            f.write(self._generate_markdown_summary(summary))
        
        return summary_file
    
    def _generate_markdown_summary(self, summary: Dict[str, Any]) -> str:
        """Generate markdown summary from results"""
        md_content = []
        
        # Header
        md_content.append("# Simulation Run Summary\n")
        
        # Metadata
        if "metadata" in summary:
            metadata = summary["metadata"]
            md_content.append("## Run Information")
            md_content.append(f"- **Scenario**: {metadata.get('scenario_name', 'Unknown')}")
            md_content.append(f"- **Timestamp**: {metadata.get('timestamp', 'Unknown')}")
            md_content.append(f"- **Execution Time**: {metadata.get('execution_time', 0):.2f}s")
            md_content.append("")
        
        # Key Metrics
        if "key_metrics" in summary:
            md_content.append("## Key Metrics")
            metrics = summary["key_metrics"]
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key.endswith("_rate") or key.endswith("_percentage"):
                        md_content.append(f"- **{key.replace('_', ' ').title()}**: {value:.2%}")
                    elif "amount" in key or "balance" in key or "capacity" in key:
                        md_content.append(f"- **{key.replace('_', ' ').title()}**: ${value:,.2f}")
                    else:
                        md_content.append(f"- **{key.replace('_', ' ').title()}**: {value:.3f}")
                else:
                    md_content.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            md_content.append("")
        
        # Risk Assessment
        if "risk_assessment" in summary:
            md_content.append("## Risk Assessment")
            assessment = summary["risk_assessment"]
            md_content.append(f"- **Overall Risk Level**: {assessment.get('risk_level', 'Unknown')}")
            md_content.append(f"- **Risk Score**: {assessment.get('risk_score', 0):.3f}")
            
            if "key_concerns" in assessment and assessment["key_concerns"]:
                md_content.append("\n### Key Concerns")
                for concern in assessment["key_concerns"]:
                    md_content.append(f"- {concern}")
            md_content.append("")
        
        # Charts Generated
        md_content.append("## Generated Charts")
        md_content.append("- Price Evolution Chart: `charts/price_evolution.png`")
        md_content.append("- Liquidation Events: `charts/liquidation_events.png`")
        md_content.append("- Protocol Health: `charts/protocol_health.png`")
        md_content.append("- Risk Metrics: `charts/risk_metrics.png`")
        
        return "\n".join(md_content)
    
    def list_scenario_runs(self, scenario_name: str) -> List[Dict[str, Any]]:
        """List all runs for a specific scenario"""
        scenario_dir = self.base_results_dir / scenario_name
        
        if not scenario_dir.exists():
            return []
        
        runs = []
        for run_dir in scenario_dir.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            
            # Load metadata if available
            metadata_file = run_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    runs.append({
                        "run_id": run_dir.name,
                        "path": str(run_dir),
                        **metadata
                    })
                except json.JSONDecodeError:
                    # Fallback to basic info
                    runs.append({
                        "run_id": run_dir.name,
                        "path": str(run_dir),
                        "scenario_name": scenario_name
                    })
        
        # Sort by run number
        runs.sort(key=lambda x: x["run_id"])
        return runs
    
    def list_all_scenarios(self) -> List[str]:
        """List all scenario directories"""
        if not self.base_results_dir.exists():
            return []
        
        scenarios = []
        for item in self.base_results_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                scenarios.append(item.name)
        
        return sorted(scenarios)
    
    def load_results(self, run_path: Path) -> Optional[Dict[str, Any]]:
        """Load results from a run directory"""
        results_file = run_path / "results.json"
        
        if not results_file.exists():
            return None
        
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    
    def load_metadata(self, run_path: Path) -> Optional[RunMetadata]:
        """Load metadata from a run directory"""
        metadata_file = run_path / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                return RunMetadata(**data)
        except (json.JSONDecodeError, TypeError):
            return None
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif isinstance(obj, dict):
            # Handle dict keys that might be enums or other non-serializable types
            serializable_dict = {}
            for k, v in obj.items():
                # Convert enum keys to strings
                if hasattr(k, 'value'):  # Enum
                    key = str(k.value)
                elif hasattr(k, 'name'):  # Enum alternative
                    key = str(k.name)
                else:
                    try:
                        json.dumps(k)  # Test if key is serializable
                        key = k
                    except (TypeError, ValueError):
                        key = str(k)
                
                serializable_dict[key] = self._make_serializable(v)
            return serializable_dict
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):  # Custom objects
            return self._make_serializable(obj.__dict__)
        else:
            try:
                json.dumps(obj)  # Test if it's already serializable
                return obj
            except (TypeError, ValueError):
                return str(obj)
