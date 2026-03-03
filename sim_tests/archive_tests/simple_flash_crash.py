#!/usr/bin/env python3
"""
Simple Flash Crash Simulation - Minimal Working Version

A simplified version of the flash crash simulation that focuses on the core mechanics
without the complex engine interdependencies.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tidal_protocol_sim.core.yield_tokens import calculate_true_yield_token_price


class SimpleFlashCrashConfig:
    """Simplified configuration for flash crash simulation"""
    
    def __init__(self, scenario: str = "moderate"):
        self.test_name = f"Simple_Flash_Crash_{scenario.title()}"
        self.simulation_duration_minutes = 3 * 24 * 60  # 3 days
        
        # Agent configuration
        self.num_agents = 150
        self.target_total_debt = 100_000_000  # $100M system debt
        
        # Flash crash timing - Day 1 at 15:00
        self.crash_start_minute = 15 * 60  # 900 minutes
        self.crash_duration_minutes = 25   # 25-minute crash
        self.crash_end_minute = self.crash_start_minute + self.crash_duration_minutes
        
        # Scenario parameters
        self.scenario = scenario
        if scenario == "mild":
            self.yt_crash_magnitude = 0.35
            self.btc_crash_magnitude = 0.12
            self.liquidity_reduction_peak = 0.60
        elif scenario == "moderate":
            self.yt_crash_magnitude = 0.50
            self.btc_crash_magnitude = 0.20
            self.liquidity_reduction_peak = 0.70
        elif scenario == "severe":
            self.yt_crash_magnitude = 0.70
            self.btc_crash_magnitude = 0.25
            self.liquidity_reduction_peak = 0.80
        
        # Pool configuration
        self.moet_yt_pool_size = 500_000  # $500K pool
        
        # Logging
        self.log_every_n_minutes = 60
        self.progress_report_every_n_minutes = 360


class SimpleAgent:
    """Simplified agent for flash crash testing"""
    
    def __init__(self, agent_id: str, btc_collateral: float, initial_hf: float):
        self.agent_id = agent_id
        self.btc_collateral = btc_collateral  # BTC amount
        self.initial_hf = initial_hf
        self.health_factor = initial_hf
        
        # Calculate debt based on collateral and HF
        btc_price = 100_000.0  # $100k BTC
        collateral_value = btc_collateral * btc_price * 0.8  # 80% collateral factor
        self.moet_debt = collateral_value / initial_hf
        
        # YT tokens purchased with MOET
        self.yt_tokens = self.moet_debt  # 1:1 initial purchase
        self.active = True
        
        # Tracking
        self.liquidation_events = []
        self.rebalancing_events = []
    
    def update_health_factor(self, btc_price: float):
        """Update health factor based on current BTC price"""
        if self.moet_debt <= 0:
            self.health_factor = float('inf')
            return
        
        collateral_value = self.btc_collateral * btc_price * 0.8  # 80% collateral factor
        self.health_factor = collateral_value / self.moet_debt
    
    def needs_liquidation(self) -> bool:
        """Check if agent needs liquidation (HF < 1.0)"""
        return self.health_factor < 1.0 and self.active
    
    def execute_liquidation(self, btc_price: float, slippage_multiplier: float = 1.0) -> Dict:
        """Execute liquidation with crash conditions"""
        if not self.needs_liquidation():
            return {}
        
        # Liquidate 50% of collateral
        btc_liquidated = self.btc_collateral * 0.5
        base_slippage = 0.02  # 2% base slippage
        total_slippage = base_slippage * slippage_multiplier
        
        # Calculate liquidation value after slippage
        gross_value = btc_liquidated * btc_price
        net_value = gross_value * (1 - total_slippage)
        
        # Reduce debt and collateral
        debt_reduction = min(net_value, self.moet_debt)
        self.moet_debt -= debt_reduction
        self.btc_collateral -= btc_liquidated
        
        # Check if agent is fully liquidated
        if self.btc_collateral <= 0.1 or self.moet_debt <= 100:
            self.active = False
        
        liquidation_event = {
            "btc_liquidated": btc_liquidated,
            "gross_value": gross_value,
            "net_value": net_value,
            "debt_reduction": debt_reduction,
            "slippage_pct": total_slippage,
            "agent_liquidated": not self.active
        }
        
        self.liquidation_events.append(liquidation_event)
        return liquidation_event


class SimpleFlashCrashSimulation:
    """Simplified flash crash simulation"""
    
    def __init__(self, config: SimpleFlashCrashConfig):
        self.config = config
        self.agents = []
        self.results = {
            "test_metadata": {
                "test_name": config.test_name,
                "timestamp": datetime.now().isoformat(),
                "scenario": config.scenario,
                "num_agents": config.num_agents,
                "crash_start_minute": config.crash_start_minute,
                "crash_duration": config.crash_duration_minutes
            },
            "detailed_logs": [],
            "liquidation_events": [],
            "oracle_events": [],
            "agent_performance": {}
        }
        
        # Initialize agents
        self._create_agents()
        
        # Set random seed
        random.seed(42)
        np.random.seed(42)
    
    def _create_agents(self):
        """Create 150 simplified agents"""
        debt_per_agent = self.config.target_total_debt / self.config.num_agents
        
        for i in range(self.config.num_agents):
            # Random BTC collateral (10-15 BTC per agent)
            btc_amount = random.uniform(10.0, 15.0)
            
            # Random health factor (1.1 - 1.2 for aggressive leverage)
            initial_hf = random.uniform(1.1, 1.2)
            
            agent = SimpleAgent(f"agent_{i+1}", btc_amount, initial_hf)
            self.agents.append(agent)
        
        total_debt = sum(agent.moet_debt for agent in self.agents)
        print(f"✅ Created {len(self.agents)} agents with ${total_debt:,.0f} total debt")
    
    def run_simulation(self) -> Dict[str, Any]:
        """Run the simplified flash crash simulation"""
        
        print("⚡ SIMPLE FLASH CRASH SIMULATION")
        print("=" * 50)
        print(f"📊 Scenario: {self.config.scenario.upper()}")
        print(f"💥 Crash: Day 1, 15:00-15:25 (minutes {self.config.crash_start_minute}-{self.config.crash_end_minute})")
        print(f"📉 YT Drop: {self.config.yt_crash_magnitude:.0%}, BTC Drop: {self.config.btc_crash_magnitude:.0%}")
        print(f"👥 Agents: {self.config.num_agents}")
        print()
        
        # Run simulation
        for minute in range(self.config.simulation_duration_minutes):
            # Calculate current prices
            btc_price = self._calculate_btc_price(minute)
            true_yt_price = calculate_true_yield_token_price(minute, 0.10, 1.0)
            
            # Apply oracle manipulation during crash
            manipulated_yt_price = self._apply_oracle_manipulation(minute, true_yt_price)
            
            # Update agent health factors
            for agent in self.agents:
                agent.update_health_factor(btc_price)
            
            # Process liquidations during crash
            if self._is_crash_active(minute):
                self._process_crash_liquidations(minute, btc_price)
            
            # Record metrics
            if minute % self.config.log_every_n_minutes == 0:
                self._record_metrics(minute, btc_price, true_yt_price, manipulated_yt_price)
            
            # Progress reporting
            if minute % self.config.progress_report_every_n_minutes == 0:
                self._print_progress(minute, btc_price)
        
        # Analyze results
        self._analyze_results()
        
        # Save results
        self._save_results()
        
        # Generate charts
        self._generate_charts()
        
        print("\n✅ Simple flash crash simulation completed!")
        self._print_summary()
        
        return self.results
    
    def _calculate_btc_price(self, minute: int) -> float:
        """Calculate BTC price with crash dynamics"""
        base_price = 100_000.0
        
        if not self._is_crash_active(minute):
            return base_price
        
        # During crash, apply crash magnitude
        crash_progress = (minute - self.config.crash_start_minute) / self.config.crash_duration_minutes
        price_reduction = self.config.btc_crash_magnitude * min(crash_progress * 2, 1.0)
        
        return base_price * (1 - price_reduction)
    
    def _apply_oracle_manipulation(self, minute: int, true_yt_price: float) -> float:
        """Apply oracle manipulation during crash"""
        if not self._is_crash_active(minute):
            return true_yt_price
        
        # Inject outliers every 5 minutes during crash
        if (minute - self.config.crash_start_minute) % 5 == 0:
            manipulated_price = true_yt_price * 0.75  # 25% negative outlier
            self.results["oracle_events"].append({
                "minute": minute,
                "true_price": true_yt_price,
                "manipulated_price": manipulated_price,
                "deviation_pct": 0.25
            })
            return manipulated_price
        
        return true_yt_price
    
    def _process_crash_liquidations(self, minute: int, btc_price: float):
        """Process liquidations during crash with higher slippage"""
        crash_progress = (minute - self.config.crash_start_minute) / self.config.crash_duration_minutes
        slippage_multiplier = 1.0 + crash_progress * 3.0  # Up to 4x slippage at peak
        
        for agent in self.agents:
            if agent.needs_liquidation():
                liquidation_event = agent.execute_liquidation(btc_price, slippage_multiplier)
                if liquidation_event:
                    liquidation_event["minute"] = minute
                    liquidation_event["agent_id"] = agent.agent_id
                    self.results["liquidation_events"].append(liquidation_event)
    
    def _record_metrics(self, minute: int, btc_price: float, true_yt_price: float, manipulated_yt_price: float):
        """Record detailed metrics"""
        active_agents = [a for a in self.agents if a.active]
        total_debt = sum(a.moet_debt for a in active_agents)
        avg_hf = np.mean([a.health_factor for a in active_agents]) if active_agents else 0
        liquidatable = sum(1 for a in active_agents if a.needs_liquidation())
        
        metrics = {
            "minute": minute,
            "hour": minute / 60,
            "phase": self._get_phase(minute),
            "btc_price": btc_price,
            "true_yt_price": true_yt_price,
            "manipulated_yt_price": manipulated_yt_price,
            "active_agents": len(active_agents),
            "total_debt": total_debt,
            "avg_health_factor": avg_hf,
            "liquidatable_agents": liquidatable
        }
        
        self.results["detailed_logs"].append(metrics)
    
    def _get_phase(self, minute: int) -> str:
        """Get current simulation phase"""
        if minute < self.config.crash_start_minute:
            return "pre_crash"
        elif minute <= self.config.crash_end_minute:
            return "crash"
        elif minute <= self.config.crash_end_minute + 120:
            return "recovery"
        else:
            return "post_recovery"
    
    def _is_crash_active(self, minute: int) -> bool:
        """Check if crash is active"""
        return self.config.crash_start_minute <= minute <= self.config.crash_end_minute
    
    def _print_progress(self, minute: int, btc_price: float):
        """Print progress report"""
        active = len([a for a in self.agents if a.active])
        liquidatable = len([a for a in self.agents if a.active and a.needs_liquidation()])
        phase = self._get_phase(minute)
        
        print(f"⏱️  Minute {minute:,} ({minute/60:.1f}h) - {phase.upper()}: "
              f"BTC=${btc_price:,.0f}, Active={active}, Liquidatable={liquidatable}")
    
    def _analyze_results(self):
        """Analyze simulation results"""
        active_agents = [a for a in self.agents if a.active]
        liquidated_agents = [a for a in self.agents if not a.active]
        
        self.results["agent_performance"] = {
            "total_agents": len(self.agents),
            "survived_agents": len(active_agents),
            "liquidated_agents": len(liquidated_agents),
            "survival_rate": len(active_agents) / len(self.agents),
            "total_liquidation_events": len(self.results["liquidation_events"]),
            "oracle_manipulation_events": len(self.results["oracle_events"])
        }
    
    def _save_results(self):
        """Save results to JSON"""
        output_dir = Path("tidal_protocol_sim/results") / self.config.test_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / f"simple_flash_crash_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"📁 Results saved to: {results_path}")
    
    def _generate_charts(self):
        """Generate simple charts"""
        if not self.results["detailed_logs"]:
            return
        
        output_dir = Path("tidal_protocol_sim/results") / self.config.test_name / "charts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.results["detailed_logs"])
        
        # Create timeline chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Simple Flash Crash Timeline - {self.config.scenario.title()} Scenario', 
                     fontsize=16, fontweight='bold')
        
        # BTC Price
        ax1.plot(df["hour"], df["btc_price"], linewidth=2, color='orange', label='BTC Price')
        ax1.axvspan(self.config.crash_start_minute/60, self.config.crash_end_minute/60, 
                   alpha=0.3, color='red', label='Crash Window')
        ax1.set_xlabel('Hours')
        ax1.set_ylabel('BTC Price ($)')
        ax1.set_title('BTC Price Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # YT Price
        ax2.plot(df["hour"], df["true_yt_price"], linewidth=2, color='green', label='True YT Price')
        ax2.axvspan(self.config.crash_start_minute/60, self.config.crash_end_minute/60, 
                   alpha=0.3, color='red')
        ax2.set_xlabel('Hours')
        ax2.set_ylabel('YT Price ($)')
        ax2.set_title('Yield Token Price (10% APR Rebasing)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Active Agents
        ax3.plot(df["hour"], df["active_agents"], linewidth=2, color='blue', label='Active Agents')
        ax3.plot(df["hour"], df["liquidatable_agents"], linewidth=2, color='red', label='Liquidatable')
        ax3.axvspan(self.config.crash_start_minute/60, self.config.crash_end_minute/60, 
                   alpha=0.3, color='red')
        ax3.set_xlabel('Hours')
        ax3.set_ylabel('Agent Count')
        ax3.set_title('Agent Status')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Health Factor
        ax4.plot(df["hour"], df["avg_health_factor"], linewidth=2, color='purple', label='Avg Health Factor')
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Liquidation Threshold')
        ax4.axvspan(self.config.crash_start_minute/60, self.config.crash_end_minute/60, 
                   alpha=0.3, color='red')
        ax4.set_xlabel('Hours')
        ax4.set_ylabel('Health Factor')
        ax4.set_title('System Health Factor')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "simple_flash_crash_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Charts saved to: {output_dir}")
    
    def _print_summary(self):
        """Print simulation summary"""
        perf = self.results["agent_performance"]
        
        print(f"\n🎯 SIMPLE FLASH CRASH SUMMARY")
        print(f"=" * 40)
        print(f"Scenario: {self.config.scenario.upper()}")
        print(f"Duration: {self.config.simulation_duration_minutes:,} minutes")
        print()
        print(f"👥 Agent Performance:")
        print(f"   Survival Rate: {perf['survival_rate']:.1%}")
        print(f"   Survived: {perf['survived_agents']}/{perf['total_agents']}")
        print(f"   Liquidation Events: {perf['total_liquidation_events']}")
        print(f"   Oracle Events: {perf['oracle_manipulation_events']}")
        print()
        print(f"💥 Crash Parameters:")
        print(f"   YT Crash: {self.config.yt_crash_magnitude:.0%}")
        print(f"   BTC Crash: {self.config.btc_crash_magnitude:.0%}")


def main():
    """Run simple flash crash simulation"""
    
    print("Simple Flash Crash Simulation")
    print("=" * 40)
    print()
    
    # Run moderate scenario
    config = SimpleFlashCrashConfig("moderate")
    simulation = SimpleFlashCrashSimulation(config)
    results = simulation.run_simulation()
    
    return results


if __name__ == "__main__":
    main()
