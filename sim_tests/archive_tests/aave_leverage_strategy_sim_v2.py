#!/usr/bin/env python3
"""
Aave Leverage Strategy Simulation (v2)

Uses the existing Aave infrastructure from balanced_scenario_monte_carlo.py
to test a leverage strategy where agents maintain maximum leverage (2.0 HF)
by weekly rebalancing to buy more BTC.

Key differences from v1:
- Uses existing AaveAgent class and AaveProtocolEngine
- Follows the same pattern as balanced_scenario_monte_carlo.py
- Integrates properly with existing Aave infrastructure
- Only leverages UP when BTC rises, lets HF decline naturally when BTC falls
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
import random
import csv

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tidal_protocol_sim.engine.aave_protocol_engine import AaveProtocolEngine, AaveConfig
from tidal_protocol_sim.agents.aave_agent import AaveAgent
from tidal_protocol_sim.core.protocol import TidalProtocol, Asset
from tidal_protocol_sim.core.yield_tokens import YieldTokenPool


class AaveLeverageConfig(AaveConfig):
    """Configuration for Aave Leverage Strategy simulation"""
    
    def __init__(self):
        super().__init__()
        
        # Override scenario name
        self.scenario_name = "Aave_Leverage_Strategy_2024"
        
        # Full year simulation parameters
        self.simulation_duration_hours = 24 * 365  # Full year: 8760 hours
        self.simulation_duration_minutes = 365 * 24 * 60  # 525,600 minutes
        
        # BTC pricing data configuration (same as full year sim)
        self.btc_csv_path = "btc-usd-max.csv"
        self.btc_2024_data = self._load_2024_btc_data()
        
        # BTC price scenario - Real 2024 data
        self.btc_initial_price = 42208.20  # 2024-01-01 price
        self.btc_final_price = 92627.28   # 2024-12-31 price (+119% over year)
        self.btc_decline_duration = self.simulation_duration_minutes  # Full year
        self.btc_final_price_range = (self.btc_final_price, self.btc_final_price)
        
        # Agent configuration - Same as High Tide for fair comparison
        self.num_aave_agents = 120  # 120 agents for consistency
        self.monte_carlo_agent_variation = False  # Use fixed count
        
        # Aave leverage strategy parameters
        self.target_health_factor = 2.0  # Maintain 2.0 HF (50% LTV)
        self.rebalance_interval_days = 7  # Weekly rebalancing
        self.aave_borrow_rate = 0.05  # 5% APR borrowing cost
        
        # Pool configurations (same as full year sim for consistency)
        self.moet_btc_pool_size = 10_000_000  # $10M liquidation pool
        self.moet_yield_pool_size = 500_000  # $500K pool
        self.yield_token_concentration = 0.95  # 95% concentration
        self.yield_token_ratio = 0.75  # 75% MOET, 25% YT
        
        # Override simulation parameters
        self.simulation_steps = self.simulation_duration_minutes
        self.price_update_frequency = 1  # Update every minute
        
        # Output configuration
        self.generate_charts = True
        self.save_detailed_csv = True
        self.log_every_n_minutes = 1440  # Daily logging
        self.progress_report_every_n_minutes = 10080  # Weekly progress
    
    def _load_2024_btc_data(self) -> List[float]:
        """Load 2024 BTC pricing data from CSV file (same as full year sim)"""
        btc_prices = []
        
        try:
            with open(self.btc_csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if '2024-' in row['snapped_at']:
                        price = float(row['price'])
                        btc_prices.append(price)
            
            print(f"📊 Loaded {len(btc_prices)} days of 2024 BTC pricing data")
            print(f"📈 2024 BTC Range: ${btc_prices[0]:,.2f} → ${btc_prices[-1]:,.2f} ({((btc_prices[-1]/btc_prices[0])-1)*100:+.1f}%)")
            
            return btc_prices
            
        except FileNotFoundError:
            print(f"⚠️  Warning: {self.btc_csv_path} not found. Using synthetic 2024 data.")
            return self._generate_synthetic_2024_data()
        except Exception as e:
            print(f"❌ Error loading BTC data: {e}. Using synthetic data.")
            return self._generate_synthetic_2024_data()
    
    def _generate_synthetic_2024_data(self) -> List[float]:
        """Generate synthetic 2024-like BTC price progression as fallback"""
        days = 366
        prices = []
        
        start_price = 42208.20
        end_price = 92627.28
        
        for day in range(days):
            progress = day / (days - 1)
            base_price = start_price + (end_price - start_price) * progress
            
            # Add volatility
            volatility = random.uniform(-0.05, 0.05)
            daily_price = base_price * (1 + volatility)
            daily_price = max(daily_price, 10000.0)
            prices.append(daily_price)
        
        return prices
        
    def get_btc_price_at_minute(self, minute: int) -> float:
        """Get BTC price at given minute using real 2024 data with interpolation"""
        if not self.btc_2024_data:
            progress = minute / self.simulation_duration_minutes
            return self.btc_initial_price + (self.btc_final_price - self.btc_initial_price) * progress
        
        minutes_per_day = 24 * 60
        day_of_year = minute // minutes_per_day
        
        if day_of_year >= len(self.btc_2024_data):
            return self.btc_2024_data[-1]
        
        current_day_price = self.btc_2024_data[day_of_year]
        
        if day_of_year + 1 < len(self.btc_2024_data):
            next_day_price = self.btc_2024_data[day_of_year + 1]
            minutes_into_day = minute % minutes_per_day
            daily_progress = minutes_into_day / minutes_per_day
            interpolated_price = current_day_price + (next_day_price - current_day_price) * daily_progress
            return interpolated_price
        else:
            return current_day_price


class AaveLeverageAnalysisEngine(AaveProtocolEngine):
    """Aave Engine with leverage strategy and analysis tracking"""
    
    def __init__(self, config: AaveLeverageConfig):
        super().__init__(config)
        self.leverage_config = config
        
        # Enhanced tracking for leverage strategy
        self.time_series_data = {
            "timestamps": [],
            "btc_prices": [],
            "agent_states": {},
            "leverage_events": [],
            "liquidation_events": []
        }
        
        # Agent health history tracking (like full year sim)
        self.agent_health_history = []
        
        # Override BTC price manager to use 2024 data
        self._setup_2024_btc_price_manager()
    
    def _setup_2024_btc_price_manager(self):
        """Setup BTC price manager to use real 2024 data"""
        from tidal_protocol_sim.engine.btc_price_manager import BTCPriceDeclineManager
        
        # Create a custom price manager that follows 2024 BTC progression
        self.btc_price_manager = BTCPriceDeclineManager(
            initial_price=self.leverage_config.btc_initial_price,
            duration=self.leverage_config.simulation_duration_minutes,
            final_price_range=(self.leverage_config.btc_final_price, self.leverage_config.btc_final_price)
        )
        
        # Override the update method to use our 2024 data
        original_update = self.btc_price_manager.update_btc_price
        
        def custom_update_btc_price(minute: int) -> float:
            return self.leverage_config.get_btc_price_at_minute(minute)
        
        self.btc_price_manager.update_btc_price = custom_update_btc_price
    
    def run_leverage_simulation(self) -> Dict[str, Any]:
        """Run Aave leverage strategy simulation with weekly rebalancing"""
        
        print(f"🏦 Starting Aave leverage simulation with {len(self.aave_agents)} agents")
        print(f"📈 BTC progression: ${self.leverage_config.btc_initial_price:,.0f} → ${self.leverage_config.btc_final_price:,.0f}")
        print(f"🎯 Strategy: Weekly rebalancing to maintain 2.0 HF (only leverage UP)")
        print()
        
        # Initialize tracking
        self.btc_price_history = []
        self.liquidation_events = []
        self.current_step = 0
        
        # Run simulation loop
        for minute in range(self.leverage_config.simulation_duration_minutes):
            self.current_step = minute
            
            # Update BTC price using 2024 data
            new_btc_price = self.btc_price_manager.update_btc_price(minute)
            self.state.current_prices[Asset.BTC] = new_btc_price
            
            # Store BTC price daily
            if minute % 1440 == 0:  # Daily
                self.btc_price_history.append(new_btc_price)
            
            # Update protocol state
            self.protocol.current_block = minute
            self.protocol.accrue_interest()
            
            # Update agent debt interest
            self._update_agent_debt_interest(minute)
            
            # Process Aave agents with leverage strategy
            self._process_aave_agents_with_leverage_strategy(minute)
            
            # Check for liquidations
            self._check_aave_liquidations(minute)
            
            # Capture agent health snapshots daily
            if minute % 1440 == 0:  # Daily
                self._capture_agent_health_snapshot(minute, new_btc_price)
            
            # Progress reporting
            if minute % self.leverage_config.progress_report_every_n_minutes == 0:  # Weekly
                days = minute / (60 * 24)
                weeks = days / 7
                active_agents = len([a for a in self.aave_agents if a.active])
                print(f"⏱️  Week {weeks:.0f}/52 (Day {days:.0f}/365) - BTC: ${new_btc_price:,.0f}, Active agents: {active_agents}")
        
        # Generate results
        results = self._generate_leverage_results()
        
        return results
    
    def _process_aave_agents_with_leverage_strategy(self, minute: int):
        """Process Aave agents with leverage strategy logic"""
        
        asset_prices = {Asset.BTC: self.state.current_prices[Asset.BTC]}
        
        for agent in self.aave_agents:
            if not agent.active:
                continue
            
            # Update agent health factor
            agent._update_health_factor(asset_prices)
            
            # Check if it's time for weekly rebalancing
            if self._should_rebalance_agent(agent, minute):
                self._execute_leverage_rebalancing(agent, minute, asset_prices)
    
    def _should_rebalance_agent(self, agent: AaveAgent, minute: int) -> bool:
        """Check if agent should rebalance (weekly schedule)"""
        
        # Get last rebalance time (stored in agent state if available)
        if not hasattr(agent.state, 'last_leverage_rebalance_minute'):
            agent.state.last_leverage_rebalance_minute = 0
        
        minutes_since_last_rebalance = minute - agent.state.last_leverage_rebalance_minute
        rebalance_interval_minutes = self.leverage_config.rebalance_interval_days * 24 * 60  # 7 days
        
        return minutes_since_last_rebalance >= rebalance_interval_minutes
    
    def _execute_leverage_rebalancing(self, agent: AaveAgent, minute: int, asset_prices: Dict[Asset, float]):
        """Execute leverage rebalancing for agent"""
        
        btc_price = asset_prices.get(Asset.BTC, 100_000.0)
        current_hf = agent.state.health_factor
        target_hf = self.leverage_config.target_health_factor
        
        # Update last rebalance time
        agent.state.last_leverage_rebalance_minute = minute
        
        if current_hf > target_hf:
            # BTC price went up, we can borrow more and buy more BTC to get back to 2.0 HF
            self._leverage_up_agent(agent, minute, asset_prices)
        else:
            # BTC price went down - DO NOT LEVERAGE DOWN
            # Let health factor naturally decline, risk liquidation
            print(f"⚠️  {agent.agent_id}: HF={current_hf:.3f} < target={target_hf:.1f}, holding position (risk liquidation)")
    
    def _leverage_up_agent(self, agent: AaveAgent, minute: int, asset_prices: Dict[Asset, float]):
        """Leverage up agent to get back to 2.0 HF"""
        
        btc_price = asset_prices.get(Asset.BTC, 100_000.0)
        
        # Calculate current position
        current_btc = agent.state.supplied_balances.get(Asset.BTC, 0.0)
        current_debt = agent.state.moet_debt
        
        # Calculate how much more we can borrow to reach 2.0 HF
        current_collateral_value = current_btc * btc_price * 0.80  # With collateral factor
        target_debt = current_collateral_value / self.leverage_config.target_health_factor
        additional_debt = target_debt - current_debt
        
        if additional_debt > 100:  # Only if meaningful amount (>$100)
            # Borrow more MOET
            agent.state.moet_debt += additional_debt
            
            # Use MOET to buy more BTC
            additional_btc = additional_debt / btc_price
            agent.state.supplied_balances[Asset.BTC] += additional_btc
            
            # Update protocol state
            btc_pool = self.protocol.asset_pools[Asset.BTC]
            btc_pool.total_supplied += additional_btc
            self.protocol.moet_system.mint(additional_debt)
            
            # Record leverage event
            leverage_event = {
                "minute": minute,
                "agent_id": agent.agent_id,
                "type": "leverage_up",
                "btc_price": btc_price,
                "hf_before": agent.state.health_factor,
                "additional_debt": additional_debt,
                "additional_btc": additional_btc,
                "total_btc_after": agent.state.supplied_balances.get(Asset.BTC, 0.0),
                "total_debt_after": agent.state.moet_debt
            }
            
            # Update health factor after leveraging
            agent._update_health_factor(asset_prices)
            leverage_event["hf_after"] = agent.state.health_factor
            
            self.time_series_data["leverage_events"].append(leverage_event)
            
            print(f"📈 {agent.agent_id}: Leveraged up - Borrowed ${additional_debt:,.0f}, bought {additional_btc:.4f} BTC (HF: {leverage_event['hf_before']:.3f} → {leverage_event['hf_after']:.3f})")
    
    def _capture_agent_health_snapshot(self, minute: int, btc_price: float):
        """Capture agent health snapshot (like full year sim)"""
        
        health_snapshot = {
            "minute": minute,
            "day": minute // 1440,
            "btc_price": btc_price,
            "agents": []
        }
        
        for agent in self.aave_agents:
            # Calculate net position value
            btc_value = agent.state.supplied_balances.get(Asset.BTC, 0.0) * btc_price
            net_position_value = btc_value - agent.state.moet_debt
            
            agent_summary = {
                "agent_id": agent.agent_id,
                "active": agent.active,
                "liquidated": hasattr(agent.state, 'liquidated') and agent.state.liquidated,
                "health_factor": agent.state.health_factor,
                "btc_amount": agent.state.supplied_balances.get(Asset.BTC, 0.0),
                "moet_debt": agent.state.moet_debt,
                "net_position_value": net_position_value,
                "yield_token_value": 0.0,  # Aave strategy doesn't have yield tokens
                "rebalancing_events": len(getattr(agent.state, 'rebalancing_events', [])),
                "liquidation_events": len(getattr(agent.state, 'liquidation_events', []))
            }
            
            health_snapshot["agents"].append(agent_summary)
        
        self.agent_health_history.append(health_snapshot)
    
    def _generate_leverage_results(self) -> Dict[str, Any]:
        """Generate comprehensive leverage strategy results"""
        
        # Get final agent summaries
        final_btc_price = self.btc_price_history[-1] if self.btc_price_history else self.leverage_config.btc_final_price
        
        agent_outcomes = []
        for agent in self.aave_agents:
            btc_value = agent.state.supplied_balances.get(Asset.BTC, 0.0) * final_btc_price
            net_position_value = btc_value - agent.state.moet_debt
            
            agent_outcome = {
                "agent_id": agent.agent_id,
                "active": agent.active,
                "liquidated": hasattr(agent.state, 'liquidated') and agent.state.liquidated,
                "health_factor": agent.state.health_factor,
                "btc_amount": agent.state.supplied_balances.get(Asset.BTC, 0.0),
                "moet_debt": agent.state.moet_debt,
                "net_position_value": net_position_value,
                "total_interest_paid": getattr(agent.state, 'total_interest_accrued', 0),
                "leverage_events": len([e for e in self.time_series_data["leverage_events"] if e["agent_id"] == agent.agent_id]),
                "liquidation_events": len([e for e in self.liquidation_events if e.get("agent_id") == agent.agent_id])
            }
            
            agent_outcomes.append(agent_outcome)
        
        # Calculate summary statistics
        total_agents = len(self.aave_agents)
        active_agents = len([a for a in self.aave_agents if a.active])
        liquidated_agents = total_agents - active_agents
        
        results = {
            "test_metadata": {
                "test_name": "Aave_Leverage_Strategy_2024",
                "timestamp": datetime.now().isoformat(),
                "duration_hours": self.leverage_config.simulation_duration_hours,
                "num_agents": total_agents,
                "strategy": "aave_leverage_2.0_hf",
                "rebalance_interval_days": self.leverage_config.rebalance_interval_days
            },
            "agent_outcomes": agent_outcomes,
            "btc_price_history": self.btc_price_history,
            "agent_health_history": self.agent_health_history,
            "leverage_events": self.time_series_data["leverage_events"],
            "liquidation_events": self.liquidation_events,
            "summary_statistics": {
                "total_agents": total_agents,
                "active_agents": active_agents,
                "liquidated_agents": liquidated_agents,
                "survival_rate": active_agents / total_agents if total_agents > 0 else 0,
                "total_leverage_events": len(self.time_series_data["leverage_events"]),
                "total_liquidations": len(self.liquidation_events)
            }
        }
        
        return results


def create_aave_leverage_agents(config: AaveLeverageConfig) -> List[AaveAgent]:
    """Create Aave leverage strategy agents using existing AaveAgent class"""
    
    agents = []
    
    for i in range(config.num_aave_agents):
        agent_id = f"aave_leverage_agent_{i:03d}"
        
        # Start all agents at 2.0 HF (maximum safe leverage)
        initial_hf = 2.0
        target_hf = 2.0  # Target to maintain
        
        # Create agent using existing AaveAgent class
        agent = AaveAgent(
            agent_id,
            initial_hf,
            target_hf  # Not used for rebalancing in AAVE, but kept for comparison
        )
        
        # Initialize with 1 BTC at 2024 starting price
        initial_btc_price = config.btc_initial_price
        agent.state.supplied_balances[Asset.BTC] = 1.5  # 1.5 BTC (1 original + 0.5 leveraged)
        agent.state.moet_debt = initial_btc_price * 0.5  # 50% LTV debt
        
        # Initialize leverage tracking fields
        agent.state.last_leverage_rebalance_minute = 0
        
        agents.append(agent)
        
        print(f"🏦 {agent_id}: Initialized with 1.5 BTC, ${agent.state.moet_debt:,.0f} MOET debt (2.0 HF) at ${initial_btc_price:,.0f}/BTC")
    
    return agents


class AaveLeverageSimulation:
    """Main simulation class for Aave Leverage Strategy testing"""
    
    def __init__(self):
        self.config = AaveLeverageConfig()
        self.results = {}
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
    def run_simulation(self) -> Dict[str, Any]:
        """Run the complete Aave leverage strategy simulation"""
        
        print("🏦 AAVE LEVERAGE STRATEGY SIMULATION (v2)")
        print("=" * 70)
        print(f"📅 Duration: {self.config.simulation_duration_hours:,} hours ({self.config.simulation_duration_hours//24} days)")
        print(f"👥 Agents: {self.config.num_aave_agents} Aave leverage agents (2.0 HF target)")
        print(f"📈 BTC 2024 Journey: ${self.config.btc_initial_price:,.0f} → ${self.config.btc_final_price:,.0f} ({((self.config.btc_final_price/self.config.btc_initial_price)-1)*100:+.1f}%)")
        print(f"🔄 Strategy: Weekly rebalancing to maintain 2.0 HF (ONLY leverage UP)")
        print(f"💰 Borrow Rate: {self.config.aave_borrow_rate:.1%} APR")
        print(f"⚠️  Risk: Liquidation if BTC drops >20% between weekly rebalances")
        print()
        
        # Create engine
        engine = AaveLeverageAnalysisEngine(self.config)
        
        # Create agents
        agents = create_aave_leverage_agents(self.config)
        engine.aave_agents = agents
        
        # Add agents to engine's agent dict
        engine.agents = {}
        for agent in agents:
            engine.agents[agent.agent_id] = agent
            agent.engine = engine  # Set engine reference
        
        print(f"✅ Created {len(agents)} Aave leverage agents")
        print(f"   Strategy: Maintain 2.0 HF through weekly rebalancing (only leverage UP)")
        print(f"   Risk Management: Let HF decline naturally when BTC falls")
        print()
        
        # Run simulation
        simulation_results = engine.run_leverage_simulation()
        
        # Store results
        self.results = simulation_results
        
        # Save results
        self._save_results()
        
        # Generate charts
        if self.config.generate_charts:
            self._generate_charts()
        
        print("\n✅ Aave leverage strategy simulation completed!")
        self._print_summary()
        
        return self.results
    
    def _save_results(self):
        """Save simulation results"""
        output_dir = Path("tidal_protocol_sim/results") / self.config.scenario_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results JSON
        results_path = output_dir / f"aave_leverage_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert for JSON serialization
        json_results = self._convert_for_json(self.results)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"📁 Results saved to: {results_path}")
        
        # Save CSV
        if self.config.save_detailed_csv:
            self._save_csv(output_dir)
    
    def _save_csv(self, output_dir: Path):
        """Save detailed CSV files"""
        agent_data = self.results["agent_outcomes"]
        agent_df = pd.DataFrame(agent_data)
        agent_csv_path = output_dir / "aave_leverage_agent_performance.csv"
        agent_df.to_csv(agent_csv_path, index=False)
        
        print(f"📊 CSV saved to: {agent_csv_path}")
    
    def _convert_for_json(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {str(key): self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif hasattr(obj, 'dtype') and 'float' in str(obj.dtype):
            return float(obj)
        elif hasattr(obj, 'dtype') and 'int' in str(obj.dtype):
            return int(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def _generate_charts(self):
        """Generate analysis charts"""
        output_dir = Path("tidal_protocol_sim/results") / self.config.scenario_name / "charts"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("📊 Generating Aave leverage strategy charts...")
        
        # Chart 1: Agent Performance Overview
        self._create_agent_performance_chart(output_dir)
        
        # Chart 2: BTC Price vs Net Position Evolution
        self._create_performance_evolution_chart(output_dir)
        
        print(f"📊 Charts saved to: {output_dir}")
    
    def _create_agent_performance_chart(self, output_dir: Path):
        """Create agent performance overview chart"""
        agent_data = self.results["agent_outcomes"]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Aave Leverage Strategy: Agent Performance Analysis', fontsize=16, fontweight='bold')
        
        # Chart 1: Survival vs Liquidation
        active_count = len([a for a in agent_data if a["active"]])
        liquidated_count = len([a for a in agent_data if a["liquidated"]])
        
        ax1.pie([active_count, liquidated_count], labels=['Active', 'Liquidated'], 
               colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Agent Survival Rate')
        
        # Chart 2: Net Position Distribution
        net_positions = [a["net_position_value"] for a in agent_data if a["active"]]
        if net_positions:
            ax2.hist(net_positions, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax2.set_xlabel('Net Position Value ($)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Net Position Distribution (Active Agents)')
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Chart 3: Leverage Events by Agent
        leverage_counts = [a["leverage_events"] for a in agent_data]
        colors = ['green' if a["active"] else 'red' for a in agent_data]
        
        bars = ax3.bar(range(len(agent_data)), leverage_counts, color=colors, alpha=0.7)
        ax3.set_xlabel('Agent')
        ax3.set_ylabel('Number of Leverage Events')
        ax3.set_title('Leverage Activity by Agent')
        ax3.set_xticks(range(0, len(agent_data), 10))
        ax3.set_xticklabels([f"A{i}" for i in range(0, len(agent_data), 10)])
        
        # Chart 4: Final Health Factor Distribution
        final_hfs = [a["health_factor"] for a in agent_data if a["active"]]
        if final_hfs:
            ax4.hist(final_hfs, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax4.axvline(x=2.0, color='red', linestyle='--', label='Target HF (2.0)')
            ax4.set_xlabel('Final Health Factor')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Final Health Factor Distribution')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "aave_leverage_agent_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_evolution_chart(self, output_dir: Path):
        """Create performance evolution over time chart"""
        btc_history = self.results.get("btc_price_history", [])
        health_history = self.results.get("agent_health_history", [])
        
        if not btc_history or not health_history:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Aave Leverage Strategy: Performance Evolution Over 2024', fontsize=16, fontweight='bold')
        
        # Extract time series data
        days = list(range(len(btc_history)))
        btc_prices = btc_history
        
        # Calculate average net position over time
        avg_net_positions = []
        for snapshot in health_history:
            agents = snapshot.get("agents", [])
            if agents:
                net_positions = [a["net_position_value"] for a in agents if a["active"]]
                avg_net_positions.append(np.mean(net_positions) if net_positions else 0)
            else:
                avg_net_positions.append(0)
        
        # Chart 1: BTC Price Evolution
        ax1.plot(days, btc_prices, linewidth=2, color='orange', label='BTC Price')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('BTC Price ($)')
        ax1.set_title('BTC Price Evolution (2024)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Chart 2: Average Net Position Evolution
        ax2.plot(days[:len(avg_net_positions)], avg_net_positions, linewidth=2, color='blue', label='Avg Net Position')
        ax2.axhline(y=42208.20, color='gray', linestyle='--', alpha=0.5, label='Initial Position (1 BTC @ $42,208)')
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Average Net Position ($)')
        ax2.set_title('Average Agent Net Position Evolution')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "aave_leverage_performance_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_summary(self):
        """Print simulation summary"""
        print("\n📊 AAVE LEVERAGE STRATEGY SUMMARY")
        print("=" * 50)
        
        summary = self.results["summary_statistics"]
        print(f"👥 Agent Performance:")
        print(f"   Total Agents: {summary['total_agents']}")
        print(f"   Active: {summary['active_agents']} ({summary['survival_rate']:.1%})")
        print(f"   Liquidated: {summary['liquidated_agents']}")
        print(f"   Total Leverage Events: {summary['total_leverage_events']:,}")
        print(f"   Total Liquidations: {summary['total_liquidations']:,}")
        
        # Calculate average final position
        active_agents = [a for a in self.results["agent_outcomes"] if a["active"]]
        if active_agents:
            avg_final_position = np.mean([a["net_position_value"] for a in active_agents])
            initial_position = 42208.20  # 1 BTC at 2024 starting price
            total_return = (avg_final_position / initial_position - 1) * 100
            print(f"   Average Final Net Position: ${avg_final_position:,.2f}")
            print(f"   Average Total Return: {total_return:+.1f}%")


def main():
    """Main execution function"""
    
    print("Aave Leverage Strategy Simulation (v2)")
    print("=" * 50)
    print()
    print("This simulation tests an alternative strategy where agents use Aave to")
    print("maintain maximum leverage (2.0 HF) through weekly rebalancing.")
    print()
    print("Key Strategy:")
    print("• Start with 1.5 BTC (1 original + 0.5 leveraged) at 2.0 HF")
    print("• Weekly rebalancing: ONLY leverage UP when BTC rises")
    print("• When BTC falls: Let HF decline naturally, risk liquidation")
    print("• Uses existing Aave infrastructure for proper integration")
    print()
    
    # Run simulation
    try:
        simulation = AaveLeverageSimulation()
        results = simulation.run_simulation()
        
        print(f"\n🎯 AAVE LEVERAGE SIMULATION COMPLETED!")
        return results
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Simulation interrupted by user.")
        return None
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
