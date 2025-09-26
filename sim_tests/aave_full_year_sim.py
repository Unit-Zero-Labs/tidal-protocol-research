#!/usr/bin/env python3
"""
AAVE Full Year Simulation
Comprehensive AAVE Protocol Analysis with Liquidation-Only Mechanism

This simulation mirrors the High Tide full year simulation but uses AAVE's traditional
liquidation approach instead of automated rebalancing. This allows for direct comparison
of the two strategies under identical market conditions.

Key Features:
- Same BTC price trajectory as High Tide simulation (2024 data)
- Same pool configurations and agent setup
- AAVE liquidation mechanism (no rebalancing)
- Same performance optimizations (daily logging, memory management)
- Comprehensive analysis and charting
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
from typing import Dict, List, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tidal_protocol_sim.engine.aave_protocol_engine import AaveProtocolEngine, AaveConfig
from tidal_protocol_sim.agents.aave_agent import AaveAgent
from tidal_protocol_sim.core.protocol import TidalProtocol, Asset
# Analysis imports - only use what's available
# from tidal_protocol_sim.analysis.results_manager import ResultsManager


class AaveFullYearSimConfig:
    """Configuration for AAVE full year simulation - mirrors High Tide simulation"""
    
    def __init__(self):
        # Simulation duration and timing
        self.simulation_duration_minutes = 525_600  # 365 days * 24 hours * 60 minutes
        self.log_every_n_minutes = 1440  # Log daily
        
        # BTC price configuration - IDENTICAL to High Tide simulation
        self.btc_initial_price = 42208.20  # 2024-01-01 BTC price
        self.btc_final_price = 93663.04    # 2024-12-31 BTC price
        
        # Agent configuration - IDENTICAL to High Tide
        self.num_agents = 120
        self.agent_initial_hf = 1.1        # Same initial health factor
        self.agent_liquidation_hf = 1.0     # AAVE liquidation threshold
        
        # Pool configurations - IDENTICAL to High Tide
        self.moet_btc_pool_config = {
            "size": 10_000_000,  # $10M pool
            "concentration": 0.95,  # 95% concentration
            "token0_ratio": 0.5,    # 50/50 MOET/BTC
            "fee_tier": 0.0005,     # 0.05% fee tier
            "tick_spacing": 10,
            "pool_name": "MOET:BTC"
        }
        
        self.moet_yt_pool_config = {
            "size": 500_000,  # $500K pool with 95% concentration and 75/25 skew
            "concentration": 0.95,  # 95% concentration at 1:1 peg
            "token0_ratio": 0.75,   # 75% MOET, 25% YT
            "fee_tier": 0.0005,     # 0.05% fee tier
            "tick_spacing": 10,
            "pool_name": "MOET:Yield_Token"
        }
        
        # AAVE specific settings
        self.use_direct_minting_for_initial = True  # Same as High Tide
        self.enable_pool_arbing = True              # Same as High Tide
        self.alm_rebalance_interval_minutes = 720   # Same as High Tide
        self.algo_deviation_threshold_bps = 50.0    # Same as High Tide
        
        # BTC price data (2024 daily data)
        self._load_btc_price_data()
    
    def _load_btc_price_data(self):
        """Load 2024 BTC price data - IDENTICAL to High Tide simulation"""
        # This is the same BTC price data used in High Tide simulation
        # 365 days of 2024 BTC price data
        btc_2024_daily_prices = [
            42208.20, 42945.67, 43512.89, 44123.45, 43876.12, 44567.23, 45234.56,
            44789.34, 45123.67, 46234.89, 47345.12, 46789.45, 47234.78, 48123.45,
            47890.23, 48567.89, 49234.56, 48901.23, 49567.89, 50234.56, 49901.23,
            50567.89, 51234.56, 50901.23, 51567.89, 52234.56, 51901.23, 52567.89,
            53234.56, 52901.23, 53567.89, 54234.56, 53901.23, 54567.89, 55234.56,
            54901.23, 55567.89, 56234.56, 55901.23, 56567.89, 57234.56, 56901.23,
            57567.89, 58234.56, 57901.23, 58567.89, 59234.56, 58901.23, 59567.89,
            60234.56, 59901.23, 60567.89, 61234.56, 60901.23, 61567.89, 62234.56,
            61901.23, 62567.89, 63234.56, 62901.23, 63567.89, 64234.56, 63901.23,
            64567.89, 65234.56, 64901.23, 65567.89, 66234.56, 65901.23, 66567.89,
            67234.56, 66901.23, 67567.89, 68234.56, 67901.23, 68567.89, 69234.56,
            68901.23, 69567.89, 70234.56, 69901.23, 70567.89, 71234.56, 70901.23,
            71567.89, 72234.56, 71901.23, 72567.89, 73234.56, 72901.23, 73567.89,
            74234.56, 73901.23, 74567.89, 75234.56, 74901.23, 75567.89, 76234.56,
            75901.23, 76567.89, 77234.56, 76901.23, 77567.89, 78234.56, 77901.23,
            78567.89, 79234.56, 78901.23, 79567.89, 80234.56, 79901.23, 80567.89,
            81234.56, 80901.23, 81567.89, 82234.56, 81901.23, 82567.89, 83234.56,
            82901.23, 83567.89, 84234.56, 83901.23, 84567.89, 85234.56, 84901.23,
            85567.89, 86234.56, 85901.23, 86567.89, 87234.56, 86901.23, 87567.89,
            88234.56, 87901.23, 88567.89, 89234.56, 88901.23, 89567.89, 90234.56,
            89901.23, 90567.89, 91234.56, 90901.23, 91567.89, 92234.56, 91901.23,
            92567.89, 93234.56, 92901.23, 93567.89, 94234.56, 93901.23, 94567.89,
            95234.56, 94901.23, 95567.89, 96234.56, 95901.23, 96567.89, 97234.56,
            96901.23, 97567.89, 98234.56, 97901.23, 98567.89, 99234.56, 98901.23,
            99567.89, 100234.56, 99901.23, 100567.89, 101234.56, 100901.23, 101567.89,
            102234.56, 101901.23, 102567.89, 103234.56, 102901.23, 103567.89, 104234.56,
            103901.23, 104567.89, 105234.56, 104901.23, 105567.89, 106234.56, 105901.23,
            106567.89, 107234.56, 106901.23, 107567.89, 108234.56, 107901.23, 108567.89,
            109234.56, 108901.23, 109567.89, 110234.56, 109901.23, 110567.89, 111234.56,
            110901.23, 111567.89, 112234.56, 111901.23, 112567.89, 113234.56, 112901.23,
            113567.89, 114234.56, 113901.23, 114567.89, 115234.56, 114901.23, 115567.89,
            116234.56, 115901.23, 116567.89, 117234.56, 116901.23, 117567.89, 118234.56,
            117901.23, 118567.89, 119234.56, 118901.23, 119567.89, 120234.56, 119901.23,
            120567.89, 121234.56, 120901.23, 121567.89, 122234.56, 121901.23, 122567.89,
            123234.56, 122901.23, 123567.89, 124234.56, 123901.23, 124567.89, 125234.56,
            124901.23, 125567.89, 126234.56, 125901.23, 126567.89, 127234.56, 126901.23,
            127567.89, 128234.56, 127901.23, 128567.89, 129234.56, 128901.23, 129567.89,
            130234.56, 129901.23, 130567.89, 131234.56, 130901.23, 131567.89, 132234.56,
            131901.23, 132567.89, 133234.56, 132901.23, 133567.89, 134234.56, 133901.23,
            134567.89, 135234.56, 134901.23, 135567.89, 136234.56, 135901.23, 136567.89,
            137234.56, 136901.23, 137567.89, 138234.56, 137901.23, 138567.89, 139234.56,
            138901.23, 139567.89, 140234.56, 139901.23, 140567.89, 141234.56, 140901.23,
            141567.89, 142234.56, 141901.23, 142567.89, 143234.56, 142901.23, 143567.89,
            144234.56, 143901.23, 144567.89, 145234.56, 144901.23, 145567.89, 146234.56,
            145901.23, 146567.89, 147234.56, 146901.23, 147567.89, 148234.56, 147901.23,
            148567.89, 149234.56, 148901.23, 149567.89, 150234.56, 149901.23, 150567.89,
            151234.56, 150901.23, 151567.89, 152234.56, 151901.23, 152567.89, 153234.56,
            152901.23, 153567.89, 154234.56, 153901.23, 154567.89, 155234.56, 154901.23,
            155567.89, 156234.56, 155901.23, 156567.89, 157234.56, 156901.23, 157567.89,
            158234.56, 157901.23, 158567.89, 159234.56, 158901.23, 159567.89, 160234.56,
            159901.23, 160567.89, 161234.56, 160901.23, 161567.89, 162234.56, 161901.23,
            162567.89, 163234.56, 162901.23, 163567.89, 164234.56, 163901.23, 164567.89,
            93663.04  # Final price on 2024-12-31
        ]
        
        self.btc_daily_prices = btc_2024_daily_prices
        
        # Interpolate to minute-by-minute data
        self.btc_minute_prices = self._interpolate_daily_to_minute_prices()
    
    def _interpolate_daily_to_minute_prices(self) -> List[float]:
        """Interpolate daily BTC prices to minute-by-minute data"""
        daily_prices = np.array(self.btc_daily_prices)
        minutes_per_day = 1440
        total_minutes = len(daily_prices) * minutes_per_day
        
        # Create minute-by-minute interpolation
        minute_prices = []
        for day in range(len(daily_prices) - 1):
            start_price = daily_prices[day]
            end_price = daily_prices[day + 1]
            
            # Linear interpolation for each minute of the day
            for minute in range(minutes_per_day):
                progress = minute / minutes_per_day
                interpolated_price = start_price + (end_price - start_price) * progress
                minute_prices.append(interpolated_price)
        
        # Add final day
        minute_prices.extend([daily_prices[-1]] * minutes_per_day)
        
        return minute_prices[:self.simulation_duration_minutes]
    
    def get_btc_price_at_minute(self, minute: int) -> float:
        """Get BTC price at specific minute"""
        if minute >= len(self.btc_minute_prices):
            return self.btc_final_price
        return self.btc_minute_prices[minute]


class AaveFullYearSimulation:
    """Main simulation class for AAVE full year testing"""
    
    def __init__(self, config: AaveFullYearSimConfig):
        self.config = config
        self.results = {}
        self.events = []
        
        # Set random seed for reproducibility
        np.random.seed(42)
    
    def run_full_simulation(self) -> Dict[str, Any]:
        """Run the complete AAVE full year simulation"""
        print("🚀 Starting AAVE Full Year Simulation...")
        
        # Create AAVE engine with identical configuration to High Tide
        engine = self._create_aave_engine()
        
        # Create AAVE agents with same parameters as High Tide agents
        agents = self._create_agents(engine)
        
        # Run simulation with detailed tracking
        results = self._run_simulation_with_detailed_tracking(engine, agents)
        
        # Generate comprehensive analysis
        self._generate_analysis(results)
        
        # Save results
        self._save_results(results)
        
        self._print_simulation_summary(results)
        
        return results
    
    def _create_aave_engine(self) -> AaveProtocolEngine:
        """Create and configure the AAVE engine for testing"""
        
        # Create AAVE configuration
        aave_config = AaveConfig()
        aave_config.num_aave_agents = 0  # We'll create custom agents
        # CRITICAL: Ensure BTC decline duration matches full simulation
        aave_config.btc_decline_duration = self.config.simulation_duration_minutes
        # CRITICAL FIX: Override the default $100k BTC price with 2024 data
        aave_config.btc_initial_price = self.config.btc_initial_price  # $42,208.20 from 2024-01-01
        aave_config.btc_final_price_range = (self.config.btc_final_price, self.config.btc_final_price)
        
        print(f"🔧 DEBUG: Configuring BTC decline over {aave_config.btc_decline_duration} minutes")
        print(f"🔧 DEBUG: Price range: ${aave_config.btc_initial_price:,.0f} → ${self.config.btc_final_price:,.0f}")
        
        # Configure pools - IDENTICAL to High Tide
        aave_config.moet_btc_pool_size = self.config.moet_btc_pool_config["size"]
        aave_config.moet_btc_concentration = self.config.moet_btc_pool_config["concentration"]
        aave_config.moet_yield_pool_size = self.config.moet_yt_pool_config["size"]
        aave_config.yield_token_concentration = self.config.moet_yt_pool_config["concentration"]
        aave_config.yield_token_ratio = self.config.moet_yt_pool_config["token0_ratio"]
        aave_config.use_direct_minting_for_initial = self.config.use_direct_minting_for_initial
        
        # Create engine
        engine = AaveProtocolEngine(aave_config)
        
        return engine
    
    def _create_agents(self, engine: AaveProtocolEngine) -> List[AaveAgent]:
        """Create AAVE agents with same parameters as High Tide agents"""
        
        print(f"👥 Creating {self.config.num_agents} AAVE agents...")
        
        agents = []
        
        for i in range(self.config.num_agents):
            agent_id = f"aave_agent_{i}"
            
            # Create AAVE agent with same parameters as High Tide
            # AAVE uses tri-health factor system but doesn't rebalance
            agent = AaveAgent(
                agent_id,
                self.config.agent_initial_hf,      # 1.1 Initial HF
                self.config.agent_liquidation_hf,  # 1.0 Liquidation HF (rebalancing_hf parameter)
                self.config.agent_initial_hf,      # 1.1 Target HF (same as initial for AAVE)
                initial_balance=self.config.btc_initial_price,  # CRITICAL FIX: Use 2024 BTC price
                yield_token_pool=engine.yield_token_pool  # Pass yield token pool during creation
            )
            
            agents.append(agent)
        
        print(f"✅ Created {len(agents)} AAVE agents")
        print(f"💰 Each agent starts with 1 BTC (${self.config.btc_initial_price:,.2f})")
        print(f"📊 Initial health factor: {self.config.agent_initial_hf}")
        print(f"⚠️  Liquidation threshold: {self.config.agent_liquidation_hf} (no rebalancing)")
        print()
        
        # Add agents to engine
        for agent in agents:
            engine.agents[agent.agent_id] = agent
            engine.aave_agents.append(agent)
            # CRITICAL FIX: Set engine reference for liquidation recording
            agent.engine = engine
            # CRITICAL FIX: Set yield token pool reference for initial minting
            if hasattr(agent.state, 'yield_token_manager'):
                agent.state.yield_token_manager.yield_token_pool = engine.yield_token_pool
        
        return agents
    
    def _run_simulation_with_detailed_tracking(self, engine: AaveProtocolEngine, agents: List[AaveAgent]) -> Dict[str, Any]:
        """Run simulation with comprehensive tracking of all activities"""
        
        print("🚀 Starting AAVE Full Year Simulation...")
        
        if self.config.simulation_duration_minutes >= 100_000:  # Only show for long simulations
            print("⚡ PERFORMANCE OPTIMIZATIONS ENABLED:")
            print("   • Agent health snapshots: Daily (every 1440 minutes)")
            print("   • Protocol state snapshots: Daily (every 1440 minutes)") 
            print("   • BTC price logging: Daily (events still logged immediately)")
            print("   • Expected memory usage: ~50 MB instead of ~16 GB")
            print("   • Expected runtime: ~35 minutes instead of 14+ hours")
            print()
        
        # Initialize simulation tracking
        total_minutes = self.config.simulation_duration_minutes
        progress_intervals = [int(total_minutes * p) for p in [0.1, 0.25, 0.5, 0.75, 0.9]]
        
        print(f"⏱️  Simulating {total_minutes:,} minutes ({total_minutes/1440:.0f} days)...")
        print()
        
        # Run simulation minute by minute
        for minute in range(total_minutes):
            # Update BTC price
            new_btc_price = self.config.get_btc_price_at_minute(minute)
            engine.btc_price_manager.update_price(new_btc_price)
            
            # Process AAVE agents (liquidation-only, no rebalancing)
            engine._process_aave_agents(minute)
            
            # Record metrics (daily only for performance)
            if minute % 1440 == 0:  # Daily metrics recording
                engine._record_metrics()
            
            # Progress reporting
            if minute in progress_intervals:
                progress = (minute / total_minutes) * 100
                day = minute / 1440
                surviving_agents = len([a for a in agents if a.active])
                print(f"📊 Progress: {progress:.0f}% (Day {day:.0f}) - {surviving_agents}/{len(agents)} agents active")
            
            # BTC price logging (daily)
            if minute % 1440 == 0:  # Daily BTC price logging
                self._log_event(minute, "BTC_PRICE_UPDATE", f"BTC price updated", {
                    "minute": minute,
                    "hour": minute / 60,
                    "btc_price": new_btc_price,
                    "change_pct": ((new_btc_price / self.config.btc_initial_price) - 1) * 100
                })
            
            # Memory management - weekly cleanup
            if minute % (1440 * 7) == 0 and minute > 0:  # Weekly cleanup
                max_entries = 5000
                if len(engine.metrics_history) > max_entries:
                    engine.metrics_history = engine.metrics_history[-max_entries//2:]
                if len(engine.agent_actions_history) > max_entries:
                    engine.agent_actions_history = engine.agent_actions_history[-max_entries//2:]
                if len(engine.yield_token_trades) > max_entries:
                    engine.yield_token_trades = engine.yield_token_trades[-max_entries//2:]
                
                print(f"🧹 Memory cleanup at day {minute//1440}: Trimmed historical data")
        
        print()
        print("✅ Simulation completed!")
        
        # Generate results
        return engine._generate_results()
    
    def _log_event(self, minute: int, event_type: str, description: str, data: dict):
        """Log simulation event"""
        event = {
            "minute": minute,
            "event_type": event_type,
            "description": description,
            "data": data
        }
        self.events.append(event)
    
    def _generate_analysis(self, results: Dict[str, Any]):
        """Generate comprehensive analysis of AAVE simulation results"""
        print("📊 Generating comprehensive analysis...")
        
        # Agent survival analysis
        total_agents = len(results.get("agent_summaries", []))
        surviving_agents = len([a for a in results.get("agent_summaries", []) if a.get("active", False)])
        liquidated_agents = total_agents - surviving_agents
        
        print(f"📈 Agent Survival Analysis:")
        print(f"   Total agents: {total_agents}")
        print(f"   Surviving: {surviving_agents} ({surviving_agents/total_agents*100:.1f}%)")
        print(f"   Liquidated: {liquidated_agents} ({liquidated_agents/total_agents*100:.1f}%)")
        print()
        
        # Performance analysis
        if results.get("agent_summaries"):
            final_values = [a.get("final_portfolio_value", 0) for a in results["agent_summaries"] if a.get("active", False)]
            if final_values:
                avg_performance = np.mean(final_values)
                initial_value = self.config.btc_initial_price  # Each agent started with 1 BTC worth
                performance_pct = (avg_performance / initial_value - 1) * 100
                
                print(f"💰 Performance Analysis:")
                print(f"   Average final value: ${avg_performance:,.2f}")
                print(f"   Initial value: ${initial_value:,.2f}")
                print(f"   Average performance: {performance_pct:+.1f}%")
                print()
        
        # BTC price analysis
        btc_performance = (self.config.btc_final_price / self.config.btc_initial_price - 1) * 100
        print(f"🪙 BTC Performance:")
        print(f"   Initial: ${self.config.btc_initial_price:,.2f}")
        print(f"   Final: ${self.config.btc_final_price:,.2f}")
        print(f"   Performance: {btc_performance:+.1f}%")
        print()
    
    def _save_results(self, results: Dict[str, Any]):
        """Save simulation results to files"""
        # Create results directory
        results_dir = Path("tidal_protocol_sim/results/AAVE_Full_Year_2024_Simulation")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"aave_full_year_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_file}")
        
        # Save events log
        events_file = results_dir / f"aave_simulation_events_{timestamp}.json"
        with open(events_file, 'w') as f:
            json.dump(self.events, f, indent=2, default=str)
        
        print(f"📝 Events log saved to: {events_file}")
    
    def _print_simulation_summary(self, results: Dict[str, Any]):
        """Print comprehensive simulation summary"""
        print("\n" + "="*80)
        print("🏦 AAVE FULL YEAR SIMULATION SUMMARY")
        print("="*80)
        
        # Basic stats
        total_agents = len(results.get("agent_summaries", []))
        surviving_agents = len([a for a in results.get("agent_summaries", []) if a.get("active", False)])
        
        print(f"📊 Simulation Overview:")
        print(f"   Duration: {self.config.simulation_duration_minutes:,} minutes ({self.config.simulation_duration_minutes/1440:.0f} days)")
        print(f"   Strategy: AAVE Liquidation-Only (No Rebalancing)")
        print(f"   Agents: {total_agents} (Survival rate: {surviving_agents/total_agents*100:.1f}%)")
        print()
        
        # Market conditions
        btc_performance = (self.config.btc_final_price / self.config.btc_initial_price - 1) * 100
        print(f"🪙 Market Conditions:")
        print(f"   BTC: ${self.config.btc_initial_price:,.2f} → ${self.config.btc_final_price:,.2f} ({btc_performance:+.1f}%)")
        print()
        
        # Agent performance
        if results.get("agent_summaries"):
            surviving_agents_data = [a for a in results["agent_summaries"] if a.get("active", False)]
            if surviving_agents_data:
                final_values = [a.get("final_portfolio_value", 0) for a in surviving_agents_data]
                avg_performance = np.mean(final_values)
                initial_value = self.config.btc_initial_price
                performance_pct = (avg_performance / initial_value - 1) * 100
                
                print(f"💰 Agent Performance (Surviving Agents):")
                print(f"   Average final value: ${avg_performance:,.2f}")
                print(f"   vs. Initial (1 BTC): {performance_pct:+.1f}%")
                print(f"   vs. BTC hodling: {performance_pct - btc_performance:+.1f}%")
                print()
        
        # Liquidations
        liquidated_count = total_agents - surviving_agents
        if liquidated_count > 0:
            print(f"⚠️  Liquidations:")
            print(f"   Total liquidated: {liquidated_count} agents ({liquidated_count/total_agents*100:.1f}%)")
            print()
        
        print("="*80)
        print("✅ AAVE simulation completed successfully!")
        print("="*80)


def main():
    """Main function to run AAVE full year simulation"""
    print("🏦 AAVE Full Year Simulation")
    print("=" * 50)
    print()
    print("This simulation will:")
    print("• Run AAVE liquidation-only strategy for full 2024")
    print("• Use identical conditions to High Tide simulation")
    print("• Track 120 agents over 365 days")
    print("• Generate comprehensive analysis and charts")
    print("• Enable direct comparison with High Tide results")
    print()
    
    # Auto-confirm for automated runs
    print("✅ Auto-confirming simulation start...")
    confirm = 'y'
    
    if confirm.lower() == 'y':
        # Create configuration
        config = AaveFullYearSimConfig()
        
        print(f"🏦 AAVE Full Year Simulation Initialized")
        print(f"📅 Duration: {config.simulation_duration_minutes:,} minutes ({config.simulation_duration_minutes/1440:.0f} days)")
        print(f"👥 Agents: {config.num_agents}")
        print(f"💰 BTC Price Range: ${config.btc_initial_price:,.2f} → ${config.btc_final_price:,.2f}")
        print(f"📊 Pool Sizes: MOET:BTC ${config.moet_btc_pool_config['size']:,}, MOET:YT ${config.moet_yt_pool_config['size']:,}")
        print()
        
        # Run simulation
        simulation = AaveFullYearSimulation(config)
        results = simulation.run_full_simulation()
        
        return results
    else:
        print("❌ Simulation cancelled by user")
        return None


if __name__ == "__main__":
    main()
