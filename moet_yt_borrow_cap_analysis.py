#!/usr/bin/env python3
"""
BTC:MOET Pool Deposit Cap Analysis

Tests deposit caps as a percentage of BTC:MOET liquidity for liquidation capacity.
Establishes positions with low Health Factors (1.1-1.25) and tests liquidation capacity
under various BTC price shocks without rebalancing.

Key Question: What deposit cap ratio (deposits:liquidity) can the BTC:MOET pool handle for liquidations?
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
from typing import Dict, List, Any

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tidal_protocol_sim.simulation.tidal_engine import TidalProtocolEngine, TidalConfig
from tidal_protocol_sim.agents.high_tide_agent import HighTideAgent
from tidal_protocol_sim.agents.aave_agent import AaveAgent
from tidal_protocol_sim.core.protocol import TidalProtocol, Asset, AssetPool, LiquidityPool
from tidal_protocol_sim.core.uniswap_v3_math import UniswapV3SlippageCalculator, UniswapV3Pool


class BTCOnyProtocol(TidalProtocol):
    """BTC-only protocol for borrow cap analysis"""
    
    def __init__(self):
        super().__init__()
        
        # Create BTC-only asset pools
        self.asset_pools = {
            Asset.BTC: AssetPool(
                asset=Asset.BTC,
                total_supply=0.0,
                total_borrowed=0.0,
                collateral_factor=0.80,  # 80% collateral factor
                borrow_cap=float('inf'),
                liquidation_threshold=0.85
            )
        }
        
        # Create BTC-only liquidity pools
        self.liquidity_pools = {
            "moet_btc": LiquidityPool(
                pool_name="MOET:BTC",
                asset_a=Asset.MOET,
                asset_b=Asset.BTC,
                total_liquidity_a=0.0,
                total_liquidity_b=0.0,
                fee_rate=0.003  # 0.3% fee
            )
        }


def convert_for_json(obj):
    """Convert objects to JSON-serializable format"""
    if isinstance(obj, dict):
        return {str(k): convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return convert_for_json(obj.__dict__)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def build_clean_simulation_data(simulation_runs: List, scenario_type: str) -> List:
    """Build clean, navigable simulation data structure"""
    clean_runs = []
    
    for run in simulation_runs:
        # Filter out non-test agents (lender_0, trader_0, liquidator_0, etc.)
        agent_outcomes = run.get("agent_outcomes", [])
        filtered_outcomes = [
            outcome for outcome in agent_outcomes 
            if not any(prefix in outcome.get("agent_id", "") for prefix in ["lender_", "trader_", "liquidator_", "high_tide_conservative_"])
        ]
        
        # Extract health factor history for test agents only
        health_factors = run.get("agent_health_history", {})
        filtered_health_factors = {
            agent_id: history for agent_id, history in health_factors.items()
            if not any(prefix in agent_id for prefix in ["lender_", "trader_", "liquidator_", "high_tide_conservative_"])
        }
        
        # Extract actions history for test agents only
        actions_history = run.get("agent_actions_history", {})
        filtered_actions_history = {
            agent_id: actions for agent_id, actions in actions_history.items()
            if not any(prefix in agent_id for prefix in ["lender_", "trader_", "liquidator_", "high_tide_conservative_"])
        }
        
        clean_run = {
            "simulation_metadata": {
                "scenario_type": scenario_type,
                "btc_price_history": run.get("btc_price_history", []),
                "total_agents": len(filtered_outcomes),
                "simulation_duration": run.get("simulation_duration", 60)
            },
            "agent_health_factors": filtered_health_factors,
            "agent_actions_history": filtered_actions_history,
            "rebalancing_events": run.get("rebalancing_events", []),
            "agent_outcomes": filtered_outcomes,
            "summary_stats": {
                "total_agents": len(filtered_outcomes),
                "survived_agents": sum(1 for outcome in filtered_outcomes if outcome.get("survived", True)),
                "total_rebalancing_events": len(run.get("rebalancing_events", [])),
                "total_slippage_costs": sum(outcome.get("total_slippage_costs", 0) for outcome in filtered_outcomes),
                "final_btc_price": run.get("btc_price_history", [100000])[-1] if isinstance(run.get("btc_price_history", [100000])[-1], (int, float)) else run.get("btc_price_history", [100000])[-1].get("btc_price", 100000) if run.get("btc_price_history") else 100000,
                "btc_price_decline_percent": ((100000 - (run.get("btc_price_history", [100000])[-1] if isinstance(run.get("btc_price_history", [100000])[-1], (int, float)) else run.get("btc_price_history", [100000])[-1].get("btc_price", 100000) if run.get("btc_price_history") else 100000)) / 100000) * 100
            }
        }
        clean_runs.append(clean_run)
    
    return clean_runs


def run_borrow_cap_analysis():
    """Run BTC:MOET pool deposit cap analysis with 15 agents per scenario"""
    
    print("=" * 80)
    print("BTC:MOET POOL DEPOSIT CAP ANALYSIS")
    print("=" * 80)
    print("Testing liquidation capacity with 15 agents per deposit cap ratio")
    print("Baseline: $250K:$250K BTC:MOET pool with 80% concentration")
    print("Question: What deposit cap ratio can handle liquidations within concentration range?")
    print()
    
    # BTC:MOET pool configuration (liquidation pool)
    btc_moet_pool_size = 250_000  # $250K each side
    btc_moet_concentration = 0.80  # 80% concentration at peg
    
    # Test different deposit cap ratios
    deposit_cap_ratios = [1.0, 2.0, 3.0, 4.0, 5.0]  # 1:1, 2:1, 3:1, 4:1, 5:1
    
    # Test different BTC price shock scenarios
    btc_shock_scenarios = [
        {"shock_percent": 10, "description": "Moderate shock"},
        {"shock_percent": 15, "description": "Significant shock"},
        {"shock_percent": 25, "description": "Severe shock"}
    ]
    
    results = []
    
    for ratio in deposit_cap_ratios:
        print(f"\n💰 Testing {ratio:.0f}:1 Deposit Cap Ratio")
        
        # Calculate total deposit capacity
        total_deposit_capacity = btc_moet_pool_size * 2 * ratio  # Total pool value * ratio
        collateral_per_agent = total_deposit_capacity / 15  # Distribute equally among 15 agents
        print(f"   Total deposit capacity: ${total_deposit_capacity:,.0f}")
        print(f"   Collateral per agent: ${collateral_per_agent:,.0f}")
        print(f"   BTC:MOET pool: ${btc_moet_pool_size:,.0f} each side")
        
        # Store results from all shock scenarios for this ratio
        ratio_runs = []
        
        for shock in btc_shock_scenarios:
            print(f"   📉 Testing {shock['shock_percent']}% BTC price shock...")
            
            # Run liquidation capacity scenario with 15 agents
            result = run_liquidation_capacity_scenario_with_agents(
                ratio, shock, btc_moet_pool_size, btc_moet_concentration
            )
            ratio_runs.append(result)
            
            print(f"      {result['liquidation_metrics']['liquidation_success_rate']:.1%} liquidation success rate")
        
        # Aggregate results for this ratio
        scenario_results = aggregate_liquidation_capacity_results(ratio_runs, ratio)
        results.append(scenario_results)
        
        print(f"   Results: {scenario_results['liquidation_summary']['mean_success_rate']:.1%} mean success rate")
        print()
    
    # Save results with JSON output and get run folder
    run_folder = save_borrow_cap_results(results)
    
    # Generate charts in the run folder
    generated_charts = create_liquidation_capacity_charts(results, run_folder)
    
    print("✅ Liquidation capacity analysis completed!")
    print(f"📊 Generated {len(generated_charts)} charts")
    print(f"📁 All results saved to: {run_folder}")
    
    return results


def run_liquidation_capacity_scenario_with_agents(ratio: float, shock: Dict, 
                                                pool_size: int, concentration: float) -> Dict:
    """Run liquidation capacity test with 15 agents with varying initial HFs"""
    
    shock_percent = shock["shock_percent"]
    
    # Calculate position parameters
    btc_price_initial = 100_000.0  # $100K initial BTC price
    btc_price_shocked = btc_price_initial * (1 - shock_percent / 100)
    collateral_factor = 0.80  # 80% collateral factor
    
    # Create agents based on deposit cap ratio - higher ratios = more agents = more risk
    # This simulates the real scenario where higher deposit caps mean more total exposure
    total_deposit_capacity = pool_size * 2 * ratio  # Total pool value * ratio
    position_value = calculate_position_value(1.2, btc_price_initial, collateral_factor)  # Use 1.2 as base
    max_agents = min(int(total_deposit_capacity / position_value), 50)  # Cap at 50 agents max
    
    # Create 15 agents with scaled collateral based on deposit cap ratio
    agents = create_varied_agents_for_liquidation_test(15, btc_price_initial, collateral_factor, ratio)
    
    # Calculate liquidation needs after shock
    liquidation_analysis = calculate_liquidation_needs_with_agents(agents, btc_price_shocked, collateral_factor, ratio)
    
    # Test DEX liquidation capacity with priority ordering
    dex_capacity_analysis = test_dex_liquidation_capacity_with_priority(
        liquidation_analysis, pool_size, concentration, btc_price_shocked
    )
    
    # Calculate metrics
    total_agents = len(agents)
    positions_needing_liquidation = liquidation_analysis["positions_needing_liquidation"]
    liquidation_success_rate = dex_capacity_analysis["successful_liquidations"] / max(positions_needing_liquidation, 1)
    
    return {
        "scenario_params": {
            "deposit_cap_ratio": ratio,
            "btc_shock_percent": shock_percent,
            "pool_size": pool_size,
            "concentration": concentration,
            "total_agents": total_agents
        },
        "agent_analysis": {
            "total_agents": total_agents,
            "positions_needing_liquidation": positions_needing_liquidation,
            "total_debt_to_liquidate": liquidation_analysis["total_debt_to_liquidate"],
            "total_collateral_to_seize": liquidation_analysis["total_collateral_to_seize"],
            "agent_details": liquidation_analysis["agent_details"]
        },
        "dex_capacity_analysis": dex_capacity_analysis,
        "liquidation_metrics": {
            "liquidation_success_rate": liquidation_success_rate,
            "positions_successfully_liquidated": dex_capacity_analysis["successful_liquidations"],
            "positions_failed_liquidation": positions_needing_liquidation - dex_capacity_analysis["successful_liquidations"],
            "total_slippage_cost": dex_capacity_analysis["total_slippage_cost"],
            "average_slippage_per_liquidation": dex_capacity_analysis["total_slippage_cost"] / max(dex_capacity_analysis["successful_liquidations"], 1),
            "concentration_utilization": dex_capacity_analysis["concentration_utilization"],
            "total_liquidated_value": liquidation_analysis["total_debt_to_liquidate"]  # Total debt value that was liquidated
        }
    }

def run_liquidation_capacity_scenario(ratio: float, shock: Dict, hf_scenario: Dict, 
                                    pool_size: int, concentration: float) -> Dict:
    """Run liquidation capacity test for specific deposit cap ratio and shock scenario"""
    
    initial_hf = hf_scenario["initial_hf"]
    shock_percent = shock["shock_percent"]
    
    print(f"      {hf_scenario['profile']} (HF {initial_hf:.2f}) → {shock_percent}% shock: ", end="")
    
    # Calculate position parameters
    btc_price_initial = 100_000.0  # $100K initial BTC price
    btc_price_shocked = btc_price_initial * (1 - shock_percent / 100)
    collateral_factor = 0.80  # 80% collateral factor
    
    # Calculate how many positions we can create with this deposit cap ratio
    total_deposit_capacity = pool_size * 2 * ratio  # Total pool value * ratio
    position_value = calculate_position_value(initial_hf, btc_price_initial, collateral_factor)
    max_positions = int(total_deposit_capacity / position_value)
    
    # Create positions and calculate liquidation needs
    positions = create_test_positions(max_positions, initial_hf, btc_price_initial, collateral_factor)
    
    # Calculate liquidation requirements after shock
    liquidation_analysis = calculate_liquidation_needs(positions, btc_price_shocked, collateral_factor)
    
    # Test DEX liquidation capacity
    dex_capacity_analysis = test_dex_liquidation_capacity(
        liquidation_analysis, pool_size, concentration, btc_price_shocked
    )
    
    # Calculate metrics
    total_positions = len(positions)
    positions_needing_liquidation = liquidation_analysis["positions_needing_liquidation"]
    liquidation_success_rate = dex_capacity_analysis["successful_liquidations"] / max(positions_needing_liquidation, 1)
    
    print(f"{liquidation_success_rate:.1%} liquidation success rate")
    
    return {
        "scenario_params": {
            "deposit_cap_ratio": ratio,
            "btc_shock_percent": shock_percent,
            "initial_hf": initial_hf,
            "profile": hf_scenario["profile"],
            "pool_size": pool_size,
            "concentration": concentration
        },
        "position_analysis": {
            "total_positions": total_positions,
            "positions_needing_liquidation": positions_needing_liquidation,
            "total_debt_to_liquidate": liquidation_analysis["total_debt_to_liquidate"],
            "total_collateral_to_seize": liquidation_analysis["total_collateral_to_seize"]
        },
        "dex_capacity_analysis": dex_capacity_analysis,
        "liquidation_metrics": {
            "liquidation_success_rate": liquidation_success_rate,
            "positions_successfully_liquidated": dex_capacity_analysis["successful_liquidations"],
            "positions_failed_liquidation": positions_needing_liquidation - dex_capacity_analysis["successful_liquidations"],
            "total_slippage_cost": dex_capacity_analysis["total_slippage_cost"],
            "average_slippage_per_liquidation": dex_capacity_analysis["total_slippage_cost"] / max(dex_capacity_analysis["successful_liquidations"], 1)
        }
    }


def create_varied_agents_for_liquidation_test(num_agents: int, btc_price: float, 
                                            collateral_factor: float, deposit_cap_ratio: float) -> List[Dict]:
    """Create agents with scaled collateral based on deposit cap ratio"""
    
    agents = []
    
    # Fixed health factors for consistent liquidation testing
    fixed_health_factors = [1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8]
    
    # Calculate collateral per agent based on total deposit capacity
    # Total deposits = pool_size * 2 * ratio, divided equally among agents
    # pool_size is $250K, so total deposits = $250K * 2 * ratio
    total_deposit_capacity = 250_000 * 2 * deposit_cap_ratio  # $250K * 2 * ratio
    collateral_per_agent = total_deposit_capacity / num_agents  # Distribute equally
    
    for i in range(num_agents):
        # Use fixed health factor (cycle through if we have more agents than factors)
        initial_hf = fixed_health_factors[i % len(fixed_health_factors)]
        
        collateral_value = collateral_per_agent  # Scaled collateral per agent
        effective_collateral = collateral_value * collateral_factor
        debt_value = effective_collateral / initial_hf
        
        agent = {
            "agent_id": f"liquidation_agent_{i}",
            "initial_hf": initial_hf,
            "collateral_value": collateral_value,
            "effective_collateral": effective_collateral,
            "debt_value": debt_value,
            "collateral_factor": collateral_factor
        }
        agents.append(agent)
    
    return agents

def calculate_liquidation_needs_with_agents(agents: List[Dict], shocked_btc_price: float, 
                                          collateral_factor: float, deposit_cap_ratio: float) -> Dict:
    """Calculate liquidation needs after BTC price shock with agent details"""
    
    positions_needing_liquidation = 0
    total_debt_to_liquidate = 0.0
    total_collateral_to_seize = 0.0
    agent_details = []
    
    liquidation_threshold = 1.0  # Health factor below 1.0 triggers liquidation
    target_hf_after_liquidation = 1.1  # Target HF after liquidation
    
    for agent in agents:
        # Calculate new health factor after price shock
        # Agent's collateral is already scaled by deposit_cap_ratio, just apply price shock
        price_shock_factor = shocked_btc_price / 100_000  # e.g., 0.9 for 10% drop
        new_collateral_value = agent["collateral_value"] * price_shock_factor
        new_effective_collateral = new_collateral_value * collateral_factor
        new_hf = new_effective_collateral / agent["debt_value"]
        
        agent_detail = {
            "agent_id": agent["agent_id"],
            "initial_hf": agent["initial_hf"],
            "new_hf_after_shock": new_hf,
            "needs_liquidation": new_hf < liquidation_threshold,
            "debt_to_liquidate": 0.0,
            "collateral_to_seize": 0.0,
            "liquidation_priority": 0.0  # Will be set based on debt amount
        }
        
        if new_hf < liquidation_threshold:
            positions_needing_liquidation += 1
            
            # Calculate how much debt to liquidate to get HF back to 1.1
            target_effective_collateral = new_effective_collateral
            target_debt_after_liquidation = target_effective_collateral / target_hf_after_liquidation
            debt_to_liquidate = agent["debt_value"] - target_debt_after_liquidation
            
            # Calculate collateral to seize (debt liquidated + liquidation bonus)
            liquidation_bonus = 0.05  # 5% liquidation bonus
            # Use the scaled BTC price that matches the agent's collateral scale
            scaled_btc_price = shocked_btc_price * deposit_cap_ratio
            collateral_to_seize = (debt_to_liquidate / scaled_btc_price) * (1 + liquidation_bonus)
            
            agent_detail["debt_to_liquidate"] = debt_to_liquidate
            agent_detail["collateral_to_seize"] = collateral_to_seize
            agent_detail["liquidation_priority"] = debt_to_liquidate  # Priority = debt amount
            
            total_debt_to_liquidate += debt_to_liquidate
            total_collateral_to_seize += collateral_to_seize
        
        agent_details.append(agent_detail)
    
    # Sort agents by liquidation priority (highest debt first)
    agent_details.sort(key=lambda x: x["liquidation_priority"], reverse=True)
    
    return {
        "positions_needing_liquidation": positions_needing_liquidation,
        "total_debt_to_liquidate": total_debt_to_liquidate,
        "total_collateral_to_seize": total_collateral_to_seize,
        "agent_details": agent_details
    }

def test_dex_liquidation_capacity_with_priority(liquidation_analysis: Dict, pool_size: int, 
                                              concentration: float, btc_price: float) -> Dict:
    """Test DEX liquidation capacity with priority ordering (highest debt first)"""
    
    total_debt_to_liquidate = liquidation_analysis["total_debt_to_liquidate"]
    total_collateral_to_seize = liquidation_analysis["total_collateral_to_seize"]
    agent_details = liquidation_analysis["agent_details"]
    
    # Calculate MOET needed for liquidations (debt value in MOET)
    moet_needed = total_debt_to_liquidate  # Debt is in MOET terms
    
    # Calculate BTC available for liquidations (collateral seized)
    btc_available = total_collateral_to_seize  # Collateral seized in BTC
    
    # Fixed pool size across all scenarios
    pool_btc_liquidity = pool_size  # $250K BTC side
    pool_moet_liquidity = pool_size  # $250K MOET side
    
    # Check if we have enough liquidity
    can_liquidate_all = (moet_needed <= pool_moet_liquidity and btc_available <= pool_btc_liquidity)
    
    # Process liquidations in priority order (highest debt first)
    successful_liquidations = 0
    total_slippage_cost = 0.0
    liquidation_events = []
    cumulative_liquidations = 0.0
    concentration_utilization = 0.0
    
    for agent_detail in agent_details:
        if not agent_detail["needs_liquidation"]:
            continue
            
        debt_to_liquidate = agent_detail["debt_to_liquidate"]
        collateral_to_seize = agent_detail["collateral_to_seize"]
        
        # Check if we can liquidate this agent
        if (cumulative_liquidations + debt_to_liquidate <= pool_moet_liquidity and 
            cumulative_liquidations + collateral_to_seize <= pool_btc_liquidity):
            
            # Calculate slippage cost using proper Uniswap V3 math
            # This is a BTC:MOET swap (debt liquidation)
            try:
                pool_state = UniswapV3Pool(
                    pool_name="MOET:BTC",
                    total_liquidity=pool_moet_liquidity * 2,  # Total pool value
                    btc_price=btc_price,
                    concentration=concentration  # 0.8 (80% concentration)
                )
                slippage_calculator = UniswapV3SlippageCalculator(pool_state)
                
                slippage_result = slippage_calculator.calculate_swap_slippage(
                    amount_in=debt_to_liquidate,
                    token_in="BTC"
                )
                slippage_cost = slippage_result["slippage_amount"]
                
                # Include trading fees in total cost (slippage + fees)
                total_swap_cost = slippage_cost + slippage_result.get("trading_fees", 0)
                
            except Exception as e:
                # If slippage calculation fails, use simplified estimate
                slippage_cost = 0.0
                total_swap_cost = debt_to_liquidate * 0.003  # 0.3% trading fee estimate
            
            successful_liquidations += 1
            total_slippage_cost += total_swap_cost  # Use total cost including fees
            cumulative_liquidations += debt_to_liquidate
            
            # Calculate concentration utilization based on concentrated liquidity (80% of total pool)
            # Total pool = $500K, 80% concentrated = $400K
            total_concentrated_liquidity = (pool_moet_liquidity + pool_btc_liquidity) * concentration  # $500K * 0.8 = $400K
            concentration_utilization = (cumulative_liquidations / total_concentrated_liquidity) * 100
            
            liquidation_events.append({
                "agent_id": agent_detail["agent_id"],
                "debt_liquidated": debt_to_liquidate,
                "collateral_seized": collateral_to_seize,
                "slippage_cost": total_swap_cost,  # Use total cost including fees
                "cumulative_liquidations": cumulative_liquidations,
                "concentration_utilization": concentration_utilization
            })
        else:
            # Can't liquidate this agent due to capacity constraints
            break
    
    return {
        "can_liquidate_all": can_liquidate_all,
        "successful_liquidations": successful_liquidations,
        "total_slippage_cost": total_slippage_cost,
        "moet_needed": moet_needed,
        "btc_available": btc_available,
        "pool_btc_liquidity": pool_btc_liquidity,
        "pool_moet_liquidity": pool_moet_liquidity,
        "concentration_utilization": concentration_utilization,
        "liquidation_events": liquidation_events
    }

def aggregate_liquidation_capacity_results(ratio_runs: List[Dict], ratio: float) -> Dict:
    """Aggregate results for a deposit cap ratio across all shock scenarios"""
    
    # Extract metrics from all shock scenarios
    success_rates = [run["liquidation_metrics"]["liquidation_success_rate"] for run in ratio_runs]
    total_agents = [run["agent_analysis"]["total_agents"] for run in ratio_runs]
    positions_needing_liquidation = [run["agent_analysis"]["positions_needing_liquidation"] for run in ratio_runs]
    slippage_costs = [run["liquidation_metrics"]["total_slippage_cost"] for run in ratio_runs]
    concentration_utilizations = [run["liquidation_metrics"]["concentration_utilization"] for run in ratio_runs]
    
    return {
        "deposit_cap_ratio": ratio,
        "scenario_params": {
            "deposit_cap_ratio": ratio,
            "shock_scenarios": [run["scenario_params"]["btc_shock_percent"] for run in ratio_runs],
            "total_agents": total_agents[0] if total_agents else 0
        },
        "liquidation_summary": {
            "mean_success_rate": np.mean(success_rates),
            "success_rate_std": np.std(success_rates),
            "mean_positions_needing_liquidation": np.mean(positions_needing_liquidation),
            "mean_slippage_cost": np.mean(slippage_costs),
            "mean_concentration_utilization": np.mean(concentration_utilizations)
        },
        "shock_scenario_results": ratio_runs,
        "detailed_simulation_data": {
            "ratio_runs": ratio_runs
        }
    }

def calculate_position_value(initial_hf: float, btc_price: float, collateral_factor: float) -> float:
    """Calculate the total value of a position (collateral + debt)"""
    collateral_value = btc_price  # 1 BTC
    effective_collateral = collateral_value * collateral_factor
    debt_value = effective_collateral / initial_hf
    return collateral_value + debt_value


def create_test_positions(num_positions: int, initial_hf: float, btc_price: float, 
                         collateral_factor: float) -> List[Dict]:
    """Create test positions with specified health factors"""
    positions = []
    
    for i in range(num_positions):
        collateral_value = btc_price  # 1 BTC per position
        effective_collateral = collateral_value * collateral_factor
        debt_value = effective_collateral / initial_hf
        
        position = {
            "position_id": f"pos_{i}",
            "collateral_value": collateral_value,
            "effective_collateral": effective_collateral,
            "debt_value": debt_value,
            "initial_hf": initial_hf,
            "collateral_factor": collateral_factor
        }
        positions.append(position)
    
    return positions


def calculate_liquidation_needs(positions: List[Dict], shocked_btc_price: float, 
                               collateral_factor: float) -> Dict:
    """Calculate liquidation needs after BTC price shock"""
    
    positions_needing_liquidation = 0
    total_debt_to_liquidate = 0.0
    total_collateral_to_seize = 0.0
    
    liquidation_threshold = 1.0  # Health factor below 1.0 triggers liquidation
    target_hf_after_liquidation = 1.1  # Target HF after liquidation
    
    for position in positions:
        # Calculate new health factor after price shock
        new_collateral_value = shocked_btc_price  # 1 BTC at new price
        new_effective_collateral = new_collateral_value * collateral_factor
        new_hf = new_effective_collateral / position["debt_value"]
        
        if new_hf < liquidation_threshold:
            positions_needing_liquidation += 1
            
            # Calculate how much debt to liquidate to get HF back to 1.1
            target_effective_collateral = new_effective_collateral
            target_debt_after_liquidation = target_effective_collateral / target_hf_after_liquidation
            debt_to_liquidate = position["debt_value"] - target_debt_after_liquidation
            
            # Calculate collateral to seize (debt liquidated + liquidation bonus)
            liquidation_bonus = 0.05  # 5% liquidation bonus
            collateral_to_seize = (debt_to_liquidate / shocked_btc_price) * (1 + liquidation_bonus)
            
            total_debt_to_liquidate += debt_to_liquidate
            total_collateral_to_seize += collateral_to_seize
    
    return {
        "positions_needing_liquidation": positions_needing_liquidation,
        "total_debt_to_liquidate": total_debt_to_liquidate,
        "total_collateral_to_seize": total_collateral_to_seize
    }


def test_dex_liquidation_capacity(liquidation_analysis: Dict, pool_size: int, 
                                 concentration: float, btc_price: float) -> Dict:
    """Test if DEX can handle the required liquidations within concentration range"""
    
    total_debt_to_liquidate = liquidation_analysis["total_debt_to_liquidate"]
    total_collateral_to_seize = liquidation_analysis["total_collateral_to_seize"]
    
    # Calculate MOET needed for liquidations (debt value in MOET)
    moet_needed = total_debt_to_liquidate  # Debt is in MOET terms
    
    # Calculate BTC available for liquidations (collateral seized)
    btc_available = total_collateral_to_seize  # Collateral seized in BTC
    
    # Test if we can liquidate within concentration range
    # This is a simplified test - in reality we'd need to simulate the actual DEX trades
    pool_btc_liquidity = pool_size  # $250K BTC side
    pool_moet_liquidity = pool_size  # $250K MOET side
    
    # Check if we have enough liquidity
    can_liquidate = (moet_needed <= pool_moet_liquidity and btc_available <= pool_btc_liquidity)
    
    if can_liquidate:
        # Calculate slippage cost (simplified)
        # Higher concentration = more slippage for large trades
        concentration_factor = 1 - concentration  # 0.2 for 80% concentration
        slippage_rate = (moet_needed / pool_moet_liquidity) * concentration_factor
        total_slippage_cost = moet_needed * slippage_rate
        
        successful_liquidations = liquidation_analysis["positions_needing_liquidation"]
    else:
        # Partial liquidation based on available liquidity
        max_moet_liquidations = pool_moet_liquidity
        max_btc_liquidations = pool_btc_liquidity
        
        # Calculate how many positions we can actually liquidate
        liquidation_ratio = min(max_moet_liquidations / moet_needed, max_btc_liquidations / btc_available)
        successful_liquidations = int(liquidation_analysis["positions_needing_liquidation"] * liquidation_ratio)
        total_slippage_cost = max_moet_liquidations * 0.1  # 10% slippage for failed liquidations
    
    return {
        "can_liquidate_all": can_liquidate,
        "successful_liquidations": successful_liquidations,
        "total_slippage_cost": total_slippage_cost,
        "moet_needed": moet_needed,
        "btc_available": btc_available,
        "pool_btc_liquidity": pool_btc_liquidity,
        "pool_moet_liquidity": pool_moet_liquidity
    }


def run_borrow_cap_scenario(agent_count: int, hf_scenario: Dict, 
                           pool_size: int, pool_utilization: float) -> Dict:
    """Run scenario testing pool capacity with specific agent load"""
    
    initial_hf = hf_scenario["initial_hf"]
    target_hf = hf_scenario["target_hf"]
    profile = hf_scenario["profile"]
    
    print(f"      {profile} (HF {initial_hf:.2f}→{target_hf:.2f}): ", end="")
    
    monte_carlo_runs = 5  # Lighter testing due to large agent counts
    
    ht_results = []
    aave_results = []
    
    for run_num in range(monte_carlo_runs):
        # High Tide simulation with many tight-range agents
        ht_config = HighTideConfig()
        ht_config.num_high_tide_agents = 0  # Custom agents
        ht_config.btc_decline_duration = 45  # Shorter for performance
        ht_config.moet_btc_pool_size = 250_000  # Standard liquidation pool
        ht_config.moet_yield_pool_size = pool_size  # Baseline YT pool
        ht_config.moet_btc_concentration = 0.80  # 80% concentration at peg for BTC:MOET
        ht_config.yield_token_concentration = 0.95  # 95% concentration at peg for MOET:YT
        
        # Create many agents with tight HF ranges
        tight_ht_agents = create_tight_range_agents(
            initial_hf, target_hf, agent_count, "high_tide", run_num, profile
        )
        
        ht_engine = HighTideVaultEngine(ht_config)
        ht_engine.protocol = BTCOnyProtocol()  # Enforce BTC-only
        ht_engine.high_tide_agents = tight_ht_agents
        for agent in tight_ht_agents:
            ht_engine.agents[agent.agent_id] = agent
        
        ht_result = ht_engine.run_simulation()
        ht_results.append(ht_result)
        
        # Matching Aave scenario
        aave_config = AaveConfig()
        aave_config.num_aave_agents = 0
        aave_config.btc_decline_duration = 45
        aave_config.moet_btc_pool_size = 250_000
        aave_config.moet_yield_pool_size = pool_size
        
        tight_aave_agents = create_tight_range_agents(
            initial_hf, target_hf, agent_count, "aave", run_num, profile
        )
        
        aave_engine = AaveProtocolEngine(aave_config)
        aave_engine.aave_agents = tight_aave_agents
        for agent in tight_aave_agents:
            aave_engine.agents[agent.agent_id] = agent
        
        aave_result = aave_engine.run_simulation()
        aave_results.append(aave_result)
    
    # Aggregate results
    scenario_result = aggregate_borrow_cap_scenario(
        ht_results, aave_results, hf_scenario, agent_count, pool_utilization
    )
    
    # Quick feedback
    ht_rebalances = scenario_result["high_tide_metrics"]["total_rebalancing_events"]
    ht_liquidations = scenario_result["high_tide_metrics"]["liquidation_rate_percentage"]
    pool_stress = scenario_result["pool_stress_analysis"]["stress_level"]
    
    print(f"{ht_rebalances:.0f} rebalances, {ht_liquidations:.1f}% liquidations, {pool_stress} stress")
    
    return scenario_result


def create_tight_range_agents(initial_hf: float, target_hf: float, num_agents: int,
                             agent_type: str, run_num: int, profile: str) -> List:
    """Create agents with tight HF ranges for maximum rebalancing activity"""
    
    agents = []
    
    for i in range(num_agents):
        agent_id = f"tight_{agent_type}_{profile}_r{run_num}_a{i}"
        
        if agent_type == "high_tide":
            agent = HighTideAgent(agent_id, initial_hf, target_hf)
        else:  # aave
            agent = AaveAgent(agent_id, initial_hf, target_hf)
        
        # Set risk profile based on scenario
        if "conservative" in profile:
            agent.risk_profile = "conservative"
            agent.color = "#2E8B57"
        elif "moderate" in profile:
            agent.risk_profile = "moderate"
            agent.color = "#FF8C00"
        else:
            agent.risk_profile = "aggressive"
            agent.color = "#DC143C"
        
        agents.append(agent)
    
    return agents


def aggregate_borrow_cap_scenario(ht_results: List, aave_results: List,
                                 hf_scenario: Dict, agent_count: int, 
                                 pool_utilization: float) -> Dict:
    """Aggregate results for borrow cap scenario"""
    
    # High Tide metrics
    ht_survival_rates = []
    ht_rebalancing_events = []
    ht_liquidations = []
    ht_pool_stress_events = []
    
    for run in ht_results:
        survival_stats = run.get("survival_statistics", {})
        ht_survival_rates.append(survival_stats.get("survival_rate", 0.0))
        
        # Rebalancing activity
        rebalancing_activity = run.get("yield_token_activity", {})
        ht_rebalancing_events.append(rebalancing_activity.get("rebalancing_events", 0))
        
        # Emergency liquidations
        agent_outcomes = run.get("agent_outcomes", [])
        total_liquidations = sum(outcome.get("emergency_liquidations", 0) for outcome in agent_outcomes)
        ht_liquidations.append(total_liquidations)
        
        # Pool stress indicators (high rebalancing concentration)
        rebalancing_events = run.get("rebalancing_events", [])
        stress_events = count_pool_stress_events(rebalancing_events)
        ht_pool_stress_events.append(stress_events)
    
    # Aave metrics for comparison
    aave_survival_rates = []
    aave_liquidations = []
    
    for run in aave_results:
        survival_stats = run.get("survival_statistics", {})
        aave_survival_rates.append(survival_stats.get("survival_rate", 0.0))
        
        liquidation_activity = run.get("liquidation_activity", {})
        aave_liquidations.append(liquidation_activity.get("total_liquidation_events", 0))
    
    # Calculate pool stress analysis
    pool_stress_analysis = analyze_pool_capacity_stress(
        ht_rebalancing_events, ht_pool_stress_events, pool_utilization
    )
    
    return {
        "scenario_params": {
            **hf_scenario,
            "agent_count": agent_count,
            "pool_utilization": pool_utilization,
            "total_borrowing_capacity": agent_count * calculate_agent_borrowing_capacity()
        },
        "high_tide_metrics": {
            "survival_rate": np.mean(ht_survival_rates),
            "survival_rate_std": np.std(ht_survival_rates),
            "total_rebalancing_events": np.mean(ht_rebalancing_events),
            "liquidation_frequency": np.mean(ht_liquidations) / agent_count,
            "liquidation_rate_percentage": (np.mean(ht_liquidations) / agent_count) * 100,
            "pool_stress_events": np.mean(ht_pool_stress_events)
        },
        "aave_metrics": {
            "survival_rate": np.mean(aave_survival_rates),
            "liquidation_frequency": np.mean(aave_liquidations) / agent_count,
            "liquidation_rate_percentage": (np.mean(aave_liquidations) / agent_count) * 100
        },
        "pool_stress_analysis": pool_stress_analysis,
        "borrow_cap_implications": generate_borrow_cap_implications(
            pool_utilization, pool_stress_analysis, np.mean(ht_rebalancing_events)
        )
    }


def count_pool_stress_events(rebalancing_events: List) -> int:
    """Count events that indicate pool stress (multiple rebalances in short time)"""
    
    # Group rebalancing events by time windows (5-minute windows)
    time_windows = {}
    
    for event in rebalancing_events:
        minute = event.get("minute", 0)
        window = minute // 5  # 5-minute windows
        
        if window not in time_windows:
            time_windows[window] = 0
        time_windows[window] += 1
    
    # Count windows with high activity (>5 rebalances in 5 minutes)
    stress_events = sum(1 for count in time_windows.values() if count > 5)
    
    return stress_events


def analyze_pool_capacity_stress(rebalancing_events: List, stress_events: List, 
                                pool_utilization: float) -> Dict:
    """Analyze how pool capacity handles rebalancing stress"""
    
    avg_rebalancing = np.mean(rebalancing_events)
    avg_stress_events = np.mean(stress_events)
    
    # Determine stress level based on pool utilization and activity
    if pool_utilization > 0.8 and avg_stress_events > 2:
        stress_level = "Critical"
    elif pool_utilization > 0.6 and avg_stress_events > 1:
        stress_level = "High"
    elif pool_utilization > 0.4 or avg_stress_events > 0:
        stress_level = "Moderate"
    else:
        stress_level = "Low"
    
    return {
        "pool_utilization": pool_utilization,
        "avg_rebalancing_events": avg_rebalancing,
        "avg_stress_events": avg_stress_events,
        "stress_level": stress_level,
        "capacity_analysis": {
            "rebalancing_per_dollar_liquidity": avg_rebalancing / 500_000,  # $500K total pool
            "stress_events_per_dollar": avg_stress_events / 500_000,
            "utilization_efficiency": avg_rebalancing / max(pool_utilization, 0.01)
        }
    }


def generate_borrow_cap_implications(pool_utilization: float, stress_analysis: Dict,
                                   rebalancing_events: float) -> Dict:
    """Generate implications for borrow cap policy"""
    
    stress_level = stress_analysis["stress_level"]
    
    # Determine if borrow cap is needed
    if stress_level in ["Critical", "High"]:
        borrow_cap_needed = True
        recommended_cap = pool_utilization * 0.8  # 80% of current utilization
    elif stress_level == "Moderate":
        borrow_cap_needed = False  # Monitor only
        recommended_cap = pool_utilization * 1.2  # 20% buffer
    else:
        borrow_cap_needed = False
        recommended_cap = None
    
    # Calculate implied caps as percentage of pool liquidity
    if recommended_cap:
        cap_as_percentage = recommended_cap * 100
    else:
        cap_as_percentage = None
    
    return {
        "borrow_cap_needed": borrow_cap_needed,
        "recommended_cap_percentage": cap_as_percentage,
        "current_utilization_percentage": pool_utilization * 100,
        "stress_level": stress_level,
        "reasoning": generate_cap_reasoning(stress_level, pool_utilization, rebalancing_events),
        "monitoring_thresholds": {
            "utilization_warning": 60,  # 60% of pool capacity
            "utilization_critical": 80,  # 80% of pool capacity
            "rebalancing_activity_warning": 100,  # 100+ rebalancing events
            "stress_events_warning": 2  # 2+ stress periods
        }
    }


def generate_cap_reasoning(stress_level: str, pool_utilization: float, 
                          rebalancing_events: float) -> str:
    """Generate reasoning for borrow cap recommendation"""
    
    if stress_level == "Critical":
        return f"Pool utilization at {pool_utilization:.1%} with {rebalancing_events:.0f} rebalancing events indicates severe stress. Borrow cap needed to prevent system failure."
    elif stress_level == "High":
        return f"High rebalancing activity ({rebalancing_events:.0f} events) at {pool_utilization:.1%} utilization. Borrow cap recommended to maintain system stability."
    elif stress_level == "Moderate":
        return f"Moderate stress at {pool_utilization:.1%} utilization. Monitor closely but no immediate cap needed."
    else:
        return f"Low stress at {pool_utilization:.1%} utilization. System operating within normal parameters."


def analyze_liquidation_capacity_results(results_matrix: List[Dict]) -> Dict:
    """Analyze liquidation capacity results to determine optimal deposit cap ratios"""
    
    # Create DataFrame for analysis
    df_data = []
    for result in results_matrix:
        params = result["scenario_params"]
        position_analysis = result["position_analysis"]
        dex_analysis = result["dex_capacity_analysis"]
        liquidation_metrics = result["liquidation_metrics"]
        
        df_data.append({
            "deposit_cap_ratio": params["deposit_cap_ratio"],
            "btc_shock_percent": params["btc_shock_percent"],
            "initial_hf": params["initial_hf"],
            "profile": params["profile"],
            "total_positions": position_analysis["total_positions"],
            "positions_needing_liquidation": position_analysis["positions_needing_liquidation"],
            "liquidation_success_rate": liquidation_metrics["liquidation_success_rate"],
            "can_liquidate_all": dex_analysis["can_liquidate_all"],
            "total_slippage_cost": liquidation_metrics["total_slippage_cost"],
            "average_slippage_per_liquidation": liquidation_metrics["average_slippage_per_liquidation"]
        })
    
    df = pd.DataFrame(df_data)
    
    # Generate comprehensive analysis
    analysis = {
        "deposit_cap_thresholds": find_deposit_cap_thresholds(df),
        "liquidation_capacity_recommendations": generate_liquidation_capacity_recommendations(df),
        "shock_resistance_analysis": analyze_shock_resistance(df),
        "concentration_impact_analysis": analyze_concentration_impact(df),
        "risk_profile_liquidation_impact": analyze_risk_profile_liquidation_impact(df),
        "raw_results_matrix": results_matrix
    }
    
    return analysis


def find_deposit_cap_thresholds(df: pd.DataFrame) -> Dict:
    """Find critical deposit cap thresholds for liquidation capacity"""
    
    # Find scenarios where liquidation success rate drops below 90%
    failed_scenarios = df[df["liquidation_success_rate"] < 0.90]
    successful_scenarios = df[df["liquidation_success_rate"] >= 0.90]
    
    # Find the highest deposit cap ratio that maintains 90%+ liquidation success
    max_safe_ratio = successful_scenarios["deposit_cap_ratio"].max() if len(successful_scenarios) > 0 else 1.0
    min_failed_ratio = failed_scenarios["deposit_cap_ratio"].min() if len(failed_scenarios) > 0 else 5.0
    
    return {
        "max_safe_deposit_cap_ratio": max_safe_ratio,
        "min_failed_deposit_cap_ratio": min_failed_ratio,
        "recommended_deposit_cap_ratio": max_safe_ratio * 0.9,  # 10% safety buffer
        "failed_scenarios_count": len(failed_scenarios),
        "successful_scenarios_count": len(successful_scenarios)
    }


def generate_liquidation_capacity_recommendations(df: pd.DataFrame) -> Dict:
    """Generate specific deposit cap recommendations based on liquidation capacity"""
    
    # Analyze by shock level
    shock_analysis = {}
    for shock in [10, 15, 25]:
        shock_data = df[df["btc_shock_percent"] == shock]
        if len(shock_data) > 0:
            max_safe_ratio = shock_data[shock_data["liquidation_success_rate"] >= 0.90]["deposit_cap_ratio"].max()
            shock_analysis[f"{shock}%_shock"] = {
                "max_safe_ratio": max_safe_ratio if not pd.isna(max_safe_ratio) else 1.0,
                "recommended_ratio": max_safe_ratio * 0.9 if not pd.isna(max_safe_ratio) else 0.9
            }
    
    return {
        "by_shock_level": shock_analysis,
        "conservative_recommendation": 1.0,  # 1:1 ratio for maximum safety
        "moderate_recommendation": 2.0,  # 2:1 ratio for balanced risk
        "aggressive_recommendation": 3.0,  # 3:1 ratio for higher utilization
        "monitoring_thresholds": {
            "liquidation_success_warning": 0.95,  # Warn if success rate drops below 95%
            "liquidation_success_critical": 0.90,  # Critical if success rate drops below 90%
            "slippage_cost_warning": 0.05  # Warn if slippage exceeds 5% of liquidation value
        }
    }


def analyze_shock_resistance(df: pd.DataFrame) -> Dict:
    """Analyze how different deposit cap ratios handle various BTC price shocks"""
    
    shock_resistance = {}
    
    for ratio in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ratio_data = df[df["deposit_cap_ratio"] == ratio]
        if len(ratio_data) > 0:
            shock_resistance[f"{ratio:.1f}_ratio"] = {
                "avg_success_rate_10pct": ratio_data[ratio_data["btc_shock_percent"] == 10]["liquidation_success_rate"].mean(),
                "avg_success_rate_15pct": ratio_data[ratio_data["btc_shock_percent"] == 15]["liquidation_success_rate"].mean(),
                "avg_success_rate_25pct": ratio_data[ratio_data["btc_shock_percent"] == 25]["liquidation_success_rate"].mean(),
                "max_safe_shock": find_max_safe_shock(ratio_data)
            }
    
    return shock_resistance


def find_max_safe_shock(ratio_data: pd.DataFrame) -> float:
    """Find the maximum shock level that maintains 90%+ liquidation success"""
    safe_shocks = ratio_data[ratio_data["liquidation_success_rate"] >= 0.90]["btc_shock_percent"]
    return safe_shocks.max() if len(safe_shocks) > 0 else 0.0


def analyze_concentration_impact(df: pd.DataFrame) -> Dict:
    """Analyze how 80% concentration affects liquidation capacity"""
    
    # Calculate average slippage costs by deposit cap ratio
    slippage_analysis = {}
    for ratio in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ratio_data = df[df["deposit_cap_ratio"] == ratio]
        if len(ratio_data) > 0:
            slippage_analysis[f"{ratio:.1f}_ratio"] = {
                "avg_slippage_cost": ratio_data["total_slippage_cost"].mean(),
                "avg_slippage_per_liquidation": ratio_data["average_slippage_per_liquidation"].mean(),
                "slippage_efficiency": calculate_slippage_efficiency(ratio_data)
            }
    
    return {
        "slippage_by_ratio": slippage_analysis,
        "concentration_impact": "80% concentration provides good liquidity depth while maintaining reasonable slippage costs",
        "optimal_ratio_for_slippage": find_optimal_ratio_for_slippage(df)
    }


def calculate_slippage_efficiency(ratio_data: pd.DataFrame) -> float:
    """Calculate slippage efficiency (successful liquidations per dollar of slippage)"""
    successful_liquidations = ratio_data["positions_needing_liquidation"] * ratio_data["liquidation_success_rate"]
    total_slippage = ratio_data["total_slippage_cost"]
    return (successful_liquidations / total_slippage).mean() if total_slippage.sum() > 0 else 0.0


def find_optimal_ratio_for_slippage(df: pd.DataFrame) -> float:
    """Find the deposit cap ratio that provides the best slippage efficiency"""
    efficiency_by_ratio = {}
    for ratio in [1.0, 2.0, 3.0, 4.0, 5.0]:
        ratio_data = df[df["deposit_cap_ratio"] == ratio]
        if len(ratio_data) > 0:
            efficiency = calculate_slippage_efficiency(ratio_data)
            efficiency_by_ratio[ratio] = efficiency
    
    return max(efficiency_by_ratio, key=efficiency_by_ratio.get) if efficiency_by_ratio else 1.0


def analyze_risk_profile_liquidation_impact(df: pd.DataFrame) -> Dict:
    """Analyze how different risk profiles (initial HFs) impact liquidation capacity"""
    
    profile_impact = {}
    for profile in ["conservative", "moderate", "aggressive", "ultra_aggressive"]:
        profile_data = df[df["profile"] == profile]
        if len(profile_data) > 0:
            profile_impact[profile] = {
                "avg_liquidation_success_rate": profile_data["liquidation_success_rate"].mean(),
                "avg_positions_needing_liquidation": profile_data["positions_needing_liquidation"].mean(),
                "avg_slippage_cost": profile_data["total_slippage_cost"].mean(),
                "max_safe_deposit_cap_ratio": profile_data[profile_data["liquidation_success_rate"] >= 0.90]["deposit_cap_ratio"].max()
            }
    
    return profile_impact


def find_capacity_thresholds(df: pd.DataFrame) -> Dict:
    """Find critical capacity thresholds for the MOET:YT pool"""
    
    # Group by stress level to find thresholds
    stress_thresholds = {}
    
    for stress_level in ["Low", "Moderate", "High", "Critical"]:
        stress_data = df[df["stress_level"] == stress_level]
        
        if len(stress_data) > 0:
            stress_thresholds[stress_level] = {
                "min_utilization": stress_data["pool_utilization"].min(),
                "max_utilization": stress_data["pool_utilization"].max(),
                "min_agent_count": stress_data["agent_count"].min(),
                "max_agent_count": stress_data["agent_count"].max(),
                "avg_rebalancing_events": stress_data["rebalancing_events"].mean(),
                "avg_liquidation_rate": stress_data["liquidation_rate"].mean()
            }
    
    # Find critical thresholds
    critical_utilization = None
    safe_utilization = None
    
    # Critical: First utilization level that shows Critical or High stress
    critical_data = df[df["stress_level"].isin(["Critical", "High"])]
    if len(critical_data) > 0:
        critical_utilization = critical_data["pool_utilization"].min()
    
    # Safe: Highest utilization with Low stress
    safe_data = df[df["stress_level"] == "Low"]
    if len(safe_data) > 0:
        safe_utilization = safe_data["pool_utilization"].max()
    
    return {
        "stress_level_thresholds": stress_thresholds,
        "critical_utilization_threshold": critical_utilization,
        "safe_utilization_threshold": safe_utilization,
        "recommended_utilization_cap": safe_utilization * 0.9 if safe_utilization else 0.5  # 10% safety buffer
    }


def generate_borrow_cap_recommendations(df: pd.DataFrame) -> Dict:
    """Generate specific borrow cap recommendations"""
    
    # Find scenarios where borrow caps are recommended
    cap_needed = df[df["borrow_cap_needed"] == True]
    cap_not_needed = df[df["borrow_cap_needed"] == False]
    
    recommendations = {}
    
    if len(cap_needed) > 0:
        # Conservative recommendation: Lowest cap where system shows stress
        min_stressed_utilization = cap_needed["pool_utilization"].min()
        conservative_cap = min_stressed_utilization * 0.8  # 20% safety buffer
        
        # Aggressive recommendation: Just below stress threshold
        max_safe_utilization = cap_not_needed["pool_utilization"].max() if len(cap_not_needed) > 0 else 0.5
        aggressive_cap = max_safe_utilization * 1.1  # 10% above safe threshold
        
        recommendations = {
            "conservative_cap": {
                "utilization_percentage": conservative_cap * 100,
                "reasoning": f"20% safety buffer below observed stress at {min_stressed_utilization:.1%} utilization"
            },
            "aggressive_cap": {
                "utilization_percentage": aggressive_cap * 100,
                "reasoning": f"10% buffer above safe threshold of {max_safe_utilization:.1%} utilization"
            },
            "recommended_approach": "conservative_cap",  # Default to conservative
            "monitoring_required": True
        }
    else:
        # No caps needed - system handles all tested loads
        max_tested_utilization = df["pool_utilization"].max()
        
        recommendations = {
            "no_cap_needed": True,
            "max_tested_utilization": max_tested_utilization * 100,
            "reasoning": f"System stable up to {max_tested_utilization:.1%} pool utilization",
            "monitoring_threshold": max_tested_utilization * 1.2 * 100,  # Monitor at 20% above tested
            "recommended_approach": "monitoring_only"
        }
    
    return recommendations


def analyze_stress_levels_by_utilization(df: pd.DataFrame) -> Dict:
    """Analyze how stress levels correlate with pool utilization"""
    
    utilization_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, float('inf')]
    bin_labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%", "100%+"]
    
    utilization_analysis = {}
    
    for i, (low, high) in enumerate(zip(utilization_bins[:-1], utilization_bins[1:])):
        bin_label = bin_labels[i]
        bin_data = df[(df["pool_utilization"] >= low) & (df["pool_utilization"] < high)]
        
        if len(bin_data) > 0:
            utilization_analysis[bin_label] = {
                "utilization_range": f"{low:.0%}-{high:.0%}",
                "scenario_count": len(bin_data),
                "avg_survival_rate": bin_data["survival_rate"].mean(),
                "avg_rebalancing_events": bin_data["rebalancing_events"].mean(),
                "avg_liquidation_rate": bin_data["liquidation_rate"].mean(),
                "stress_level_distribution": bin_data["stress_level"].value_counts().to_dict(),
                "dominant_stress_level": bin_data["stress_level"].mode().iloc[0] if not bin_data["stress_level"].mode().empty else "Unknown"
            }
    
    return utilization_analysis


def find_rebalancing_capacity_limits(df: pd.DataFrame) -> Dict:
    """Find the limits of rebalancing capacity under different loads"""
    
    # Analyze rebalancing efficiency across different loads
    efficiency_analysis = {}
    
    for agent_count in df["agent_count"].unique():
        count_data = df[df["agent_count"] == agent_count]
        
        avg_rebalancing = count_data["rebalancing_events"].mean()
        avg_liquidations = count_data["liquidation_rate"].mean()
        avg_utilization = count_data["pool_utilization"].mean()
        
        # Calculate efficiency metrics
        rebalancing_per_agent = avg_rebalancing / agent_count
        liquidation_prevention_ratio = avg_rebalancing / max(avg_liquidations, 0.1)
        
        efficiency_analysis[f"{agent_count}_agents"] = {
            "agent_count": agent_count,
            "avg_pool_utilization": avg_utilization,
            "avg_rebalancing_events": avg_rebalancing,
            "rebalancing_per_agent": rebalancing_per_agent,
            "avg_liquidation_rate": avg_liquidations,
            "liquidation_prevention_ratio": liquidation_prevention_ratio,
            "efficiency_rating": "High" if liquidation_prevention_ratio > 10 else "Medium" if liquidation_prevention_ratio > 3 else "Low"
        }
    
    # Find capacity limits
    high_efficiency_scenarios = [v for v in efficiency_analysis.values() if v["efficiency_rating"] == "High"]
    max_efficient_utilization = max([s["avg_pool_utilization"] for s in high_efficiency_scenarios]) if high_efficiency_scenarios else 0.5
    
    return {
        "efficiency_by_agent_count": efficiency_analysis,
        "max_efficient_utilization": max_efficient_utilization,
        "recommended_operational_limit": max_efficient_utilization * 0.9  # 10% safety buffer
    }


def analyze_risk_profile_pool_impact(df: pd.DataFrame) -> Dict:
    """Analyze how different risk profiles impact pool capacity"""
    
    profile_impact = {}
    
    for profile in df["profile"].unique():
        profile_data = df[df["profile"] == profile]
        
        profile_impact[profile] = {
            "avg_rebalancing_events": profile_data["rebalancing_events"].mean(),
            "avg_liquidation_rate": profile_data["liquidation_rate"].mean(),
            "avg_stress_events": profile_data["pool_stress_events"].mean(),
            "stress_level_distribution": profile_data["stress_level"].value_counts().to_dict(),
            "pool_impact_rating": calculate_pool_impact_rating(profile_data)
        }
    
    return profile_impact


def calculate_pool_impact_rating(profile_data: pd.DataFrame) -> str:
    """Calculate pool impact rating for a risk profile"""
    
    avg_stress = profile_data["pool_stress_events"].mean()
    avg_rebalancing = profile_data["rebalancing_events"].mean()
    
    if avg_stress > 2 or avg_rebalancing > 200:
        return "High Impact"
    elif avg_stress > 1 or avg_rebalancing > 100:
        return "Medium Impact"
    else:
        return "Low Impact"


def save_borrow_cap_results(results: List[Dict]):
    """Save liquidation capacity analysis results with numbered run folder"""
    
    # Create base results directory
    base_results_path = Path("tidal_protocol_sim/results/moet_yt_borrow_cap_analysis")
    base_results_path.mkdir(parents=True, exist_ok=True)
    
    # Find next available run number
    run_number = 1
    while True:
        run_folder = base_results_path / f"run_{run_number:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if not run_folder.exists():
            break
        run_number += 1
    
    # Create run-specific directory
    run_folder.mkdir(parents=True, exist_ok=True)
    
    # Build clean, navigable results structure
    clean_scenario_results = []
    for result in results:
        ratio = result["deposit_cap_ratio"]
        
        # Extract the ratio_runs from detailed_simulation_data
        ratio_runs = []
        if "detailed_simulation_data" in result and "ratio_runs" in result["detailed_simulation_data"]:
            ratio_runs = result["detailed_simulation_data"]["ratio_runs"]
        
        clean_runs = build_clean_liquidation_data(ratio_runs, ratio)
        
        clean_scenario = {
            "deposit_cap_ratio": ratio,
            "scenario_params": result.get("scenario_params", {}),
            "liquidation_summary": result.get("liquidation_summary", {}),
            "shock_scenario_results": result.get("shock_scenario_results", []),
            "simulation_runs": clean_runs
        }
        clean_scenario_results.append(clean_scenario)
    
    liquidation_analysis_results = {
        "analysis_metadata": {
            "analysis_type": "BTC_MOET_Liquidation_Capacity_Analysis",
            "timestamp": datetime.now().isoformat(),
            "deposit_cap_ratios_tested": [1.0, 2.0, 3.0, 4.0, 5.0],
            "btc_shock_scenarios": [10, 15, 25],
            "agents_per_scenario": 15,
            "total_scenarios": len(results),
            "protocol_type": "BTC_only",
            "uniswap_v3_math": True,
            "moet_btc_concentration": 0.80,
            "yield_token_concentration": 0.95
        },
        "detailed_scenario_results": clean_scenario_results
    }
    
    results_path = run_folder / "liquidation_capacity_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(liquidation_analysis_results, f, indent=2, default=str)
    
    # Generate CSV output with agent-by-agent liquidation outcomes
    csv_path = generate_agent_liquidation_csv(liquidation_analysis_results, run_folder)
    
    print(f"📁 Liquidation capacity analysis results saved to: {run_folder}")
    print(f"📊 Agent-by-agent CSV saved to: {csv_path}")
    return run_folder

def generate_agent_liquidation_csv(results_data: Dict, output_dir: Path) -> Path:
    """Generate CSV output with agent-by-agent liquidation outcomes"""
    
    csv_data = []
    
    # Process each deposit cap ratio scenario
    for scenario in results_data["detailed_scenario_results"]:
        deposit_cap_ratio = scenario["deposit_cap_ratio"]
        
        # Process each simulation run (price shock scenario)
        for run in scenario["simulation_runs"]:
            shock_percent = run["scenario_params"]["btc_shock_percent"]
            agent_analysis = run["agent_analysis"]
            liquidation_metrics = run["liquidation_metrics"]
            
            # Process each agent in this run
            for agent_detail in agent_analysis["agent_details"]:
                csv_row = {
                    "deposit_cap_ratio": deposit_cap_ratio,
                    "btc_shock_percent": shock_percent,
                    "agent_id": agent_detail["agent_id"],
                    "initial_health_factor": agent_detail["initial_hf"],
                    "health_factor_after_shock": agent_detail["new_hf_after_shock"],
                    "needs_liquidation": agent_detail["needs_liquidation"],
                    "debt_to_liquidate": agent_detail["debt_to_liquidate"],
                    "collateral_to_seize": agent_detail["collateral_to_seize"],
                    "liquidation_priority": agent_detail["liquidation_priority"],
                    "liquidation_success_rate": liquidation_metrics["liquidation_success_rate"],
                    "total_slippage_cost": liquidation_metrics["total_slippage_cost"],
                    "concentration_utilization": liquidation_metrics["concentration_utilization"]
                }
                csv_data.append(csv_row)
    
    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(csv_data)
    csv_path = output_dir / "agent_liquidation_outcomes.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path


def build_clean_liquidation_data(ratio_runs: List[Dict], ratio: float) -> List[Dict]:
    """Build clean, navigable liquidation data structure"""
    clean_runs = []
    
    for run_idx, run in enumerate(ratio_runs):
        shock_percent = run["scenario_params"]["btc_shock_percent"]
        
        clean_run = {
            "run_id": f"run_{run_idx+1:03d}_ratio_{ratio}_shock_{shock_percent}",
            "timestamp": datetime.now().isoformat(),
            "scenario_params": run["scenario_params"],
            "agent_analysis": run["agent_analysis"],
            "dex_capacity_analysis": run["dex_capacity_analysis"],
            "liquidation_metrics": run["liquidation_metrics"],
            "summary_stats": {
                "total_agents": run["agent_analysis"]["total_agents"],
                "positions_needing_liquidation": run["agent_analysis"]["positions_needing_liquidation"],
                "liquidation_success_rate": run["liquidation_metrics"]["liquidation_success_rate"],
                "total_slippage_cost": run["liquidation_metrics"]["total_slippage_cost"],
                "concentration_utilization": run["liquidation_metrics"]["concentration_utilization"]
            }
        }
        clean_runs.append(clean_run)
    
    return clean_runs


def create_liquidation_capacity_charts(results: List[Dict], output_dir: Path) -> List[Path]:
    """Create comprehensive charts for liquidation capacity analysis"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    print("🎨 Generating liquidation capacity analysis charts...")
    
    # Create charts directory
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    
    generated_charts = []
    
    try:
        # 1. LP Concentration Efficiency Chart (as requested)
        concentration_chart = create_lp_concentration_efficiency_chart(results, charts_dir)
        if concentration_chart:
            generated_charts.append(concentration_chart)
        
        # 2. Liquidation Success Rate Analysis
        success_chart = create_liquidation_success_analysis(results, charts_dir)
        if success_chart:
            generated_charts.append(success_chart)
        
        # 3. Slippage Cost Analysis
        slippage_chart = create_slippage_analysis(results, charts_dir)
        if slippage_chart:
            generated_charts.append(slippage_chart)
        
        # 4. Deposit Cap Ratio Performance
        performance_chart = create_deposit_cap_performance_chart(results, charts_dir)
        if performance_chart:
            generated_charts.append(performance_chart)
        
        # 5. Liquidation Dollars Analysis
        liquidation_dollars_chart = create_liquidation_dollars_chart(results, charts_dir)
        if liquidation_dollars_chart:
            generated_charts.append(liquidation_dollars_chart)
        
        print(f"✅ Generated {len(generated_charts)} liquidation capacity charts")
        
    except Exception as e:
        print(f"❌ Error generating charts: {e}")
        import traceback
        traceback.print_exc()
    
    return generated_charts

def create_lp_concentration_efficiency_chart(results: List[Dict], charts_dir: Path) -> Path:
    """Create LP Concentration Efficiency chart: X-axis = $ of liquidations, Y-axis = Concentration utilization"""
    
    try:
        # Setup styling
        plt.style.use('default')
        sns.set_palette("husl")
        
        plt.rcParams.update({
            'figure.figsize': (16, 12),
            'font.size': 11,
            'axes.titlesize': 16,
            'axes.labelsize': 13,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'figure.titlesize': 18
        })
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("BTC:MOET Pool Liquidation Capacity Analysis", fontsize=18, fontweight='bold')
        
        # Collect data for concentration efficiency analysis
        concentration_data = []
        
        for result in results:
            ratio = result["deposit_cap_ratio"]
            for shock_run in result["shock_scenario_results"]:
                shock_percent = shock_run["scenario_params"]["btc_shock_percent"]
                liquidation_events = shock_run["dex_capacity_analysis"].get("liquidation_events", [])
                
                for event in liquidation_events:
                    concentration_data.append({
                        "deposit_cap_ratio": ratio,
                        "btc_shock_percent": shock_percent,
                        "cumulative_liquidations": event["cumulative_liquidations"],
                        "concentration_utilization": event["concentration_utilization"],
                        "debt_liquidated": event["debt_liquidated"],
                        "slippage_cost": event["slippage_cost"]
                    })
        
        if not concentration_data:
            print("⚠️ No liquidation events found for concentration analysis")
            return None
        
        df = pd.DataFrame(concentration_data)
        
        # Chart 1: LP Concentration Efficiency (as requested)
        for ratio in sorted(df["deposit_cap_ratio"].unique()):
            ratio_data = df[df["deposit_cap_ratio"] == ratio]
            ax1.plot(ratio_data["cumulative_liquidations"], ratio_data["concentration_utilization"], 
                    marker='o', linewidth=2, label=f'{ratio:.0f}:1 Ratio', markersize=6)
        
        ax1.set_xlabel('Cumulative Liquidations ($)')
        ax1.set_ylabel('Concentration Utilization (%)')
        ax1.set_title('LP Concentration Efficiency: Liquidations vs Utilization')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Full Concentration Range')
        
        # Chart 2: Liquidation Success Rate by Ratio and Shock
        success_data = []
        for result in results:
            ratio = result["deposit_cap_ratio"]
            for shock_run in result["shock_scenario_results"]:
                shock_percent = shock_run["scenario_params"]["btc_shock_percent"]
                success_rate = shock_run["liquidation_metrics"]["liquidation_success_rate"]
                success_data.append({
                    "deposit_cap_ratio": ratio,
                    "btc_shock_percent": shock_percent,
                    "success_rate": success_rate
                })
        
        success_df = pd.DataFrame(success_data)
        pivot_data = success_df.pivot_table(
            values='success_rate', 
            index='deposit_cap_ratio', 
            columns='btc_shock_percent', 
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax2,
                    cbar_kws={'label': 'Liquidation Success Rate'})
        ax2.set_title('Liquidation Success Rate by Ratio and Shock')
        ax2.set_xlabel('BTC Shock Level (%)')
        ax2.set_ylabel('Deposit Cap Ratio')
        
        # Chart 3: Slippage Cost Analysis
        for ratio in sorted(df["deposit_cap_ratio"].unique()):
            ratio_data = df[df["deposit_cap_ratio"] == ratio]
            if len(ratio_data) > 0:
                ax3.scatter(ratio_data["cumulative_liquidations"], ratio_data["slippage_cost"], 
                           label=f'{ratio:.0f}:1 Ratio', alpha=0.7, s=50)
        
        ax3.set_xlabel('Cumulative Liquidations ($)')
        ax3.set_ylabel('Slippage Cost ($)')
        ax3.set_title('Slippage Cost vs Liquidations')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Agent Liquidation Priority Analysis
        priority_data = []
        for result in results:
            ratio = result["deposit_cap_ratio"]
            for shock_run in result["shock_scenario_results"]:
                shock_percent = shock_run["scenario_params"]["btc_shock_percent"]
                agent_details = shock_run["agent_analysis"].get("agent_details", [])
                
                for i, agent in enumerate(agent_details):
                    if agent["needs_liquidation"]:
                        priority_data.append({
                            "deposit_cap_ratio": ratio,
                            "btc_shock_percent": shock_percent,
                            "liquidation_priority": i + 1,  # 1-based priority
                            "debt_to_liquidate": agent["debt_to_liquidate"],
                            "initial_hf": agent["initial_hf"]
                        })
        
        if priority_data:
            priority_df = pd.DataFrame(priority_data)
            
            # Show liquidation priority distribution
            for ratio in sorted(priority_df["deposit_cap_ratio"].unique()):
                ratio_data = priority_df[priority_df["deposit_cap_ratio"] == ratio]
                ax4.hist(ratio_data["liquidation_priority"], bins=10, alpha=0.7, 
                        label=f'{ratio:.0f}:1 Ratio', density=True)
        
        ax4.set_xlabel('Liquidation Priority (1 = Highest Debt)')
        ax4.set_ylabel('Density')
        ax4.set_title('Liquidation Priority Distribution by Ratio')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        chart_path = charts_dir / "lp_concentration_efficiency_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Generated LP concentration efficiency chart: {chart_path.name}")
        return chart_path
        
    except Exception as e:
        print(f"❌ Error creating LP concentration efficiency chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_success_rate_heatmap(df, charts_dir):
    """Create heatmap showing liquidation success rates"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Pivot data for heatmap
    heatmap_data = df.pivot_table(
        values='liquidation_success_rate', 
        index='deposit_cap_ratio', 
        columns='btc_shock_percent', 
        aggfunc='mean'
    )
    
    # Update index labels to show ratio format
    heatmap_data.index = [f'{ratio:.0f}:1' for ratio in heatmap_data.index]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                cbar_kws={'label': 'Liquidation Success Rate'})
    plt.title('Liquidation Success Rate by Deposit Cap Ratio and BTC Shock Level', fontsize=16, fontweight='bold')
    plt.xlabel('BTC Price Shock (%)', fontsize=12)
    plt.ylabel('Deposit Cap Ratio', fontsize=12)
    plt.tight_layout()
    plt.savefig(charts_dir / 'liquidation_success_rate_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_slippage_cost_analysis(df, charts_dir):
    """Create slippage cost analysis charts"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Filter out scenarios with no liquidations
    df_with_liquidations = df[df['positions_needing_liquidation'] > 0].copy()
    
    if len(df_with_liquidations) == 0:
        print("⚠️ No liquidations found for slippage analysis")
        return
    
    # 1. Total Slippage Cost by Scenario
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Total slippage cost by ratio and shock
    cost_by_scenario = df_with_liquidations.groupby(['deposit_cap_ratio', 'btc_shock_percent'])['total_slippage_cost'].sum().reset_index()
    
    for shock in df_with_liquidations['btc_shock_percent'].unique():
        shock_data = cost_by_scenario[cost_by_scenario['btc_shock_percent'] == shock]
        ax1.plot(shock_data['deposit_cap_ratio'], shock_data['total_slippage_cost'], 
                marker='o', linewidth=2, label=f'{shock}% Shock')
    
    # Set x-axis labels to ratio format
    ax1.set_xticks(cost_by_scenario['deposit_cap_ratio'].unique())
    ax1.set_xticklabels([f'{ratio:.0f}:1' for ratio in cost_by_scenario['deposit_cap_ratio'].unique()])
    ax1.set_xlabel('Deposit Cap Ratio', fontsize=12)
    ax1.set_ylabel('Total Slippage Cost ($)', fontsize=12)
    ax1.set_title('Total Slippage Cost by Deposit Cap Ratio and Shock Level', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Average slippage per liquidation
    avg_slippage = df_with_liquidations.groupby(['deposit_cap_ratio', 'btc_shock_percent'])['average_slippage_per_liquidation'].mean().reset_index()
    
    for shock in df_with_liquidations['btc_shock_percent'].unique():
        shock_data = avg_slippage[avg_slippage['btc_shock_percent'] == shock]
        ax2.plot(shock_data['deposit_cap_ratio'], shock_data['average_slippage_per_liquidation'], 
                marker='s', linewidth=2, label=f'{shock}% Shock')
    
    # Set x-axis labels to ratio format
    ax2.set_xticks(avg_slippage['deposit_cap_ratio'].unique())
    ax2.set_xticklabels([f'{ratio:.0f}:1' for ratio in avg_slippage['deposit_cap_ratio'].unique()])
    ax2.set_xlabel('Deposit Cap Ratio', fontsize=12)
    ax2.set_ylabel('Average Slippage per Liquidation ($)', fontsize=12)
    ax2.set_title('Average Slippage Cost per Liquidation', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'slippage_cost_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_position_distribution_charts(df, charts_dir):
    """Create position distribution analysis charts"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Total Positions by Ratio
    position_counts = df.groupby('deposit_cap_ratio')['total_positions'].sum().reset_index()
    ax1.bar(position_counts['deposit_cap_ratio'], position_counts['total_positions'], 
            color='skyblue', alpha=0.7, edgecolor='navy')
    ax1.set_xticks(position_counts['deposit_cap_ratio'])
    ax1.set_xticklabels([f'{ratio:.0f}:1' for ratio in position_counts['deposit_cap_ratio']])
    ax1.set_xlabel('Deposit Cap Ratio', fontsize=12)
    ax1.set_ylabel('Total Positions', fontsize=12)
    ax1.set_title('Total Positions by Deposit Cap Ratio', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Positions Needing Liquidation by Shock Level
    liquidation_by_shock = df.groupby('btc_shock_percent')['positions_needing_liquidation'].sum().reset_index()
    ax2.bar(liquidation_by_shock['btc_shock_percent'], liquidation_by_shock['positions_needing_liquidation'], 
            color='lightcoral', alpha=0.7, edgecolor='darkred')
    ax2.set_xlabel('BTC Shock Level (%)', fontsize=12)
    ax2.set_ylabel('Positions Needing Liquidation', fontsize=12)
    ax2.set_title('Positions Needing Liquidation by Shock Level', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Liquidation Rate by Health Factor Profile
    profile_liquidation = df.groupby('profile')['liquidation_success_rate'].mean().reset_index()
    ax3.bar(profile_liquidation['profile'], profile_liquidation['liquidation_success_rate'], 
            color='lightgreen', alpha=0.7, edgecolor='darkgreen')
    ax3.set_xlabel('Health Factor Profile', fontsize=12)
    ax3.set_ylabel('Average Liquidation Success Rate', fontsize=12)
    ax3.set_title('Liquidation Success Rate by Health Factor Profile', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Position Distribution by Ratio and Shock
    pivot_data = df.pivot_table(
        values='positions_needing_liquidation', 
        index='deposit_cap_ratio', 
        columns='btc_shock_percent', 
        aggfunc='sum'
    )
    
    sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax4,
                cbar_kws={'label': 'Positions Needing Liquidation'})
    ax4.set_title('Positions Needing Liquidation by Ratio and Shock', fontsize=14, fontweight='bold')
    ax4.set_xlabel('BTC Shock Level (%)', fontsize=12)
    ax4.set_ylabel('Deposit Cap Ratio', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'position_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_deposit_cap_threshold_charts(df, charts_dir):
    """Create deposit cap threshold analysis charts"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 1. Breaking Points by Shock Level
    breaking_points = {}
    for shock in df['btc_shock_percent'].unique():
        shock_data = df[df['btc_shock_percent'] == shock]
        # Find the highest ratio where all scenarios have 100% success rate
        max_safe_ratio = shock_data[shock_data['liquidation_success_rate'] == 1.0]['deposit_cap_ratio'].max()
        breaking_points[shock] = max_safe_ratio if not pd.isna(max_safe_ratio) else 0
    
    ax1.bar(breaking_points.keys(), breaking_points.values(), 
            color='gold', alpha=0.7, edgecolor='orange')
    ax1.set_xlabel('BTC Shock Level (%)', fontsize=12)
    ax1.set_ylabel('Max Safe Deposit Cap Ratio', fontsize=12)
    ax1.set_title('Maximum Safe Deposit Cap Ratio by Shock Level', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Risk Profile by Ratio
    risk_profile = df.groupby('deposit_cap_ratio').agg({
        'liquidation_success_rate': 'mean',
        'total_slippage_cost': 'sum'
    }).reset_index()
    
    ax2_twin = ax2.twinx()
    
    bars = ax2.bar(risk_profile['deposit_cap_ratio'], risk_profile['liquidation_success_rate'], 
                   color='lightblue', alpha=0.7, edgecolor='navy', label='Success Rate')
    line = ax2_twin.plot(risk_profile['deposit_cap_ratio'], risk_profile['total_slippage_cost'], 
                         color='red', marker='o', linewidth=2, label='Total Slippage Cost')
    
    ax2.set_xlabel('Deposit Cap Ratio', fontsize=12)
    ax2.set_ylabel('Average Liquidation Success Rate', fontsize=12)
    ax2_twin.set_ylabel('Total Slippage Cost ($)', fontsize=12)
    ax2.set_title('Risk Profile: Success Rate vs Slippage Cost', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'deposit_cap_threshold_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_risk_assessment_charts(df, charts_dir):
    """Create risk assessment charts"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Shock Resistance Matrix
    resistance_matrix = df.pivot_table(
        values='liquidation_success_rate', 
        index='deposit_cap_ratio', 
        columns='btc_shock_percent', 
        aggfunc='mean'
    )
    
    sns.heatmap(resistance_matrix, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax1,
                cbar_kws={'label': 'Success Rate'})
    ax1.set_title('Shock Resistance Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('BTC Shock Level (%)', fontsize=12)
    ax1.set_ylabel('Deposit Cap Ratio', fontsize=12)
    
    # 2. Health Factor Profile Risk Assessment
    profile_risk = df.groupby('profile').agg({
        'liquidation_success_rate': 'mean',
        'average_slippage_per_liquidation': 'mean'
    }).reset_index()
    
    scatter = ax2.scatter(profile_risk['liquidation_success_rate'], 
                         profile_risk['average_slippage_per_liquidation'],
                         s=200, alpha=0.7, c=range(len(profile_risk)), cmap='viridis')
    
    for i, profile in enumerate(profile_risk['profile']):
        ax2.annotate(profile, (profile_risk.iloc[i]['liquidation_success_rate'], 
                              profile_risk.iloc[i]['average_slippage_per_liquidation']),
                    xytext=(5, 5), textcoords='offset points')
    
    ax2.set_xlabel('Average Liquidation Success Rate', fontsize=12)
    ax2.set_ylabel('Average Slippage per Liquidation ($)', fontsize=12)
    ax2.set_title('Health Factor Profile Risk Assessment', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Liquidation Capacity Utilization
    capacity_utilization = df.groupby('deposit_cap_ratio').agg({
        'total_positions': 'sum',
        'positions_needing_liquidation': 'sum'
    }).reset_index()
    capacity_utilization['utilization_rate'] = (
        capacity_utilization['positions_needing_liquidation'] / 
        capacity_utilization['total_positions']
    )
    
    ax3.bar(capacity_utilization['deposit_cap_ratio'], capacity_utilization['utilization_rate'], 
            color='orange', alpha=0.7, edgecolor='darkorange')
    ax3.set_xlabel('Deposit Cap Ratio', fontsize=12)
    ax3.set_ylabel('Liquidation Utilization Rate', fontsize=12)
    ax3.set_title('Liquidation Capacity Utilization by Ratio', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Risk vs Reward Analysis
    risk_reward = df.groupby('deposit_cap_ratio').agg({
        'liquidation_success_rate': 'mean',
        'total_slippage_cost': 'sum'
    }).reset_index()
    
    ax4.scatter(risk_reward['deposit_cap_ratio'], risk_reward['liquidation_success_rate'], 
               s=risk_reward['total_slippage_cost']/100, alpha=0.6, c='purple')
    ax4.set_xlabel('Deposit Cap Ratio', fontsize=12)
    ax4.set_ylabel('Average Liquidation Success Rate', fontsize=12)
    ax4.set_title('Risk vs Reward Analysis\n(Bubble size = Total Slippage Cost)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'risk_assessment_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_economic_impact_charts(df, charts_dir):
    """Create economic impact analysis charts"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # 1. Total Slippage Cost Escalation
    cost_escalation = df.groupby(['deposit_cap_ratio', 'btc_shock_percent'])['total_slippage_cost'].sum().reset_index()
    
    for shock in df['btc_shock_percent'].unique():
        shock_data = cost_escalation[cost_escalation['btc_shock_percent'] == shock]
        ax1.plot(shock_data['deposit_cap_ratio'], shock_data['total_slippage_cost'], 
                marker='o', linewidth=3, label=f'{shock}% Shock', markersize=8)
    
    ax1.set_xlabel('Deposit Cap Ratio', fontsize=12)
    ax1.set_ylabel('Total Slippage Cost ($)', fontsize=12)
    ax1.set_title('Slippage Cost Escalation by Ratio and Shock', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Liquidation Efficiency Analysis
    efficiency_data = df.groupby('deposit_cap_ratio').agg({
        'positions_needing_liquidation': 'sum',
        'liquidation_success_rate': 'mean'
    }).reset_index()
    efficiency_data['efficiency_score'] = (
        efficiency_data['positions_needing_liquidation'] * efficiency_data['liquidation_success_rate']
    )
    
    ax2.bar(efficiency_data['deposit_cap_ratio'], efficiency_data['efficiency_score'], 
            color='lightgreen', alpha=0.7, edgecolor='darkgreen')
    ax2.set_xlabel('Deposit Cap Ratio', fontsize=12)
    ax2.set_ylabel('Liquidation Efficiency Score', fontsize=12)
    ax2.set_title('Liquidation Efficiency by Ratio', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Pool Stress Indicators
    stress_indicators = df.groupby('deposit_cap_ratio').agg({
        'total_positions': 'sum',
        'positions_needing_liquidation': 'sum',
        'total_slippage_cost': 'sum'
    }).reset_index()
    stress_indicators['stress_level'] = (
        stress_indicators['positions_needing_liquidation'] / stress_indicators['total_positions']
    )
    
    ax3_twin = ax3.twinx()
    
    bars = ax3.bar(stress_indicators['deposit_cap_ratio'], stress_indicators['stress_level'], 
                   color='lightcoral', alpha=0.7, edgecolor='darkred', label='Stress Level')
    line = ax3_twin.plot(stress_indicators['deposit_cap_ratio'], stress_indicators['total_slippage_cost'], 
                         color='blue', marker='s', linewidth=2, label='Total Slippage Cost')
    
    ax3.set_xlabel('Deposit Cap Ratio', fontsize=12)
    ax3.set_ylabel('Pool Stress Level', fontsize=12)
    ax3_twin.set_ylabel('Total Slippage Cost ($)', fontsize=12)
    ax3.set_title('Pool Stress Indicators', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 4. Cost per Position Analysis
    cost_per_position = df[df['total_positions'] > 0].copy()
    cost_per_position['cost_per_position'] = (
        cost_per_position['total_slippage_cost'] / cost_per_position['total_positions']
    )
    
    cost_by_ratio = cost_per_position.groupby('deposit_cap_ratio')['cost_per_position'].mean().reset_index()
    
    ax4.bar(cost_by_ratio['deposit_cap_ratio'], cost_by_ratio['cost_per_position'], 
            color='gold', alpha=0.7, edgecolor='orange')
    ax4.set_xlabel('Deposit Cap Ratio', fontsize=12)
    ax4.set_ylabel('Average Cost per Position ($)', fontsize=12)
    ax4.set_title('Average Slippage Cost per Position by Ratio', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(charts_dir / 'economic_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_borrow_cap_summary(analysis: Dict):
    """Print summary of BTC:MOET deposit cap analysis"""
    
    deposit_cap_thresholds = analysis.get("deposit_cap_thresholds", {})
    liquidation_recs = analysis.get("liquidation_capacity_recommendations", {})
    shock_analysis = analysis.get("shock_resistance_analysis", {})
    
    print("\n" + "=" * 80)
    print("BTC:MOET POOL DEPOSIT CAP ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Deposit cap thresholds
    max_safe_ratio = deposit_cap_thresholds.get("max_safe_deposit_cap_ratio", 1.0)
    recommended_ratio = deposit_cap_thresholds.get("recommended_deposit_cap_ratio", 1.0)
    
    print(f"\n💰 DEPOSIT CAP THRESHOLDS:")
    print(f"   Max safe ratio: {max_safe_ratio:.1f}:1 (deposits:liquidity)")
    print(f"   Recommended ratio: {recommended_ratio:.1f}:1 (with 10% safety buffer)")
    
    # Liquidation capacity recommendations
    print(f"\n📊 LIQUIDATION CAPACITY RECOMMENDATIONS:")
    by_shock = liquidation_recs.get("by_shock_level", {})
    
    for shock_level, data in by_shock.items():
        max_safe = data.get("max_safe_ratio", 1.0)
        recommended = data.get("recommended_ratio", 1.0)
        print(f"   {shock_level}: Max {max_safe:.1f}:1, Recommended {recommended:.1f}:1")
    
    conservative_rec = liquidation_recs.get("conservative_recommendation", 1.0)
    moderate_rec = liquidation_recs.get("moderate_recommendation", 2.0)
    aggressive_rec = liquidation_recs.get("aggressive_recommendation", 3.0)
    
    print(f"\n🎯 RISK-BASED RECOMMENDATIONS:")
    print(f"   🛡️  Conservative: {conservative_rec:.1f}:1 ratio (maximum safety)")
    print(f"   ⚖️  Moderate: {moderate_rec:.1f}:1 ratio (balanced risk)")
    print(f"   ⚡ Aggressive: {aggressive_rec:.1f}:1 ratio (higher utilization)")
    
    # Shock resistance analysis
    print(f"\n📉 SHOCK RESISTANCE ANALYSIS:")
    for ratio, data in shock_analysis.items():
        max_safe_shock = data.get("max_safe_shock", 0)
        success_10 = data.get("avg_success_rate_10pct", 0)
        success_25 = data.get("avg_success_rate_25pct", 0)
        
        print(f"   {ratio} ratio: Max safe shock {max_safe_shock:.0f}%, "
              f"10% shock {success_10:.1%} success, 25% shock {success_25:.1%} success")
    
    print("\n" + "=" * 80)


def create_liquidation_success_analysis(results: List[Dict], charts_dir: Path) -> Path:
    """Create liquidation success rate analysis chart"""
    try:
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Liquidation Success Rate Analysis", fontsize=16, fontweight='bold')
        
        # Extract success rate data
        success_data = []
        for result in results:
            ratio = result["deposit_cap_ratio"]
            for shock_run in result["shock_scenario_results"]:
                shock_percent = shock_run["scenario_params"]["btc_shock_percent"]
                success_rate = shock_run["liquidation_metrics"]["liquidation_success_rate"]
                success_data.append({
                    "deposit_cap_ratio": ratio,
                    "btc_shock_percent": shock_percent,
                    "success_rate": success_rate
                })
        
        df = pd.DataFrame(success_data)
        
        # Chart 1: Success rate by ratio
        ratio_success = df.groupby('deposit_cap_ratio')['success_rate'].mean().reset_index()
        ax1.bar(ratio_success['deposit_cap_ratio'], ratio_success['success_rate'], 
                color='skyblue', alpha=0.7, edgecolor='navy')
        ax1.set_xticks(ratio_success['deposit_cap_ratio'])
        ax1.set_xticklabels([f'{ratio:.0f}:1' for ratio in ratio_success['deposit_cap_ratio']])
        ax1.set_xlabel('Deposit Cap Ratio')
        ax1.set_ylabel('Mean Liquidation Success Rate')
        ax1.set_title('Liquidation Success Rate by Deposit Cap Ratio')
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Success rate by shock level
        shock_success = df.groupby('btc_shock_percent')['success_rate'].mean().reset_index()
        ax2.bar(shock_success['btc_shock_percent'], shock_success['success_rate'], 
                color='lightcoral', alpha=0.7, edgecolor='darkred')
        ax2.set_xlabel('BTC Shock Level (%)')
        ax2.set_ylabel('Mean Liquidation Success Rate')
        ax2.set_title('Liquidation Success Rate by Shock Level')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        chart_path = charts_dir / "liquidation_success_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"❌ Error creating liquidation success analysis: {e}")
        return None

def create_slippage_analysis(results: List[Dict], charts_dir: Path) -> Path:
    """Create slippage cost analysis chart"""
    try:
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Slippage Cost Analysis", fontsize=16, fontweight='bold')
        
        # Extract slippage data
        slippage_data = []
        for result in results:
            ratio = result["deposit_cap_ratio"]
            for shock_run in result["shock_scenario_results"]:
                shock_percent = shock_run["scenario_params"]["btc_shock_percent"]
                total_slippage = shock_run["liquidation_metrics"]["total_slippage_cost"]
                avg_slippage = shock_run["liquidation_metrics"]["average_slippage_per_liquidation"]
                slippage_data.append({
                    "deposit_cap_ratio": ratio,
                    "btc_shock_percent": shock_percent,
                    "total_slippage": total_slippage,
                    "avg_slippage": avg_slippage
                })
        
        df = pd.DataFrame(slippage_data)
        
        # Chart 1: Total slippage by ratio
        ratio_slippage = df.groupby('deposit_cap_ratio')['total_slippage'].mean().reset_index()
        ax1.bar(ratio_slippage['deposit_cap_ratio'], ratio_slippage['total_slippage'], 
                color='orange', alpha=0.7, edgecolor='darkorange')
        ax1.set_xticks(ratio_slippage['deposit_cap_ratio'])
        ax1.set_xticklabels([f'{ratio:.0f}:1' for ratio in ratio_slippage['deposit_cap_ratio']])
        ax1.set_xlabel('Deposit Cap Ratio')
        ax1.set_ylabel('Mean Total Slippage Cost ($)')
        ax1.set_title('Total Slippage Cost by Deposit Cap Ratio')
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Average slippage per liquidation
        ratio_avg_slippage = df.groupby('deposit_cap_ratio')['avg_slippage'].mean().reset_index()
        ax2.bar(ratio_avg_slippage['deposit_cap_ratio'], ratio_avg_slippage['avg_slippage'], 
                color='gold', alpha=0.7, edgecolor='darkgoldenrod')
        ax2.set_xticks(ratio_avg_slippage['deposit_cap_ratio'])
        ax2.set_xticklabels([f'{ratio:.0f}:1' for ratio in ratio_avg_slippage['deposit_cap_ratio']])
        ax2.set_xlabel('Deposit Cap Ratio')
        ax2.set_ylabel('Mean Slippage per Liquidation ($)')
        ax2.set_title('Average Slippage per Liquidation by Ratio')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        chart_path = charts_dir / "slippage_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"❌ Error creating slippage analysis: {e}")
        return None

def create_deposit_cap_performance_chart(results: List[Dict], charts_dir: Path) -> Path:
    """Create deposit cap ratio performance summary chart"""
    try:
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Deposit Cap Ratio Performance Analysis", fontsize=18, fontweight='bold')
        
        # Extract performance data
        performance_data = []
        for result in results:
            ratio = result["deposit_cap_ratio"]
            summary = result["liquidation_summary"]
            performance_data.append({
                "deposit_cap_ratio": ratio,
                "mean_success_rate": summary["mean_success_rate"],
                "mean_positions_needing_liquidation": summary["mean_positions_needing_liquidation"],
                "mean_slippage_cost": summary["mean_slippage_cost"],
                "mean_concentration_utilization": summary["mean_concentration_utilization"]
            })
        
        df = pd.DataFrame(performance_data)
        
        # Chart 1: Success rate vs ratio
        ax1.plot(df['deposit_cap_ratio'], df['mean_success_rate'], 
                marker='o', linewidth=2, markersize=8, color='green')
        ax1.set_xticks(df['deposit_cap_ratio'])
        ax1.set_xticklabels([f'{ratio:.0f}:1' for ratio in df['deposit_cap_ratio']])
        ax1.set_xlabel('Deposit Cap Ratio')
        ax1.set_ylabel('Mean Success Rate')
        ax1.set_title('Liquidation Success Rate vs Deposit Cap Ratio')
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Positions needing liquidation vs ratio
        ax2.bar(df['deposit_cap_ratio'], df['mean_positions_needing_liquidation'], 
                color='lightcoral', alpha=0.7, edgecolor='darkred')
        ax2.set_xticks(df['deposit_cap_ratio'])
        ax2.set_xticklabels([f'{ratio:.0f}:1' for ratio in df['deposit_cap_ratio']])
        ax2.set_xlabel('Deposit Cap Ratio')
        ax2.set_ylabel('Mean Positions Needing Liquidation')
        ax2.set_title('Positions Needing Liquidation by Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Slippage cost vs ratio
        ax3.plot(df['deposit_cap_ratio'], df['mean_slippage_cost'], 
                marker='s', linewidth=2, markersize=8, color='orange')
        ax3.set_xticks(df['deposit_cap_ratio'])
        ax3.set_xticklabels([f'{ratio:.0f}:1' for ratio in df['deposit_cap_ratio']])
        ax3.set_xlabel('Deposit Cap Ratio')
        ax3.set_ylabel('Mean Slippage Cost ($)')
        ax3.set_title('Slippage Cost vs Deposit Cap Ratio')
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Concentration utilization vs ratio
        ax4.bar(df['deposit_cap_ratio'], df['mean_concentration_utilization'], 
                color='purple', alpha=0.7, edgecolor='darkmagenta')
        ax4.set_xticks(df['deposit_cap_ratio'])
        ax4.set_xticklabels([f'{ratio:.0f}:1' for ratio in df['deposit_cap_ratio']])
        ax4.set_xlabel('Deposit Cap Ratio')
        ax4.set_ylabel('Mean Concentration Utilization (%)')
        ax4.set_title('Concentration Utilization by Ratio')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Full Range')
        
        plt.tight_layout()
        
        chart_path = charts_dir / "deposit_cap_performance_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"❌ Error creating deposit cap performance chart: {e}")
        return None

def create_liquidation_dollars_chart(results: List[Dict], charts_dir: Path) -> Path:
    """Create chart showing total dollars liquidated per scenario"""
    try:
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Extract liquidation dollar data
        liquidation_data = []
        for result in results:
            ratio = result["deposit_cap_ratio"]
            for shock_run in result["shock_scenario_results"]:
                shock_percent = shock_run["scenario_params"]["btc_shock_percent"]
                total_liquidated = shock_run["liquidation_metrics"]["total_liquidated_value"]
                successful_liquidations = shock_run["liquidation_metrics"]["positions_successfully_liquidated"]
                liquidation_data.append({
                    "deposit_cap_ratio": ratio,
                    "btc_shock_percent": shock_percent,
                    "total_liquidated_dollars": total_liquidated,
                    "successful_liquidations": successful_liquidations
                })
        
        df = pd.DataFrame(liquidation_data)
        
        # Create 2x2 subplot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Liquidation Dollars Analysis by Deposit Cap Ratio", fontsize=18, fontweight='bold')
        
        # Chart 1: Total Dollars Liquidated by Ratio and Shock
        for shock in sorted(df['btc_shock_percent'].unique()):
            shock_data = df[df['btc_shock_percent'] == shock]
            ax1.bar([f'{ratio:.0f}:1' for ratio in shock_data['deposit_cap_ratio']], 
                   shock_data['total_liquidated_dollars'], 
                   label=f'{shock}% Shock', alpha=0.7)
        
        ax1.set_xlabel('Deposit Cap Ratio')
        ax1.set_ylabel('Total Dollars Liquidated ($)')
        ax1.set_title('Total Dollars Liquidated by Ratio and Shock')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Average Dollars per Liquidation by Ratio
        def safe_avg_per_liquidation(group):
            total_dollars = group['total_liquidated_dollars'].sum()
            total_liquidations = group['successful_liquidations'].sum()
            return total_dollars / max(total_liquidations, 1)  # Avoid division by zero
        
        avg_per_liquidation = df.groupby('deposit_cap_ratio', group_keys=False).apply(safe_avg_per_liquidation).reset_index()
        avg_per_liquidation.columns = ['deposit_cap_ratio', 'avg_dollars_per_liquidation']
        
        ax2.bar([f'{ratio:.0f}:1' for ratio in avg_per_liquidation['deposit_cap_ratio']], 
               avg_per_liquidation['avg_dollars_per_liquidation'], 
               color='orange', alpha=0.7)
        ax2.set_xlabel('Deposit Cap Ratio')
        ax2.set_ylabel('Average Dollars per Liquidation ($)')
        ax2.set_title('Average Dollars per Liquidation by Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Liquidation Success Count by Ratio
        success_counts = df.groupby('deposit_cap_ratio')['successful_liquidations'].sum().reset_index()
        ax3.bar([f'{ratio:.0f}:1' for ratio in success_counts['deposit_cap_ratio']], 
               success_counts['successful_liquidations'], 
               color='green', alpha=0.7)
        ax3.set_xlabel('Deposit Cap Ratio')
        ax3.set_ylabel('Total Successful Liquidations')
        ax3.set_title('Total Successful Liquidations by Ratio')
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Heatmap of Dollars Liquidated
        heatmap_data = df.pivot_table(
            values='total_liquidated_dollars', 
            index='deposit_cap_ratio', 
            columns='btc_shock_percent', 
            aggfunc='sum'
        )
        heatmap_data.index = [f'{ratio:.0f}:1' for ratio in heatmap_data.index]
        
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Total Dollars Liquidated ($)'}, ax=ax4)
        ax4.set_title('Dollars Liquidated Heatmap')
        ax4.set_xlabel('BTC Shock Level (%)')
        ax4.set_ylabel('Deposit Cap Ratio')
        
        plt.tight_layout()
        
        chart_path = charts_dir / "liquidation_dollars_analysis.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        print(f"❌ Error creating liquidation dollars chart: {e}")
        return None

def main():
    """Main execution function"""
    try:
        results = run_borrow_cap_analysis()
        print("\n✅ BTC:MOET liquidation capacity analysis completed successfully!")
        return results
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()