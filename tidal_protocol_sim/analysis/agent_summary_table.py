#!/usr/bin/env python3
"""
Agent Summary Table Generator

Creates comprehensive agent-by-agent summary tables for High Tide scenario analysis
showing all key metrics including collateral, debt, yield tokens, and performance.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional


class AgentSummaryTableGenerator:
    """Generates detailed agent summary tables for High Tide scenario"""
    
    def __init__(self):
        pass
    
    def generate_agent_summary_table(
        self, 
        results: Dict[str, Any], 
        btc_initial_price: float = 100_000.0,
        output_dir: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive agent summary table
        
        Columns:
        - Agent ID
        - Risk Profile  
        - Collateral Deposited (BTC)
        - Effective Collateral ($)
        - Initial Borrowed MOET
        - Final Debt (with interest)
        - Total Interest Accrued
        - Initial Health Factor
        - Target Health Factor (rebalancing trigger)
        - Final Health Factor
        - Initial Yield Tokens Acquired
        - Yield Tokens Sold ($)
        - Final Collateral Value ($)
        - Final Yield Token Value ($)
        - Net Position Value ($)
        - Cost of Rebalancing ($)
        - Survival Status
        - Rebalancing Events
        """
        
        agent_outcomes = results.get("agent_outcomes", [])
        agent_health_history = results.get("agent_health_history", [])
        btc_price_history = results.get("btc_price_history", [])
        
        if not agent_outcomes or not agent_health_history:
            return pd.DataFrame()
        
        final_minute = len(btc_price_history) - 1 if btc_price_history else 0
        final_btc_price = btc_price_history[-1] if btc_price_history else btc_initial_price
        
        # Extract agent details from the final state
        final_agent_data = agent_health_history[-1]["agents"] if agent_health_history else []
        
        table_data = []
        
        for outcome in agent_outcomes:
            agent_id = outcome["agent_id"]
            
            # Find corresponding agent data in health history
            agent_data = next((a for a in final_agent_data if a["agent_id"] == agent_id), None)
            
            if not agent_data:
                continue
            
            # Calculate metrics
            collateral_btc = 1.0  # Each agent deposits exactly 1 BTC
            collateral_value_initial = collateral_btc * btc_initial_price  # $100,000
            effective_collateral_usd = collateral_value_initial * 0.8  # $80,000 (borrowing capacity)
            
            # Current collateral value (for final calculations)
            collateral_value_current = collateral_btc * final_btc_price
            
            # Get initial data from first health history entry
            initial_agent_data = None
            if agent_health_history:
                initial_agents = agent_health_history[0]["agents"]
                initial_agent_data = next((a for a in initial_agents if a["agent_id"] == agent_id), None)
            
            # Get actual debt amounts from agent data
            initial_hf = outcome["target_health_factor"]  # This is the initial HF the agent starts with
            initial_debt = outcome.get("initial_debt", 0)  # Get actual initial debt
            final_debt = outcome.get("final_debt", 0)
            
            # Calculate actual initial health factor to verify math
            actual_initial_hf = effective_collateral_usd / initial_debt if initial_debt > 0 else float('inf')
            
            # Get the target (rebalancing trigger) health factor
            # This should come from target_health_factor in the outcome data
            target_rebalancing_hf = outcome.get("target_health_factor", initial_hf)
            
            initial_yield_tokens = initial_debt  # 1:1 conversion MOET to yield tokens
            
            # Extract metrics from agent data and outcomes
            row = {
                "Agent ID": agent_id,
                "Risk Profile": outcome["risk_profile"].title(),
                "Collateral Deposited (BTC)": f"{collateral_btc:.1f}",
                "Effective Collateral ($)": f"${effective_collateral_usd:,.0f}",
                "Initial Borrowed MOET ($)": f"${initial_debt:,.0f}",
                "Final Debt w/ Interest ($)": f"${final_debt:,.0f}",
                "Total Interest Accrued ($)": f"${outcome.get('interest_accrued', 0):,.0f}",
                "Initial Health Factor": f"{actual_initial_hf:.2f}",
                "Target Health Factor": f"{target_rebalancing_hf:.2f}",
                "Final Health Factor": f"{outcome['final_health_factor']:.2f}",
                "Initial Yield Tokens ($)": f"${initial_yield_tokens:,.0f}",
                "Yield Tokens Sold ($)": f"${outcome['total_yield_sold']:,.0f}",
                "Final Collateral Value ($)": f"${collateral_value_current:,.0f}",
                "Final Yield Token Value ($)": f"${outcome['yield_token_value']:,.0f}",
                "Net Position Value ($)": f"${outcome['net_position_value']:,.0f}",
                "Cost of Rebalancing ($)": f"${outcome['cost_of_rebalancing']:,.0f}",
                "Survival Status": "✅ Survived" if outcome["survived"] else "❌ Liquidated",
                "Rebalancing Events": outcome["rebalancing_events"],
                "Emergency Liquidations": outcome["emergency_liquidations"]
            }
            
            table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Sort by risk profile and then by final health factor
        risk_order = {"Conservative": 1, "Moderate": 2, "Aggressive": 3}
        df["_sort_risk"] = df["Risk Profile"].map(risk_order)
        df = df.sort_values(["_sort_risk", "Final Health Factor"], ascending=[True, False])
        df = df.drop("_sort_risk", axis=1)
        
        # Save to file if output directory provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as CSV
            csv_path = output_dir / "agent_summary_table.csv"
            df.to_csv(csv_path, index=False)
            print(f"Agent summary table saved to: {csv_path}")
            
            # Save as formatted Excel with styling
            excel_path = output_dir / "agent_summary_table.xlsx"
            try:
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Agent Summary', index=False)
                    
                    # Get the workbook and worksheet for formatting
                    workbook = writer.book
                    worksheet = writer.sheets['Agent Summary']
                    
                    # Auto-adjust column widths
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width
                
                print(f"Formatted Excel table saved to: {excel_path}")
                
            except ImportError:
                print("openpyxl not available, skipping Excel export")
        
        return df
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics from the agent summary table"""
        
        if df.empty:
            return {}
        
        # Extract numeric values (remove $ and commas)
        def extract_numeric(series):
            return series.str.replace('$', '').str.replace(',', '').astype(float)
        
        # Calculate statistics by risk profile
        risk_profiles = df["Risk Profile"].unique()
        
        summary_stats = {
            "overall": {
                "total_agents": len(df),
                "survivors": len(df[df["Survival Status"].str.contains("Survived")]),
                "survival_rate": len(df[df["Survival Status"].str.contains("Survived")]) / len(df),
                "avg_cost_of_rebalancing": extract_numeric(df["Cost of Rebalancing ($)"]).mean(),
                "avg_final_health_factor": df["Final Health Factor"].astype(float).mean(),
                "total_yield_sold": extract_numeric(df["Yield Tokens Sold ($)"]).sum(),
                "total_interest_accrued": extract_numeric(df["Total Interest Accrued ($)"]).sum(),
                "avg_rebalancing_events": df["Rebalancing Events"].mean()
            },
            "by_risk_profile": {}
        }
        
        for profile in risk_profiles:
            profile_df = df[df["Risk Profile"] == profile]
            
            summary_stats["by_risk_profile"][profile] = {
                "count": len(profile_df),
                "survivors": len(profile_df[profile_df["Survival Status"].str.contains("Survived")]),
                "survival_rate": len(profile_df[profile_df["Survival Status"].str.contains("Survived")]) / len(profile_df),
                "avg_cost_of_rebalancing": extract_numeric(profile_df["Cost of Rebalancing ($)"]).mean(),
                "avg_final_health_factor": profile_df["Final Health Factor"].astype(float).mean(),
                "avg_yield_sold": extract_numeric(profile_df["Yield Tokens Sold ($)"]).mean(),
                "avg_interest_accrued": extract_numeric(profile_df["Total Interest Accrued ($)"]).mean(),
                "avg_rebalancing_events": profile_df["Rebalancing Events"].mean()
            }
        
        return summary_stats
    
    def print_agent_summary_table(self, df: pd.DataFrame, max_width: int = 150):
        """Print a formatted agent summary table to console"""
        
        if df.empty:
            print("No agent data available for summary table")
            return
        
        print("\n" + "=" * max_width)
        print("HIGH TIDE AGENT SUMMARY TABLE")
        print("=" * max_width)
        
        # Configure pandas display options for better formatting
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', max_width)
        pd.set_option('display.max_colwidth', 20)
        
        print(df.to_string(index=False))
        
        print("\n" + "=" * max_width)
        
        # Print summary statistics
        stats = self.generate_summary_statistics(df)
        
        if stats:
            print("SUMMARY STATISTICS")
            print("-" * 50)
            
            overall = stats["overall"]
            print(f"Total Agents: {overall['total_agents']}")
            print(f"Survivors: {overall['survivors']} ({overall['survival_rate']:.1%})")
            print(f"Average Cost of Rebalancing: ${overall['avg_cost_of_rebalancing']:,.0f}")
            print(f"Average Final Health Factor: {overall['avg_final_health_factor']:.2f}")
            print(f"Total Yield Sold: ${overall['total_yield_sold']:,.0f}")
            print(f"Total Interest Accrued: ${overall['total_interest_accrued']:,.0f}")
            print(f"Average Rebalancing Events: {overall['avg_rebalancing_events']:.1f}")
            
            print("\nBY RISK PROFILE:")
            print("-" * 30)
            
            for profile, data in stats["by_risk_profile"].items():
                print(f"\n{profile}:")
                print(f"  Count: {data['count']}")
                print(f"  Survival Rate: {data['survival_rate']:.1%}")
                print(f"  Avg Cost: ${data['avg_cost_of_rebalancing']:,.0f}")
                print(f"  Avg Final HF: {data['avg_final_health_factor']:.2f}")
                print(f"  Avg Yield Sold: ${data['avg_yield_sold']:,.0f}")
                print(f"  Avg Interest: ${data['avg_interest_accrued']:,.0f}")
                print(f"  Avg Rebalancing: {data['avg_rebalancing_events']:.1f}")
        
        print("\n" + "=" * max_width)
    
    def generate_interest_analysis_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Generate detailed interest analysis showing borrow rates over time"""
        
        agent_health_history = results.get("agent_health_history", [])
        btc_price_history = results.get("btc_price_history", [])
        
        if not agent_health_history:
            return pd.DataFrame()
        
        interest_data = []
        
        for i, entry in enumerate(agent_health_history):
            minute = entry["minute"]
            btc_price = btc_price_history[i] if i < len(btc_price_history) else 100_000
            
            # Calculate average utilization and interest for this minute
            total_debt = sum(agent["current_moet_debt"] for agent in entry["agents"])
            total_interest = sum(agent.get("total_interest_accrued", 0) for agent in entry["agents"])
            
            # Estimate BTC pool utilization (simplified)
            # In a real implementation, this would come from protocol state
            btc_supplied = len(entry["agents"]) * 1.0  # 1 BTC per agent
            utilization_rate = min(0.95, total_debt / (btc_supplied * btc_price * 0.8))
            
            # Calculate borrow rate using kink model
            kink = 0.80
            base_rate = 0.02  # 2% base
            slope1 = 0.10  # 10% slope before kink
            slope2 = 1.00  # 100% jump rate after kink
            
            if utilization_rate <= kink:
                borrow_rate = base_rate + (utilization_rate * slope1)
            else:
                borrow_rate = base_rate + (kink * slope1) + ((utilization_rate - kink) * slope2)
            
            interest_data.append({
                "Minute": minute,
                "BTC Price ($)": f"${btc_price:,.0f}",
                "Total Debt ($)": f"${total_debt:,.0f}",
                "Pool Utilization (%)": f"{utilization_rate:.1%}",
                "Borrow Rate (%)": f"{borrow_rate:.2%}",
                "Cumulative Interest ($)": f"${total_interest:,.0f}",
                "Active Agents": len(entry["agents"])
            })
        
        return pd.DataFrame(interest_data)


# Convenience function for easy access
def generate_agent_summary_table(results: Dict[str, Any], output_dir: Optional[Path] = None) -> pd.DataFrame:
    """Convenience function to generate agent summary table"""
    generator = AgentSummaryTableGenerator()
    return generator.generate_agent_summary_table(results, output_dir=output_dir)
