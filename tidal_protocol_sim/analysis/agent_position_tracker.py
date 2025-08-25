"""
Agent Position Tracker - Tracks individual agent positions minute-by-minute
for detailed analysis of rebalancing mechanics and health factor changes.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from ..core.protocol import Asset


class AgentPositionTracker:
    """Tracks minute-by-minute position data for a single agent"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.tracking_data = []
        self.is_tracking = False
        
    def start_tracking(self, agent_id: str = None):
        """Start tracking an agent (optionally change which agent to track)"""
        if agent_id:
            self.agent_id = agent_id
        self.is_tracking = True
        self.tracking_data = []
        print(f"📊 Started tracking agent: {self.agent_id}")
        
    def stop_tracking(self):
        """Stop tracking"""
        self.is_tracking = False
        print(f"📊 Stopped tracking agent: {self.agent_id}")
        
    def record_minute_data(self, minute: int, btc_price: float, agent, simulation_engine, swap_data: Dict = None):
        """Record data for the current minute"""
        if not self.is_tracking or agent.agent_id != self.agent_id:
            return
            
        # Calculate all the metrics
        btc_collateral_amount = agent.state.btc_amount
        collateral_usd = btc_collateral_amount * btc_price
        effective_collateral = agent._calculate_effective_collateral_value({Asset.BTC: btc_price})
        current_debt = agent.state.moet_debt
        
        # IMPORTANT: Always force update health factor to ensure accuracy
        # The agent's stored health_factor can be stale, so we recalculate it
        agent._update_health_factor({Asset.BTC: btc_price, Asset.MOET: 1.0})
        health_factor = agent.state.health_factor
        

        
        # Calculate yield token value with accrued interest
        yield_token_value = 0.0
        if hasattr(agent.state, 'yield_token_manager'):
            yield_token_value = agent.state.yield_token_manager.calculate_total_value(minute)
        
        # Get swap data if provided
        yt_swapped = swap_data.get('yt_swapped', 0.0) if swap_data else 0.0
        moet_received = swap_data.get('moet_received', 0.0) if swap_data else 0.0
        
        # Calculate protocol utilization and borrow rate
        total_borrowed = 0.0
        total_liquidity = 0.0
        
        # Sum across all asset pools
        for pool in simulation_engine.protocol.asset_pools.values():
            total_borrowed += pool.total_borrowed
            total_liquidity += pool.total_supplied
            
        utilization = (total_borrowed / total_liquidity) * 100 if total_liquidity > 0 else 0.0
        
        # Simple interest rate calculation (could be enhanced with kink model)
        base_rate = 0.02  # 2% base rate
        borrow_rate = base_rate * (1 + utilization / 100)
        
        # Record the data
        row_data = {
            'Minute': minute,
            'BTC Price': f"${btc_price:,.2f}",
            'Total Borrowed': f"${total_borrowed:,.0f}",
            'Total Liquidity': f"${total_liquidity:,.2f}",
            'Utilization %': f"{utilization:.2f}%",
            'Borrow APR': f"{borrow_rate:.2f}",
            'Agent Debt': f"${current_debt:.2f}",
            'BTC Collateral': f"{btc_collateral_amount:.6f}",
            'Collateral USD': f"${collateral_usd:.2f}",
            'Effective Collateral': f"${effective_collateral:.2f}",
            'Health Factor': f"{health_factor:.6f}",
            'Yield Token Value': f"${yield_token_value:.2f}",
            'YT Swapped': f"${yt_swapped:.2f}" if yt_swapped > 0 else "$-",
            'MOET Received': f"${moet_received:.2f}" if moet_received > 0 else "$-",
            'Agent Risk Profile': getattr(agent, 'risk_profile', 'unknown'),
            'Initial Health Factor': f"{agent.state.initial_health_factor:.6f}",
            'Target Health Factor': f"{agent.state.target_health_factor:.6f}"
        }
        
        self.tracking_data.append(row_data)
        
    def generate_position_table(self, output_dir: Optional[Path] = None) -> pd.DataFrame:
        """Generate a detailed position tracking table"""
        if not self.tracking_data:
            print("⚠️ No tracking data available")
            return pd.DataFrame()
            
        df = pd.DataFrame(self.tracking_data)
        
        if output_dir:
            output_path = output_dir / f"agent_position_tracker_{self.agent_id}.csv"
            df.to_csv(output_path, index=False)
            print(f"💾 Agent position tracker saved: {output_path}")
            
        return df
        
    def get_rebalancing_summary(self) -> Dict[str, Any]:
        """Get summary of rebalancing events"""
        if not self.tracking_data:
            return {}
            
        rebalancing_events = []
        total_yt_swapped = 0.0
        total_moet_received = 0.0
        
        for row in self.tracking_data:
            yt_swapped = row.get('YT Swapped', '$-')
            if yt_swapped != '$-':
                # Parse the dollar amount
                yt_amount = float(yt_swapped.replace('$', '').replace(',', ''))
                moet_amount = float(row.get('MOET Received', '$0').replace('$', '').replace(',', ''))
                
                rebalancing_events.append({
                    'minute': row['Minute'],
                    'btc_price': row['BTC Price'],
                    'yt_swapped': yt_amount,
                    'moet_received': moet_amount,
                    'health_factor_before': row['Health Factor']
                })
                
                total_yt_swapped += yt_amount
                total_moet_received += moet_amount
        
        return {
            'agent_id': self.agent_id,
            'total_rebalancing_events': len(rebalancing_events),
            'total_yt_swapped': total_yt_swapped,
            'total_moet_received': total_moet_received,
            'rebalancing_events': rebalancing_events
        }
        
    def print_tracking_summary(self):
        """Print a summary of the tracking data"""
        if not self.tracking_data:
            print("⚠️ No tracking data available")
            return
            
        summary = self.get_rebalancing_summary()
        
        print(f"\n📊 AGENT POSITION TRACKING SUMMARY")
        print(f"═══════════════════════════════════")
        print(f"Agent ID: {self.agent_id}")
        print(f"Total Minutes Tracked: {len(self.tracking_data)}")
        print(f"Rebalancing Events: {summary['total_rebalancing_events']}")
        print(f"Total YT Swapped: ${summary['total_yt_swapped']:,.2f}")
        print(f"Total MOET Received: ${summary['total_moet_received']:,.2f}")
        
        if summary['rebalancing_events']:
            print(f"\nRebalancing Timeline:")
            for event in summary['rebalancing_events']:
                print(f"  Minute {event['minute']}: Swapped ${event['yt_swapped']:,.2f} YT → ${event['moet_received']:,.2f} MOET")
                
        # Show first and last health factors
        first_hf = self.tracking_data[0]['Health Factor']
        last_hf = self.tracking_data[-1]['Health Factor']
        print(f"\nHealth Factor Journey: {first_hf} → {last_hf}")
