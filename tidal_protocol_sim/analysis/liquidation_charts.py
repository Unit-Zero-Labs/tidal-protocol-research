#!/usr/bin/env python3
"""
Liquidation Analysis and Visualization

Creates comprehensive charts analyzing liquidation capacity, profitability,
and debt liquidation percentages under various market scenarios for the
Tidal Protocol with $2.5M liquidity pool setup.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd
from dataclasses import dataclass

from ..core.protocol import Asset, TidalProtocol
from ..core.math import TidalMath
from ..simulation.engine import TidalSimulationEngine
from ..simulation.config import SimulationConfig


@dataclass
class LiquidationScenario:
    """Single liquidation scenario analysis"""
    price_drop: float
    asset: Asset
    liquidation_capacity: float
    debt_liquidatable: float
    liquidation_percentage: float
    avg_profit_per_liquidation: float
    total_slippage: float


class LiquidationAnalyzer:
    """Comprehensive liquidation analysis and visualization"""
    
    def __init__(self, protocol: TidalProtocol = None):
        self.protocol = protocol or TidalProtocol()
        self.initial_prices = {
            Asset.ETH: 4400.0,
            Asset.BTC: 118_000.0,
            Asset.FLOW: 0.40,
            Asset.USDC: 1.0,
            Asset.MOET: 1.0
        }
    
    def analyze_liquidation_scenarios(self) -> List[LiquidationScenario]:
        """Analyze liquidation capacity across various price drop scenarios"""
        
        scenarios = []
        price_drops = np.arange(0.05, 0.60, 0.05)  # 5% to 55% drops
        assets_to_test = [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]
        
        for asset in assets_to_test:
            for price_drop in price_drops:
                scenario = self._analyze_single_scenario(asset, price_drop)
                scenarios.append(scenario)
        
        return scenarios
    
    def _analyze_single_scenario(self, asset: Asset, price_drop: float) -> LiquidationScenario:
        """Analyze single asset price drop scenario"""
        
        # Create price scenario
        current_prices = self.initial_prices.copy()
        current_prices[asset] *= (1 - price_drop)
        
        # Calculate liquidation capacity for this asset's pool
        pool_key = f"MOET_{asset.value}"
        if pool_key in self.protocol.liquidity_pools:
            pool = self.protocol.liquidity_pools[pool_key]
            liquidation_capacity = pool.get_liquidation_capacity()
        else:
            liquidation_capacity = 0.0
        
        # Estimate debt at risk (simplified model)
        # In real scenario, this would come from agent positions
        total_debt_in_asset = self._estimate_debt_at_risk(asset, price_drop)
        
        # Calculate liquidation percentage
        if total_debt_in_asset > 0:
            liquidation_percentage = min(liquidation_capacity / total_debt_in_asset, 1.0)
        else:
            liquidation_percentage = 1.0
        
        # Calculate average profit per liquidation
        avg_liquidation_size = min(liquidation_capacity * 0.1, 10000)  # $10k max per liquidation
        profit = self._calculate_liquidation_profit(asset, avg_liquidation_size, current_prices)
        
        # Estimate slippage impact
        slippage = self._estimate_slippage(asset, liquidation_capacity)
        
        return LiquidationScenario(
            price_drop=price_drop,
            asset=asset,
            liquidation_capacity=liquidation_capacity,
            debt_liquidatable=liquidation_capacity,
            liquidation_percentage=liquidation_percentage,
            avg_profit_per_liquidation=profit,
            total_slippage=slippage
        )
    
    def _estimate_debt_at_risk(self, asset: Asset, price_drop: float) -> float:
        """Estimate total debt at risk for given price drop"""
        # Simplified model: assume debt scales with asset pool size and price drop severity
        asset_pool = self.protocol.asset_pools.get(asset)
        if not asset_pool:
            return 0.0
        
        # Estimate based on typical utilization and collateral ratios
        base_debt = asset_pool.total_supplied * 0.6  # 60% utilization
        risk_multiplier = max(0, (price_drop - 0.2) / 0.3)  # Risk increases after 20% drop
        
        return base_debt * risk_multiplier
    
    def _calculate_liquidation_profit(self, asset: Asset, liquidation_size: float, prices: Dict[Asset, float]) -> float:
        """Calculate profit for a single liquidation"""
        liquidation_penalty = 0.08  # 8% penalty
        gas_cost = 50.0  # $50 gas cost estimate
        
        profit = liquidation_size * liquidation_penalty - gas_cost
        return max(0, profit)
    
    def _estimate_slippage(self, asset: Asset, liquidation_amount: float) -> float:
        """Estimate slippage for liquidation amount"""
        pool_key = f"MOET_{asset.value}"
        if pool_key not in self.protocol.liquidity_pools:
            return 0.0
        
        pool = self.protocol.liquidity_pools[pool_key]
        moet_reserve = pool.reserves.get(Asset.MOET, 0.0)
        
        if moet_reserve <= 0:
            return 0.0
        
        # Simplified slippage calculation
        return liquidation_amount / (moet_reserve + liquidation_amount)
    
    def create_liquidation_capacity_chart(self, scenarios: List[LiquidationScenario]) -> plt.Figure:
        """Create liquidation capacity vs price drop chart"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Liquidation Capacity Analysis - $2.5M Liquidity Pool Setup', fontsize=16, fontweight='bold')
        
        # Prepare data
        df = pd.DataFrame([
            {
                'price_drop': s.price_drop * 100,
                'asset': s.asset.value,
                'liquidation_capacity': s.liquidation_capacity,
                'liquidation_percentage': s.liquidation_percentage * 100,
                'profit': s.avg_profit_per_liquidation,
                'slippage': s.total_slippage * 100
            }
            for s in scenarios
        ])
        
        # Chart 1: Liquidation Capacity by Asset
        for asset in [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]:
            asset_data = df[df['asset'] == asset.value]
            ax1.plot(asset_data['price_drop'], asset_data['liquidation_capacity'], 
                    marker='o', label=asset.value, linewidth=2)
        
        ax1.set_xlabel('Price Drop (%)')
        ax1.set_ylabel('Liquidation Capacity ($)')
        ax1.set_title('Liquidation Capacity vs Price Drop')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Chart 2: Debt Liquidation Percentage
        pivot_data = df.pivot(index='price_drop', columns='asset', values='liquidation_percentage')
        sns.heatmap(pivot_data.T, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax2, cbar_kws={'label': 'Liquidation %'})
        ax2.set_title('Debt Liquidation Coverage (%)')
        ax2.set_xlabel('Price Drop (%)')
        ax2.set_ylabel('Asset')
        
        # Chart 3: Profit per Liquidation
        for asset in [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]:
            asset_data = df[df['asset'] == asset.value]
            ax3.plot(asset_data['price_drop'], asset_data['profit'], 
                    marker='s', label=asset.value, linewidth=2)
        
        ax3.set_xlabel('Price Drop (%)')
        ax3.set_ylabel('Avg Profit per Liquidation ($)')
        ax3.set_title('Liquidation Profitability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Chart 4: Slippage Impact
        for asset in [Asset.ETH, Asset.BTC, Asset.FLOW, Asset.USDC]:
            asset_data = df[df['asset'] == asset.value]
            ax4.plot(asset_data['price_drop'], asset_data['slippage'], 
                    marker='^', label=asset.value, linewidth=2)
        
        ax4.set_xlabel('Price Drop (%)')
        ax4.set_ylabel('Average Slippage (%)')
        ax4.set_title('Liquidation Slippage Impact')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_stress_test_comparison(self, scenarios: List[LiquidationScenario]) -> plt.Figure:
        """Create stress test scenario comparison chart"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Stress Test Liquidation Analysis', fontsize=16, fontweight='bold')
        
        # Define stress scenarios
        stress_scenarios = {
            'ETH Flash Crash': {'asset': Asset.ETH, 'drop': 0.40},
            'BTC Crash': {'asset': Asset.BTC, 'drop': 0.35},
            'FLOW Collapse': {'asset': Asset.FLOW, 'drop': 0.50},
            'Multi-Asset Crisis': {'combined': True, 'drops': {'ETH': 0.30, 'BTC': 0.25, 'FLOW': 0.45}}
        }
        
        # Chart 1: Liquidation Capacity by Scenario
        scenario_names = []
        capacities = []
        profits = []
        coverage = []
        
        for name, config in stress_scenarios.items():
            if config.get('combined'):
                # Multi-asset scenario
                total_capacity = 0
                total_profit = 0
                total_coverage = 0
                count = 0
                
                for asset_name, drop in config['drops'].items():
                    asset = Asset(asset_name)
                    matching_scenarios = [s for s in scenarios if s.asset == asset and abs(s.price_drop - drop) < 0.01]
                    if matching_scenarios:
                        s = matching_scenarios[0]
                        total_capacity += s.liquidation_capacity
                        total_profit += s.avg_profit_per_liquidation
                        total_coverage += s.liquidation_percentage
                        count += 1
                
                if count > 0:
                    scenario_names.append(name)
                    capacities.append(total_capacity)
                    profits.append(total_profit / count)
                    coverage.append(total_coverage / count)
            else:
                # Single asset scenario
                asset = config['asset']
                drop = config['drop']
                matching_scenarios = [s for s in scenarios if s.asset == asset and abs(s.price_drop - drop) < 0.01]
                
                if matching_scenarios:
                    s = matching_scenarios[0]
                    scenario_names.append(name)
                    capacities.append(s.liquidation_capacity)
                    profits.append(s.avg_profit_per_liquidation)
                    coverage.append(s.liquidation_percentage * 100)
        
        # Bar charts
        x_pos = np.arange(len(scenario_names))
        
        bars1 = ax1.bar(x_pos, capacities, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_xlabel('Stress Scenario')
        ax1.set_ylabel('Liquidation Capacity ($)')
        ax1.set_title('Liquidation Capacity by Stress Scenario')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'${height:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        bars2 = ax2.bar(x_pos, profits, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_xlabel('Stress Scenario')
        ax2.set_ylabel('Avg Profit per Liquidation ($)')
        ax2.set_title('Liquidation Profitability by Scenario')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'${height:,.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        bars3 = ax3.bar(x_pos, coverage, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax3.set_xlabel('Stress Scenario')
        ax3.set_ylabel('Debt Coverage (%)')
        ax3.set_title('Debt Liquidation Coverage by Scenario')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax3.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100% Coverage')
        ax3.legend()
        
        for bar in bars3:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        # Chart 4: Liquidity Pool Utilization
        pool_names = ['MOET/USDC', 'MOET/ETH', 'MOET/BTC', 'MOET/FLOW']
        pool_liquidity = [2500000] * 4  # $2.5M each
        max_liquidation = [52000] * 4   # ~$52k capacity each
        utilization = [x/y*100 for x, y in zip(max_liquidation, pool_liquidity)]
        
        bars4 = ax4.bar(pool_names, utilization, color=['#FFD93D', '#6BCF7F', '#4D96FF', '#FF6B9D'])
        ax4.set_xlabel('Liquidity Pool')
        ax4.set_ylabel('Max Single Liquidation (% of Pool)')
        ax4.set_title('Pool Utilization for Max Liquidation')
        ax4.set_xticklabels(pool_names, rotation=45, ha='right')
        
        for bar in bars4:
            height = bar.get_height()
            ax4.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def create_liquidation_efficiency_chart(self) -> plt.Figure:
        """Create liquidation efficiency analysis chart"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Liquidation Efficiency Analysis - Pool Configuration Impact', fontsize=16, fontweight='bold')
        
        # Chart 1: Liquidation Capacity vs Pool Size
        pool_sizes = np.arange(1000000, 10000000, 500000)  # $1M to $10M
        liquidation_capacities = []
        
        for size in pool_sizes:
            # Calculate capacity with 4% max slippage
            moet_reserve = size / 2  # Half the pool is MOET
            capacity = (0.04 * moet_reserve) / (1 - 0.04) * 0.8  # With safety factor
            liquidation_capacities.append(capacity)
        
        ax1.plot(pool_sizes / 1000000, liquidation_capacities, marker='o', linewidth=2, color='#4ECDC4')
        ax1.axvline(x=2.5, color='red', linestyle='--', alpha=0.7, label='Current Setup ($2.5M)')
        ax1.set_xlabel('Pool Size ($ Millions)')
        ax1.set_ylabel('Liquidation Capacity ($)')
        ax1.set_title('Liquidation Capacity vs Pool Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Chart 2: Slippage vs Liquidation Size
        liquidation_sizes = np.arange(5000, 100000, 5000)
        slippages = []
        
        for size in liquidation_sizes:
            # Using current pool setup ($1.25M MOET reserve)
            moet_reserve = 1250000
            slippage = size / (moet_reserve + size) * 100
            slippages.append(slippage)
        
        ax2.plot(liquidation_sizes / 1000, slippages, marker='s', linewidth=2, color='#FF6B6B')
        ax2.axhline(y=4, color='orange', linestyle='--', alpha=0.7, label='4% Max Slippage')
        ax2.axvline(x=52, color='red', linestyle='--', alpha=0.7, label='Current Max Capacity')
        ax2.set_xlabel('Liquidation Size ($K)')
        ax2.set_ylabel('Slippage (%)')
        ax2.set_title('Slippage vs Liquidation Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Profit Margin Analysis
        liquidation_sizes = np.arange(1000, 50000, 1000)
        profit_margins = []
        net_profits = []
        
        for size in liquidation_sizes:
            gross_profit = size * 0.08  # 8% penalty
            gas_cost = 50
            net_profit = gross_profit - gas_cost
            profit_margin = (net_profit / size) * 100 if size > 0 else 0
            
            profit_margins.append(profit_margin)
            net_profits.append(net_profit)
        
        ax3_twin = ax3.twinx()
        line1 = ax3.plot(liquidation_sizes / 1000, profit_margins, marker='o', linewidth=2, color='#45B7D1', label='Profit Margin %')
        line2 = ax3_twin.plot(liquidation_sizes / 1000, net_profits, marker='^', linewidth=2, color='#96CEB4', label='Net Profit $')
        
        ax3.set_xlabel('Liquidation Size ($K)')
        ax3.set_ylabel('Profit Margin (%)', color='#45B7D1')
        ax3_twin.set_ylabel('Net Profit ($)', color='#96CEB4')
        ax3.set_title('Liquidation Profitability Analysis')
        ax3.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='upper left')
        
        # Chart 4: Multi-Asset Liquidation Scenario
        assets = ['ETH', 'BTC', 'FLOW', 'USDC']
        current_capacity = [52000] * 4  # Current capacity per pool
        optimal_capacity = [75000, 70000, 60000, 80000]  # Hypothetical optimal
        
        x_pos = np.arange(len(assets))
        width = 0.35
        
        bars1 = ax4.bar(x_pos - width/2, current_capacity, width, label='Current ($2.5M pools)', color='#FF6B6B', alpha=0.8)
        bars2 = ax4.bar(x_pos + width/2, optimal_capacity, width, label='Optimized pools', color='#4ECDC4', alpha=0.8)
        
        ax4.set_xlabel('Asset Pool')
        ax4.set_ylabel('Liquidation Capacity ($)')
        ax4.set_title('Current vs Optimized Pool Configuration')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(assets)
        ax4.legend()
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.annotate(f'${height:,.0f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def generate_liquidation_report(self, scenarios: List[LiquidationScenario]) -> Dict:
        """Generate comprehensive liquidation analysis report"""
        
        # Calculate summary statistics
        total_capacity = sum(s.liquidation_capacity for s in scenarios if s.price_drop == 0.20)  # At 20% drop
        avg_profit = np.mean([s.avg_profit_per_liquidation for s in scenarios if s.avg_profit_per_liquidation > 0])
        
        # Critical scenarios (>30% price drops)
        critical_scenarios = [s for s in scenarios if s.price_drop >= 0.30]
        
        # Coverage analysis
        full_coverage_scenarios = [s for s in scenarios if s.liquidation_percentage >= 1.0]
        partial_coverage_scenarios = [s for s in scenarios if 0.5 <= s.liquidation_percentage < 1.0]
        insufficient_coverage_scenarios = [s for s in scenarios if s.liquidation_percentage < 0.5]
        
        return {
            'summary': {
                'total_liquidation_capacity': total_capacity,
                'average_profit_per_liquidation': avg_profit,
                'total_pools': 4,
                'pool_liquidity_each': 2500000
            },
            'coverage_analysis': {
                'full_coverage_count': len(full_coverage_scenarios),
                'partial_coverage_count': len(partial_coverage_scenarios),
                'insufficient_coverage_count': len(insufficient_coverage_scenarios),
                'coverage_rate': len(full_coverage_scenarios) / len(scenarios) * 100
            },
            'risk_analysis': {
                'critical_scenarios_count': len(critical_scenarios),
                'avg_capacity_under_stress': np.mean([s.liquidation_capacity for s in critical_scenarios]) if critical_scenarios else 0,
                'worst_case_coverage': min([s.liquidation_percentage for s in scenarios]) * 100
            },
            'recommendations': [
                f"Current setup provides ${total_capacity:,.0f} total liquidation capacity",
                f"Average profit per liquidation: ${avg_profit:,.0f}",
                f"Coverage rate: {len(full_coverage_scenarios) / len(scenarios) * 100:.1f}% of scenarios have full coverage",
                "Consider increasing pool sizes for assets with higher volatility (FLOW, ETH)",
                "Monitor gas costs - they significantly impact small liquidation profitability"
            ]
        }


def main():
    """Main function to generate liquidation analysis charts"""
    
    print("Generating Liquidation Analysis Charts...")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = LiquidationAnalyzer()
    
    # Run analysis
    print("Analyzing liquidation scenarios...")
    scenarios = analyzer.analyze_liquidation_scenarios()
    
    # Generate charts
    print("Creating liquidation capacity chart...")
    fig1 = analyzer.create_liquidation_capacity_chart(scenarios)
    fig1.savefig('liquidation_capacity_analysis.png', dpi=300, bbox_inches='tight')
    
    print("Creating stress test comparison chart...")
    fig2 = analyzer.create_stress_test_comparison(scenarios)
    fig2.savefig('stress_test_liquidation_comparison.png', dpi=300, bbox_inches='tight')
    
    print("Creating liquidation efficiency chart...")
    fig3 = analyzer.create_liquidation_efficiency_chart()
    fig3.savefig('liquidation_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    
    # Generate report
    print("Generating liquidation analysis report...")
    report = analyzer.generate_liquidation_report(scenarios)
    
    print("\n" + "=" * 60)
    print("LIQUIDATION ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"Total Liquidation Capacity: ${report['summary']['total_liquidation_capacity']:,.0f}")
    print(f"Average Profit per Liquidation: ${report['summary']['average_profit_per_liquidation']:,.0f}")
    print(f"Pool Configuration: {report['summary']['total_pools']} pools @ ${report['summary']['pool_liquidity_each']:,.0f} each")
    
    print(f"\nCoverage Analysis:")
    print(f"  Full Coverage Scenarios: {report['coverage_analysis']['full_coverage_count']}")
    print(f"  Partial Coverage Scenarios: {report['coverage_analysis']['partial_coverage_count']}")
    print(f"  Insufficient Coverage Scenarios: {report['coverage_analysis']['insufficient_coverage_count']}")
    print(f"  Overall Coverage Rate: {report['coverage_analysis']['coverage_rate']:.1f}%")
    
    print(f"\nRisk Analysis:")
    print(f"  Critical Scenarios (>30% drops): {report['risk_analysis']['critical_scenarios_count']}")
    print(f"  Avg Capacity Under Stress: ${report['risk_analysis']['avg_capacity_under_stress']:,.0f}")
    print(f"  Worst Case Coverage: {report['risk_analysis']['worst_case_coverage']:.1f}%")
    
    print(f"\nRecommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print(f"\nCharts saved:")
    print(f"  • liquidation_capacity_analysis.png")
    print(f"  • stress_test_liquidation_comparison.png") 
    print(f"  • liquidation_efficiency_analysis.png")
    
    plt.show()


if __name__ == "__main__":
    main()
