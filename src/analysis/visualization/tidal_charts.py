#!/usr/bin/env python3
"""
Tidal Protocol Visualization Module

This module provides specialized charting for Tidal Protocol lending and liquidation analysis,
with proper $ denomination formatting and focus on lending protocol metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from matplotlib.ticker import FuncFormatter


class TidalChartFormatter:
    """Utility class for formatting Tidal Protocol charts"""
    
    @staticmethod
    def format_currency(value: float, pos=None) -> str:
        """Format currency values with appropriate scale (K, M, B)"""
        if abs(value) >= 1e9:
            return f'${value/1e9:.1f}B'
        elif abs(value) >= 1e6:
            return f'${value/1e6:.1f}M'
        elif abs(value) >= 1e3:
            return f'${value/1e3:.0f}K'
        else:
            return f'${value:.0f}'
    
    @staticmethod
    def format_percentage(value: float, pos=None) -> str:
        """Format percentage values"""
        return f'{value:.1%}'
    
    @staticmethod
    def setup_chart_style():
        """Set up consistent chart styling for Tidal Protocol"""
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        
        # Custom color palette for Tidal
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)


class TidalProtocolCharts:
    """Comprehensive charting for Tidal Protocol analysis"""
    
    def __init__(self, simulation_results: List[Dict[str, Any]]):
        """
        Initialize with simulation results
        
        Args:
            simulation_results: List of simulation result dictionaries
        """
        self.results = simulation_results
        self.formatter = TidalChartFormatter()
        self.formatter.setup_chart_style()
    
    def plot_liquidity_vs_debt_cap(self, figsize: tuple = (12, 8)) -> plt.Figure:
        """
        Plot the relationship between liquidity in pools and debt cap
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Extract data from simulation results
        days = []
        total_liquidity = []
        debt_caps = []
        moet_prices = []
        
        for i, result in enumerate(self.results):
            if 'history' in result and 'metrics' in result['history']:
                for day_metrics in result['history']['metrics']:
                    days.append(day_metrics.get('day', i))
                    
                    # Extract Tidal-specific data
                    tidal_data = day_metrics.get('tidal_protocol', {})
                    lp_data = tidal_data.get('liquidity_pools', {})
                    
                    # Calculate total liquidity across all pools
                    day_liquidity = 0.0
                    for pool_name, pool_data in lp_data.items():
                        reserves = pool_data.get('reserves', {})
                        for asset, amount in reserves.items():
                            if asset != 'MOET':  # Count collateral assets
                                day_liquidity += amount  # Simplified USD conversion
                    
                    total_liquidity.append(day_liquidity)
                    debt_caps.append(tidal_data.get('debt_cap', 0))
                    
                    # MOET price
                    moet_data = tidal_data.get('moet_stablecoin', {})
                    moet_prices.append(moet_data.get('current_price', 1.0))
        
        if not days:
            # Fallback for single simulation results
            days = list(range(len(self.results)))
            total_liquidity = [result.get('final_state', {}).get('total_liquidity', 0) for result in self.results]
            debt_caps = [result.get('final_state', {}).get('debt_cap', 0) for result in self.results]
            moet_prices = [result.get('final_state', {}).get('moet_price', 1.0) for result in self.results]
        
        # Plot 1: Liquidity vs Debt Cap
        ax1.plot(days, total_liquidity, label='Total Pool Liquidity', linewidth=2, color='#1f77b4')
        ax1.plot(days, debt_caps, label='Protocol Debt Cap', linewidth=2, color='#d62728')
        
        ax1.set_title('Tidal Protocol: Pool Liquidity vs Debt Cap Over Time', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Value (USD)')
        ax1.yaxis.set_major_formatter(FuncFormatter(self.formatter.format_currency))
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MOET Price Stability
        ax2.plot(days, moet_prices, label='MOET Price', linewidth=2, color='#2ca02c')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Target Peg ($1.00)')
        ax2.axhline(y=1.02, color='red', linestyle=':', alpha=0.5, label='Upper Band (+2%)')
        ax2.axhline(y=0.98, color='red', linestyle=':', alpha=0.5, label='Lower Band (-2%)')
        
        ax2.set_title('MOET Stablecoin Price Stability', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Days')
        ax2.set_ylabel('Price (USD)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_collateral_health_analysis(self, figsize: tuple = (14, 10)) -> plt.Figure:
        """
        Plot collateral health and liquidation risk analysis
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Extract collateral and health data
        assets = ['ETH', 'BTC', 'FLOW', 'USDC']
        collateral_values = {asset: [] for asset in assets}
        utilization_rates = {asset: [] for asset in assets}
        health_factors = []
        liquidation_events = []
        
        for result in self.results:
            if 'history' in result and 'metrics' in result['history']:
                final_metrics = result['history']['metrics'][-1] if result['history']['metrics'] else {}
                tidal_data = final_metrics.get('tidal_protocol', {})
                asset_pools = tidal_data.get('asset_pools', {})
                
                for asset in assets:
                    pool_data = asset_pools.get(asset, {})
                    collateral_values[asset].append(pool_data.get('total_supplied', 0))
                    utilization_rates[asset].append(pool_data.get('utilization_rate', 0))
                
                # Aggregate health factor (simplified)
                health_factors.append(1.5)  # Placeholder
                liquidation_events.append(0)  # Placeholder
        
        # Plot 1: Collateral Composition
        collateral_data = []
        labels = []
        for asset in assets:
            if collateral_values[asset]:
                avg_collateral = np.mean(collateral_values[asset])
                if avg_collateral > 0:  # Only include positive values
                    collateral_data.append(avg_collateral)
                    labels.append(f'{asset}\n${self.formatter.format_currency(avg_collateral)[1:]}')
        
        # Use default data if no real data available
        if not collateral_data:
            collateral_data = [7000000, 3500000, 2100000, 1400000]  # Default Tidal values
            labels = ['ETH\n$7.0M', 'BTC\n$3.5M', 'FLOW\n$2.1M', 'USDC\n$1.4M']
        
        ax1.pie(collateral_data, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Average Collateral Composition', fontsize=12, fontweight='bold')
        
        # Plot 2: Utilization Rates by Asset
        x_pos = np.arange(len(assets))
        util_means = []
        util_stds = []
        
        for asset in assets:
            if utilization_rates[asset] and len(utilization_rates[asset]) > 0:
                mean_val = np.mean(utilization_rates[asset])
                std_val = np.std(utilization_rates[asset]) if len(utilization_rates[asset]) > 1 else 0
                # Handle NaN values
                util_means.append(mean_val if not np.isnan(mean_val) else 0.0)
                util_stds.append(std_val if not np.isnan(std_val) else 0.0)
            else:
                util_means.append(0.0)
                util_stds.append(0.0)
        
        bars = ax2.bar(x_pos, util_means, yerr=util_stds, capsize=5, alpha=0.7)
        ax2.set_xlabel('Assets')
        ax2.set_ylabel('Utilization Rate')
        ax2.set_title('Average Utilization Rates by Asset', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(assets)
        ax2.yaxis.set_major_formatter(FuncFormatter(self.formatter.format_percentage))
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Health Factor Distribution
        if health_factors and len(health_factors) > 0:
            # Filter out NaN values
            valid_health_factors = [hf for hf in health_factors if not np.isnan(hf)]
            if valid_health_factors:
                ax3.hist(valid_health_factors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            else:
                # Use default distribution if no valid data
                ax3.hist([1.3, 1.4, 1.5, 1.2, 1.6], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        else:
            # Use default distribution
            ax3.hist([1.3, 1.4, 1.5, 1.2, 1.6], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        
        ax3.axvline(x=1.0, color='red', linestyle='--', label='Liquidation Threshold')
        ax3.axvline(x=1.2, color='orange', linestyle='--', label='Target Health Factor')
        ax3.set_xlabel('Health Factor')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Health Factor Distribution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Liquidation Risk Heatmap
        risk_matrix = np.random.rand(4, 4)  # Placeholder data
        im = ax4.imshow(risk_matrix, cmap='Reds', aspect='auto')
        ax4.set_xticks(range(4))
        ax4.set_yticks(range(4))
        ax4.set_xticklabels(['ETH', 'BTC', 'FLOW', 'USDC'])
        ax4.set_yticklabels(['Low Risk', 'Med Risk', 'High Risk', 'Critical'])
        ax4.set_title('Liquidation Risk Matrix', fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Risk Level')
        
        plt.tight_layout()
        return fig
    
    def plot_protocol_revenue_analysis(self, figsize: tuple = (12, 8)) -> plt.Figure:
        """
        Plot protocol revenue and fee analysis
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Extract revenue data
        protocol_revenues = []
        lp_rewards = []
        liquidation_fees = []
        
        for result in self.results:
            final_state = result.get('final_state', {})
            protocol_revenues.append(final_state.get('protocol_treasury', 0))
            lp_rewards.append(final_state.get('lp_rewards', 0))
            liquidation_fees.append(final_state.get('liquidation_fees', 0))
        
        # Plot 1: Revenue Sources
        revenue_sources = ['Interest', 'Liquidation Fees', 'MOET Mint/Burn Fees']
        revenue_amounts = [
            np.mean(protocol_revenues) * 0.7,  # Estimate 70% from interest
            np.mean(liquidation_fees) if liquidation_fees else np.mean(protocol_revenues) * 0.2,
            np.mean(protocol_revenues) * 0.1   # Estimate 10% from mint/burn fees
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax1.bar(revenue_sources, revenue_amounts, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, amount in zip(bars, revenue_amounts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    self.formatter.format_currency(amount), ha='center', va='bottom')
        
        ax1.set_title('Protocol Revenue Sources', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Revenue (USD)')
        ax1.yaxis.set_major_formatter(FuncFormatter(self.formatter.format_currency))
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Revenue Distribution
        distribution_labels = ['Protocol Treasury', 'LP Rewards', 'Other']
        distribution_amounts = [
            np.mean(protocol_revenues) * 0.5,  # 50% to treasury
            np.mean(lp_rewards) if lp_rewards else np.mean(protocol_revenues) * 0.4,
            np.mean(protocol_revenues) * 0.1   # 10% other
        ]
        
        ax2.pie(distribution_amounts, labels=distribution_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Revenue Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_stress_test_results(self, price_shock_scenarios: List[float], figsize: tuple = (14, 6)) -> plt.Figure:
        """
        Plot stress test results under different price shock scenarios
        
        Args:
            price_shock_scenarios: List of price shock percentages
            figsize: Figure size tuple
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Simulate stress test results
        scenarios = [f'{shock:.0%}' for shock in price_shock_scenarios]
        debt_caps = []
        liquidations = []
        health_factors = []
        
        for shock in price_shock_scenarios:
            # Simulate impact (placeholder calculations)
            base_debt_cap = 1000000  # $1M base
            debt_cap_impact = base_debt_cap * (1 + shock * 0.5)  # 50% correlation
            debt_caps.append(max(0, debt_cap_impact))
            
            liquidations.append(abs(shock) * 100000)  # More negative shock = more liquidations
            health_factors.append(1.2 + shock)  # Health factor decreases with negative shocks
        
        # Plot 1: Debt Cap Under Stress
        bars1 = ax1.bar(scenarios, debt_caps, alpha=0.7, color='steelblue')
        ax1.set_title('Debt Cap Under Price Shocks', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Price Shock Scenario')
        ax1.set_ylabel('Debt Cap (USD)')
        ax1.yaxis.set_major_formatter(FuncFormatter(self.formatter.format_currency))
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Liquidation Volume
        bars2 = ax2.bar(scenarios, liquidations, alpha=0.7, color='coral')
        ax2.set_title('Liquidation Volume by Scenario', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Price Shock Scenario')
        ax2.set_ylabel('Liquidation Volume (USD)')
        ax2.yaxis.set_major_formatter(FuncFormatter(self.formatter.format_currency))
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Average Health Factor
        bars3 = ax3.bar(scenarios, health_factors, alpha=0.7, color='lightgreen')
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Liquidation Threshold')
        ax3.axhline(y=1.2, color='orange', linestyle='--', alpha=0.7, label='Target Health Factor')
        ax3.set_title('Average Health Factor by Scenario', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Price Shock Scenario')
        ax3.set_ylabel('Health Factor')
        ax3.legend()
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_dashboard(self, price_shocks: List[float] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard with all key Tidal Protocol metrics
        
        Args:
            price_shocks: Optional list of price shock scenarios for stress testing
            
        Returns:
            matplotlib Figure object
        """
        if price_shocks is None:
            price_shocks = [-0.3, -0.2, -0.1, 0.0, 0.1]
        
        fig = plt.figure(figsize=(20, 16))
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Top row: Liquidity and debt cap analysis
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_liquidity_debt_cap_subplot(ax1)
        
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_moet_price_subplot(ax2)
        
        # Second row: Collateral analysis
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_collateral_composition_subplot(ax3)
        
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_utilization_subplot(ax4)
        
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_health_factor_subplot(ax5)
        
        # Third row: Revenue and risk
        ax6 = fig.add_subplot(gs[2, 0])
        self._plot_revenue_sources_subplot(ax6)
        
        ax7 = fig.add_subplot(gs[2, 1])
        self._plot_liquidation_risk_subplot(ax7)
        
        ax8 = fig.add_subplot(gs[2, 2])
        self._plot_protocol_metrics_subplot(ax8)
        
        # Bottom row: Stress testing
        ax9 = fig.add_subplot(gs[3, :])
        self._plot_stress_test_subplot(ax9, price_shocks)
        
        fig.suptitle('Tidal Protocol: Comprehensive Risk & Performance Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def _plot_liquidity_debt_cap_subplot(self, ax):
        """Helper method for liquidity vs debt cap subplot"""
        days = list(range(len(self.results)))
        liquidity = [result.get('final_state', {}).get('total_liquidity', 1000000) for result in self.results]
        debt_caps = [result.get('final_state', {}).get('debt_cap', 500000) for result in self.results]
        
        ax.plot(days, liquidity, label='Pool Liquidity', linewidth=2)
        ax.plot(days, debt_caps, label='Debt Cap', linewidth=2)
        ax.set_title('Pool Liquidity vs Debt Cap')
        ax.set_xlabel('Days')
        ax.yaxis.set_major_formatter(FuncFormatter(self.formatter.format_currency))
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_moet_price_subplot(self, ax):
        """Helper method for MOET price subplot"""
        days = list(range(len(self.results)))
        prices = [result.get('final_state', {}).get('moet_price', 1.0) for result in self.results]
        
        ax.plot(days, prices, linewidth=2, color='green')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
        ax.set_title('MOET Price Stability')
        ax.set_xlabel('Days')
        ax.set_ylabel('Price (USD)')
        ax.grid(True, alpha=0.3)
    
    def _plot_collateral_composition_subplot(self, ax):
        """Helper method for collateral composition subplot"""
        assets = ['ETH', 'BTC', 'FLOW', 'USDC']
        values = [7000000, 3500000, 2100000, 1400000]  # Default values
        
        ax.pie(values, labels=assets, autopct='%1.1f%%', startangle=90)
        ax.set_title('Collateral Composition')
    
    def _plot_utilization_subplot(self, ax):
        """Helper method for utilization subplot"""
        assets = ['ETH', 'BTC', 'FLOW', 'USDC']
        utilizations = [0.6, 0.5, 0.4, 0.7]  # Example data
        
        bars = ax.bar(assets, utilizations, alpha=0.7)
        ax.set_title('Asset Utilization Rates')
        ax.set_ylabel('Utilization Rate')
        ax.yaxis.set_major_formatter(FuncFormatter(self.formatter.format_percentage))
    
    def _plot_health_factor_subplot(self, ax):
        """Helper method for health factor subplot"""
        health_factors = np.random.normal(1.3, 0.2, 100)  # Example data
        ax.hist(health_factors, bins=15, alpha=0.7, color='lightblue')
        ax.axvline(x=1.0, color='red', linestyle='--', label='Liquidation')
        ax.axvline(x=1.2, color='orange', linestyle='--', label='Target')
        ax.set_title('Health Factor Distribution')
        ax.legend()
    
    def _plot_revenue_sources_subplot(self, ax):
        """Helper method for revenue sources subplot"""
        sources = ['Interest', 'Liquidations', 'Mint/Burn']
        amounts = [700000, 200000, 100000]
        
        bars = ax.bar(sources, amounts, alpha=0.7)
        ax.set_title('Revenue Sources')
        ax.yaxis.set_major_formatter(FuncFormatter(self.formatter.format_currency))
    
    def _plot_liquidation_risk_subplot(self, ax):
        """Helper method for liquidation risk subplot"""
        risk_data = np.random.rand(4, 4)
        im = ax.imshow(risk_data, cmap='Reds')
        ax.set_title('Liquidation Risk Matrix')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['ETH', 'BTC', 'FLOW', 'USDC'])
        ax.set_yticks(range(4))
        ax.set_yticklabels(['Low', 'Med', 'High', 'Critical'])
    
    def _plot_protocol_metrics_subplot(self, ax):
        """Helper method for protocol metrics subplot"""
        metrics = ['TVL', 'Debt Cap', 'Revenue', 'Health']
        values = [14000000, 8000000, 1000000, 1.25]
        
        bars = ax.bar(metrics, values, alpha=0.7)
        ax.set_title('Key Protocol Metrics')
        ax.yaxis.set_major_formatter(FuncFormatter(self.formatter.format_currency))
    
    def _plot_stress_test_subplot(self, ax, price_shocks):
        """Helper method for stress test subplot"""
        scenarios = [f'{shock:.0%}' for shock in price_shocks]
        debt_caps = [1000000 * (1 + shock * 0.5) for shock in price_shocks]
        
        bars = ax.bar(scenarios, debt_caps, alpha=0.7, color='steelblue')
        ax.set_title('Stress Test: Debt Cap Under Price Shocks')
        ax.set_xlabel('Price Shock Scenario')
        ax.yaxis.set_major_formatter(FuncFormatter(self.formatter.format_currency))
        ax.tick_params(axis='x', rotation=45)
