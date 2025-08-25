"""
Tidal Protocol Streamlined Simulation

A focused simulation system for stress testing the Tidal lending mechanism,
liquidity parameters, liquidation methodology, and protocol stability.
"""

__version__ = "1.0.0"
__author__ = "Tidal Protocol Team"

# Core components
from .core.protocol import TidalProtocol, Asset, AssetPool, LiquidityPool
from .core.moet import MoetStablecoin
# Removed TidalMath - functionality integrated into protocol.py

# Agents
from .agents.base_agent import BaseAgent, AgentAction, AgentState
from .agents.tidal_lender import TidalLender
from .agents.trader import BasicTrader
from .agents.liquidator import Liquidator

# Simulation
from .simulation.engine import TidalSimulationEngine
from .simulation.config import SimulationConfig, StressTestScenarios
from .simulation.state import SimulationState

# Stress Testing
from .stress_testing.runner import StressTestRunner, QuickStressTest
from .stress_testing.scenarios import TidalStressTestSuite

# Analysis
from .analysis.metrics import TidalMetricsCalculator

__all__ = [
    # Core
    "TidalProtocol", "Asset", "AssetPool", "LiquidityPool",
    "MoetStablecoin",
    
    # Agents
    "BaseAgent", "AgentAction", "AgentState",
    "TidalLender", "BasicTrader", "Liquidator",
    
    # Simulation
    "TidalSimulationEngine", "SimulationConfig", "StressTestScenarios", "SimulationState",
    
    # Stress Testing
    "StressTestRunner", "QuickStressTest", "TidalStressTestSuite",
    
    # Analysis
    "TidalMetricsCalculator"
]