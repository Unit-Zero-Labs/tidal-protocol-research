"""
Tidal Protocol Streamlined Simulation

A focused simulation system for stress testing the Tidal lending mechanism,
liquidity parameters, liquidation methodology, and protocol stability.
"""

__version__ = "1.0.0"
__author__ = "Tidal Protocol Team"

# Core components
from .core.protocol import TidalProtocol, Asset, AssetPool
from .core.moet import MoetStablecoin
from .core.yield_tokens import YieldToken, YieldTokenManager, YieldTokenPool
# Removed TidalMath - functionality integrated into protocol.py

# Agents
from .agents.base_agent import BaseAgent, AgentAction, AgentState
from .agents.tidal_lender import TidalLender
from .agents.trader import BasicTrader
from .agents.liquidator import Liquidator
from .agents.high_tide_agent import HighTideAgent, create_high_tide_agents

# Engine
from .engine.tidal_engine import TidalProtocolEngine, TidalConfig
from .engine.high_tide_vault_engine import HighTideVaultEngine, HighTideConfig
from .engine.aave_protocol_engine import AaveProtocolEngine, AaveConfig
from .engine.config import SimulationConfig, StressTestScenarios
from .engine.state import SimulationState

# Stress Testing
from .stress_testing.runner import StressTestRunner, QuickStressTest
from .stress_testing.scenarios import TidalStressTestSuite

# Analysis
from .analysis.metrics import TidalMetricsCalculator

__all__ = [
    # Core
    "TidalProtocol", "Asset", "AssetPool",
    "MoetStablecoin", "YieldToken", "YieldTokenManager", "YieldTokenPool",
    
    # Agents
    "BaseAgent", "AgentAction", "AgentState",
    "TidalLender", "BasicTrader", "Liquidator",
    "HighTideAgent", "create_high_tide_agents",
    
    # Simulation
    "TidalProtocolEngine", "TidalConfig", "HighTideVaultEngine", "HighTideConfig", "AaveProtocolEngine", "AaveConfig",
    "SimulationConfig", "StressTestScenarios", "SimulationState",
    
    # Stress Testing
    "StressTestRunner", "QuickStressTest", "TidalStressTestSuite",
    
    # Analysis
    "TidalMetricsCalculator"
]