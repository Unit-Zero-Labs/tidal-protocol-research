#!/usr/bin/env python3
"""
Base agent and policy interfaces for the Tidal Protocol simulation.

This module implements the Agent-Action pattern where agents hold only state
and delegate decision-making to pure policy functions.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..simulation.primitives import Action, MarketSnapshot, AgentState


class BasePolicy(ABC):
    """Abstract base class for agent decision policies"""
    
    @abstractmethod
    def decide(self, agent_state: AgentState, snapshot: MarketSnapshot) -> List[Action]:
        """
        Pure decision logic - no state mutations
        
        Args:
            agent_state: Current state of the agent
            snapshot: Read-only market state
            
        Returns:
            List of actions the agent wants to perform
        """
        pass
    
    def can_execute_action(self, action: Action, agent_state: AgentState, snapshot: MarketSnapshot) -> bool:
        """
        Check if an action can be executed given current state
        
        Args:
            action: Action to validate
            agent_state: Current agent state
            snapshot: Current market snapshot
            
        Returns:
            True if action is valid and executable
        """
        # Default implementation - can be overridden by specific policies
        return True


class BaseAgent:
    """Universal agent container - holds only state, no business logic"""
    
    def __init__(self, agent_id: str, initial_state: AgentState, policy: BasePolicy):
        """
        Initialize agent with state and policy
        
        Args:
            agent_id: Unique identifier for this agent
            initial_state: Initial state of the agent
            policy: Decision policy for this agent
        """
        self.agent_id = agent_id
        self.state = initial_state
        self.policy = policy
        self.action_history: List[Action] = []
        self.created_at = snapshot.timestamp if 'snapshot' in locals() else 0
    
    def decide_actions(self, snapshot: MarketSnapshot) -> List[Action]:
        """
        Delegate decision-making to policy
        
        Args:
            snapshot: Current market state
            
        Returns:
            List of actions this agent wants to perform
        """
        actions = self.policy.decide(self.state, snapshot)
        
        # Validate actions before returning
        valid_actions = []
        for action in actions:
            if self.policy.can_execute_action(action, self.state, snapshot):
                # Ensure action has correct agent_id
                action.agent_id = self.agent_id
                valid_actions.append(action)
        
        # Store actions in history
        self.action_history.extend(valid_actions)
        
        return valid_actions
    
    def update_state(self, **kwargs):
        """
        Update agent state - should only be called by markets after action execution
        
        Args:
            **kwargs: State updates to apply
        """
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
    
    def get_portfolio_summary(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """
        Get a summary of the agent's current portfolio
        
        Args:
            snapshot: Current market snapshot for pricing
            
        Returns:
            Dictionary with portfolio summary
        """
        total_value = self.state.get_total_value(snapshot.token_prices)
        
        return {
            'agent_id': self.agent_id,
            'total_value_usd': total_value,
            'token_balances': dict(self.state.token_balances),
            'staked_balance': self.state.staked_balance,
            'lp_balance': self.state.lp_balance,
            'cash_balance': self.state.cash_balance,
            'total_fees_paid': self.state.total_fees_paid,
            'total_rewards_earned': self.state.total_rewards_earned,
            'actions_taken': len(self.action_history)
        }


class HoldPolicy(BasePolicy):
    """Default policy that takes no actions (HOLD)"""
    
    def decide(self, agent_state: AgentState, snapshot: MarketSnapshot) -> List[Action]:
        """Always return HOLD action"""
        from ..simulation.primitives import ActionKind
        
        return [Action(
            kind=ActionKind.HOLD,
            agent_id="",  # Will be set by agent
            params={}
        )]


class AgentPopulation:
    """Manages a collection of agents"""
    
    def __init__(self):
        self.agents: List[BaseAgent] = []
        self._agent_lookup: Dict[str, BaseAgent] = {}
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent to the population"""
        if agent.agent_id in self._agent_lookup:
            raise ValueError(f"Agent with ID {agent.agent_id} already exists")
        
        self.agents.append(agent)
        self._agent_lookup[agent.agent_id] = agent
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get agent by ID"""
        return self._agent_lookup.get(agent_id)
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove agent by ID"""
        agent = self._agent_lookup.pop(agent_id, None)
        if agent:
            self.agents.remove(agent)
            return True
        return False
    
    def get_all_actions(self, snapshot: MarketSnapshot) -> List[Action]:
        """Get actions from all agents in the population"""
        all_actions = []
        for agent in self.agents:
            actions = agent.decide_actions(snapshot)
            all_actions.extend(actions)
        return all_actions
    
    def get_population_summary(self, snapshot: MarketSnapshot) -> Dict[str, Any]:
        """Get summary statistics for the entire population"""
        if not self.agents:
            return {
                'total_agents': 0,
                'total_value_usd': 0.0,
                'average_value_usd': 0.0,
                'policy_distribution': {}
            }
        
        total_value = 0.0
        policy_counts = {}
        
        for agent in self.agents:
            portfolio = agent.get_portfolio_summary(snapshot)
            total_value += portfolio['total_value_usd']
            
            policy_name = agent.policy.__class__.__name__
            policy_counts[policy_name] = policy_counts.get(policy_name, 0) + 1
        
        return {
            'total_agents': len(self.agents),
            'total_value_usd': total_value,
            'average_value_usd': total_value / len(self.agents),
            'policy_distribution': policy_counts
        }
    
    def __len__(self):
        return len(self.agents)
    
    def __iter__(self):
        return iter(self.agents)
