# JSON Structure and Formatting Guidelines

## Overview

JSON creation and structuring is **CRITICAL** to the success of our simulation analysis. All charts, visuals, and analysis depend on properly structured JSON output. Every simulation script must follow these guidelines to ensure consistent, navigable, and analyzable data.

## Core Principles

### 1. **JSON is the Foundation**
- All charts and visuals are generated from JSON data
- Analysis success depends on clean, structured JSON
- JSON must be navigable and collapsible in viewers
- Data must be easily accessible for programmatic analysis

### 2. **Consistency Across Scripts**
- All simulation scripts must use the same JSON structure
- Naming conventions must be consistent
- Data hierarchy must be uniform
- Metadata must be comprehensive

## Required JSON Structure

### Top-Level Structure
```json
{
  "analysis_metadata": {
    "analysis_type": "string",
    "timestamp": "ISO_8601_string",
    "target_hfs_tested": [array_of_floats],
    "monte_carlo_runs_per_scenario": integer,
    "agents_per_run": integer,
    "total_scenarios": integer,
    "additional_metadata": "as_needed"
  },
  "detailed_scenario_results": [
    // Array of scenario results
  ]
}
```

### Scenario Structure
```json
{
  "target_hf": float,
  "scenario_params": {
    "target_hf": float,
    "initial_hf_range": [min_float, max_float],
    "variation_type": "string"
  },
  "high_tide_summary": {
    "mean_survival_rate": float,
    "survival_rate_std": float,
    "mean_liquidations": float,
    "mean_rebalancing_events": float,
    "liquidation_frequency": float
  },
  "ht_agent_outcomes": [
    // Array of individual agent outcomes
  ],
  "simulation_runs": [
    // Array of individual simulation runs
  ]
}
```

### Individual Run Structure
```json
{
  "run_id": "run_001_target_hf_1.01",
  "timestamp": "ISO_8601_string",
  "btc_price_history": [
    {
      "minute": integer,
      "btc_price": float
    }
  ],
  "agent_health_history": [
    {
      "minute": integer,
      "agents": [
        {
          "agent_id": "string",
          "health_factor": float,
          "collateral_value": float,
          "effective_collateral": float,
          "debt_value": float,
          "btc_amount": float,
          "moet_debt": float,
          "yield_token_value": float,
          "rebalancing_triggered": boolean
        }
      ]
    }
  ],
  "agent_actions_history": [
    {
      "step": integer,
      "agent_id": "string",
      "action_type": "string",
      "parameters": {},
      "agent_health_factor": float,
      "timestamp": integer
    }
  ],
  "rebalancing_events": [
    {
      "minute": integer,
      "agent_id": "string",
      "trigger_reason": "string",
      "health_factor_before": float,
      "health_factor_after": float,
      "yield_tokens_sold": float,
      "moet_received": float,
      "slippage_cost": float,
      "debt_repaid": float,
      "new_debt_balance": float,
      "pool_utilization_before": float,
      "pool_utilization_after": float
    }
  ],
  "agent_outcomes": [
    {
      "agent_id": "string",
      "risk_profile": "string",
      "target_health_factor": float,
      "initial_health_factor": float,
      "final_health_factor": float,
      "cost_of_rebalancing": float,
      "net_position_value": float,
      "total_yield_earned": float,
      "total_yield_sold": float,
      "total_slippage_costs": float,
      "rebalancing_events": integer,
      "survived": boolean,
      "emergency_liquidations": integer,
      "yield_token_value": float,
      "initial_debt": float,
      "final_debt": float,
      "interest_accrued": float,
      "rebalancing_history": []
    }
  ],
  "summary_stats": {
    "total_agents": integer,
    "survived_agents": integer,
    "total_rebalancing_events": integer,
    "total_slippage_costs": float,
    "final_btc_price": float,
    "btc_price_decline_percent": float
  }
}
```

## Naming Conventions

### Agent IDs
- **Main Analysis**: `hf_test_ht_{target_hf}_run{run_num}_agent{i}`
- **Quick Test**: `quick_test_ht_{target_hf}_{i}`
- **Other Scripts**: `{script_name}_ht_{target_hf}_run{run_num}_agent{i}`

### Run IDs
- **Format**: `run_{run_num:03d}_target_hf_{target_hf}`
- **Example**: `run_001_target_hf_1.01`

### File Names
- **Main Analysis**: `target_hf_analysis_results.json`
- **Quick Test**: `quick_test_results.json`
- **Other Scripts**: `{script_name}_results.json`

## Data Filtering Requirements

### 1. **Agent Filtering**
- **ONLY** include agents created by the simulation script
- **EXCLUDE** all legacy agents: `lender_*`, `trader_*`, `liquidator_*`, `high_tide_conservative_*`, `high_tide_moderate_*`, `high_tide_aggressive_*`
- **INCLUDE** only test agents matching the naming convention

### 2. **Data Cleanup**
- Remove `pool_utilization_history` (not needed for current analysis)
- Filter `agent_health_factors` to only include test agents
- Filter `agent_actions_history` to only include test agents
- Filter `agent_health_history` to only include test agents

### 3. **Protocol Configuration**
- Use BTC-only protocol for Target Health Factor analysis
- Initialize only BTC asset pools
- Include only MOET:BTC liquidity pools

## JSON Serialization Requirements

### 1. **Handle Non-Serializable Objects**
```python
def convert_for_json(obj):
    """Recursively convert objects to JSON-serializable format"""
    if isinstance(obj, dict):
        converted_dict = {}
        for key, value in obj.items():
            if hasattr(key, 'name'):
                converted_key = key.name
            else:
                converted_key = str(key)
            converted_dict[converted_key] = convert_for_json(value)
        return converted_dict
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif hasattr(obj, 'name'):  # Handle enums
        return obj.name
    elif isinstance(obj, bool):  # Handle Python boolean values
        return bool(obj)
    elif hasattr(obj, 'dtype') and 'bool' in str(obj.dtype):  # Handle numpy boolean arrays/scalars
        if hasattr(obj, 'shape') and len(obj.shape) > 0:  # It's an array
            return obj.tolist()  # Convert array to list
        else:  # It's a scalar
            return bool(obj)
    elif str(type(obj)).startswith('<class \'numpy.'):  # Handle any numpy type
        if hasattr(obj, 'shape') and len(obj.shape) > 0:  # It's an array
            return obj.tolist()  # Convert array to list
        else:  # It's a scalar
            try:
                return bool(obj)
            except:
                return str(obj)
    elif hasattr(obj, '__dict__'):  # Handle objects with __dict__
        return str(obj)
    else:
        return obj
```

### 2. **Fallback Serialization**
```python
with open(results_path, 'w', encoding='utf-8') as f:
    try:
        json.dump(json_safe_results, f, indent=2)
    except TypeError as e:
        if 'not JSON serializable' in str(e):
            print(f"⚠️  JSON serialization warning: {e}")
            print("   Converting problematic objects to strings...")
            # Fallback: convert everything to string representation
            def force_convert(obj):
                if isinstance(obj, dict):
                    return {str(k): force_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [force_convert(item) for item in obj]
                else:
                    return str(obj)
            
            json_safe_results = force_convert(final_results)
            json.dump(json_safe_results, f, indent=2)
        else:
            raise
```

## Required Data Elements

### 1. **Minute-by-Minute Health Factors** ✅ REQUIRED
- Track health factor changes over time
- Essential for understanding rebalancing triggers
- Must include all test agents

### 2. **Step-by-Step Action Logging** ✅ REQUIRED
- Track when agents trigger rebalancing
- Show hold, swap, and other actions
- Essential for understanding agent behavior

### 3. **BTC Price History** ✅ REQUIRED
- Track price changes over simulation
- Essential for correlation analysis
- Must be minute-by-minute

### 4. **Rebalancing Events** ✅ REQUIRED
- Track all rebalancing actions
- Include slippage costs
- Show before/after health factors

### 5. **Agent Outcomes** ✅ REQUIRED
- Final state of each agent
- Include all financial metrics
- Show survival status

## Prohibited Data Elements

### 1. **Pool Utilization History** ❌ PROHIBITED
- Not needed for current analysis
- Adds unnecessary bloat

### 2. **Non-Test Agents** ❌ PROHIBITED
- Exclude all legacy agents
- Only include agents created by the script

### 3. **Multi-Asset Data** ❌ PROHIBITED (for Target HF analysis)
- Use BTC-only protocol
- Exclude ETH, FLOW, USDC data

## Implementation Checklist

### Before Creating Any Simulation Script:

- [ ] Define clear agent naming convention
- [ ] Implement BTC-only protocol configuration
- [ ] Create data filtering functions
- [ ] Implement JSON serialization handling
- [ ] Define run ID format
- [ ] Plan data hierarchy structure

### During Script Development:

- [ ] Filter out non-test agents from all data
- [ ] Store detailed simulation data for clean processing
- [ ] Implement minute-by-minute health factor tracking
- [ ] Implement step-by-step action logging
- [ ] Track rebalancing events with slippage
- [ ] Calculate comprehensive agent outcomes

### Before Script Completion:

- [ ] Verify JSON structure matches guidelines
- [ ] Test JSON serialization with all data types
- [ ] Ensure data is navigable and collapsible
- [ ] Verify all required data elements are present
- [ ] Confirm no prohibited data elements are included
- [ ] Test chart generation from JSON data

## Examples

### Good JSON Structure
```json
{
  "analysis_metadata": {
    "analysis_type": "Target_Health_Factor_Analysis",
    "timestamp": "2025-01-02T14:30:22.123456",
    "target_hfs_tested": [1.01, 1.05, 1.075, 1.1, 1.15],
    "monte_carlo_runs_per_scenario": 5,
    "agents_per_run": 15,
    "total_scenarios": 5
  },
  "detailed_scenario_results": [
    {
      "target_hf": 1.01,
      "simulation_runs": [
        {
          "run_id": "run_001_target_hf_1.01",
          "agent_health_history": [...],
          "agent_actions_history": [...],
          "rebalancing_events": [...],
          "agent_outcomes": [...]
        }
      ]
    }
  ]
}
```

### Bad JSON Structure
```json
{
  "results": [
    {
      "target_hf": 1.01,
      "data": {
        "agents": [...], // Mixed test and legacy agents
        "pool_utilization_history": [...], // Unnecessary data
        "all_assets": {...} // Multi-asset data
      }
    }
  ]
}
```

## Conclusion

Proper JSON structure is **ESSENTIAL** for:
- Chart generation
- Data analysis
- Visualization creation
- Simulation validation
- Research insights

Every simulation script must follow these guidelines to ensure consistent, analyzable, and navigable data output. The JSON structure is the foundation of our entire analysis pipeline.
