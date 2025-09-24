# Rebalancing Health Factor Comparative Analysis

**Analysis Date:** September 2025  
**Comparison:** Rebalancing HF 1.01 vs 1.025  
**Configuration:** 20 agents ($2M in High Tide TVL per Yield Token), Fixed Initial HF = 1.25  
**Pool Size:** $500K ($250K MOET + $250K Yield Tokens)

## Executive Summary

This comparative analysis examines the impact of rebalancing health factor settings on pool utilization and system performance. The study compares two configurations: an aggressive **Rebalancing HF = 1.01** versus a more conservative **Rebalancing HF = 1.025**. The more aggresive target had 2 simulations hit 100% capacity and fail to raise enough capital while the moderate configuration achieved **100% survival rates** for High Tide agents.

## Key Performance Comparison

### Pool Utilization Analysis

| Metric | Rebalancing HF 1.01 | Rebalancing HF 1.025 | Difference |
|--------|---------------------|----------------------|------------|
| **Average Pool Utilization** | 99.4% | 92.9% | **-6.5%** |
| **Maximum Pool Utilization** | 100.0% | 93.7% | **-6.3%** |
| **Total MOET Extracted** | $2,732K | $2,090K | **-$642K (-23.5%)** |
| **Scenarios Tested** | 11 | 9 | -2 |
| **Failed Trades** | 2 | 0 | 2 |

### Agent Survival Rates
- **High Tide Agents**: 81% survival vs. 100% survival between each test

## Detailed Scenario Analysis

### Rebalancing HF 1.01 Results
| Scenario | Target HF | Pool Utilization | MOET Extracted |
|----------|-----------|------------------|----------------|
| 1.10 Conservative | 1.100 | 99.5% | $248,787 |
| 1.09 Conservative | 1.090 | 98.7% | $246,640 |
| 1.08 Conservative | 1.080 | 99.9% | $249,767 |
| 1.07 Conservative | 1.070 | 99.2% | $247,888 |
| 1.06 Conservative | 1.060 | **100.0%%** | $250,00 |
| 1.05 Moderate | 1.050 | 99.8% | $249,455 |
| 1.04 Moderate | 1.040 | 98.3% | $245,716 |
| 1.03 Moderate | 1.030 | 98.5% | $246,359 |
| 1.02 Aggressive | 1.020 | 98.8% | $246,897 |
| 1.015 Aggressive | 1.015 | **100.0%** | $250,000 |
| **1.011 Maximum Aggressive** | **1.011** | 98.8% | $247,043 |

### Rebalancing HF 1.025 Results
| Scenario | Target HF | Pool Utilization | MOET Extracted |
|----------|-----------|------------------|----------------|
| 1.10 Conservative | 1.100 | 92.2% | $230,400 |
| 1.09 Conservative | 1.090 | 92.9% | $232,366 |
| 1.08 Conservative | 1.080 | 92.8% | $232,123 |
| 1.07 Conservative | 1.070 | **93.7%** | $234,131 |
| 1.06 Conservative | 1.060 | 93.1% | $232,761 |
| 1.05 Moderate | 1.050 | 93.1% | $232,720 |
| 1.04 Moderate | 1.040 | 93.0% | $232,472 |
| 1.03 Moderate | 1.030 | 92.3% | $230,691 |
| **1.026 Aggressive** | **1.026** | 93.0% | $232,468 |

## Key Insights

### 1. Capital Efficiency Trade-offs
**Rebalancing HF 1.01** demonstrates superior capital efficiency but with some failed scenarios:
- **23.5% more MOET extracted** from the same pool
- **Near-perfect utilization** at 99.4% average
- **Maximum aggressiveness** with Target HF as low as 1.011

**Rebalancing HF 1.025** provides conservative operation:
- **Consistent ~93% utilization** across all scenarios
- **More predictable** pool usage patterns

### 2. Rebalancing Timing Impact
The 1.5 percentage point difference in rebalancing triggers creates significant behavioral changes:

**Earlier Rebalancing (1.025):**
- Agents rebalance at minute 30-35 when HF = 1.024
- **Lower MOET requirements** per agent (~$11.6K vs ~$12.4K)
- **Reduced competition** for pool liquidity
- **Lower slippage costs** due to earlier intervention

**Later Rebalancing (1.01):**
- Agents rebalance at minute 40-44 when HF = 1.005-1.010  
- **Higher MOET requirements** per agent (~$12.4K)
- **Intense competition** for remaining pool liquidity
- **Higher slippage costs** but maximum capital utilization

### 3. Risk vs Efficiency Analysis

| Aspect | Rebalancing HF 1.01 | Rebalancing HF 1.025 | Winner |
|--------|---------------------|----------------------|--------|
| **Capital Efficiency** | 99.4% utilization | 92.9% utilization | **1.01** |
| **Safety Buffer** | 1% above liquidation | 2.5% above liquidation | **1.025** |
| **Predictability** | Variable (98-100%) | Consistent (~93%) | **1.025** |
| **Maximum Capacity** | 1.011 Target HF | 1.026 Target HF | **1.01** |
| **Pool Stress** | Some exhaustion | Comfortable margin | **1.025** |

### 4. Operational Implications

**For Maximum Capital Efficiency (1.01):**
- Suitable for **deeper liquiidty** or with rebalancing mechanisms with **proven reliability**
- Requires **robust monitoring** due to tight margins
- **Greater slippage costs** during high-stress periods

**For Reliable Operations (1.025):**
- **Predictable performance** with consistent ~93% utilization
- **Lower operational stress** with comfortable safety margins
- **Reduced slippage** due to earlier intervention

## Recommendations

### Hybrid Approach Consideration
A **dynamic rebalancing system** could potentially combine benefits:
- Use 1.025 to start and during periods of intense market stress
- Switch to 1.01 during stable market conditions
- Adjust based on pool utilization metrics in real-time
