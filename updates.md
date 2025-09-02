Your task is to refactor some of the code in our simulation system, which currently has some broken assumptions, in order for us to use the system to actual risk questions based on the results of our simulations. You will review the code and instructions here thoroughly, and then work to streamline, refactor, and improve our code in order to solve the gaps that currently exist in our simulation results. 

Core components to understand:

High Tide is an automated yield aggregation layer built atop the Tidal Protocol, designed to enable crypto holders to deposit assets (BTC, ETH, FLOW, stables) and earn optimized in-kind yield without manual management. Unlike passive lending protocols that rely on forced liquidations, Tidal Protocol leverages active position management: automated rebalancing  proactively defend user health factors during market downturns, minimizing principal loss and maximizing capital efficiency. This allows High Tide to take advantage of this engine while offering users market-leading yield

Our simulation evaluates how different liquidity pool configurations—both external (MOET:BTC) and internal (MOET:Yield Token)—impact the protocol’s ability to execute rebalancing operations. We focus on three key metrics:

## **Technical Specification of High Tide Protocol**

### **2.1 Foundation: Tidal Protocol**

High Tide builds on the Tidal Protocol, an actively managed lending engine for the Flow blockchain. Core features include:

- **Collateral Supply:** Users deposit BTC as collateral
- **Stablecoin Borrowing:** MOET stablecoins are borrowed against collateral
- **Active Management:** Automated callbacks and health factor thresholds trigger position rebalancing, leveraging Flow’s scheduled transaction infrastructure

Tidal’s kinked interest rate model increases borrowing rates sharply above 80% pool utilization, incentivizing agents to maintain healthy positions and avoid costly liquidations.

### **2.2 High Tide Enhancement: Yield Token Strategy**

High Tide introduces an automated yield token strategy:

- **Immediate Yield Token Purchase:** Borrowed MOET is converted to yield-bearing tokens at 10% APR
- **Continuous Compounding:** Yield tokens accrue interest every minute
- **Health Factor Monitoring:** The system tracks each agent’s health factor (collateral value / debt value)
- **Active Rebalancing:** Yield tokens are automatically sold to repay debt when health factors fall below maintenance thresholds

**Liquidation as Last Resort:** Forced liquidation only occurs if yield token sales cannot restore solvency. Forced liquidations seize collateral and swap it for MOET to pay down DEBT

## **Things that need to immediately get fixed:**

Currently our system handles automated rebalancing by swapping Yield Tokens for MOET and then swapping MOET for BTC to pay down collateral. This is structurally incorrect. The rebalances need to swap Yield Tokens for MOET and pay down the debt asset directly rather than an extra swap into the collateral asset. The BTC:MOET pool is only used for liquidations

All of our scenario analyses need to have two simulations run and compared against each other. The HighTide/Tidal simulation vs. the Aave simulation.

Specific Files to Review:

- comprehensive_realistic_pool_analysis.py
- high_tide_engine.py
- lp_curve_analysis.py
- uniswap_v3_math.py
- yield_tokens.py

The results for our ‘comprehensive_realistic_pool_analysis.py’ script are under ‘comprehensive_realistic_analysis’ folder

## **Questions we need answered:**

We need to be thinking about supply and borrow behavior in the frame of ratios:
- What is the right Deposit Cap of BTC as a % of Liquidity of BTC:MOET available externally. Based on our earliest simulations it appears that most of our agents do not get liquidated because our automated rebalancing mechanism prevents it. However, this leads us to question two.

- How low can we make the Target Health Factor (the one that triggers rebalancing) before agents begin to get liquidated frequently? We should test target health factors of 1.01, 1.05, 1.1, 1.15 for this. We should also make it so aggressive agents take an initially large LTV loan (low health factor, say 1.1-1.2) with an aggressive target HF (1.05) to see what happens.

- Is there a borrow cap that should be set as a % of Liquidity in the MOET:YT pool? We are initially establishing a $250:$250k MOET:YT as our baseline. We should be testing this against a large number of agents with tight ranges between their initial health factor and their target health factor who will initiate a lot of rebalances.

- Given the $250k:$250k MOET:YT pool, what is the most aggressive initial health factor and target health factor we can take without forcing *any* liquidations given our aggressive BTC Price Decline scenario which is back dated as the most aggressive price decline in BTC history? I guess this is the same as question number 2 above.

- Given that Tidal Protocol and High Tide are separate. There may be users of Tidal not using High Tide; thus, after answering all of these scenarios above for High Tide specifically. What happens as we increase the % of users just borrowing MOET without any Rebalancing Mechanism given the BEST Deposit Cap answer to question #1.

## How we are currently thinking about these questions:

- **Rebalancing Cost Optimization:** Minimizing slippage and trading fees during position adjustments
- **Agent Survival Rates:** Maintaining healthy borrowing positions through market stress
- **Protocol Efficiency:** Maximizing capital utilization while preserving system stability

Sixteen pool sizing combinations were tested to determine optimal allocation strategies for High Tide’s architecture.

## **Structural Things:**

We need all of our simulations to create a simulation results json file that is used to create all of our charts. All charts must have the right data in the json file to meet our needs. No hallucination of data into the charts can occur. No hardcoded values or mock data should be implemented in order to achieve desirable results. The visualizations MUST use the system-generated simulation results.

Additionally, we need a report builder that for each analysis provides an Introduction to the simulation outlining the questions we want answered, the technical methodologies and the results of the simulation.

We use the right Uniswap V3 math already baked in so no need to add anything there just streamline if needed. If not needed to do not add extra changes.