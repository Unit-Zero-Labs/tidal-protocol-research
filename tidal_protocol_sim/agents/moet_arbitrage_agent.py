#!/usr/bin/env python3
"""
MOET Arbitrage Agent Implementation

Implements arbitrage agents that maintain MOET's $1.00 peg by monitoring
MOET:USDC and MOET:USDF pools and using the Enhanced Redeemer system.

Arbitrage Strategies:
1. MOET > $1.00: Mint MOET from Redeemer → Sell on DEX → Profit
2. MOET < $1.00: Buy MOET from DEX → Redeem via Redeemer → Profit
"""

import random
from typing import Dict, Tuple, Optional, List
from .base_agent import BaseAgent, AgentAction, AgentState
from ..core.protocol import Asset


class MoetArbitrageAgentState(AgentState):
    """Extended agent state for MOET arbitrage operations"""
    
    def __init__(self, agent_id: str, initial_balance: float):
        super().__init__(agent_id, initial_balance, "moet_arbitrage_agent")
        
        # Store initial balance explicitly (needed for results generation)
        self.initial_balance = initial_balance
        
        # Arbitrage-specific balances (using USDC as proxy for both USDC and USDF)
        self.token_balances = {
            Asset.USDC: initial_balance,  # Total balance (represents both USDC and USDF)
            Asset.MOET: 0.0,
            Asset.BTC: 0.0,
            Asset.ETH: 0.0,
            Asset.FLOW: 0.0
        }
        
        # Initialize empty balances for compatibility with base class expectations
        self.supplied_balances = {}
        self.borrowed_balances = {}
        
        # Enhanced arbitrage tracking
        self.arbitrage_events = []
        self.arbitrage_attempts = []  # Track ALL attempts, not just successful ones
        self.total_profit = 0.0
        self.total_fees_generated = 0.0  # Track fees generated for protocol
        self.successful_arbitrages = 0
        self.failed_arbitrages = 0
        self.total_attempts = 0
        self.total_mint_attempts = 0
        self.total_redeem_attempts = 0
        self.total_volume_traded = 0.0
        
        # Risk management - removed for arbitrage since it's theoretically risk-free
        self.max_trade_size = initial_balance  # Allow full balance for arbitrage
        self.min_profit_threshold = 0.001  # 0.1% minimum profit
        self.last_arbitrage_minute = -10  # Initialize to allow immediate arbitrage
        self.arbitrage_cooldown = 5  # 5 minutes between arbitrages


class MoetArbitrageAgent(BaseAgent):
    """
    Arbitrage agent that maintains MOET peg using the Enhanced Redeemer system
    
    Strategy:
    - Monitor MOET prices in MOET:USDC and MOET:USDF pools
    - When MOET > $1: Mint MOET → Sell on DEX → Profit
    - When MOET < $1: Buy MOET from DEX → Redeem → Profit
    - Use Enhanced Redeemer's dynamic fee structure
    """
    
    def __init__(self, agent_id: str, initial_balance: float = 100_000.0):
        super().__init__(agent_id, "moet_arbitrage_agent", initial_balance)
        
        # Replace state with arbitrage-specific state
        self.state = MoetArbitrageAgentState(agent_id, initial_balance)
        
        # Engine reference for pool access
        self.engine = None  # Will be set by engine during initialization
        
        # Arbitrage parameters
        self.profit_threshold = 0.0  # No minimum profit threshold - take any profitable opportunity
        self.max_trade_percentage = 0.05  # Max 5% of balance per trade
        self.pool_preference = None  # Will randomly choose USDC or USDF
        
    def decide_action(self, protocol_state: dict, asset_prices: Dict[Asset, float]) -> tuple:
        """Decide arbitrage action based on MOET pool prices"""
        current_minute = protocol_state.get("current_minute", 0)
        
        # Check if agent is still active
        if not self.active:
            return (AgentAction.HOLD, {})
        
        # Cooldown check
        if current_minute - self.state.last_arbitrage_minute < self.state.arbitrage_cooldown:
            return (AgentAction.HOLD, {})
        
        # Check for arbitrage opportunities using Redeemer system
        arbitrage_opportunity = self._detect_redeemer_arbitrage_opportunity(current_minute)
        
        if arbitrage_opportunity:
            return self._execute_arbitrage_decision(arbitrage_opportunity, current_minute)
        
        return (AgentAction.HOLD, {})
    
    def _detect_redeemer_arbitrage_opportunity(self, current_minute: int) -> Optional[Dict]:
        """Detect arbitrage opportunities using 9 bps deviation threshold (no profitability check)"""
        if not self.engine:
            print(f"   ⚠️  {self.agent_id}: No engine reference")
            return None
        
        # Check if advanced MOET system is available
        has_protocol = hasattr(self.engine, 'protocol')
        advanced_moet = getattr(self.engine.protocol, 'enable_advanced_moet', False) if has_protocol else False
        
        if not advanced_moet:
            print(f"   ⚠️  {self.agent_id}: Advanced MOET system not enabled")
            return None
        
        # Check if pools exist
        has_usdc_pool = hasattr(self.engine, 'moet_usdc_pool') and hasattr(self.engine, 'moet_usdc_calculator')
        has_usdf_pool = hasattr(self.engine, 'moet_usdf_pool') and hasattr(self.engine, 'moet_usdf_calculator')
        
        if not (has_usdc_pool or has_usdf_pool):
            print(f"   ⚠️  {self.agent_id}: No MOET pools available")
            return None
        
        redeemer = getattr(self.engine.protocol.moet_system, 'redeemer', None)
        if not redeemer:
            print(f"   ⚠️  {self.agent_id}: No redeemer available")
            return None
        
        # Only log arbitrage checks daily to prevent console bloat (following existing pattern)
        if current_minute % 1440 == 0:  # Daily logging like other metrics
            print(f"   🔍 {self.agent_id} checking Pool vs Redeemer arbitrage opportunities")
        
        opportunities = []
        
        # Check MOET:USDC pool arbitrage using 9 bps deviation threshold (no profitability check)
        if has_usdc_pool:
            usdc_price = self.engine.moet_usdc_pool.get_price()  # USDC per MOET
            usdc_deviation_bps = abs(usdc_price - 1.0) * 10000
            
            if usdc_deviation_bps > 9.0:  # Execute if > 9 bps deviation
                usdc_opportunity = self._check_pool_arbitrage(self.engine.moet_usdc_pool, "USDC", current_minute)
                if usdc_opportunity:
                    opportunities.append(usdc_opportunity)  # No profitability check
                    self._record_arbitrage_attempt(usdc_opportunity, current_minute, executed=False)
        
        # Check MOET:USDF pool arbitrage using 9 bps deviation threshold (no profitability check)
        if has_usdf_pool:
            usdf_price = self.engine.moet_usdf_pool.get_price()  # USDF per MOET
            usdf_deviation_bps = abs(usdf_price - 1.0) * 10000
            
            if usdf_deviation_bps > 9.0:  # Execute if > 9 bps deviation
                usdf_opportunity = self._check_pool_arbitrage(self.engine.moet_usdf_pool, "USDF", current_minute)
                if usdf_opportunity:
                    opportunities.append(usdf_opportunity)  # No profitability check
                    self._record_arbitrage_attempt(usdf_opportunity, current_minute, executed=False)
        
        if not opportunities:
            # Still record the attempt for tracking
            null_attempt = {
                'type': 'no_opportunity',
                'pool': 'redeemer_vs_pools',
                'moet_price': 1.0,
                'trade_size': 0.0,
                'expected_profit': 0.0
            }
            self._record_arbitrage_attempt(null_attempt, current_minute, executed=False)
            # Extremely reduced logging - only log weekly or for small agent counts
            total_agents = len(getattr(self.engine, 'high_tide_agents', [])) if hasattr(self, 'engine') else 0
            should_log = False
            if total_agents <= 150:
                should_log = current_minute % 1440 == 0  # Daily for small counts
            else:
                should_log = current_minute % (1440 * 7) == 0  # Weekly for large counts
            
            if should_log:
                print(f"   📊 {self.agent_id}: No profitable Pool vs Redeemer arbitrage found")
            return None
        
        # Return the most profitable opportunity
        best_opportunity = max(opportunities, key=lambda x: x['expected_profit'])
        print(f"   💰 {self.agent_id}: Best arbitrage: {best_opportunity['type']} with ${best_opportunity['expected_profit']:.2f} profit")
        return best_opportunity
    
    def _calculate_pool_vs_redeemer_arbitrage_with_exact_math(self, stablecoin: str, current_minute: int) -> Optional[Dict]:
        """Calculate arbitrage using exact Uniswap V3 mathematical formulas"""
        try:
            # Get pool and calculator
            if stablecoin == "USDC":
                pool = self.engine.moet_usdc_pool
                calculator = self.engine.moet_usdc_calculator
            else:  # USDF
                pool = self.engine.moet_usdf_pool
                calculator = self.engine.moet_usdf_calculator
            
            # Get current MOET price from pool (USDC per MOET)
            pool_price = pool.get_price()  # P = USDC per MOET
            active_liquidity = pool.liquidity  # L = active liquidity
            
            # Agent's available capital
            max_usdc = self.state.token_balances[Asset.USDC]
            
            if max_usdc < 100.0:  # Minimum $100 trade
                return None
            
            # Fee constants from mathematical specification
            MINT_FEE = 0.0002      # m = 0.02% for Redeemer minting
            POOL_FEE = 0.0005      # f = 0.05% for Uniswap V3 swaps
            OVERVALUED_THRESHOLD = 1.0007005   # P > 1/[(1-m)(1-f)]
            UNDERVALUED_THRESHOLD = 0.9995     # P < (1-f)
            
            # Only log pool prices daily to prevent console bloat
            if current_minute % 1440 == 0:
                print(f"     🔍 {stablecoin} pool: P=${pool_price:.6f}, L={active_liquidity:,}")
                print(f"     🎯 Thresholds: overvalued>{OVERVALUED_THRESHOLD:.6f}, undervalued<{UNDERVALUED_THRESHOLD:.6f}")
            
            # Determine arbitrage strategy based on exact thresholds
            if pool_price > OVERVALUED_THRESHOLD:
                # OVERVALUED: USDC → mint MOET → sell MOET in pool
                return self._calculate_overvalued_arbitrage(pool_price, active_liquidity, max_usdc, stablecoin)
                
            elif pool_price < UNDERVALUED_THRESHOLD:
                # UNDERVALUED: USDC → buy MOET in pool → redeem MOET for USDC
                return self._calculate_undervalued_arbitrage(pool_price, active_liquidity, max_usdc, stablecoin)
            
            else:
                # Price within acceptable range, no arbitrage opportunity
                return None
                
        except Exception as e:
            print(f"   ❌ Arbitrage calculation error: {e}")
            return None
    
    def _calculate_overvalued_arbitrage(self, P: float, L: int, max_usdc: float, stablecoin: str) -> Optional[Dict]:
        """
        Calculate overvalued MOET arbitrage: USDC → mint MOET → sell MOET in pool
        Using exact formula: U_max = L × (1-m)(1-f) × (P-1) / (1 - P(1-m)(1-f))
        """
        m = 0.0002  # mint fee
        f = 0.0005  # pool fee
        
        try:
            # Calculate maximum profitable trade size
            numerator = L * (1 - m) * (1 - f) * (P - 1)
            denominator = 1 - P * (1 - m) * (1 - f)
            
            if denominator <= 0:
                return None  # Invalid calculation
                
            U_max = numerator / denominator
            U_max = min(U_max, max_usdc)  # Constrain by available capital
            
            if U_max < 100.0:  # Minimum trade size
                return None
            
            # Calculate profit using exact formula
            # Profit(U) = L × (1 / (1/P - U(1-m)(1-f)/L) - P) - U
            effective_amount = U_max * (1 - m) * (1 - f)
            new_price = 1.0 / (1.0/P - effective_amount/L)
            usdc_received = L * (new_price - P)
            profit = usdc_received - U_max
            
            return {
                'strategy': 'overvalued_mint_sell',
                'stablecoin': stablecoin,
                'pool_price': P,
                'optimal_usdc_amount': U_max,
                'expected_profit': profit,
                'expected_usdc_received': usdc_received,
                'new_pool_price': new_price
            }
            
        except Exception as e:
            print(f"   ❌ Overvalued arbitrage calculation error: {e}")
            return None
    
    def _calculate_undervalued_arbitrage(self, P: float, L: int, max_usdc: float, stablecoin: str) -> Optional[Dict]:
        """
        Calculate undervalued MOET arbitrage: USDC → buy MOET in pool → redeem MOET for USDC
        Using exact formula: U_max = L × (1-f) × (1-P) / P
        """
        f = 0.0005  # pool fee
        
        try:
            # Calculate maximum profitable trade size
            U_max = L * (1 - f) * (1 - P) / P
            U_max = min(U_max, max_usdc)  # Constrain by available capital
            
            if U_max < 100.0:  # Minimum trade size
                return None
            
            # Calculate profit using exact formula
            # Profit(U) = L × (1/P - 1/(P + U(1-f)/L)) - U
            effective_amount = U_max * (1 - f)
            new_price = P + effective_amount / L
            moet_received = L * (1.0/P - 1.0/new_price)
            usdc_redeemed = moet_received  # 1:1 redemption at peg
            profit = usdc_redeemed - U_max
            
            return {
                'strategy': 'undervalued_buy_redeem',
                'stablecoin': stablecoin,
                'pool_price': P,
                'optimal_usdc_amount': U_max,
                'expected_profit': profit,
                'expected_moet_received': moet_received,
                'expected_usdc_redeemed': usdc_redeemed,
                'new_pool_price': new_price
            }
            
        except Exception as e:
            print(f"   ❌ Undervalued arbitrage calculation error: {e}")
            return None

    def _calculate_pool_vs_redeemer_arbitrage(self, stablecoin: str, current_minute: int) -> Optional[Dict]:
        """Calculate arbitrage between pool price and Redeemer (1:1) pricing"""
        try:
            # Get pool and calculator
            if stablecoin == "USDC":
                pool = self.engine.moet_usdc_pool
                calculator = self.engine.moet_usdc_calculator
            else:  # USDF
                pool = self.engine.moet_usdf_pool
                calculator = self.engine.moet_usdf_calculator
            
            # Get current MOET price from pool (price of MOET in terms of stablecoin)
            # For a MOET:USDC pool, this tells us how many USDC we get for 1 MOET
            moet_price_in_pool = pool.get_price()  # MOET price in stablecoin terms
            
            # Only log pool prices daily to prevent console bloat
            if current_minute % 1440 == 0:
                print(f"     🔍 {stablecoin} pool: MOET price = ${moet_price_in_pool:.4f}")
            
            # Calculate trade size - use full balance for arbitrage (no artificial limits)
            max_trade = self.state.token_balances[Asset.USDC]  # Use full balance for arbitrage
            
            if max_trade < 100.0:  # Minimum $100 trade
                return None
            
            redeemer = self.engine.protocol.moet_system.redeemer
            
            # Strategy 1: MOET > $1.00 in pool → Mint from Redeemer → Sell in pool
            if moet_price_in_pool > 1.0:  # MOET overpriced (any amount above peg)
                # Estimate minting from Redeemer using stablecoin
                if stablecoin == "USDC":
                    fee_estimate = redeemer.estimate_deposit_fee(max_trade, 0.0)
                else:
                    fee_estimate = redeemer.estimate_deposit_fee(0.0, max_trade)
                
                if 'error' in fee_estimate:
                    return None
                
                # Calculate MOET that would be minted (deposit amount, since Redeemer mints 1:1)
                mint_fee = fee_estimate['total_fee']
                moet_minted = max_trade  # Redeemer mints 1:1 (deposit amount = MOET minted)
                
                # Sell MOET in pool for stablecoin
                swap_result = calculator.calculate_swap_slippage(moet_minted, "MOET")
                stablecoin_received = swap_result.get("amount_out", 0.0)
                swap_fees = swap_result.get("fees", 0.0)
                
                # Calculate net profit
                net_profit = stablecoin_received - max_trade - mint_fee - swap_fees
                
                return {
                    'type': f'mint_and_sell_{stablecoin.lower()}',
                    'pool': stablecoin,
                    'moet_price': moet_price_in_pool,
                    'trade_size': max_trade,
                    'strategy': 'mint_redeemer_sell_pool',
                    'moet_minted': moet_minted,
                    'mint_fee': mint_fee,
                    'stablecoin_received': stablecoin_received,
                    'swap_fees': swap_fees,
                    'expected_profit': net_profit,
                    'current_minute': current_minute
                }
            
            # Strategy 2: MOET < $1.00 in pool → Buy from pool → Redeem via Redeemer  
            elif moet_price_in_pool < 1.0:  # MOET underpriced (any amount below peg)
                # Buy MOET from pool using stablecoin
                swap_result = calculator.calculate_swap_slippage(max_trade, stablecoin)
                moet_received = swap_result.get("amount_out", 0.0)
                swap_fees = swap_result.get("fees", 0.0)
                
                # Redeem MOET via Redeemer for stablecoin
                redeem_result = redeemer.estimate_redemption_fee(moet_received, stablecoin)
                
                if 'error' in redeem_result:
                    return None
                
                # Calculate stablecoin received (1:1 redemption minus fees)
                redeem_fee = redeem_result['total_fee']
                stablecoin_redeemed = moet_received - redeem_fee  # 1:1 redemption minus fees
                
                # Calculate net profit (redemption fee already deducted from stablecoin_redeemed)
                net_profit = stablecoin_redeemed - max_trade - swap_fees
                
                # Debug logging
                print(f"     📊 Buy-and-redeem calculation (FIXED):")
                print(f"       Trade size: ${max_trade:,.0f}")
                print(f"       MOET received: {moet_received:.2f}")
                print(f"       Redemption fee: ${redeem_fee:.2f}")
                print(f"       Stablecoin redeemed: ${stablecoin_redeemed:.2f}")
                print(f"       Net profit: ${net_profit:.2f}")
                
                return {
                    'type': f'buy_and_redeem_{stablecoin.lower()}',
                    'pool': stablecoin,
                    'moet_price': moet_price_in_pool,
                    'trade_size': max_trade,
                    'strategy': 'buy_pool_redeem_redeemer',
                    'moet_received': moet_received,
                    'swap_fees': swap_fees,
                    'stablecoin_redeemed': stablecoin_redeemed,
                    'redeem_fee': redeem_fee,
                    'expected_profit': net_profit,
                    'current_minute': current_minute
                }
            
            # No arbitrage opportunity (price too close to $1.00)
            return {
                'type': 'no_arbitrage',
                'pool': stablecoin,
                'moet_price': moet_price_in_pool,
                'trade_size': 0.0,
                'expected_profit': 0.0,
                'current_minute': current_minute
            }
            
        except Exception as e:
            print(f"   ❌ Error calculating pool vs Redeemer arbitrage: {e}")
            return None
    
    def _calculate_redeemer_arbitrage(self, arb_type: str, current_minute: int, reserve_state) -> Optional[Dict]:
        """Calculate potential profit from Redeemer-based arbitrage"""
        try:
            # Calculate trade size (conservative approach)
            max_trade = min(
                self.state.token_balances[Asset.USDC] * 0.05,  # 5% of balance
                10000.0  # Max $10K per trade
            )
            
            if max_trade < 100.0:  # Minimum $100 trade
                return None
            
            # Estimate fees and profit based on reserve imbalance
            if arb_type == "USDF_to_USDC":
                # Mint MOET with USDF, redeem for USDC
                # Profit comes from the imbalance fee differential
                
                # Estimate deposit fee (minting with USDF when USDC is over-weighted)
                deposit_fee_estimate = self.engine.protocol.moet_system.redeemer.estimate_deposit_fee(0.0, max_trade)
                mint_fee = deposit_fee_estimate.get('total_fee', max_trade * 0.0002)
                
                # Estimate redemption fee (redeeming for USDC when USDC is over-weighted)
                redemption_fee_estimate = self.engine.protocol.moet_system.redeemer.estimate_redemption_fee(max_trade, "USDC")
                redeem_fee = redemption_fee_estimate.get('total_fee', max_trade * 0.0002)
                
                # Net profit = imbalance benefit - fees
                imbalance_benefit = max_trade * (reserve_state.current_usdc_ratio - 0.50) * 0.1  # 10% of imbalance
                net_profit = imbalance_benefit - mint_fee - redeem_fee
                
                return {
                    'type': 'redeemer_usdf_to_usdc',
                    'pool': 'redeemer',
                    'moet_price': 1.0,
                    'trade_size': max_trade,
                    'mint_asset': 'USDF',
                    'redeem_asset': 'USDC',
                    'mint_fee': mint_fee,
                    'redeem_fee': redeem_fee,
                    'expected_profit': net_profit,
                    'current_minute': current_minute
                }
                
            elif arb_type == "USDC_to_USDF":
                # Mint MOET with USDC, redeem for USDF
                
                # Estimate deposit fee (minting with USDC when USDF is over-weighted)
                deposit_fee_estimate = self.engine.protocol.moet_system.redeemer.estimate_deposit_fee(max_trade, 0.0)
                mint_fee = deposit_fee_estimate.get('total_fee', max_trade * 0.0002)
                
                # Estimate redemption fee (redeeming for USDF when USDF is over-weighted)
                redemption_fee_estimate = self.engine.protocol.moet_system.redeemer.estimate_redemption_fee(max_trade, "USDF")
                redeem_fee = redemption_fee_estimate.get('total_fee', max_trade * 0.0002)
                
                # Net profit = imbalance benefit - fees
                imbalance_benefit = max_trade * (reserve_state.current_usdf_ratio - 0.50) * 0.1  # 10% of imbalance
                net_profit = imbalance_benefit - mint_fee - redeem_fee
                
                return {
                    'type': 'redeemer_usdc_to_usdf',
                    'pool': 'redeemer',
                    'moet_price': 1.0,
                    'trade_size': max_trade,
                    'mint_asset': 'USDC',
                    'redeem_asset': 'USDF',
                    'mint_fee': mint_fee,
                    'redeem_fee': redeem_fee,
                    'expected_profit': net_profit,
                    'current_minute': current_minute
                }
            
            return None
            
        except Exception as e:
            print(f"   ❌ Error calculating Redeemer arbitrage: {e}")
            return None
    
    def _execute_redeemer_arbitrage(self, opportunity: Dict, current_minute: int) -> Optional[Dict]:
        """Execute arbitrage between pools and Redeemer system"""
        try:
            arb_type = opportunity['type']
            trade_size = opportunity['trade_size']
            strategy = opportunity.get('strategy', '')
            pool = opportunity['pool']
            
            print(f"   🔄 {self.agent_id}: Executing {arb_type} arbitrage: ${trade_size:.2f}")
            
            if strategy == 'mint_redeemer_sell_pool':
                # Strategy 1: Mint MOET from Redeemer → Sell in pool
                
                # Step 1: Mint MOET from Redeemer
                if pool == 'USDC':
                    mint_result = self.engine.protocol.mint_moet_from_deposit(trade_size, 0.0)
                else:  # USDF
                    mint_result = self.engine.protocol.mint_moet_from_deposit(0.0, trade_size)
                
                if not mint_result.get('success', False):
                    print(f"   ❌ {self.agent_id}: Mint failed")
                    return None
                
                moet_minted = mint_result['moet_minted']
                mint_fee = mint_result['total_fee']
                
                # Step 2: Sell MOET in pool for stablecoin
                if pool == 'USDC':
                    calculator = self.engine.moet_usdc_calculator
                else:
                    calculator = self.engine.moet_usdf_calculator
                
                swap_result = calculator.calculate_swap_slippage(moet_minted, "MOET")
                stablecoin_received = swap_result.get("amount_out", 0.0)
                swap_fees = swap_result.get("fees", 0.0)
                
                # Update pool state
                calculator.update_pool_state(swap_result)
                
                # Calculate actual profit
                actual_profit = stablecoin_received - trade_size
                total_fees_generated = mint_fee + swap_fees
                
            elif strategy == 'buy_pool_redeem_redeemer':
                # Strategy 2: Buy MOET from pool → Redeem via Redeemer
                
                # Step 1: Buy MOET from pool
                if pool == 'USDC':
                    calculator = self.engine.moet_usdc_calculator
                else:
                    calculator = self.engine.moet_usdf_calculator
                
                swap_result = calculator.calculate_swap_slippage(trade_size, pool)
                moet_received = swap_result.get("amount_out", 0.0)
                swap_fees = swap_result.get("fees", 0.0)
                
                # Update pool state
                calculator.update_pool_state(swap_result)
                
                # Step 2: Redeem MOET via Redeemer
                redeem_result = self.engine.protocol.redeem_moet_for_assets(moet_received, pool)
                
                if not redeem_result.get('success', False):
                    print(f"   ❌ {self.agent_id}: Redeem failed")
                    return None
                
                if pool == 'USDC':
                    stablecoin_received = redeem_result['usdc_received']
                else:
                    stablecoin_received = redeem_result['usdf_received']
                
                redeem_fee = redeem_result['total_fee']
                
                # Calculate actual profit
                actual_profit = stablecoin_received - trade_size
                total_fees_generated = swap_fees + redeem_fee
                
            else:
                print(f"   ❌ {self.agent_id}: Unknown strategy: {strategy}")
                return None
            
            # Update agent balances
            self.state.token_balances[Asset.USDC] -= trade_size  # Spent initial asset
            self.state.token_balances[Asset.USDC] += stablecoin_received  # Received target asset
            
            # Track fees generated for protocol
            self.state.total_fees_generated += total_fees_generated
            
            # Record the arbitrage event
            self._record_arbitrage_attempt(opportunity, current_minute, executed=True, actual_profit=actual_profit)
            
            print(f"   ✅ {self.agent_id}: Arbitrage executed - Profit: ${actual_profit:.2f}, Fees generated: ${total_fees_generated:.2f}")
            
            return {
                'success': True,
                'actual_profit': actual_profit,
                'fees_generated': total_fees_generated,
                'trade_size': trade_size,
                'type': arb_type
            }
            
        except Exception as e:
            print(f"   ❌ {self.agent_id}: Error executing arbitrage: {e}")
            return None
    
    def _check_pool_arbitrage(self, pool, stablecoin: str, current_minute: int) -> Optional[Dict]:
        """Check for arbitrage opportunity in a specific MOET:stablecoin pool"""
        try:
            # Get current MOET price from the pool
            current_moet_price = pool.get_price()  # This should give MOET price in stablecoin terms
            
            print(f"   🔍 Checking {stablecoin} pool: MOET price = ${current_moet_price:.4f}")
            
            # MOET should be $1.00 - check for deviations
            peg_deviation = abs(current_moet_price - 1.0)
            deviation_pct = peg_deviation * 100
            
            # Proceed with any deviation (no minimum threshold)
            
            # Calculate potential profit for different arbitrage strategies
            if current_moet_price > 1.0:
                # MOET overvalued: Mint MOET from Redeemer → Sell on DEX
                opportunity = self._calculate_mint_arbitrage(
                    pool, stablecoin, current_moet_price, current_minute
                )
                if opportunity:
                    opportunity.update({
                        'type': 'mint_arbitrage',
                        'pool_price': current_moet_price,
                        'peg_deviation': deviation_pct,
                        'stablecoin': stablecoin
                    })
                    if opportunity['expected_profit'] > 0:
                        print(f"   💰 Mint arbitrage opportunity: ${opportunity['expected_profit']:.2f} profit")
                    else:
                        print(f"   📊 Mint arbitrage checked: ${opportunity['expected_profit']:.2f} profit (unprofitable)")
                    return opportunity
            else:
                # MOET undervalued: Buy MOET from DEX → Redeem via Redeemer  
                opportunity = self._calculate_redeem_arbitrage(
                    pool, stablecoin, current_moet_price, current_minute
                )
                if opportunity:
                    opportunity.update({
                        'type': 'redeem_arbitrage',
                        'pool_price': current_moet_price,
                        'peg_deviation': deviation_pct,
                        'stablecoin': stablecoin
                    })
                    if opportunity['expected_profit'] > 0:
                        print(f"   💰 Redeem arbitrage opportunity: ${opportunity['expected_profit']:.2f} profit")
                    else:
                        print(f"   📊 Redeem arbitrage checked: ${opportunity['expected_profit']:.2f} profit (unprofitable)")
                    return opportunity
                    
        except Exception as e:
            print(f"   ❌ Error checking {stablecoin} pool arbitrage: {e}")
            return None
        
        return None
    
    def _calculate_optimal_trade_size(self, pool, stablecoin: str, current_price: float, arbitrage_type: str) -> float:
        """Calculate optimal trade size to restore MOET price to $1.00 using Uniswap V3 math"""
        target_price = 1.0  # MOET should be $1.00
        
        try:
            import math
            # Import Uniswap V3 math functions
            from ..core.uniswap_v3_math import (
                get_amount0_delta, get_amount1_delta, Q96, MIN_SQRT_RATIO, MAX_SQRT_RATIO
            )
            
            # Get actual active liquidity from the pool (not total pool size)
            if hasattr(pool, '_calculate_active_liquidity_from_ticks'):
                liquidity = pool._calculate_active_liquidity_from_ticks(pool.tick_current)
            elif hasattr(pool, 'liquidity'):
                liquidity = pool.liquidity
            else:
                # Fallback to estimated liquidity based on pool size
                liquidity = int(250_000 * 1e6)  # $250k pool scaled
            
            # Convert prices to sqrt_price_x96 format
            current_sqrt_price = int(math.sqrt(current_price) * Q96)
            current_sqrt_price = max(MIN_SQRT_RATIO, min(MAX_SQRT_RATIO, current_sqrt_price))
            
            target_sqrt_price = int(math.sqrt(target_price) * Q96)
            target_sqrt_price = max(MIN_SQRT_RATIO, min(MAX_SQRT_RATIO, target_sqrt_price))
            
            print(f"     🔍 Optimal sizing: current=${current_price:.6f}, target=${target_price:.6f}, liquidity={liquidity:,}")
            
            if arbitrage_type == "mint":
                # MOET overvalued: Need to sell MOET to lower price
                # This is a token0 -> token1 swap (MOET -> stablecoin)
                amount_moet_needed_scaled = get_amount0_delta(
                    target_sqrt_price, current_sqrt_price, liquidity, True
                )
                amount_needed = amount_moet_needed_scaled / 1e6  # Convert from scaled amount
                print(f"     💡 MINT: Need to sell ${amount_needed:.2f} MOET to restore peg")
                
            elif arbitrage_type == "redeem":
                # MOET undervalued: Need to buy MOET to raise price  
                # This is a token1 -> token0 swap (stablecoin -> MOET)
                amount_stablecoin_needed_scaled = get_amount1_delta(
                    current_sqrt_price, target_sqrt_price, liquidity, True
                )
                amount_needed = amount_stablecoin_needed_scaled / 1e6  # Convert from scaled amount
                print(f"     💡 REDEEM: Need to spend ${amount_needed:.2f} stablecoin to restore peg")
                
            else:
                return 10_000.0  # Fallback
            
            # Ensure reasonable bounds
            amount_needed = max(100.0, min(amount_needed, 100_000.0))  # Between $100 and $100k
            
            return amount_needed
            
        except Exception as e:
            print(f"     ❌ Error calculating optimal size: {e}")
            # Fallback to price-gap based estimation
            price_gap = abs(current_price - target_price)
            fallback_amount = min(price_gap * 50_000.0, 25_000.0)  # Scale by $50k per 1% gap, max $25k
            print(f"     🔄 Using fallback: ${fallback_amount:.2f}")
            return fallback_amount
    
    def _calculate_mint_arbitrage(self, pool, stablecoin: str, moet_price: float, current_minute: int) -> Optional[Dict]:
        """Calculate profitability of mint arbitrage (MOET overvalued)"""
        # Strategy: Mint MOET from Redeemer → Sell MOET on DEX for stablecoin
        
        # Calculate optimal trade size based on price deviation and pool mechanics
        optimal_trade_size = self._calculate_optimal_trade_size(pool, stablecoin, moet_price, "mint")
        available_balance = self.state.token_balances[Asset.USDC]
        
        # Execute arbitrage regardless of trade size (track P&L over time)
        
        # Use optimal size but respect balance limits (no artificial caps for arbitrage)
        max_trade = min(optimal_trade_size, available_balance)  # Use full balance if needed for arbitrage
        
        print(f"     🔍 Mint arbitrage calc: optimal=${optimal_trade_size:.2f}, max_trade=${max_trade:.2f}, MOET_price=${moet_price:.6f}")
        
        if max_trade < 100:  # Minimum $100 trade
            print(f"     ❌ Trade too small: ${max_trade:.2f}")
            return None
        
        # Estimate Redeemer minting cost (deposit fee)
        if stablecoin == "USDC":
            deposit_amount_usdc = max_trade
            deposit_amount_usdf = 0.0
        else:
            deposit_amount_usdc = 0.0
            deposit_amount_usdf = max_trade
        
        # Get minting cost estimate from protocol
        if not hasattr(self.engine, 'protocol'):
            return None
        
        try:
            # Estimate deposit fee for minting MOET
            mint_result = self.engine.protocol.mint_moet_from_deposit(deposit_amount_usdc, deposit_amount_usdf)
            if not mint_result['success']:
                return None
            
            moet_minted = mint_result['moet_minted']
            mint_fee = mint_result['total_fee']
            
            # Use actual pool swap calculation (like ALM rebalancer does)
            pool = self.engine.moet_usdc_pool if stablecoin == "USDC" else self.engine.moet_usdf_pool
            
            # Store original state
            original_sqrt_price = pool.sqrt_price_x96
            original_tick = pool.tick_current
            original_liquidity = pool.liquidity
            
            # Execute actual swap: MOET (token0) → USDC/USDF (token1)
            amount_in_scaled = int(moet_minted * 1e6)
            amount_in_actual, amount_out_actual = pool.swap(
                zero_for_one=True,  # MOET (token0) → stablecoin (token1)
                amount_specified=amount_in_scaled,
                sqrt_price_limit_x96=0  # No limit
            )
            
            # Convert back to USD amounts
            stablecoin_received = abs(amount_out_actual) / 1e6
            swap_fees = moet_minted * pool.fee_tier  # Calculate actual fees
            
            # Keep the new pool state (arbitrage should update pool prices permanently)
            
            # Calculate net profit
            total_cost = max_trade + mint_fee + swap_fees
            total_revenue = stablecoin_received
            net_profit = total_revenue - total_cost
            profit_percentage = net_profit / total_cost if total_cost > 0 else 0
            
            # Check if profitable (any positive profit)
            print(f"     💰 MINT Cost Breakdown:")
            print(f"       Initial investment: ${max_trade:.2f}")
            print(f"       Mint fees: ${mint_fee:.2f}")
            print(f"       DEX swap fees: ${swap_fees:.2f}")
            print(f"       Total cost: ${total_cost:.2f}")
            print(f"       MOET minted: {moet_minted:.2f}")
            print(f"       Stablecoin from DEX sale: ${stablecoin_received:.2f}")
            print(f"       Net profit: ${net_profit:.2f}")
            if net_profit <= 0:
                print(f"     ❌ Not profitable: ${net_profit:.2f}")
            
            return {
                'type': 'mint_arbitrage',
                'pool': stablecoin,
                'moet_price': moet_price,
                'trade_size': max_trade,
                'moet_minted': moet_minted,
                'mint_fee': mint_fee,
                'stablecoin_received': stablecoin_received,
                'swap_fees': swap_fees,
                'expected_profit': net_profit,
                'profit_percentage': profit_percentage,
                'current_minute': current_minute
            }
            
        except Exception as e:
            print(f"   ❌ Error calculating mint arbitrage: {e}")
            return None
    
    def _calculate_redeem_arbitrage(self, pool, stablecoin: str, moet_price: float, current_minute: int) -> Optional[Dict]:
        """Calculate profitability of redeem arbitrage (MOET undervalued)"""
        # Strategy: Buy MOET from DEX with stablecoin → Redeem MOET for stablecoin
        
        # Calculate optimal trade size based on price deviation and pool mechanics
        optimal_trade_size = self._calculate_optimal_trade_size(pool, stablecoin, moet_price, "redeem")
        available_balance = self.state.token_balances[Asset.USDC]
        
        # Execute arbitrage regardless of trade size (track P&L over time)
        
        # Use optimal size but respect balance limits (no artificial caps for arbitrage)
        max_trade = min(optimal_trade_size, available_balance)  # Use full balance if needed for arbitrage
        
        print(f"     🔍 Redeem arbitrage calc: optimal=${optimal_trade_size:.2f}, max_trade=${max_trade:.2f}, MOET_price=${moet_price:.6f}")
        
        if max_trade < 100:  # Minimum $100 trade
            print(f"     ❌ Trade too small: ${max_trade:.2f}")
            return None
        
        try:
            # Use actual pool swap calculation (like ALM rebalancer does)
            pool = self.engine.moet_usdc_pool if stablecoin == "USDC" else self.engine.moet_usdf_pool
            
            # Store original state
            original_sqrt_price = pool.sqrt_price_x96
            original_tick = pool.tick_current
            original_liquidity = pool.liquidity
            
            # Execute actual swap: USDC/USDF (token1) → MOET (token0)
            amount_in_scaled = int(max_trade * 1e6)
            amount_in_actual, amount_out_actual = pool.swap(
                zero_for_one=False,  # stablecoin (token1) → MOET (token0)
                amount_specified=amount_in_scaled,
                sqrt_price_limit_x96=0  # No limit
            )
            
            # Convert back to USD amounts
            moet_received = abs(amount_out_actual) / 1e6
            swap_fees = max_trade * pool.fee_tier  # Calculate actual fees
            
            # Keep the new pool state (arbitrage should update pool prices permanently)
            
            if moet_received <= 0:
                return None
            
            # Estimate Redeemer redemption proceeds
            if not hasattr(self.engine, 'protocol'):
                return None
            
            redemption_result = self.engine.protocol.redeem_moet_for_assets(moet_received, stablecoin)
            if not redemption_result['success']:
                return None
            
            stablecoin_redeemed = redemption_result['usdc_received'] if stablecoin == "USDC" else redemption_result['usdf_received']
            redemption_fee = redemption_result['total_fee']
            
            # Calculate net profit
            total_cost = max_trade + swap_fees + redemption_fee
            total_revenue = stablecoin_redeemed
            net_profit = total_revenue - total_cost
            profit_percentage = net_profit / total_cost if total_cost > 0 else 0
            
            # Check if profitable (any positive profit)
            print(f"     💰 REDEEM Cost Breakdown:")
            print(f"       Initial investment: ${max_trade:.2f}")
            print(f"       DEX swap fees: ${swap_fees:.2f}")
            print(f"       Redemption fees: ${redemption_fee:.2f}")
            print(f"       Total cost: ${total_cost:.2f}")
            print(f"       MOET received from DEX: {moet_received:.2f}")
            print(f"       Stablecoin from redemption: ${stablecoin_redeemed:.2f}")
            print(f"       Net profit: ${net_profit:.2f}")
            if net_profit <= 0:
                print(f"     ❌ Not profitable: ${net_profit:.2f}")
            
            return {
                'type': 'redeem_arbitrage',
                'pool': stablecoin,
                'moet_price': moet_price,
                'trade_size': max_trade,
                'moet_received': moet_received,
                'swap_fees': swap_fees,
                'stablecoin_redeemed': stablecoin_redeemed,
                'redemption_fee': redemption_fee,
                'expected_profit': net_profit,
                'profit_percentage': profit_percentage,
                'current_minute': current_minute
            }
            
        except Exception as e:
            print(f"   ❌ Error calculating redeem arbitrage: {e}")
            return None
    
    def _execute_arbitrage_decision(self, opportunity: Dict, current_minute: int) -> tuple:
        """Execute the arbitrage decision"""
        if opportunity['type'] == 'mint_arbitrage':
            return ("arbitrage_mint", opportunity)
        else:
            return ("arbitrage_redeem", opportunity)
    
    def execute_mint_arbitrage(self, params: dict, current_minute: int) -> bool:
        """Execute mint arbitrage: Mint MOET → Sell on DEX"""
        stablecoin = params['pool']
        trade_size = params['trade_size']
        
        print(f"🔄 {self.agent_id}: Executing mint arbitrage via {stablecoin} pool")
        print(f"   MOET price: ${params['moet_price']:.4f} (overvalued)")
        print(f"   Trade size: ${trade_size:,.0f}")
        
        try:
            # Step 1: Mint MOET from Redeemer
            if stablecoin == "USDC":
                mint_result = self.engine.protocol.mint_moet_from_deposit(trade_size, 0.0)
            else:
                mint_result = self.engine.protocol.mint_moet_from_deposit(0.0, trade_size)
            
            if not mint_result['success']:
                print(f"   ❌ Failed to mint MOET")
                self.state.failed_arbitrages += 1
                return False
            
            moet_minted = mint_result['moet_minted']
            mint_fee = mint_result['total_fee']
            
            # Update agent balances
            self.state.token_balances[Asset.USDC] -= trade_size  # Using USDC as proxy for both
            self.state.token_balances[Asset.MOET] += moet_minted
            
            print(f"   ✅ Step 1: Minted ${moet_minted:,.0f} MOET (fee: ${mint_fee:.2f})")
            
            # Step 2: Sell MOET on DEX
            if stablecoin == "USDC":
                stablecoin_received = self._swap_moet_to_usdc(moet_minted)
            else:
                stablecoin_received = self._swap_moet_to_usdf(moet_minted)
            
            if stablecoin_received <= 0:
                print(f"   ❌ Failed to sell MOET on DEX")
                self.state.failed_arbitrages += 1
                return False
            
            print(f"   ✅ Step 2: Sold MOET → ${stablecoin_received:,.0f} {stablecoin}")
            
            # Calculate actual profit
            total_cost = trade_size + mint_fee
            net_profit = stablecoin_received - total_cost
            
            # Record arbitrage event
            self._record_arbitrage_event(params, net_profit, current_minute, "mint_arbitrage")
            
            print(f"   📊 Arbitrage complete: Profit ${net_profit:.2f} ({(net_profit/total_cost)*100:.2f}%)")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Mint arbitrage error: {e}")
            self.state.failed_arbitrages += 1
            return False
    
    def execute_redeem_arbitrage(self, params: dict, current_minute: int) -> bool:
        """Execute redeem arbitrage: Buy MOET → Redeem for stablecoin"""
        stablecoin = params['pool']
        trade_size = params['trade_size']
        
        print(f"🔄 {self.agent_id}: Executing redeem arbitrage via {stablecoin} pool")
        print(f"   MOET price: ${params['moet_price']:.4f} (undervalued)")
        print(f"   Trade size: ${trade_size:,.0f}")
        
        try:
            # Step 1: Buy MOET from DEX
            if stablecoin == "USDC":
                moet_received = self._swap_usdc_to_moet(trade_size)
            else:
                moet_received = self._swap_usdf_to_moet(trade_size)
            
            if moet_received <= 0:
                print(f"   ❌ Failed to buy MOET from DEX")
                self.state.failed_arbitrages += 1
                return False
            
            print(f"   ✅ Step 1: Bought ${moet_received:,.0f} MOET with {stablecoin}")
            
            # Step 2: Redeem MOET via Redeemer
            redemption_result = self.engine.protocol.redeem_moet_for_assets(moet_received, stablecoin)
            
            if not redemption_result['success']:
                print(f"   ❌ Failed to redeem MOET")
                self.state.failed_arbitrages += 1
                return False
            
            stablecoin_redeemed = redemption_result['usdc_received'] if stablecoin == "USDC" else redemption_result['usdf_received']
            redemption_fee = redemption_result['total_fee']
            
            # Update agent balances
            self.state.token_balances[Asset.USDC] += stablecoin_redeemed  # Using USDC as proxy
            
            print(f"   ✅ Step 2: Redeemed MOET → ${stablecoin_redeemed:,.0f} {stablecoin} (fee: ${redemption_fee:.2f})")
            
            # Calculate actual profit
            total_cost = trade_size + redemption_fee
            net_profit = stablecoin_redeemed - total_cost
            
            # Record arbitrage event
            self._record_arbitrage_event(params, net_profit, current_minute, "redeem_arbitrage")
            
            print(f"   📊 Arbitrage complete: Profit ${net_profit:.2f} ({(net_profit/total_cost)*100:.2f}%)")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Redeem arbitrage error: {e}")
            self.state.failed_arbitrages += 1
            return False
    
    def _swap_moet_to_usdc(self, moet_amount: float) -> float:
        """Swap MOET to USDC using engine's MOET:USDC pool"""
        if not hasattr(self.engine, 'moet_usdc_calculator'):
            return 0.0
        
        try:
            swap_result = self.engine.moet_usdc_calculator.calculate_swap_slippage(moet_amount, "MOET")
            usdc_received = swap_result.get("amount_out", 0.0)
            
            # Update agent balances
            self.state.token_balances[Asset.MOET] -= moet_amount
            self.state.token_balances[Asset.USDC] += usdc_received
            
            # Update pool state
            self.engine.moet_usdc_calculator.update_pool_state(swap_result)
            
            return usdc_received
            
        except Exception as e:
            print(f"   ❌ MOET→USDC swap error: {e}")
            return 0.0
    
    def _swap_moet_to_usdf(self, moet_amount: float) -> float:
        """Swap MOET to USDF using engine's MOET:USDF pool"""
        if not hasattr(self.engine, 'moet_usdf_calculator'):
            return 0.0
        
        try:
            swap_result = self.engine.moet_usdf_calculator.calculate_swap_slippage(moet_amount, "MOET")
            usdf_received = swap_result.get("amount_out", 0.0)
            
            # Update agent balances
            self.state.token_balances[Asset.MOET] -= moet_amount
            self.state.token_balances[Asset.USDC] += usdf_received  # Using USDC as proxy
            
            # Update pool state
            self.engine.moet_usdf_calculator.update_pool_state(swap_result)
            
            return usdf_received
            
        except Exception as e:
            print(f"   ❌ MOET→USDF swap error: {e}")
            return 0.0
    
    def _swap_usdc_to_moet(self, usdc_amount: float) -> float:
        """Swap USDC to MOET using engine's MOET:USDC pool"""
        if not hasattr(self.engine, 'moet_usdc_calculator'):
            return 0.0
        
        try:
            swap_result = self.engine.moet_usdc_calculator.calculate_swap_slippage(usdc_amount, "USDC")
            moet_received = swap_result.get("amount_out", 0.0)
            
            # Update agent balances
            self.state.token_balances[Asset.USDC] -= usdc_amount
            self.state.token_balances[Asset.MOET] += moet_received
            
            # Update pool state
            self.engine.moet_usdc_calculator.update_pool_state(swap_result)
            
            return moet_received
            
        except Exception as e:
            print(f"   ❌ USDC→MOET swap error: {e}")
            return 0.0
    
    def _swap_usdf_to_moet(self, usdf_amount: float) -> float:
        """Swap USDF to MOET using engine's MOET:USDF pool"""
        if not hasattr(self.engine, 'moet_usdf_calculator'):
            return 0.0
        
        try:
            swap_result = self.engine.moet_usdf_calculator.calculate_swap_slippage(usdf_amount, "USDF")
            moet_received = swap_result.get("amount_out", 0.0)
            
            # Update agent balances
            self.state.token_balances[Asset.USDC] -= usdf_amount  # Using USDC as proxy
            self.state.token_balances[Asset.MOET] += moet_received
            
            # Update pool state
            self.engine.moet_usdf_calculator.update_pool_state(swap_result)
            
            return moet_received
            
        except Exception as e:
            print(f"   ❌ USDF→MOET swap error: {e}")
            return 0.0
    
    def _record_arbitrage_attempt(self, opportunity: dict, current_minute: int, executed: bool = False, actual_profit: float = 0.0):
        """Record ALL arbitrage attempts for comprehensive tracking"""
        attempt = {
            "minute": current_minute,
            "type": opportunity['type'],
            "pool": opportunity['pool'],
            "moet_price": opportunity['moet_price'],
            "trade_size": opportunity['trade_size'],
            "expected_profit": opportunity['expected_profit'],
            "executed": executed,
            "actual_profit": actual_profit if executed else 0.0,
            "profit_percentage": (actual_profit / opportunity['trade_size']) * 100 if opportunity['trade_size'] > 0 and executed else 0,
            "reason_not_executed": "unprofitable" if not executed else None
        }
        
        # Track ALL attempts
        self.state.arbitrage_attempts.append(attempt)
        self.state.total_attempts += 1
        
        # Track attempt types
        if opportunity['type'] == 'mint_arbitrage':
            self.state.total_mint_attempts += 1
        else:
            self.state.total_redeem_attempts += 1
        
        # Track volume (even for unsuccessful attempts)
        self.state.total_volume_traded += opportunity['trade_size']
        
        # Only record as event if executed
        if executed:
            self.state.arbitrage_events.append(attempt)
        self.state.total_profit += actual_profit
        self.state.last_arbitrage_minute = current_minute
        
        if actual_profit > 0:
            self.state.successful_arbitrages += 1
        else:
            self.state.failed_arbitrages += 1
    
    def _record_arbitrage_event(self, params: dict, actual_profit: float, current_minute: int, arb_type: str):
        """Legacy method - now calls enhanced tracking"""
        opportunity = {
            'type': arb_type,
            'pool': params['pool'],
            'moet_price': params['moet_price'],
            'trade_size': params['trade_size'],
            'expected_profit': params['expected_profit']
        }
        self._record_arbitrage_attempt(opportunity, current_minute, executed=True, actual_profit=actual_profit)
    
    def get_summary(self) -> dict:
        """Get comprehensive arbitrage agent performance summary"""
        total_executed = self.state.successful_arbitrages + self.state.failed_arbitrages
        
        return {
            # Legacy fields (for compatibility)
            "total_profit": self.state.total_profit,
            "successful_arbitrages": self.state.successful_arbitrages,
            "failed_arbitrages": self.state.failed_arbitrages,
            "total_arbitrage_events": len(self.state.arbitrage_events),
            
            # Enhanced tracking fields
            "total_attempts": self.state.total_attempts,
            "total_mint_attempts": self.state.total_mint_attempts,
            "total_redeem_attempts": self.state.total_redeem_attempts,
            "total_volume_traded": self.state.total_volume_traded,
            "total_fees_generated": self.state.total_fees_generated,
            
            # Calculated metrics
            "execution_rate": (total_executed / max(1, self.state.total_attempts)) * 100,
            "success_rate": (self.state.successful_arbitrages / max(1, total_executed)) * 100,
            "average_profit": self.state.total_profit / max(1, self.state.successful_arbitrages),
            "average_trade_size": self.state.total_volume_traded / max(1, self.state.total_attempts),
            
            # Detailed breakdown
            "attempts_breakdown": {
                "total": self.state.total_attempts,
                "mint": self.state.total_mint_attempts,
                "redeem": self.state.total_redeem_attempts,
                "executed": total_executed,
                "profitable": self.state.successful_arbitrages,
                "unprofitable": self.state.failed_arbitrages
            }
        }
    
    def get_detailed_portfolio_summary(self) -> dict:
        """Get detailed portfolio summary compatible with High Tide agent format"""
        summary = self.get_summary()
        
        # Add portfolio-style fields for compatibility
        summary.update({
            "agent_id": self.agent_id,
            "agent_type": "moet_arbitrage_agent",
            "initial_balance": self.state.initial_balance,
            "current_balance": self.state.token_balances.get(Asset.USDC, 0),
            "net_profit": self.state.total_profit,
            "total_trades": self.state.total_attempts,
            "active": self.active
        })
        
        return summary


def create_moet_arbitrage_agents(num_agents: int, initial_balance: float = 100_000.0) -> List[MoetArbitrageAgent]:
    """Create MOET arbitrage agents for peg maintenance"""
    agents = []
    
    for i in range(num_agents):
        agent_id = f"moet_arbitrage_{i+1}"
        agent = MoetArbitrageAgent(agent_id, initial_balance)
        agents.append(agent)
    
    return agents
