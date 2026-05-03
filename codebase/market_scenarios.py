import numpy as np

# ============================================================
# Classical Payoff Matrices for various Market Scenarios
# ============================================================

# 1. Prisoner's Dilemma (Standard Benchmark)
# 0: Cooperate (L), 1: Defect (S)
PD_ROW = np.array([[3, 0], [5, 1]], dtype=float)
PD_COL = np.array([[3, 5], [0, 1]], dtype=float)

# 2. Chicken (Market Volatility / Volatility War)
# 0: Swerve (L), 1: Straight (S)
CH_ROW = np.array([[2, 0], [3, -1]], dtype=float)
CH_COL = np.array([[2, 3], [0, -1]], dtype=float)

# 3. Market Maker vs. Trader (Week 3 Deliverable)
# Player 1 (MM): 0=Tight Spread, 1=Wide Spread
# Player 2 (Trader): 0=Trade, 1=Wait
# (Tight, Trade) -> (1, 1) : Mutual benefit
# (Tight, Wait)  -> (-0.5, 0) : MM pays inventory cost
# (Wide, Trade)  -> (2, -1) : MM exploits Trader
# (Wide, Wait)   -> (0, 0) : No execution
MM_T_ROW = np.array([[1, -0.5], [2, 0]], dtype=float)
MM_T_COL = np.array([[1, 0], [-1, 0]], dtype=float)

# 4. Arbitrage Coordination (Stag Hunt style)
# Player 1 & 2: 0=Complex Arbitrage (Cooperate), 1=Simple Arbitrage (Alone)
# (Complex, Complex) -> (4, 4) : High Profit, High Risk
# (Complex, Alone)   -> (0, 2) : Risk failure, Alone succeeds
# (Alone, Complex)   -> (2, 0) : Alone succeeds, Risk failure
# (Alone, Alone)     -> (2, 2) : Safe, Low Profit
ARB_COORD_ROW = np.array([[4, 0], [2, 2]], dtype=float)
ARB_COORD_COL = np.array([[4, 2], [0, 2]], dtype=float)

# 5. Inverse Market (Anti-Cooperation)
# Player 1 & 2: 0=Long, 1=Short
# High payoff when both go Short (Market is crashing)
# (Long, Long)   -> (0, 0)
# (Long, Short)  -> (2, 2)
# (Short, Long)  -> (2, 2)
# (Short, Short) -> (5, 5)
INV_MKT_ROW = np.array([[0, 2], [2, 5]], dtype=float)
INV_MKT_COL = np.array([[0, 2], [2, 5]], dtype=float)

# 6. Liquidity Complementarity (Opposite Positions)
# Player 1 & 2: 0=Long, 1=Short
# High payoff when positions are opposite (complementary liquidity)
# (Long, Long)   -> (1, 1)
# (Long, Short)  -> (5, 5)
# (Short, Long)  -> (5, 5)
# (Short, Short) -> (1, 1)
LIQ_COMP_ROW = np.array([[1, 5], [5, 1]], dtype=float)
LIQ_COMP_COL = np.array([[1, 5], [5, 1]], dtype=float)

SCENARIOS = {
    "Prisoner's Dilemma": (PD_ROW, PD_COL),
    "Chicken": (CH_ROW, CH_COL),
    "Market Maker vs. Trader": (MM_T_ROW, MM_T_COL),
    "Arbitrage Coordination": (ARB_COORD_ROW, ARB_COORD_COL),
    "Inverse Market": (INV_MKT_ROW, INV_MKT_COL),
    "Liquidity Complementarity": (LIQ_COMP_ROW, LIQ_COMP_COL)
}
