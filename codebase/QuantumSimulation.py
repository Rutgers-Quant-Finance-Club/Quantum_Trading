import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

# ============================================================
# 1) Classical payoff matrices from the paper
#    Basis ordering:
#       |00> = (Long,  Long)
#       |01> = (Long,  Short)
#       |10> = (Short, Long)
#       |11> = (Short, Short)
# ============================================================

# Prisoner's Dilemma
PD_ROW = np.array([
    [3, 0],
    [5, 1]
], dtype=float)

PD_COL = np.array([
    [3, 5],
    [0, 1]
], dtype=float)

# Chicken
CH_ROW = np.array([
    [2, 0],
    [3, -1]
], dtype=float)

CH_COL = np.array([
    [2, 3],
    [0, -1]
], dtype=float)

# ============================================================
# 2) Paper's strategy gates
# ============================================================

def U(theta: float, phi: float) -> np.ndarray:
    """
    Paper's two-parameter strategy family:
        U(theta, phi) = [[e^{i phi} cos(theta/2),  sin(theta/2)],
                         [-sin(theta/2),          e^{-i phi} cos(theta/2)]]
    """
    return np.array([
        [np.exp(1j * phi) * np.cos(theta / 2),  np.sin(theta / 2)],
        [-np.sin(theta / 2),                    np.exp(-1j * phi) * np.cos(theta / 2)]
    ], dtype=complex)

# Named strategies from the paper
LONG = U(0, 0)                 # Identity
SHORT = U(np.pi, 0)            # D = [[0,1],[-1,0]]
Q_LONG = U(0, np.pi / 2)       # quantum Long = [[i,0],[0,-i]]

STRATEGIES = {
    "L": LONG,
    "S": SHORT,
    "Q": Q_LONG
}

# ============================================================
# 3) EWL entangling gate J and its inverse J_dag
#    The paper defines:
#       J = exp(-i*pi/4 * (D ⊗ D))
#    where D = SHORT
# ============================================================

D = SHORT
I4 = np.eye(4, dtype=complex)
DD = np.kron(D, D)

# Because (D ⊗ D)^2 = I, exp(-i*pi/4 A) = cos(pi/4)I - i sin(pi/4)A
J = (1 / np.sqrt(2)) * (I4 - 1j * DD)
J_dag = J.conj().T

# Basis state |00>
ket00 = np.array([1, 0, 0, 0], dtype=complex)

# ============================================================
# 4) Build the actual quantum circuit (for visualization / execution)
# ============================================================

def build_ewl_circuit(U1: np.ndarray, U2: np.ndarray, measure: bool = False) -> QuantumCircuit:
    """
    EWL circuit:
        |00> -- J -- U1 ⊗ U2 -- J† -- measure
    """
    if measure:
        qc = QuantumCircuit(2, 2)
    else:
        qc = QuantumCircuit(2)

    qc.append(UnitaryGate(J, label="J"), [0, 1])
    qc.append(UnitaryGate(U1, label="U1"), [0])
    qc.append(UnitaryGate(U2, label="U2"), [1])
    qc.append(UnitaryGate(J_dag, label="J†"), [0, 1])

    if measure:
        qc.measure([0, 1], [0, 1])

    return qc

# ============================================================
# 5) Compute exact outcome probabilities
#    We use direct linear algebra to avoid bit-order confusion.
# ============================================================

def outcome_probabilities(U1: np.ndarray, U2: np.ndarray) -> dict:
    """
    Returns probabilities of |00>, |01>, |10>, |11>
    in the standard basis ordering.
    """
    full_strategy = np.kron(U1, U2)
    final_state = J_dag @ full_strategy @ J @ ket00
    probs = np.abs(final_state) ** 2

    return {
        "00": float(np.real_if_close(probs[0])),
        "01": float(np.real_if_close(probs[1])),
        "10": float(np.real_if_close(probs[2])),
        "11": float(np.real_if_close(probs[3]))
    }

# ============================================================
# 6) Convert outcome probabilities to expected payoffs
# ============================================================

def expected_payoffs(prob_dict: dict, row_payoff: np.ndarray, col_payoff: np.ndarray):
    """
    Maps:
        00 -> (L,L)
        01 -> (L,S)
        10 -> (S,L)
        11 -> (S,S)
    """
    row = (
        prob_dict["00"] * row_payoff[0, 0] +
        prob_dict["01"] * row_payoff[0, 1] +
        prob_dict["10"] * row_payoff[1, 0] +
        prob_dict["11"] * row_payoff[1, 1]
    )

    col = (
        prob_dict["00"] * col_payoff[0, 0] +
        prob_dict["01"] * col_payoff[0, 1] +
        prob_dict["10"] * col_payoff[1, 0] +
        prob_dict["11"] * col_payoff[1, 1]
    )

    return row, col

# ============================================================
# 7) One helper to analyze a given game and strategy pair
# ============================================================

def analyze_game(game_name: str, row_payoff: np.ndarray, col_payoff: np.ndarray,
                 s1_name: str, s1_gate: np.ndarray, s2_name: str, s2_gate: np.ndarray):
    probs = outcome_probabilities(s1_gate, s2_gate)
    row_exp, col_exp = expected_payoffs(probs, row_payoff, col_payoff)

    print(f"\n=== {game_name}: Player 1 = {s1_name}, Player 2 = {s2_name} ===")
    print("Outcome probabilities:")
    for outcome, p in probs.items():
        print(f"  P({outcome}) = {p:.4f}")
    print(f"Expected payoff -> Player 1: {row_exp:.4f}, Player 2: {col_exp:.4f}")

# ============================================================
# 8) Compare the paper's key cases
# ============================================================

# Prisoner's Dilemma
analyze_game("Prisoner's Dilemma", PD_ROW, PD_COL, "L", LONG, "L", LONG)
analyze_game("Prisoner's Dilemma", PD_ROW, PD_COL, "S", SHORT, "S", SHORT)
analyze_game("Prisoner's Dilemma", PD_ROW, PD_COL, "Q", Q_LONG, "Q", Q_LONG)

# Chicken
analyze_game("Chicken", CH_ROW, CH_COL, "L", LONG, "L", LONG)
analyze_game("Chicken", CH_ROW, CH_COL, "S", SHORT, "S", SHORT)
analyze_game("Chicken", CH_ROW, CH_COL, "Q", Q_LONG, "Q", Q_LONG)
analyze_game("Chicken", CH_ROW, CH_COL, "S", SHORT, "Q", Q_LONG)

# ============================================================
# 9) Optional: full 3x3 table for each game over {L, S, Q}
# ============================================================

def print_strategy_table(game_name: str, row_payoff: np.ndarray, col_payoff: np.ndarray):
    names = ["L", "S", "Q"]
    print(f"\n\n===== {game_name}: expected payoff table over {{L, S, Q}} =====")
    print("Each entry is (row payoff, col payoff)")
    for s1 in names:
        row_entries = []
        for s2 in names:
            probs = outcome_probabilities(STRATEGIES[s1], STRATEGIES[s2])
            r, c = expected_payoffs(probs, row_payoff, col_payoff)
            row_entries.append(f"{s1} vs {s2}: ({r:.2f}, {c:.2f})")
        print(" | ".join(row_entries))

print_strategy_table("Prisoner's Dilemma", PD_ROW, PD_COL)
print_strategy_table("Chicken", CH_ROW, CH_COL)

# ============================================================
# 10) If you want to see the actual circuit for Q vs Q:
# ============================================================

qc_demo = build_ewl_circuit(Q_LONG, Q_LONG, measure=False)
print("\nQuantum circuit for Q vs Q:")
print(qc_demo.draw())