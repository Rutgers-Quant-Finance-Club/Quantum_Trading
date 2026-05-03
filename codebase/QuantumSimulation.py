import numpy as np
import functools
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator

# ============================================================
# Quantum Engine Components
# ============================================================

def get_strategy_unitary(theta: float, phi: float) -> np.ndarray:
    """The paper's two-parameter strategy family U(theta, phi)."""
    return np.array([
        [np.exp(1j * phi) * np.cos(theta / 2),  np.sin(theta / 2)],
        [-np.sin(theta / 2),                    np.exp(-1j * phi) * np.cos(theta / 2)]
    ], dtype=complex)

# Strategy Presets
STRATEGIES = {
    "L": get_strategy_unitary(0, 0),
    "S": get_strategy_unitary(np.pi, 0),
    "Q": get_strategy_unitary(0, np.pi / 2)
}

# EWL Operators
_D = STRATEGIES["S"]
_I4 = np.eye(4, dtype=complex)
_DD = np.kron(_D, _D)
J_GATE = (1 / np.sqrt(2)) * (_I4 - 1j * _DD)
J_DAG = J_GATE.conj().T
KET00 = np.array([1, 0, 0, 0], dtype=complex)

class QuantumGameEngine:
    """Core engine for EWL Quantum Game simulation."""
    
    @staticmethod
    def build_circuit(U1, U2, measure=True):
        qc = QuantumCircuit(2, 2 if measure else 0)
        qc.append(UnitaryGate(J_GATE, label="J"), [0, 1])
        qc.append(UnitaryGate(U1, label="U1"), [0])
        qc.append(UnitaryGate(U2, label="U2"), [1])
        qc.append(UnitaryGate(J_DAG, label="J†"), [0, 1])
        if measure: qc.measure([0, 1], [0, 1])
        return qc

    @staticmethod
    def calculate_exact_probs(U1, U2):
        full_strategy = np.kron(U1, U2)
        final_state = J_DAG @ full_strategy @ J_GATE @ KET00
        probs = np.abs(final_state) ** 2
        return {
            "00": float(np.real_if_close(probs[0])),
            "01": float(np.real_if_close(probs[1])),
            "10": float(np.real_if_close(probs[2])),
            "11": float(np.real_if_close(probs[3]))
        }

    @staticmethod
    def simulate_circuit(U1, U2, shots=1024):
        qc = QuantumGameEngine.build_circuit(U1, U2, measure=True)
        simulator = AerSimulator()
        tqc = transpile(qc, simulator)
        result = simulator.run(tqc, shots=shots).result()
        counts = result.get_counts()
        return {b: counts.get(b, 0) / shots for b in ["00", "01", "10", "11"]}

    @staticmethod
    def map_to_payoff(prob_dict, row_mat, col_mat):
        row = (prob_dict.get("00", 0) * row_mat[0,0] + prob_dict.get("01", 0) * row_mat[0,1] +
               prob_dict.get("10", 0) * row_mat[1,0] + prob_dict.get("11", 0) * row_mat[1,1])
        col = (prob_dict.get("00", 0) * col_mat[0,0] + prob_dict.get("01", 0) * col_mat[0,1] +
               prob_dict.get("10", 0) * col_mat[1,0] + prob_dict.get("11", 0) * col_mat[1,1])
        return row, col

class QuantumLookupEngine:
    """Pre-calculated lookup table for speed (Week 8)."""
    def __init__(self):
        self.table = {}
        for s1 in ["L", "S", "Q"]:
            for s2 in ["L", "S", "Q"]:
                self.table[(s1, s2)] = QuantumGameEngine.calculate_exact_probs(STRATEGIES[s1], STRATEGIES[s2])

    @functools.lru_cache(maxsize=128)
    def get_payoff(self, s1_name, s2_name, row_mat_bytes, col_mat_bytes):
        # Convert bytes back to numpy for hashable caching
        row_mat = np.frombuffer(row_mat_bytes).reshape(2, 2)
        col_mat = np.frombuffer(col_mat_bytes).reshape(2, 2)
        probs = self.table.get((s1_name, s2_name))
        return QuantumGameEngine.map_to_payoff(probs, row_mat, col_mat)

# Global Lookup Engine
LOOKUP_ENGINE = QuantumLookupEngine()