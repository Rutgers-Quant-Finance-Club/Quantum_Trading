# `QuantumSimulation.py`

Small reference implementation of the **Eisert–Wilkens–Lewenstein (EWL)** entangled quantum game setup: two-player **Prisoner’s Dilemma** and **Chicken** with classical payoff matrices and quantum strategies from the literature.

## What it does

- Defines payoff matrices `PD_*` and `CH_*` with basis ordering  
  `|00⟩=(L,L)`, `|01⟩=(L,S)`, `|10⟩=(S,L)`, `|11⟩=(S,S)`.
- Builds the paper’s strategy unitaries `U(θ, φ)` and named strategies **L** (Long), **S** (Short), **Q** (quantum Long).
- Implements the entangling gate **J** and circuit `J → U₁⊗U₂ → J†` via `build_ewl_circuit`.
- Computes outcome probabilities with linear algebra (`outcome_probabilities`) and maps them to expected payoffs (`expected_payoffs`, `analyze_game`).
- On **run**, prints a few strategy comparisons and full **3×3** `{L,S,Q}` payoff tables, plus a sample **Q vs Q** circuit diagram.

## Requirements

Install from this folder:

```bash
pip install -r requirements.txt
```

Uses **NumPy** and **Qiskit** (circuit drawing for the demo at the bottom).

## Run

```bash
python QuantumSimulation.py
```

## Main API (importable)

| Piece | Role |
|--------|------|
| `U`, `LONG`, `SHORT`, `Q_LONG`, `STRATEGIES` | Strategy gates |
| `J`, `J_dag` | EWL entangler |
| `build_ewl_circuit(U1, U2, measure=...)` | Qiskit `QuantumCircuit` |
| `outcome_probabilities(U1, U2)` | `{"00":…,"01":…,"10":…,"11":…}` |
| `expected_payoffs(prob_dict, row_mat, col_mat)` | `(row_exp, col_exp)` |
| `analyze_game(...)`, `print_strategy_table(...)` | Console analysis helpers |
