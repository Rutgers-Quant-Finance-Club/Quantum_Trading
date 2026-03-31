from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
import nashpy as nash

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

sampler = StatevectorSampler()
result = sampler.run([qc], shots=10000).result()

counts = result[0].data.meas.get_counts()
print(counts)

plot_histogram(counts)
plt.show()

#Classical Scenario of the Prisoner's Dilemma
prisoner1 = np.array([[-8, 0], [-10, -1]])
prisoner2 = np.array([[-8, -10], [0, -1]])

game_prisoner = nash.Game(prisoner1, prisoner2)
equilibria = list(game_prisoner.support_enumeration())
print("Nash Equilibrium for the Prisoners Dilemma:", equilibria)

#Classical Scenario of Chicken
chicken1 = np.array([[0, -1], [1, -10]])
chicken2 = np.array([[0, 1], [-1, -10]])

game_chicken = nash.Game(chicken1, chicken2)
equilibria = list(game_chicken.support_enumeration())
for eq in equilibria:
    print(f"Nash Equilibria for Chicken: {eq}")