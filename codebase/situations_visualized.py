import numpy as np
import matplotlib.pyplot as plt
from QuantumSimulation import QuantumGameEngine, get_strategy_unitary
from market_scenarios import SCENARIOS
from scipy.optimize import minimize
import os
from rich.console import Console

console = Console()

def find_optimal_strategy(row, col):
    """Finds the Quantum Nash Equilibrium / Optimal Strategy."""
    def objective(params):
        t, p = params
        u = get_strategy_unitary(t, p)
        probs = QuantumGameEngine.calculate_exact_probs(u, u)
        p1, p2 = QuantumGameEngine.map_to_payoff(probs, row, col)
        return -(p1 + p2)
    
    best_res = None
    # We prioritize the Q-strategy (0, pi/2) as a starting point if it's optimal
    starts = [[0, np.pi/2], [0, 0], [np.pi, 0], [np.pi/2, np.pi/4]]
    
    for start in starts:
        # Note: We still optimize over 2*pi to find the true global optimum, 
        # but we will only plot up to pi as requested.
        res = minimize(objective, start, bounds=[(0, np.pi), (0, 2*np.pi)], method='L-BFGS-B')
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    return best_res.x

def sweep_parameters(scenario_name, resolution=100):
    row, col = SCENARIOS[scenario_name]
    thetas = np.linspace(0, np.pi, resolution)
    phis = np.linspace(0, np.pi, resolution) # Changed x-axis range to [0, pi]
    payoff_grid = np.zeros((resolution, resolution))
    for i, t in enumerate(thetas):
        for j, p in enumerate(phis):
            u = get_strategy_unitary(t, p)
            probs = QuantumGameEngine.calculate_exact_probs(u, u)
            p1, p2 = QuantumGameEngine.map_to_payoff(probs, row, col)
            payoff_grid[i, j] = (p1 + p2) / 2
    return thetas, phis, payoff_grid

def plot_landscape(scenario_name):
    console.print(f"[bold yellow]Generating Strategy Landscape for {scenario_name}...[/bold yellow]")
    thetas, phis, grid = sweep_parameters(scenario_name)
    row, col = SCENARIOS[scenario_name]
    
    # Find the specific Optimal Strategy
    opt_t, opt_p = find_optimal_strategy(row, col)
    
    # If the optimal phi is > pi, we might need to wrap it or mention it, 
    # but the user specifically asked for the axis to be pi.
    
    plt.figure(figsize=(10, 8))
    # Heatmap
    cp = plt.contourf(phis, thetas, grid, levels=30, cmap='magma')
    plt.colorbar(cp, label='Symmetric Payoff')
    
    # Highlight ONLY the specific Optimal Strategy point (if it falls within [0, pi])
    if opt_p <= np.pi + 0.1:
        plt.scatter([opt_p], [opt_t], color='cyan', marker='*', s=300, 
                    label=f'Optimal Strategy ({opt_t:.2f}, {opt_p:.2f})', 
                    edgecolors='white', linewidths=1.5, zorder=5)

    plt.title(f"Market Landscape: {scenario_name}\n[X-axis: 0 to Pi]", fontsize=14, fontweight='bold')
    plt.xlabel("Phi (Phase Angle)", fontsize=12)
    plt.ylabel("Theta (Rotation Angle)", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.1)
    
    filename = f"landscape_{scenario_name.replace(' ', '_').lower().replace('.', '')}.png"
    plt.savefig(filename)
    plt.close()
    console.print(f"[bold green]Landscape saved to {filename}[/bold green]")

if __name__ == "__main__":
    for name in SCENARIOS.keys():
        plot_landscape(name)
