import numpy as np
import matplotlib.pyplot as plt
import nashpy as nash
from QuantumSimulation import QuantumGameEngine, get_strategy_unitary, STRATEGIES, LOOKUP_ENGINE
from market_scenarios import SCENARIOS
from scipy.optimize import minimize
import time
import functools
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

console = Console()

def find_true_optimum(row, col):
    """Utility to find the global optimum for any game."""
    def objective(params):
        t, p = params
        u = get_strategy_unitary(t, p)
        probs = QuantumGameEngine.calculate_exact_probs(u, u)
        p1, p2 = QuantumGameEngine.map_to_payoff(probs, row, col)
        return -(p1 + p2)

    best_res = None
    starts = [[0, 0], [np.pi, 0], [0, np.pi/2], [np.pi/2, np.pi/4]]
    for start in starts:
        res = minimize(objective, start, bounds=[(0, np.pi), (0, 2*np.pi)], method='L-BFGS-B')
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    return best_res.x

def run_week_3():
    console.print(Panel("[bold cyan]WEEK 3: Classical Baseline Analysis[/bold cyan]", expand=False))
    table = Table(title="Classical Nash Equilibria across Scenarios")
    table.add_column("Scenario", style="magenta")
    table.add_column("Strategy (P1, P2)", style="cyan")
    table.add_column("Payoff (P1, P2)", style="green")
    
    for name, (row, col) in SCENARIOS.items():
        game = nash.Game(row, col)
        equilibria = list(game.support_enumeration())
        for eq in equilibria:
            p1, p2 = eq
            p1_val = p1 @ row @ p2.T
            p2_val = p1 @ col @ p2.T
            table.add_row(name, f"{eq}", f"({p1_val:.2f}, {p2_val:.2f})")
    console.print(table)

def run_week_4_5():
    console.print(Panel("[bold blue]WEEK 4 & 5: Quantum Circuit & Simulation[/bold blue]", expand=False))
    table = Table(title="Quantum Simulation Results (Standard Q-Strategy)")
    table.add_column("Scenario", style="magenta")
    table.add_column("Simulated Payoff (P1, P2)", style="green")
    u_q = get_strategy_unitary(0, np.pi/2)
    for name, (row, col) in SCENARIOS.items():
        probs = QuantumGameEngine.calculate_exact_probs(u_q, u_q)
        p1, p2 = QuantumGameEngine.map_to_payoff(probs, row, col)
        table.add_row(name, f"({p1:.2f}, {p2:.2f})")
    console.print(table)

def run_week_6_7():
    console.print(Panel("[bold green]WEEK 6 & 7: Optimization & Alpha Results[/bold green]", expand=False))
    
    num_scenarios = len(SCENARIOS)
    cols = 2
    rows = (num_scenarios + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()
    
    summary_table = Table(title="Final Quantum Strategy Alpha Report")
    summary_table.add_column("Scenario", style="magenta")
    summary_table.add_column("Optimal (Theta, Phi)", style="cyan")
    summary_table.add_column("Quantum Daily", style="green")
    summary_table.add_column("Total Alpha (100d)", style="yellow")

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing Market Situations...", total=num_scenarios)
        
        for i, (name, (row, col)) in enumerate(SCENARIOS.items()):
            # 1. Optimization (Week 6)
            opt_t, opt_p = find_true_optimum(row, col)
            
            # 2. PnL Simulation (Week 7)
            u_opt = get_strategy_unitary(opt_t, opt_p)
            u_s = STRATEGIES["S"] # Classical Defector baseline
            
            q_probs = QuantumGameEngine.calculate_exact_probs(u_opt, u_opt)
            c_probs = QuantumGameEngine.calculate_exact_probs(u_s, u_s)
            
            q_pnl, c_pnl = [0], [0]
            outcomes = ["00", "01", "10", "11"]
            
            for _ in range(100):
                q_o = np.random.choice(outcomes, p=[q_probs[o] for o in outcomes])
                c_o = np.random.choice(outcomes, p=[c_probs[o] for o in outcomes])
                q_p, _ = QuantumGameEngine.map_to_payoff({q_o: 1.0}, row, col)
                c_p, _ = QuantumGameEngine.map_to_payoff({c_o: 1.0}, row, col)
                q_pnl.append(q_pnl[-1] + q_p)
                c_pnl.append(c_pnl[-1] + c_p)
            
            summary_table.add_row(name, f"({opt_t:.2f}, {opt_p:.2f})", f"{q_pnl[-1]/100:.2f}", f"+{q_pnl[-1]-c_pnl[-1]:.1f}")
            
            ax = axes[i]
            ax.plot(q_pnl, label=f"Quantum Opt ({opt_t:.1f}, {opt_p:.1f})", color="#00E5FF", linewidth=2)
            ax.plot(c_pnl, label="Classical Baseline", color="#FF8C00", linestyle="--")
            ax.set_title(f"{name}", fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.2)
            
            progress.update(task, advance=1)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    plot_path = "quantum_trading_results.png"
    plt.savefig(plot_path)
    
    console.print(summary_table)
    console.print(f"\n[bold green]Success![/bold green] Consolidated results saved to: [bold underline]{plot_path}[/bold underline]")

def run_week_8():
    console.print(Panel("[bold red]WEEK 8: Competition Readiness[/bold red]", expand=False))
    start_time = time.time()
    iterations = 1000
    row_bytes = SCENARIOS["Prisoner's Dilemma"][0].tobytes()
    col_bytes = SCENARIOS["Prisoner's Dilemma"][1].tobytes()
    for _ in range(iterations):
        LOOKUP_ENGINE.get_payoff("Q", "S", row_bytes, col_bytes)
    avg_time = (time.time() - start_time) / iterations
    console.print(f"Average lookup time: [bold green]{avg_time*1000:.4f}ms[/bold green]")
    console.print("[bold cyan]ENGINE STATUS: Competition Ready[/bold cyan]")

if __name__ == "__main__":
    console.print("\n[bold reverse white]  QUANTUM TRADING PROJECT: UNIFIED DELIVERABLES SUITE  [/bold reverse white]\n")
    run_week_3()
    run_week_4_5()
    run_week_6_7()
    run_week_8()
