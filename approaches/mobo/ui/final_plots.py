import numpy as np
import matplotlib.pyplot as plt


def plot_pareto(Y_all, pareto_Y, out_png):
    plt.figure()
    plt.scatter(Y_all[:, 0], Y_all[:, 1], alpha=0.4, label="candidates")
    # Sort pareto by first objective for a clean line
    P = pareto_Y[np.argsort(pareto_Y[:, 0])]
    plt.plot(P[:, 0], P[:, 1], marker='o', linestyle='-', label="pareto")
    plt.xlabel("Power (min)")
    plt.ylabel("Delay (min)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)


def plot_attainment(pareto_history, out_png):
    """Empirical attainment: show several intermediate Pareto fronts.
    pareto_history: list of ndarray fronts (k_i, m), m=2
    """
    plt.figure()
    for i, F in enumerate(pareto_history):
        F = F[np.argsort(F[:, 0])]
        alpha = 0.2 + 0.6 * (i + 1) / len(pareto_history)
        label = "iter %d" % i if i in {0, len(pareto_history) - 1} else None
        plt.plot(F[:, 0], F[:, 1], marker='.', linestyle='-', alpha=alpha, label=label)
    plt.xlabel("Power (min)"); plt.ylabel("Delay (min)")
    if len(pareto_history) >= 2:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)