"""Simple plotting helpers for sweep results."""
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def plot_energies_vs_param(results: List[Dict[str, Any]], states: int = 5):
    ps = [r["param"] for r in results]
    for i in range(states):
        ys = [r["energies"][i] for r in results]
        plt.plot(ps, ys, label=f'state {i}')
    plt.xlabel('parameter')
    plt.ylabel('energy')
    plt.legend()
    return plt.gcf()


def plot_observable_vs_param(results: List[Dict[str, Any]], key: str, states: int = 5):
    ps = [r["param"] for r in results]
    for i in range(states):
        ys = [r[key][i] for r in results]
        plt.plot(ps, ys, label=f'state {i}')
    plt.xlabel('parameter')
    plt.ylabel(key)
    plt.legend()
    return plt.gcf()


def plot_magnetisation_diagnostics(results: List[Dict[str, Any]], states: int = 5):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ps = [r["param"] for r in results]

    for i in range(states):
        axes[0].plot(ps, [r["magnetisation_z"][i] for r in results], label=f'state {i}')
        axes[0].plot(ps, [r["magnetisation_z_squared"][i] for r in results], linestyle='--', alpha=0.8)
    axes[0].set_ylabel('magnetisation diagnostics')
    axes[0].legend()

    if any("tracking_overlap" in r for r in results):
        overlaps = [r.get("tracking_overlap", [float('nan')])[0] for r in results]
        axes[1].plot(ps, overlaps, color='black')
        axes[1].set_ylabel('tracking overlap')

    axes[1].set_xlabel('parameter')
    return fig


def plot_entanglement_vs_energy(results: List[Dict[str, Any]], states: int = 5):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(states):
        xs = [r['energies'][i] for r in results]
        ys = [r['entanglement'][i] for r in results]
        ax.scatter(xs, ys, label=f'state {i}', alpha=0.7)
    ax.set_xlabel('energy')
    ax.set_ylabel('entanglement')
    ax.legend()
    return fig

