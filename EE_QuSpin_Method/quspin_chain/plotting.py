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

