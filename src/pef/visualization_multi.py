import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    import scienceplots  # noqa: F401  # side effect on plt.style

    plt.style.use(["science", "ieee", "no-latex"])
except ImportError:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.figsize": (3.5, 2.5),
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )
plt.rcParams["text.usetex"] = False

COLORS = ["#2878B5", "#C82423", "#F8AC8C", "#6A5ACD"]
MARKERS = ["o", "s", "^", "D"]
OUTPUT_DIR = "results/figures"

DATASETS = [
    ("cifar10", "CIFAR-10"),
    ("mnist", "MNIST"),
    ("fashion_mnist", "Fashion-MNIST"),
]
THETA = 0.5


def _load(ds):
    path = os.path.join("results", ds, "experiment_log.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _round_start_mi(log):
    out = []
    for r_mi in log["mi_history"]:
        if isinstance(r_mi, list) and r_mi:
            out.append(r_mi[0][0] if isinstance(r_mi[0], list) else r_mi[0])
        else:
            out.append(0.0)
    return out


def plot_accuracy_comparison(logs):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, (ds, label) in enumerate(DATASETS):
        log = logs.get(ds)
        if log is None:
            continue
        acc = np.array(log["accuracy"])
        rounds = np.arange(1, len(acc) + 1)
        ax.plot(
            rounds, acc, color=COLORS[i], marker=MARKERS[i], markevery=2, linewidth=1.2, label=label
        )
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("(a) Accuracy Across Datasets")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/accuracy_comparison.pdf", format="pdf")
    plt.savefig(f"{OUTPUT_DIR}/accuracy_comparison.png", format="png", dpi=300)
    plt.close()


def plot_mi_comparison(logs):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, (ds, label) in enumerate(DATASETS):
        log = logs.get(ds)
        if log is None:
            continue
        mi = np.array(_round_start_mi(log))
        rounds = np.arange(1, len(mi) + 1)
        ax.plot(
            rounds, mi, color=COLORS[i], marker=MARKERS[i], markevery=2, linewidth=1.2, label=label
        )
    ax.axhline(y=THETA, color="gray", linestyle="--", linewidth=0.8, label=r"Threshold $\theta$")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel(r"R\'{e}nyi Mutual Information")
    ax.set_title("(b) MI Convergence Across Datasets")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/mi_comparison.pdf", format="pdf")
    plt.savefig(f"{OUTPUT_DIR}/mi_comparison.png", format="png", dpi=300)
    plt.close()


def write_summary(logs):
    rows = []
    for ds, label in DATASETS:
        log = logs.get(ds)
        if log is None:
            continue
        mi = _round_start_mi(log)
        rows.append(
            {
                "dataset": ds,
                "label": label,
                "final_accuracy": round(log["accuracy"][-1], 4),
                "rounds": len(log["accuracy"]),
                "final_mi": round(mi[-1], 4),
                "converged": bool(mi[-1] <= THETA + 1e-3),
                "all_verified": bool(all(log["verification"])),
                "avg_round_time": round(float(np.mean(log["round_time"])), 1),
            }
        )
    with open("results/cross_dataset_summary.json", "w") as f:
        json.dump({"theta": THETA, "datasets": rows}, f, indent=2)
    for r in rows:
        print(
            f"  {r['label']:16s} | acc={r['final_accuracy']} | "
            f"MI={r['final_mi']} | conv={r['converged']} | "
            f"verif={r['all_verified']} | {r['avg_round_time']}s/round"
        )


def plot_ablation():
    path = "results/ablation.json"
    if not os.path.exists(path):
        return
    with open(path) as f:
        ab = json.load(f)
    configs = ab["configs"]
    names = ["Full\n(Ours)", "w/o\nShapley", "w/o\nFolding", "w/o\nBoth"]
    acc = [c["accuracy"] for c in configs]
    mi = [c["residual_mi"] for c in configs]
    x = np.arange(len(configs))
    width = 0.38
    fig, ax1 = plt.subplots(figsize=(3.5, 2.5))
    b1 = ax1.bar(
        x - width / 2,
        acc,
        width,
        color=COLORS[0],
        edgecolor="black",
        linewidth=0.5,
        label="Accuracy",
    )
    ax1.set_ylabel("Test Accuracy")
    ax1.set_ylim(0.70, 0.82)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax2 = ax1.twinx()
    b2 = ax2.bar(
        x + width / 2,
        mi,
        width,
        color=COLORS[1],
        edgecolor="black",
        linewidth=0.5,
        label=r"Residual $\hat{I}_\alpha$",
    )
    ax2.axhline(y=THETA, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylabel(r"Residual MI $\hat{I}_\alpha$")
    ax2.set_ylim(0.0, 0.6)
    ax1.set_title("Component Ablation")
    ax1.legend(
        [b1, b2],
        ["Accuracy", r"Residual $\hat{I}_\alpha$"],
        loc="upper center",
        framealpha=0.9,
        ncol=2,
        fontsize=6,
    )
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/ablation.pdf", format="pdf")
    plt.savefig(f"{OUTPUT_DIR}/ablation.png", format="png", dpi=300)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logs = {ds: _load(ds) for ds, _ in DATASETS}
    available = [ds for ds in logs if logs[ds] is not None]
    print(f"Loaded datasets: {available}")
    plot_accuracy_comparison(logs)
    plot_mi_comparison(logs)
    plot_ablation()
    write_summary(logs)
    print("Cross-dataset figures + summary saved.")


if __name__ == "__main__":
    main()
