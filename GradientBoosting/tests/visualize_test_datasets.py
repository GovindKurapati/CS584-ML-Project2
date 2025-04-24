import sys
import os
from sklearn.datasets import (
    make_moons,
    make_circles,
    make_classification,
    make_gaussian_quantiles,
    make_multilabel_classification,
    load_breast_cancer,
    load_iris,
)

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model")))
from GradientBoosting import GradientBoostingClassifier

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

# Define the save location for the plot
save_dir = os.path.join(ROOT_DIR, "plots")
os.makedirs(save_dir, exist_ok=True)

results = []


# Generate 4 custom datasets
def generate_datasets():
    datasets = {}

    # Moons
    X1, y1 = make_moons(n_samples=300, noise=0.25, random_state=42)
    datasets["Moons"] = (X1, y1)

    # Circles
    X2, y2 = make_circles(n_samples=300, noise=0.15, factor=0.5, random_state=42)
    datasets["Circles"] = (X2, y2)

    # Linearly separable
    X3, y3 = make_classification(
        n_samples=300,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42,
    )
    datasets["Linearly Separable"] = (X3, y3)

    # Overlapping Gaussian blobs
    X4, y4 = make_classification(
        n_samples=300,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=0.5,
        random_state=42,
    )
    datasets["Overlapping Blobs"] = (X4, y4)

    X5, y5 = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=2,
        n_redundant=5,
        flip_y=0.1,
        random_state=42,
    )
    datasets["High-Dimensional (noisy)"] = (X5, y5)
    # Gaussian Quantiles
    X6, y6 = make_gaussian_quantiles(
        n_samples=300, n_features=2, n_classes=2, random_state=42
    )
    datasets["Gaussian Quantiles"] = (X6, y6)
    # Multi-Label Extracted
    X7, y7 = (
        lambda: (
            make_multilabel_classification(
                n_samples=300, n_features=10, n_classes=3, random_state=42
            )[0],
            make_multilabel_classification(
                n_samples=300, n_features=10, n_classes=3, random_state=42
            )[1][:, 0],
        )
    )()
    datasets["Multi-Label Extracted"] = (X7, y7)
    # Breast Cancer
    X8, y8 = (load_breast_cancer().data[:300], load_breast_cancer().target[:300])
    datasets["Breast Cancer"] = (X8, y8)
    # Iris Binary (Setosa vs Rest)
    X9, y9 = (load_iris().data[:300], (load_iris().target[:300] == 2).astype(int))
    datasets["Iris Binary (Setosa vs Rest)"] = (X9, y9)

    return datasets


# Simple visualizer
def plot_dataset(X, y, title):
    cmap_bold = ["red", "green"]
    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(cmap_bold), edgecolor="k")
    plt.title(f"{title} Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.show()


def plot_predictions(X, y_true, y_pred, title):
    correct = y_true == y_pred
    cmap = ListedColormap(["green", "red"])  # green = correct, red = incorrect
    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(X[:, 0], X[:, 1], c=correct, cmap=cmap, edgecolor="k")
    plt.title(f"{title} Predictions\n(Green = Correct, Red = Incorrect)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    # plt.show()
    filename = f"{title.replace(' ', '_').lower()}_predictions.png"

    # Define the save location for the plot
    dataset_dir = os.path.join(save_dir, title)
    os.makedirs(dataset_dir, exist_ok=True)

    save_path = os.path.join(dataset_dir, filename)
    # plt.savefig(filename, dpi=300)
    plt.savefig(save_path)
    plt.close()


def plot_decision_boundary(model, X, y, title):
    if X.shape[1] != 2:
        return  # skip high-dimensional data

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_bg = ListedColormap(["#A0C4FF", "#FFADAD"])  # light blue and light red
    cmap_pts = ListedColormap(["#3B0A45", "#FFD60A"])  # dark purple and yellow

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pts, edgecolor="k", s=50)
    plt.title(f"Decision Boundary for {title} Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    # Define the save location for the plot
    dataset_dir = os.path.join(save_dir, title)
    os.makedirs(dataset_dir, exist_ok=True)
    save_path = os.path.join(
        dataset_dir, f"{title.replace(' ', '_').lower()}_decision_boundary.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_decision_boundary_comparison(model, X, y_true, y_pred, title):
    if X.shape[1] != 2:
        print(f"Skipping decision boundary plot for {title} (not 2D)")
        return

    # Create grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Background and point colors
    cmap_bg = ListedColormap(["#A0C4FF", "#FFADAD"])  # light blue / red
    cmap_pts = ListedColormap(["#3B0A45", "#FFD60A"])  # purple / yellow

    # Plot side-by-side: Ground truth and Predictions
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    for ax, label_set, subtitle in zip(
        axs, [y_true, y_pred], ["True Labels", "Predicted Labels"]
    ):
        ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.6)
        ax.scatter(X[:, 0], X[:, 1], c=label_set, cmap=cmap_pts, edgecolor="k", s=50)
        ax.set_title(f"{title} – {subtitle}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    # Overlay misclassified points in the second plot
    wrong = y_true != y_pred
    axs[1].scatter(
        X[wrong, 0],
        X[wrong, 1],
        edgecolor="red",
        facecolor="none",
        s=100,
        linewidths=2,
        label="Misclassified",
    )
    axs[1].legend(loc="upper right")

    # Finalize and save
    plt.suptitle(f"Decision Boundary Comparison: {title}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Define the save location for the plot
    dataset_dir = os.path.join(save_dir, title)
    os.makedirs(dataset_dir, exist_ok=True)
    save_path = os.path.join(
        dataset_dir, f"{title.replace(' ', '_').lower()}_boundary_comparison.png"
    )
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    # Generate and plot all datasets
    datasets = generate_datasets()
    for name, (X, y) in datasets.items():
        # plot_dataset(X, y, name)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"\n📊 Evaluation for {name} Dataset:")
        print(f"Accuracy : {acc:.2f}")
        print(f"Precision: {prec:.2f}")
        print(f"Recall   : {rec:.2f}")
        print(f"F1 Score : {f1:.2f}")

        # Visualize predictions on test set
        plot_predictions(X_test, y_test, y_pred, name)
        plot_decision_boundary(model, X_test, y_pred, name)
        plot_decision_boundary_comparison(model, X_test, y_test, y_pred, name)

        results.append(
            {
                "Dataset": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
            }
        )

    # Create a summary table
    df = pd.DataFrame(results)
    print("\n📋 Summary Table:")
    print(df.to_markdown(index=False))
