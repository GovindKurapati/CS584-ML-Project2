import sys
import os
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model")))
from GradientBoosting import GradientBoostingClassifier


def generate_data():
    np.random.seed(42)
    mean0, cov0 = [0, 0], [[1, 0.3], [0.3, 1]]
    mean1, cov1 = [2, 2], [[1, -0.2], [-0.2, 1]]
    X0 = np.random.multivariate_normal(mean0, cov0, 50)
    X1 = np.random.multivariate_normal(mean1, cov1, 50)
    y0, y1 = np.zeros(50), np.ones(50)
    X = np.vstack((X0, X1))
    y = np.concatenate((y0, y1))
    return X, y


def test_basic_accuracy():
    X, y = generate_data()
    indices = np.random.permutation(len(X))
    train_idx, test_idx = indices[:80], indices[80:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=2)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    assert accuracy >= 0.8, f"Expected accuracy >= 0.8 but got {accuracy:.2f}"


def test_output_shape():
    X, y = generate_data()
    model = GradientBoostingClassifier(n_estimators=5)
    model.fit(X, y)
    proba = model.predict_probability(X)
    assert proba.shape == (X.shape[0],), "Probability output shape mismatch"


def test_prediction_range():
    X, y = generate_data()
    model = GradientBoostingClassifier(n_estimators=10)
    model.fit(X, y)
    probs = model.predict_probability(X)
    assert np.all((0 <= probs) & (probs <= 1)), "Predicted probabilities out of range"


def test_accuracy_on_moon_dataset():
    # Generate dataset
    X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Instantiate and train model
    model = GradientBoostingClassifier(n_estimators=30, learning_rate=0.1, max_depth=3)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model using multiple metrics:
    # 1. Accuracy: Overall, how often is the classifier correct?
    accuracy = accuracy_score(y_test, y_pred)

    # 2. Precision: Of all the predicted positive labels, how many were actually positive?
    precision = precision_score(y_test, y_pred)

    # 3. Recall: Of all actual positive labels, how many did the classifier correctly predict?
    recall = recall_score(y_test, y_pred)

    # 4. F1 Score: Harmonic mean of Precision and Recall. Balances both metrics.
    f1 = f1_score(y_test, y_pred)

    # 5. Confusion Matrix: Breakdown of prediction outcomes by class
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print evaluation results
    print(f"Model Evaluation on Moons Dataset:")
    print(f"Accuracy  : {accuracy:.2f}  # Overall correctness")
    print(
        f"Precision : {precision:.2f}  # True Positives / (True Positives + False Positives)"
    )
    print(
        f"Recall    : {recall:.2f}  # True Positives / (True Positives + False Negatives)"
    )
    print(f"F1 Score  : {f1:.2f}  # Balance between Precision and Recall")
    print("Confusion Matrix:")
    print(conf_matrix)


if __name__ == "__main__":
    test_basic_accuracy()
    test_output_shape()
    test_prediction_range()
    test_accuracy_on_moon_dataset()
    print("All tests passed.")
