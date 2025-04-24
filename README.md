# Gradient Boosting Classifier

This project demonstrates a complete from scratch implementation of a **Gradient Boosting Classifier** using Python and NumPy. It includes training, evaluating, and visualizing the classifier on various synthetic and real-world datasets.

---

## 🔍 Project Overview

- Built a binary classifier using **gradient boosting over decision trees**.
- Implemented core logic without using sklearn’s model implementations (but used sklearn for datasets and evaluation).
- Tested the model across **diverse datasets** including:
  - Moons
  - Circles
  - Linearly Separable Data
  - Gaussian Quantiles
  - High-Dimensional Noisy Data
  - Multi-label dataset (converted to binary)
  - Breast Cancer dataset
  - Iris dataset (Setosa vs Rest)

---

## 📊 Visualizations

We generate several types of visualizations:

### ✅ Prediction Performance

- **Green points**: Correct predictions
- **Red points**: Incorrect predictions

### ✅ Decision Boundary (2D datasets only)

- Left: Decision regions overlaid with **ground truth**
- Right: Same regions overlaid with **model predictions**
- ❌ Red circles highlight **misclassified points**

All visualizations are saved under the `plots/` folder.

---

## 📂 Folder Structure

```
.
├── model/
│   └── GradientBoosting.py         # Our custom model
├── plots/                          # Contains all visual output
├── tests/                          # Contains testing files
│   └── general_test.py             # Basic model testing
│   └── visualize_test_datasets.py  # Generates, Evaluates, Compares and Plots multiple datasets
└── README.md                       # You're reading this! :)
```

---

## 🚀 How to Run

### 1. Clone or download this project

```bash
git clone https://github.com/GovindKurapati/CS584-ML-Project2.git
cd CS584-ML-Project2
```

### 2. Create a Virtual Environment

```sh
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install Required Dependencies

```sh
pip install -r requirements.txt
```

### 4. Run the evaluation + visualization script

```bash
python GradientBoosting/tests/general_test.py
python GradientBoosting/tests/visualize_test_datasets.py
```

This will:

- Train your model on all datasets
- Evaluate it using Accuracy, Precision, Recall, F1 Score
- Generate prediction plots + decision boundaries
- Save everything inside the `plots/` directory

---

## 📈 Example Output

- `plots/Moons/moons_predictions.png`
- `plots/Circles/circles_decision_boundary_comparison.png`
- `plots/Breast Cancer/breast_cancer_predictions.png`

---

## Team Members

1. Krishna Ram Saravanakumar - A20578833 (ksaravanakumar@hawk.iit.edu)
2. Govind Kurapati - A20581868 (gkurapati@hawk.iit.edu)

---

## Assignment Questions & Answers

### 1. What does the model you have implemented do and when should it be used?

The model is a **Gradient Boosting Classifier** implemented from first principles. It builds an ensemble of weak learners (decision trees) in a stage-wise manner, optimizing the model to correct previous errors. It should be used when:

- You need to handle complex, non-linear datasets.
- You want a balance between accuracy and interpretability.
- You are working with tabular data (binary classification) and want to capture subtle patterns with boosting.

---

### 2. How did you test your model to determine if it is working reasonably correctly?

We tested the model on:

- A variety of **synthetic datasets** (moons, circles, Gaussian quantiles, linearly separable).
- **Real-world datasets** like Breast Cancer and Iris.
- **Noisy and high-dimensional** data to evaluate robustness.

For each dataset, we:

- Split the data into training and test sets.
- Trained the model and evaluated using **Accuracy, Precision, Recall, and F1 Score**.
- Visualized predictions and decision boundaries.
- Highlighted **misclassified examples** for manual inspection.

---

### 3. What parameters have you exposed to users of your implementation in order to tune performance?

The `GradientBoostingClassifier` supports:

| Parameter       | Description                                          |
| --------------- | ---------------------------------------------------- |
| `n_estimators`  | Number of boosting rounds (trees)                    |
| `learning_rate` | Shrinks the contribution of each tree (default: 0.1) |
| `max_depth`     | Depth of individual decision trees                   |

#### 🔧 Example usage:

```python
model = GradientBoostingClassifier(n_estimators=30, learning_rate=0.05, max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

### 4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

Yes. The current implementation can struggle with:

- **High-dimensional and sparse data**: Our from-scratch trees are simple and not fully optimized, which can slow down training.
- **Multi-class classification**: The model currently supports binary classification only.
- **Very large datasets**: Since the implementation lacks performance optimizations (e.g., feature subsampling, parallelism), it may be slow for massive datasets.

#### 🔧 With more time, improvements could include:

- Switching to decision tree classes with pruning and greedy split optimizations.
- Adding support for multi-class boosting (e.g., one-vs-rest).
- Implementing regularization techniques to improve generalization.

---
