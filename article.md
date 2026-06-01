---
author: "Kyle Jones"
date_published: "July 17, 2025"
date_exported_from_medium: "November 10, 2025"
canonical_link: "https://medium.com/@kyle-t-jones/predicting-pipeline-burst-conditions-using-machine-learning-on-opensees-simulation-data-pvt-6e0edc8a9caa"
---

# Predicting Pipeline Burst Conditions Using Machine Learning on OpenSees Simulation Data (PVT) Physical pipelines can burst due to internal pressure. The conditions of
bursting can be studied in advance using simulation. This project...

### Predicting Pipeline Burst Conditions Using Machine Learning on OpenSees Simulation Data (PVT)
Physical pipelines can burst due to internal pressure. The conditions of bursting can be studied in advance using simulation. This project takes a simulated dataset of pipe response under temperature and pressure variations, and trains machine learning models to predict burst outcomes.

We used a dataset manually generated in [OpenSees](https://opensees.berkeley.edu/) and shared on [Kaggle](https://www.kaggle.com/code/ayomidezulkazeem/opensees-data-gen/input). This article shows how to turn that dataset into a working classification system to predict pipe burst conditions.

### The Business Case
Pipeline operators face constant risk from aging infrastructure, shifting soil, and uncertain environmental conditions. Predicting burst conditions is essential to prioritize maintenance, manage operating pressures, and plan retrofits.

This project shows how a small engineering dataset can support a predictive system. This is a fully simulated dataset, not real-world sensors or SCADA. The advantage is that you can simulate failure, model it, and build an interactive decision tool for your engineers.

Pressure, temperature, and volume (PVT) metrics are commonly used for pipeline integrity. Internal pressure fluctuates with flow rates and pump operations. Temperature shifts from seasonal change or fluid composition can cause expansion or contraction in the pipe wall. Volume changes reflect the pipe's deformation and stress response. Together, these variables influence burst risk by altering the mechanical balance inside the pipeline. Monitoring and modeling these interactions helps operators anticipate failure before it occurs and adjust operations within safe bounds.

### The Dataset
The dataset was generated using OpenSees to simulate the physical behavior of a pipe under variable temperature and pressure conditions. Each row records the outcome of a pipe segment at a specific node during a simulation cycle.

Key Columns

- `Temp`, `Prev Temp`: Environmental temperature conditions
- `Soil Modulus`: Stiffness of the surrounding soil
- `Pressure at Node`: Internal pressure
- `Spring Stiffness`: Simulated support from surroundings
- `Displacement`, `New Radius`, `New Volume`: Pipe response
- `Burst Occurred`: The binary target

Each row captures a physical state. Aggregating across nodes gives a full view of a simulation cycle.

### Analysis
We followed a standard supervised learning pipeline.

1.  [Group by simulation run and average values across nodes]
2.  [Create a clean feature matrix and target vector]
3.  [Train three models: Logistic Regression, Random Forest, and Neural Network]
4.  [Visualize results using confusion matrices, feature importances, and decision boundaries]
5.  [Augment the dataset synthetically to improve model generalization]

This lets us turn sparse simulation data into a usable predictive system. All the data is simulated which leads to unrealistically pristine results.

All models predicted burst risk with strong accuracy. Pressure at Node and Displacement were the strongest predictors. The Neural Network and Random Forest gave near-identical performance, though the tree model was easier to interpret.


Feature importance matches what we would expect.


The decision boundary plot illustrates how the model separates burst from non-burst conditions.


### Wrap up
This project shows how physics-based simulations can bootstrap predictive systems for infrastructure. With only simulated data, we trained reliable classifiers that predict burst failure. The next step is to connect this to real-time telemetry or build an interactive tool using Streamlit or Databricks.

Pipeline companies can use similar methods to:

- Screen for burst risk
- Prioritize repair schedules
- Set safe operating thresholds

As always, ML does not replace engineering judgment. It augments it.

### Full Code (Python)
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load and preprocess
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['Burst Occurred'] = df['Burst Occurred'].astype(int)
    df = df.drop(columns=[
        'Dummy', 'Simulation Cycle', 'Pipe Node', 
        'Pipe Burst Pressure', 'Constant Burst Pressure'
    ])
    grouped = df.groupby([
        'Temp', 'Prev Temp', 'Soil Modulus (E_soil)', 
        'Pressure at Node', 'Spring Stiffness', 'Burst Occurred'
    ]).mean().reset_index()
    return grouped

def enlarge_dataset(df, n=30, noise=0.02, seed=42):
    np.random.seed(seed)
    num_cols = df.select_dtypes(include=[np.number]).columns.drop('Burst Occurred')
    stds = df[num_cols].std()
    return pd.concat([
        df.assign(**{
            col: df[col] + np.random.normal(0, noise * stds[col], size=len(df))
            for col in num_cols
        })
        for _ in range(n)
    ], ignore_index=True)

def train_models(X_train, y_train):
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000).fit(X_train, y_train),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(20,), max_iter=1000, random_state=42).fit(X_train, y_train)
    }

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        results[name] = {
            'model': model,
            'report': classification_report(y_test, preds, output_dict=True),
            'confusion': confusion_matrix(y_test, preds)
        }
    return results

def plot_confusion_matrices(results):
    fig, axs = plt.subplots(1, len(results), figsize=(15, 4))
    for ax, (name, res) in zip(axs, results.items()):
        sns.heatmap(res['confusion'], annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=importances[sorted_idx],
        y=np.array(feature_names)[sorted_idx]
    )
    plt.title("Random Forest Feature Importances")
    plt.savefig('feature_importances.png')
    plt.show()
    return importances

def plot_decision_boundary(X, y, importances, feature_names):
    top2 = np.argsort(importances)[-2:]
    X_sub = X.iloc[:, top2]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_scaled, y)

    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    zz = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, zz, alpha=0.3, cmap="coolwarm")
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolor='k', cmap="coolwarm")
    plt.xlabel(feature_names[top2[0]])
    plt.ylabel(feature_names[top2[1]])
    plt.title("Decision Boundary (Top 2 Features)")
    plt.savefig('decision_boundary.png')
    plt.show()

# Run pipeline
df = load_and_prepare_data("Opensees_Pipe_Burst_Condition_Dataset.csv")
df_expanded = enlarge_dataset(df)
X = df_expanded.drop(columns='Burst Occurred')
y = df_expanded['Burst Occurred']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = train_models(X_train_scaled, y_train)
results = evaluate_models(models, X_test_scaled, y_test)

plot_confusion_matrices(results)
rf_importances = plot_feature_importance(results['Random Forest']['model'], X.columns)
plot_decision_boundary(X, y, rf_importances, X.columns)
```
