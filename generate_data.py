"""
Generates a synthetic credit card transaction dataset for the fraud detection project.

Mirrors the structure of the well-known Kaggle "Credit Card Fraud Detection" dataset:
- Anonymized PCA-style features V1..V28
- Amount, Time
- Class (0 = legitimate, 1 = fraud), with a severe ~0.7% fraud rate

Run this once before opening fraud_detection.ipynb:
    python generate_data.py
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

np.random.seed(42)

n_samples = 50000
n_features = 28

X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=12,
    n_redundant=6,
    n_clusters_per_class=2,
    weights=[0.9935, 0.0065],
    flip_y=0.001,
    class_sep=1.1,
    random_state=42,
)

feature_names = [f"V{i+1}" for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)

df["Amount"] = np.round(np.random.exponential(scale=60, size=n_samples) + 1, 2)
df.loc[y == 1, "Amount"] = np.round(np.random.exponential(scale=120, size=(y == 1).sum()) + 1, 2)
df["Time"] = np.sort(np.random.randint(0, 172800, size=n_samples))
df["Class"] = y

df.to_csv("data/transactions.csv", index=False)

print("Saved data/transactions.csv")
print("Shape:", df.shape)
print("Fraud rate: {:.3f}%".format(100 * df["Class"].mean()))
