import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train ProbCut models from CSV data")
parser.add_argument("csv_path", type=str, help="Path to the input CSV file")
args = parser.parse_args()

df = pd.read_csv(args.csv_path)

models = {}
epsilon = 1e-8

X_train = df[["shallow_depth", "deep_depth"]].values
y_train = df["diff"].values

# ===== Model 1: Predicting mean error (diff) =====
mean_model = LinearRegression()
mean_model.fit(X_train, y_train)

# Prediction and residual calculation for training data
y_train_pred = mean_model.predict(X_train)
residuals_train = y_train - y_train_pred

# ===== Calculation of local mean absolute residuals (training data) =====
avg_abs_residuals_train = []
X_train_df = pd.DataFrame(X_train, columns=["shallow_depth", "deep_depth"])

for i in range(len(X_train)):
    current_shallow_depth = X_train[i, 0]
    current_deep_depth = X_train[i, 1]

    # Find indices of all points with the same shallow_depth and deep_depth
    neighbor_indices = X_train_df[
        (X_train_df["shallow_depth"] == current_shallow_depth)
        & (X_train_df["deep_depth"] == current_deep_depth)
    ].index.tolist()

    # Calculate average absolute residuals for these neighbors
    avg_abs_residual = np.mean(np.abs(residuals_train[neighbor_indices]))
    avg_abs_residuals_train.append(avg_abs_residual)

avg_abs_residuals_train = np.array(avg_abs_residuals_train)

# For normal distribution, E(|ε|) = σ√(2/π), so σ = (mean absolute residual)*√(π/2)
std_targets_train = avg_abs_residuals_train * np.sqrt(np.pi / 2)
log_std_targets_train = np.log(std_targets_train + epsilon)

# ===== Model 2: Predicting log standard deviation =====
std_model = LinearRegression()
std_model.fit(X_train, log_std_targets_train)

print("    ProbcutParams {")
print(f"        mean_intercept: {mean_model.intercept_:.10},")
print(f"        mean_coef_shallow: {mean_model.coef_[0]:.10},")
print(f"        mean_coef_deep: {mean_model.coef_[1]:.10},")
print(f"        std_intercept: {std_model.intercept_:.10},")
print(f"        std_coef_shallow: {std_model.coef_[0]:.10},")
print(f"        std_coef_deep: {std_model.coef_[1]:.10},")
print("    },")
