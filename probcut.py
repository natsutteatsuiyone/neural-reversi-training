import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, r2_score
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train ProbCut models from CSV data')
parser.add_argument('csv_path', type=str, help='Path to the input CSV file')
args = parser.parse_args()

df = pd.read_csv(args.csv_path)

models = {}
epsilon = 1e-8

for ply, group in df.groupby('ply'):
    X = group[['shallow_depth', 'deep_depth']].values
    y = group['diff'].values

    # Split data (training/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ===== Model 1: Predicting mean error (diff) =====
    mean_model = LinearRegression()
    mean_model.fit(X_train, y_train)

    # Prediction and residual calculation for training data
    y_train_pred = mean_model.predict(X_train)
    residuals_train = y_train - y_train_pred

    # ===== Calculation of local mean absolute residuals (training data) =====
    k_neighbors_train = min(50, len(X_train))
    nbrs_train = NearestNeighbors(n_neighbors=k_neighbors_train, algorithm='auto').fit(X_train)
    distances_train, indices_train = nbrs_train.kneighbors(X_train)
    avg_abs_residuals_train = np.array([np.mean(np.abs(residuals_train[idx])) for idx in indices_train])
    # For normal distribution, E(|ε|) = σ√(2/π), so σ = (mean absolute residual)*√(π/2)
    std_targets_train = avg_abs_residuals_train * np.sqrt(np.pi / 2)
    log_std_targets_train = np.log(std_targets_train + epsilon)

    # ===== Model 2: Predicting log standard deviation =====
    std_model = LinearRegression()
    std_model.fit(X_train, log_std_targets_train)

    # Save models and test data to dictionary
    models[ply] = {
        'mean_model': mean_model,
        'std_model': std_model,
        'X_test': X_test,
        'y_test': y_test
    }

    print(f"Model construction completed for Ply {ply}.")

print("\n--- Evaluation on test data ---")
for ply, model_data in models.items():
    mean_model = model_data['mean_model']
    std_model = model_data['std_model']
    X_test = model_data['X_test']
    y_test = model_data['y_test']

    # Evaluate mean error model
    y_pred_mean = mean_model.predict(X_test)
    mse_mean = mean_squared_error(y_test, y_pred_mean)
    r2_mean = r2_score(y_test, y_pred_mean)
    print(f"Ply {ply} - Mean Model: MSE: {mse_mean:.3f}, R^2: {r2_mean:.3f}")

print("const PROBCUT_PARAMS: [ProbcutParams; 60] = [")

for ply in range(60):
    mean_model = models[ply]['mean_model']
    std_model = models[ply]['std_model']

    print("    ProbcutParams {")
    print(f"        mean_intercept: {mean_model.intercept_:.10},")
    print(f"        mean_coef_shallow: {mean_model.coef_[0]:.10},")
    print(f"        mean_coef_deep: {mean_model.coef_[1]:.10},")
    print(f"        std_intercept: {std_model.intercept_:.10},")
    print(f"        std_coef_shallow: {std_model.coef_[0]:.10},")
    print(f"        std_coef_deep: {std_model.coef_[1]:.10},")
    print("    },")

print("];")
