
# -*- coding: utf-8 -*-
"""
California Housing Regression with PyTorch
  - 將圖檔與最佳模型權重存到 ./outputs/
"""
import math
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

def load_data(test_size=0.2, val_size=0.1, random_state=42):
    data = fetch_california_housing(as_frame=True)
    X = data.data.values.astype(np.float32)          # (N, 8)
    y = data.target.values.astype(np.float32).reshape(-1, 1)  # 房價中位數 (N,1)

    # 先切 train/test，再從 train 中切出 val
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    # 標準化（fit 只在訓練集，避免資料外洩）
    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train).astype(np.float32)
    X_val_s   = x_scaler.transform(X_val).astype(np.float32)
    X_test_s  = x_scaler.transform(X_test).astype(np.float32)

    y_scaler = StandardScaler().fit(y_train)
    y_train_s = y_scaler.transform(y_train).astype(np.float32)
    y_val_s   = y_scaler.transform(y_val).astype(np.float32)
    y_test_s  = y_scaler.transform(y_test).astype(np.float32)

    return (X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test_s, y_scaler, x_scaler)


class MLP(nn.Module):
    def __init__(self, in_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 24), nn.ReLU(),
            nn.Linear(24, 12), nn.ReLU(),
            nn.Linear(12, 6),  nn.ReLU(),
            nn.Linear(6, 1)  # 線性輸出（回歸）
        )
    def forward(self, x):
        return self.net(x)

def plot_curve(values, title, path):
    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_true_vs_pred(y_true, y_pred, title, path):
    plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    plt.title(title)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def plot_hist(values, title, path, bins=50):
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def main():
    (X_tr, y_tr, X_val, y_val, X_te, y_te, y_scaler, _) = load_data()

    batch_size = 10
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                              batch_size=256, shuffle=False)

    model = MLP(in_dim=X_tr.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 300

    best_val = float("inf")
    best_state = None
    train_curve, val_curve = [], []

    for epoch in range(1, epochs+1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(train_loader.dataset)
        train_curve.append(train_loss)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = criterion(pred, yb)
                val_total += loss.item() * xb.size(0)
        val_loss = val_total / len(val_loader.dataset)
        val_curve.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | train MSE={train_loss:.6f}  val MSE={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        y_pred_s = model(torch.from_numpy(X_te)).numpy()
    y_pred = y_scaler.inverse_transform(y_pred_s)
    y_true = y_scaler.inverse_transform(y_te)

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"Test MAE: {mae:.6f}  MSE: {mse:.6f}  RMSE: {rmse:.6f}  R2: {r2:.4f}")

    plot_curve(train_curve, "Train Loss (MSE, standardized target)", "outputs/california_train_loss.png")
    plot_curve(val_curve,   "Val Loss (MSE, standardized target)",   "outputs/california_val_loss.png")
    plot_true_vs_pred(y_true.flatten(), y_pred.flatten(),
                      "True vs Pred (Median House Value)", "outputs/california_true_vs_pred.png")
    residuals = y_true.flatten() - y_pred.flatten()
    plot_hist(residuals, "Residuals Histogram", "outputs/california_residuals_hist.png")

    torch.save(model.state_dict(), "outputs/california_regression_model.pt")
    print("Saved charts and model to ./outputs")

if __name__ == "__main__":
    main()
