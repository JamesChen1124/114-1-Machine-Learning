# -*- coding: utf-8 -*-
"""
Titanic Binary Classification with PyTorch
-----------------------------------------
功能：
  - 從 seaborn-data 下載 titanic.csv
  - 清理/特徵工程（數值標準化、類別 one-hot、缺失值處理）
  - stratified train/val/test 切分（0.7/0.15/0.15）
  - MLP + BCEWithLogitsLoss（含 pos_weight 應對類別不平衡）
  - 指標：Accuracy / Precision / Recall / F1 / ROC-AUC
  - 圖表：學習曲線、ROC、混淆矩陣
  - 保存最佳權重至 ./outputs/titanic_binary_model.pt
"""
import os, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)
TITANIC_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

def load_titanic():
    df = pd.read_csv(TITANIC_URL)
    cols = ["survived","pclass","sex","age","sibsp","parch","fare","embarked"]
    df = df[cols].copy()
    df["age"] = df["age"].fillna(df["age"].median())
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
    df["sex"] = (df["sex"] == "male").astype(int)
    df["pclass"] = df["pclass"].astype("category")
    df["embarked"] = df["embarked"].astype("category")
    df = pd.get_dummies(df, columns=["pclass","embarked"], drop_first=True)
    y = df["survived"].astype(int).values
    X = df.drop(columns=["survived"]).values.astype(np.float32)
    return X, y, df.drop(columns=["survived"]).columns.tolist()

class MLP(nn.Module):
    def __init__(self, in_dim, h1=32, h2=16, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, out_dim)
        )
    def forward(self, x):
        return self.net(x)

def plot_loss(curve, title, path):
    plt.figure(); plt.plot(curve); plt.title(title); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.tight_layout(); plt.savefig(path); plt.close()

def plot_conf_mat(cm, classes, title, path):
    plt.figure(); plt.imshow(cm, interpolation="nearest"); plt.title(title)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(len(classes)), classes); plt.yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout(); plt.savefig(path); plt.close()

def plot_roc(fpr, tpr, title, path):
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1], linestyle="--")
    plt.title(title); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.tight_layout(); plt.savefig(path); plt.close()

def main():
    X, y, feat_names = load_titanic()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.1765, random_state=42, stratify=y_tr)  # ~0.7/0.15/0.15

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_te_s  = scaler.transform(X_te).astype(np.float32)

    y_tr_f = y_tr.reshape(-1,1).astype(np.float32)
    y_val_f = y_val.reshape(-1,1).astype(np.float32)
    y_te_f  = y_te.reshape(-1,1).astype(np.float32)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr_f)), batch_size=64, shuffle=True)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val_s), torch.from_numpy(y_val_f)), batch_size=256, shuffle=False)

    model = MLP(in_dim=X_tr_s.shape[1])
    n_pos = (y_tr == 1).sum(); n_neg = (y_tr == 0).sum()
    pos_weight = torch.tensor([ n_neg / max(1, n_pos) ], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val = float("inf"); best_state = None
    epochs = 1000
    tr_curve, val_curve = [], []

    for ep in range(1, epochs+1):
        model.train(); total=0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward(); optimizer.step()
            total += loss.item() * xb.size(0)
        tr_loss = total / len(train_loader.dataset); tr_curve.append(tr_loss)

        model.eval(); total=0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                total += loss.item() * xb.size(0)
        val_loss = total / len(val_loader.dataset); val_curve.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss; best_state = model.state_dict()

        if ep % 10 == 0:
            print(f"Epoch {ep:3d} | train {tr_loss:.4f}  val {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_te_s)).numpy().flatten()
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_te, preds)
    prec = precision_score(y_te, preds, zero_division=0)
    rec  = recall_score(y_te, preds, zero_division=0)
    f1   = f1_score(y_te, preds, zero_division=0)
    try:
        auc = roc_auc_score(y_te, probs)
    except Exception:
        auc = float("nan")
    print(f"Test -> Acc {acc:.4f}  Prec {prec:.4f}  Rec {rec:.4f}  F1 {f1:.4f}  ROC-AUC {auc:.4f}")

    plot_loss(tr_curve, "Binary Train Loss", "outputs/titanic_train_loss.png")
    plot_loss(val_curve, "Binary Val Loss", "outputs/titanic_val_loss.png")
    cm = confusion_matrix(y_te, preds)
    plot_conf_mat(cm, ["Not Survive (0)","Survive (1)"], "Confusion Matrix", "outputs/titanic_confusion_matrix.png")
    fpr, tpr, _ = roc_curve(y_te, probs)
    plot_roc(fpr, tpr, "ROC Curve", "outputs/titanic_roc.png")

    torch.save(model.state_dict(), "outputs/titanic_binary_model.pt")
    print("Saved charts & model under ./outputs")

if __name__ == "__main__":
    main()
