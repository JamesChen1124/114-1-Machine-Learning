# -*- coding: utf-8 -*-
# Iris multiclass classification with PyTorch
import os, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs("outputs", exist_ok=True)

# 1) 資料：Iris(150x4, 3類)  :contentReference[oaicite:0]{index=0}
iris = load_iris()
X, y = iris.data.astype(np.float32), iris.target.astype(np.int64)

# 分層切分 70/15/15（先切 test，再從 train 切 val）  :contentReference[oaicite:1]{index=1}
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.1765, random_state=42, stratify=y_tr)

# 標準化（僅以訓練集 fit）
scaler = StandardScaler().fit(X_tr)
X_tr = scaler.transform(X_tr).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
X_te  = scaler.transform(X_te ).astype(np.float32)

# Tensor / DataLoader
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=64, shuffle=False)

# 2) 模型：輸出 logits（不加 Softmax）；訓練用 CrossEntropyLoss  :contentReference[oaicite:2]{index=2}
class Net(nn.Module):
    def __init__(self, in_dim=4, h=32, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, h//2),   nn.ReLU(),
            nn.Linear(h//2, out_dim)  # logits
        )
    def forward(self, x): return self.net(x)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# 3) 訓練（以驗證損失挑最佳權重）
best_val, best_state = float("inf"), None
tr_curve, val_curve = [], []
epochs = 120
for ep in range(1, epochs+1):
    model.train(); total=0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward(); optimizer.step()
        total += loss.item()*xb.size(0)
    tr_loss = total/len(train_loader.dataset); tr_curve.append(tr_loss)

    model.eval(); total=0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            total += loss.item()*xb.size(0)
    val_loss = total/len(val_loader.dataset); val_curve.append(val_loss)

    if val_loss < best_val:
        best_val, best_state = val_loss, model.state_dict()

# 回滾最佳權重，於測試集評估
if best_state is not None: model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    logits = model(torch.from_numpy(X_te)).numpy()
preds = logits.argmax(axis=1)

acc = accuracy_score(y_te, preds)
print(f"Test Accuracy: {acc:.4f}")

# 每類 Precision/Recall/F1 與支援數  :contentReference[oaicite:3]{index=3}
print(classification_report(y_te, preds, target_names=iris.target_names, digits=4))

# 4) 圖表：學習曲線 + 混淆矩陣  :contentReference[oaicite:4]{index=4}
plt.figure(); plt.plot(tr_curve, label="train"); plt.plot(val_curve, label="val")
plt.title("Iris CrossEntropy Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.tight_layout(); plt.savefig("outputs/iris_loss_curves.png"); plt.close()

cm = confusion_matrix(y_te, preds)
plt.figure()
plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.xticks(range(3), iris.target_names); plt.yticks(range(3), iris.target_names)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.tight_layout(); plt.savefig("outputs/iris_confusion_matrix.png"); plt.close()

print("Saved charts under ./outputs")
