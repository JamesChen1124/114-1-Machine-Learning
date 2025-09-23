import torch
import pandas as pd
import matplotlib.pyplot as plt
#讀csv
df = pd.read_csv("student_score.csv")
x = torch.tensor(df["hours"].values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(df["score"].values, dtype=torch.float32).unsqueeze(1)
#建模型 y_hat=w*x+b
model = torch.nn.Linear(1, 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
losses = []
for _ in range(100):
    y_hat = model(x)
    loss = loss_fn(y_hat, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
w = model.weight.item()
b = model.bias.item()
print(f"Learned line: y = {w:.2f}x + {b:.2f}")
#plot
plt.figure(figsize=(10,4))
# 左邊是回歸線
plt.subplot(1,2,1)
plt.scatter(x.numpy(), y.numpy(), label="data")
x_line = torch.linspace(x.min(), x.max(), 100).unsqueeze(1)
y_line = model(x_line).detach().numpy()
plt.plot(x_line.numpy(), y_line, "r", label="fit")
plt.xlabel("Hours"); plt.ylabel("Score")
plt.title("Linear Regression")
plt.legend()
# 右邊是loss 曲線
plt.subplot(1,2,2)
plt.plot(losses)
plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
plt.title("Loss Curve")
plt.tight_layout()
plt.show()

