import torch

torch.manual_seed(0)
N, D = 200, 2
X = torch.randn(N, D)
w_true = torch.tensor([2.0, -3.0])
b_true = 0.5
y = (torch.sigmoid(X @ w_true + b_true) > 0.5).float()
#Parameters to learn (weights & bias)
w = torch.zeros(D, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
#Loss & optimizer
loss_fn = torch.nn.BCEWithLogitsLoss()
opt = torch.optim.SGD([w, b], lr=0.1)
#minimize loss to solve w&b
for epoch in range(100):
    opt.zero_grad()
    logits = X @ w + b                        
    loss = loss_fn(logits, y)                
    loss.backward()                             # ∂loss/∂w, ∂loss/∂b
    opt.step()                                  # gradient descent update

with torch.no_grad():
    print("Learned w:", w.numpy())
    print("Learned b:", b.item())
    preds = (torch.sigmoid(X @ w + b) > 0.5).float()
    acc = (preds == y).float().mean().item()
    print("Train accuracy:", acc, "Final loss:", loss.item())
