import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
CSV_PATH = "ML_HW7/CSV_1__2_3__1.csv"  
SEQ_LEN = 48                 
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_EPOCHS = 200
LR = 1e-3

TARGET_YEAR = 2026          
TARGET_MONTH = 1               

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

df = pd.read_csv(CSV_PATH)
print("Columns:", df.columns)

needed_cols = ["Country/Area", "Month (abbr)", "Year", "Visitor Arrivals"]
df = df[needed_cols].copy()

df["Visitor Arrivals"] = (
    df["Visitor Arrivals"]
    .astype(str)           
    .str.replace(",", "") 
    .str.strip()         
)

df["Visitor Arrivals"] = pd.to_numeric(df["Visitor Arrivals"], errors="coerce")
df = df.dropna(subset=["Visitor Arrivals"])

month_map = {
    "Jan.": 1, "Feb.": 2, "Mar.": 3, "Apr.": 4,
    "May": 5, "Jun.": 6, "Jul.": 7, "Aug.": 8,
    "Sep.": 9, "Oct.": 10, "Nov.": 11, "Dec.": 12,

    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

df["Month (abbr)"] = df["Month (abbr)"].astype(str).str.strip()
df["month_num"] = df["Month (abbr)"].map(month_map)

df = df.dropna(subset=["month_num"])
df["month_num"] = df["month_num"].astype(int)

df["date"] = pd.to_datetime(
    dict(year=df["Year"].astype(int), month=df["month_num"], day=1)
)

monthly = (
    df.groupby("date")["Visitor Arrivals"]
    .sum()
    .sort_index()
)

print("Monthly series head:")
print(monthly.head())
print("Monthly series tail:")
print(monthly.tail())
print("月份區間：", monthly.index[0].strftime("%Y-%m"), "~", monthly.index[-1].strftime("%Y-%m"))
print("前幾個月總人數:", monthly.head().values)
print("最後幾個月總人數:", monthly.tail().values)

values = monthly.values.astype(np.float32)

data_min = values.min()
data_max = values.max()
scaled = (values - data_min) / (data_max - data_min + 1e-8)  # 避免除以 0

def create_sequences(series, seq_len):
    xs, ys = [], []
    for i in range(len(series) - seq_len):
        x = series[i : i + seq_len]
        y = series[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

X, y = create_sequences(scaled, SEQ_LEN)
print("X shape:", X.shape)  # (N, seq_len)
print("y shape:", y.shape)  # (N,)

if len(X) > 30:
    test_size = 12
else:
    test_size = max(1, len(X) // 5)

train_size = len(X) - test_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Train size: {train_size}, Test size: {test_size}")


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, seq_len) → (N, seq_len, 1)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        # y: (N,) → (N, 1)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = TimeSeriesDataset(X_train, y_train)
test_ds  = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)         
        last_hidden = out[:, -1, :]    
        out = self.fc(last_hidden)     
        return out

model = LSTMPredictor(
    input_size=1,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=1
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(model)

train_losses = []
test_losses = []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_x.size(0)

    train_loss /= len(train_loader.dataset)

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            test_loss += loss.item() * batch_x.size(0)

    test_loss /= len(test_loader.dataset)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.6f}  Test Loss: {test_loss:.6f}")


plt.figure()
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Train Loss")
plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Train vs Test Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


def multi_step_forecast(model, scaled_series, seq_len, steps, device):
    model.eval()
    history = scaled_series.copy()  
    preds_scaled = []

    with torch.no_grad():
        for _ in range(steps):
            last_seq = history[-seq_len:]  
            inp = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            pred = model(inp).cpu().item()  
            preds_scaled.append(pred)
            history = np.append(history, pred)  

    return np.array(preds_scaled, dtype=np.float32)


last_date = monthly.index[-1]
print("\n目前資料最後一筆月份:", last_date.strftime("%Y-%m"))

target_date = pd.Timestamp(year=TARGET_YEAR, month=TARGET_MONTH, day=1)
months_diff = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)

if months_diff <= 0:
    print(f"⚠️ 目前資料已經包含或超過 {target_date.strftime('%Y-%m')}，無需往前預測。")
    future_dates = []
    future_values = np.array([])
else:
    print(f"將往未來預測 {months_diff} 個月，直到 {target_date.strftime('%Y-%m')}。")

    future_scaled = multi_step_forecast(
        model=model,
        scaled_series=scaled,
        seq_len=SEQ_LEN,
        steps=months_diff,
        device=DEVICE
    )
    future_values = future_scaled * (data_max - data_min + 1e-8) + data_min

    future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(months_diff)]

    print("\n未來各月份預測的訪日總人數：")
    for d, v in zip(future_dates, future_values):
        print(f"{d.strftime('%Y-%m')}  ≈ {v:,.0f} 人")

    final_idx = future_dates.index(target_date)
    pred_target = future_values[final_idx]

    print("\n==========")
    print(f"預測 {target_date.strftime('%Y-%m')} 的訪日總人數 ≈ {pred_target:,.0f} 人")
    print("==========")


if len(future_dates) > 0:
    all_dates = list(monthly.index) + future_dates
    all_values = np.concatenate([values, future_values])

    plt.figure()
    plt.plot(monthly.index, values, label="History")
    plt.plot(future_dates, future_values, linestyle="--", marker="o", label="Forecast")
    plt.xlabel("Date")
    plt.ylabel("Visitor Arrivals to Japan")
    plt.title("Monthly Visitors: History + LSTM Forecast")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    years_to_show = 5  
    cutoff_date = last_date - pd.DateOffset(years=years_to_show)

    hist_mask = monthly.index >= cutoff_date
    plt.figure()
    plt.plot(monthly.index[hist_mask], values[hist_mask], label="History (last 4 years)")
    plt.plot(future_dates, future_values, linestyle="--", marker="o", label="Forecast")
    plt.xlabel("Date")
    plt.ylabel("Visitor Arrivals to Japan")
    plt.title("Last 4 Years + Forecast")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
