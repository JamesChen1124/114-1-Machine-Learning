import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
train_losses, train_accs = [], []
val_losses, val_accs = [], []

CSV_PATH = "phases.csv"         # 資料集
MAX_LEN = 50                    # 限制的字數

BATCH_SIZE = 32
EPOCHS = 100
EMBED_DIM = 64
HIDDEN_SIZE = 64
LR = 1e-3
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#讀資料跟取第一個標籤
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["text", "labels"])
df["text"] = df["text"].astype(str)

def get_primary_label(s):
    parts = str(s).split(";")
    first = parts[0].strip()
    return first if first != "" else None

df["primary_label"] = df["labels"].apply(get_primary_label)
df = df.dropna(subset=["primary_label"])
label_list = sorted(df["primary_label"].unique())
label2id = {lbl: i for i, lbl in enumerate(label_list)}
id2label = {i: lbl for lbl, i in label2id.items()}

df["label_id"] = df["primary_label"].map(label2id)
label_counts = df["label_id"].value_counts()
valid_labels = label_counts[label_counts >= 3].index

print("Total classes before filtering:", len(label_counts))
print("Classes kept (>=3 samples):", len(valid_labels))
df = df[df["label_id"].isin(valid_labels)].reset_index(drop=True)

num_classes = len(label_list)
print("Num classes:", num_classes)
print("Label mapping examples:", list(label2id.items())[:15])

counter = Counter()
for text in df["text"]:
    for ch in text:
        counter[ch] += 1

PAD_IDX = 0
UNK_IDX = 1

char2id = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
for i, (ch, _) in enumerate(counter.most_common(), start=2):
    char2id[ch] = i

vocab_size = len(char2id)
print("Vocab size:", vocab_size)

def encode_text(text, max_len=MAX_LEN):
    ids = []
    for ch in text:
        ids.append(char2id.get(ch, UNK_IDX))
        if len(ids) >= max_len:
            break
    if len(ids) < max_len:
        ids += [PAD_IDX] * (max_len - len(ids))
    return ids

df["input_ids"] = df["text"].apply(lambda x: encode_text(x, MAX_LEN))

#Train/Val/Test
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df["label_id"]
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=train_df["label_id"]
)

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))

# Dataset & DataLoader
class PhasesDataset(Dataset):
    def __init__(self, df_subset):
        self.inputs = df_subset["input_ids"].tolist()
        self.labels = df_subset["label_id"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_ids = torch.tensor(self.inputs[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x_ids, y

train_dataset = PhasesDataset(train_df)
val_dataset = PhasesDataset(val_df)
test_dataset = PhasesDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

#RNN model
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, pad_idx=0):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)  
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, hn = self.rnn(x, h0)          
        out = out[:, -1, :]                 
        logits = self.fc(out)               
        return logits

model = RNNClassifier(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    hidden_size=HIDDEN_SIZE,
    num_classes=num_classes,
    pad_idx=PAD_IDX
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == batch_y).sum().item()
        total_examples += batch_x.size(0)

    avg_loss = total_loss / total_examples
    acc = total_correct / total_examples
    print(f"[Epoch {epoch}] Train loss: {avg_loss:.4f}, acc: {acc:.4f}")
    return avg_loss, acc 
def eval_loader(dataloader, name="Val"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_examples += batch_x.size(0)

    avg_loss = total_loss / total_examples
    acc = total_correct / total_examples
    print(f"[{name}] loss: {avg_loss:.4f}, acc: {acc:.4f}")
    return avg_loss, acc

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(epoch)
    val_loss, val_acc = eval_loader(val_loader, name="Val")
    train_losses.append(tr_loss)
    train_accs.append(tr_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

# 畫 Loss 圖
epochs = range(1, EPOCHS + 1)

plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.close()

# 畫 Accuracy 圖
plt.figure()
plt.plot(epochs, train_accs, label="Train Acc")
plt.plot(epochs, val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_curve.png")
plt.close()
print("Saved loss_curve.png and accuracy_curve.png")

print("=== Final evaluation on TEST set ===")
test_loss, test_acc = eval_loader(test_loader, name="Test")
print(f"Test accuracy: {test_acc:.4f}")

torch.save(model.state_dict(), "HW6_model.pt")
print("HW6 Model saved: HW6_model.pt")