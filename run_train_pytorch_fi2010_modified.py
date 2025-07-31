import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# import json
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from datetime import datetime

# === File paths ===
snapshot_file = '../../datasets/Cong/BTCUSDT_orderbook_snapshot_2024_04_16-556.json'
trade_file = '../../datasets/Cong/BTCUSDT_trade_2024_04_16-886.json'

# === Load snapshot data ===
with open(snapshot_file, "r") as f:
    lines = f.readlines()
parsed_data = [json.loads(line) for line in lines]

snapshots = []
for entry in parsed_data:
    snapshots.append({
        "lastUpdateId": entry["lastUpdateId"],
        "asks": entry.get("asks", []),
        "bids": entry.get("bids", []),
        "timestamp": entry.get("record_utc_time")  # Optional: if available
    })

# === Load trade data ===
with open(trade_file, "r") as f:
    trade_lines = f.readlines()
trade_data = [json.loads(line) for line in trade_lines]
trade_records = [{"U": t["U"], "u": t["u"], "a": t["a"], "b": t["b"]} for t in trade_data]

# === Match snapshots to trades and attach trade_price ===
matched = []
for snap in tqdm(snapshots):
    snap_id = snap["lastUpdateId"]
    for t in trade_records:
        if t["U"] <= snap_id <= t["u"]:
            try:
                best_ask = float(t["a"][0][0])
                best_bid = float(t["b"][0][0])
                trade_price = (best_ask + best_bid) / 2
            except Exception:
                continue
            snap["trade_price"] = trade_price
            matched.append(snap)
            break

# === Feature engineering using snapshots ===
records = []
for snap in matched:
    row = {"timestamp": snap.get("timestamp")}
    asks = sorted([[float(p), float(v)] for p, v in snap["asks"]], key=lambda x: x[0])
    bids = sorted([[float(p), float(v)] for p, v in snap["bids"]], key=lambda x: -x[0])

    for i in range(10):
        row[f'ask_price_{i+1}'] = asks[i][0] if i < len(asks) else None
        row[f'ask_volume_{i+1}'] = asks[i][1] if i < len(asks) else None
        row[f'bid_price_{i+1}'] = bids[i][0] if i < len(bids) else None
        row[f'bid_volume_{i+1}'] = bids[i][1] if i < len(bids) else None

    row["mid_price"] = (row["ask_price_1"] + row["bid_price_1"]) / 2
    row["spread"] = row["ask_price_1"] - row["bid_price_1"]
    row["ask_depth"] = sum([row[f'ask_volume_{j+1}'] for j in range(10) if row[f'ask_volume_{j+1}'] is not None])
    row["bid_depth"] = sum([row[f'bid_volume_{j+1}'] for j in range(10) if row[f'bid_volume_{j+1}'] is not None])
    row["imbalance_5"] = (
        sum([row[f'bid_volume_{j+1}'] for j in range(5)]) -
        sum([row[f'ask_volume_{j+1}'] for j in range(5)])
    ) / (
        sum([row[f'bid_volume_{j+1}'] for j in range(5)]) +
        sum([row[f'ask_volume_{j+1}'] for j in range(5)]) + 1e-8
    )
    row["imbalance_10"] = (
        sum([row[f'bid_volume_{j+1}'] for j in range(10)]) -
        sum([row[f'ask_volume_{j+1}'] for j in range(10)])
    ) / (
        sum([row[f'bid_volume_{j+1}'] for j in range(10)]) +
        sum([row[f'ask_volume_{j+1}'] for j in range(10)]) + 1e-8
    )
    row["trade_price"] = snap["trade_price"]
    records.append(row)

df = pd.DataFrame(records)
df = df.dropna(subset=["ask_price_1", "bid_price_1", "trade_price"]).reset_index(drop=True)

# === Normalize features ===
price_volume_cols = [col for col in df.columns if col.startswith(('ask_', 'bid_')) or col in [
    "mid_price", "spread", "ask_depth", "bid_depth", "imbalance_5", "imbalance_10"
]]
scaler = StandardScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df[price_volume_cols]), columns=price_volume_cols)
df_norm["trade_price"] = df["trade_price"]
df_norm["mid_price_raw"] = df["mid_price"]
df_norm["timestamp"] = pd.to_datetime(df["timestamp"]) if "timestamp" in df else pd.NaT

# === Step-based labeling using trade price ===
step_horizons = [5, 10, 20, 30, 50]
epsilon = 0.0001
usable_df = df_norm.iloc[:-max(step_horizons)].copy()

def label_movement(p_now, p_future):
    delta = (p_future - p_now) / p_now
    if delta > epsilon:
        return 1
    elif delta < -epsilon:
        return 3
    else:
        return 2

for h in step_horizons:
    future_price = df_norm["trade_price"].shift(-h)
    usable_df[f"label_{h}"] = [
        label_movement(p, f) if pd.notna(f) else np.nan
        for p, f in zip(usable_df["trade_price"], future_price)
    ]

label_cols = [f"label_{h}" for h in step_horizons]
final_df = usable_df.dropna(subset=label_cols).reset_index(drop=True)

# === Coin Context Vector ===
df['log_return'] = np.log(df['mid_price']).diff()
coin_context = {
    "tick_size": 0.01,
    "spread_median": df['spread'].median(),
    "spread_std": df['spread'].std(),
    "ask_depth_mean": df['ask_depth'].mean(),
    "ask_depth_std": df['ask_depth'].std(),
    "bid_depth_mean": df['bid_depth'].mean(),
    "bid_depth_std": df['bid_depth'].std(),
    "imbalance_5_mean": df['imbalance_5'].mean(),
    "imbalance_5_std": df['imbalance_5'].std(),
}

for h in step_horizons:
    coin_context[f"realized_vol_{h}"] = df['log_return'].rolling(window=h).std().mean()
    coin_context[f"price_range_{h}"] = (df['mid_price'].rolling(window=h).max() - df['mid_price'].rolling(window=h).min()).mean()

coin_context_df = pd.DataFrame([coin_context])
context_scaler = StandardScaler()
context_numeric = coin_context_df.select_dtypes(include=[np.number])
context_normalized = pd.DataFrame(
    context_scaler.fit_transform(context_numeric),
    columns=context_numeric.columns
)

# === Merge context with final_df ===
context_broadcast = pd.concat([context_normalized] * len(final_df), ignore_index=True)
context_broadcast.index = final_df.index
final_df = pd.concat([final_df.drop(columns=label_cols), context_broadcast, final_df[label_cols]], axis=1)

final_df = final_df.drop(columns=["timestamp"])
print("Processed dataset shape:", final_df.shape)

final_df = final_df.T

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def prepare_x(data):
    df1 = data[:67, :].T
    return np.array(df1)

def get_label(data):
    lob = data[-5:, :].T
    return lob

def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX, dataY

def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = F.one_hot(y, num_classes=3)
    return x, y


class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, data, k, num_classes, T):
        """Initialization"""
        self.k = k
        self.num_classes = num_classes
        self.T = T

        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        y = y[:, self.k] - 1
        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]

# please change the data_path to your local path
# data_path = '/nfs/home/zihaoz/limit_order_book/data'

# dec_data = np.loadtxt('../data/data/Train_Dst_NoAuction_DecPre_CF_7.txt')
# dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.8))]
# dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.8)):]

# dec_test1 = np.loadtxt('../data/data/Test_Dst_NoAuction_DecPre_CF_7.txt')
# dec_test2 = np.loadtxt('../data/data/Test_Dst_NoAuction_DecPre_CF_8.txt')
# dec_test3 = np.loadtxt('../data/data/Test_Dst_NoAuction_DecPre_CF_9.txt')
# dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

# Split into training (80%), validation (10%), and test (10%) without shuffling (time series)
num_samples = len(final_df.T)
train_end = int(num_samples * 0.8)
val_end = int(num_samples * 0.9)
final_df_np = final_df.values

dec_train = final_df_np[:, :train_end]
dec_val = final_df_np[:, train_end:val_end]
dec_test = final_df_np[:, val_end:]

print(dec_train.shape, dec_val.shape, dec_test.shape)

batch_size = 64

dataset_train = Dataset(data=dec_train, k=4, num_classes=3, T=50)
dataset_val = Dataset(data=dec_val, k=4, num_classes=3, T=50)
dataset_test = Dataset(data=dec_test, k=4, num_classes=3, T=50)

train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

print(dataset_train.x.shape, dataset_train.y.shape)

tmp_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=1, shuffle=True)

for x, y in tmp_loader:
    print(x)
    print(y)
    print(x.shape, y.shape)
    break


class deeplob(nn.Module):
    def __init__(self, y_len):
        super().__init__()
        self.y_len = y_len

        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            #             nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # lstm layers
        self.lstm = nn.LSTM(input_size=1344, hidden_size=64, num_layers=1, batch_first=True)
        # self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x):
        batch_size = x.size(0)

        # Initial hidden and cell state for LSTM
        h0 = torch.zeros(1, batch_size, 64).to(x.device)
        c0 = torch.zeros(1, batch_size, 64).to(x.device)

        # Convolutional layers
        x = self.conv1(x)  # (B, C1, T, D)
        x = self.conv2(x)  # (B, C2, T, D)
        x = self.conv3(x)  # (B, C3, T, D)

        # Inception-like branches
        x_inp1 = self.inp1(x)  # e.g., 1x1 conv
        x_inp2 = self.inp2(x)  # e.g., 3x1 conv
        x_inp3 = self.inp3(x)  # e.g., 5x1 conv

        # Concatenate across the channel dimension
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)  # shape: (B, C_total, T, D)

        # Prepare input for LSTM
        x = x.permute(0, 2, 1, 3)  # (B, T, C, D)
        x = x.reshape(batch_size, x.shape[1], -1)  # (B, T, C*D)

        # LSTM expects input of shape (B, T, input_size)
        x, _ = self.lstm(x, (h0, c0))  # output: (B, T, hidden_size)

        # Use last time step's hidden state
        x = x[:, -1, :]  # (B, hidden_size)

        # Fully connected + softmax for classification
        x = self.fc1(x)  # (B, num_classes)
        forecast_y = torch.softmax(x, dim=1)

        return forecast_y

#     def forward(self, x):
#         # h0: (number of hidden layers, batch size, hidden size)
#         h0 = torch.zeros(1, x.size(0), 64).to(device)
#         c0 = torch.zeros(1, x.size(0), 64).to(device)

#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)

#         x_inp1 = self.inp1(x)
#         x_inp2 = self.inp2(x)
#         x_inp3 = self.inp3(x)

#         x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

# #         x = torch.transpose(x, 1, 2)
#         x = x.permute(0, 2, 1, 3)
#         x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

#         x, _ = self.lstm(x, (h0, c0))
#         x = x[:, -1, :]
#         x = self.fc1(x)
#         forecast_y = torch.softmax(x, dim=1)

#         return forecast_y

model = deeplob(y_len = dataset_train.num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


# A function to encapsulate the training loop
def batch_gd(model, criterion, optimizer, train_loader, test_loader, epochs):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in tqdm(range(epochs)):

        model.train()
        t0 = datetime.now()
        train_loss = []
        for inputs, targets in train_loader:
            # move data to GPU
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            # print("inputs.shape:", inputs.shape)
            # zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            # print("about to get model output")
            outputs = model(inputs)
            # print("done getting model output")
            # print("outputs.shape:", outputs.shape, "targets.shape:", targets.shape)
            loss = criterion(outputs, targets)
            # Backward and optimize
            # print("about to optimize")
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        # Get train loss and test loss
        train_loss = np.mean(train_loss)  # a little misleading

        model.eval()
        test_loss = []
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        if test_loss < best_test_loss:
            torch.save(model, './best_val_model_pytorch')
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')

        dt = datetime.now() - t0
        print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')

    return train_losses, test_losses

train_losses, val_losses = batch_gd(model, criterion, optimizer,
                                    train_loader, val_loader, epochs=50)

plt.figure(figsize=(15,6))
plt.plot(train_losses, label='train loss')
plt.plot(val_losses, label='validation loss')
plt.legend()

model = torch.load('best_val_model_pytorch')

n_correct = 0.
n_total = 0.
for inputs, targets in test_loader:
    # Move to GPU
    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

    # Forward pass
    outputs = model(inputs)

    # Get prediction
    # torch.max returns both max and argmax
    _, predictions = torch.max(outputs, 1)

    # update counts
    n_correct += (predictions == targets).sum().item()
    n_total += targets.shape[0]

test_acc = n_correct / n_total
print(f"Test acc: {test_acc:.4f}")

# model = torch.load('best_val_model_pytorch')
all_targets = []
all_predictions = []

for inputs, targets in test_loader:
    # Move to GPU
    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

    # Forward pass
    outputs = model(inputs)

    # Get prediction
    # torch.max returns both max and argmax
    _, predictions = torch.max(outputs, 1)

    all_targets.append(targets.cpu().numpy())
    all_predictions.append(predictions.cpu().numpy())

all_targets = np.concatenate(all_targets)
all_predictions = np.concatenate(all_predictions)

print('accuracy_score:', accuracy_score(all_targets, all_predictions))
print(classification_report(all_targets, all_predictions, digits=4))