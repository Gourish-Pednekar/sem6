# =========================================================
# train.py — Heterogeneous GraphSAGE + LightGBM Hybrid
# =========================================================
# Architecture:
#   1. Build heterogeneous graph (user, device, ip nodes)
#   2. Train GraphSAGE to get node embeddings
#   3. Concatenate user embeddings with tabular features
#   4. Train LightGBM on the combined representation
#
# Node classification target: user nodes → is_fraud
# =========================================================

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix)
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =========================================================
# CONFIG
# =========================================================
CSV_PATH      = "fraud_transactions_v6.csv"
EMB_DIM       = 128
SAGE_HIDDEN   = 128
SAGE_LAYERS   = 2
SAGE_EPOCHS   = 50
SAGE_LR       = 0.005
BATCH_SIZE    = 2048
LGBM_ROUNDS   = 500
SEED          = 42

TABULAR_FEATURES = [
    "user_velocity_1h", "user_velocity_24h",
    "device_velocity_1h", "ip_velocity_1h",
    "is_new_device", "device_user_count_capped",
    "hour_of_day", "is_night", "is_weekend", "amount_inr"
]

# =========================================================
# 1. LOAD & PREP
# =========================================================
print("\n" + "=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
print(f"Loaded {len(df):,} transactions, fraud rate: {df['is_fraud'].mean():.4f}")

# Encode entity IDs to integer indices
user_enc   = LabelEncoder().fit(df["user_id"])
device_enc = LabelEncoder().fit(df["device_id"])
ip_enc     = LabelEncoder().fit(df["ip_address"])

df["user_idx"]   = user_enc.transform(df["user_id"])
df["device_idx"] = device_enc.transform(df["device_id"])
df["ip_idx"]     = ip_enc.transform(df["ip_address"])

n_users   = df["user_idx"].nunique()
n_devices = df["device_idx"].nunique()
n_ips     = df["ip_idx"].nunique()

print(f"Unique users: {n_users:,}  devices: {n_devices:,}  IPs: {n_ips:,}")

# User-level labels (a user is fraud if ANY of their txns is fraud)
user_labels = df.groupby("user_idx")["is_fraud"].max().reset_index()
user_labels = user_labels.sort_values("user_idx").reset_index(drop=True)
y_user = torch.tensor(user_labels["is_fraud"].values, dtype=torch.long)

# User tabular features (mean-aggregated per user)
user_tab = df.groupby("user_idx")[TABULAR_FEATURES].mean().reset_index()
user_tab = user_tab.sort_values("user_idx").reset_index(drop=True)
user_tab_np = user_tab[TABULAR_FEATURES].values.astype(np.float32)

# =========================================================
# 2. BUILD HETEROGENEOUS GRAPH
# =========================================================
print("\n" + "=" * 60)
print("STEP 2: Building heterogeneous graph")
print("=" * 60)

data = HeteroData()

# Node features
# Users: tabular features (normalized)
user_feat = torch.tensor(user_tab_np, dtype=torch.float)
user_feat = (user_feat - user_feat.mean(0)) / (user_feat.std(0) + 1e-8)
data["user"].x = user_feat
data["user"].y = y_user

# Devices: degree as feature (how many users share it)
device_degrees = df.groupby("device_idx")["user_idx"].nunique().reset_index()
device_degrees = device_degrees.sort_values("device_idx").reset_index(drop=True)
device_feat = torch.tensor(
    device_degrees["user_idx"].values.reshape(-1, 1).astype(np.float32)
)
data["device"].x = device_feat

# IPs: degree as feature
ip_degrees = df.groupby("ip_idx")["user_idx"].nunique().reset_index()
ip_degrees = ip_degrees.sort_values("ip_idx").reset_index(drop=True)
ip_feat = torch.tensor(
    ip_degrees["user_idx"].values.reshape(-1, 1).astype(np.float32)
)
data["ip"].x = ip_feat

# Edges: user ↔ device, user ↔ ip
edges_ud = df[["user_idx", "device_idx"]].drop_duplicates()
edges_ui = df[["user_idx", "ip_idx"]].drop_duplicates()

# user → device
data["user", "uses", "device"].edge_index = torch.tensor(
    edges_ud[["user_idx", "device_idx"]].values.T, dtype=torch.long
)
# device → user (reverse)
data["device", "used_by", "user"].edge_index = torch.tensor(
    edges_ud[["device_idx", "user_idx"]].values.T, dtype=torch.long
)
# user → ip
data["user", "connects", "ip"].edge_index = torch.tensor(
    edges_ui[["user_idx", "ip_idx"]].values.T, dtype=torch.long
)
# ip → user (reverse)
data["ip", "connected_by", "user"].edge_index = torch.tensor(
    edges_ui[["ip_idx", "user_idx"]].values.T, dtype=torch.long
)

print(f"Graph built:")
print(f"  user nodes:   {data['user'].x.shape[0]:,}")
print(f"  device nodes: {data['device'].x.shape[0]:,}")
print(f"  ip nodes:     {data['ip'].x.shape[0]:,}")
print(f"  user-device edges: {data['user','uses','device'].edge_index.shape[1]:,}")
print(f"  user-ip edges:     {data['user','connects','ip'].edge_index.shape[1]:,}")

# =========================================================
# 3. TRAIN/VAL/TEST SPLIT (temporal)
# =========================================================
print("\n" + "=" * 60)
print("STEP 3: Train/val/test split (temporal)")
print("=" * 60)

# Temporal split: train on first 8 months, val on month 9-10, test on 11-12
# This is critical for fraud — no future leakage
df_sorted = df.sort_values("timestamp")
n = len(df_sorted)
train_cutoff = df_sorted["timestamp"].quantile(0.67)
val_cutoff   = df_sorted["timestamp"].quantile(0.84)

train_users = set(df_sorted[df_sorted["timestamp"] <= train_cutoff]["user_idx"])
val_users   = set(df_sorted[(df_sorted["timestamp"] > train_cutoff) &
                             (df_sorted["timestamp"] <= val_cutoff)]["user_idx"])
test_users  = set(df_sorted[df_sorted["timestamp"] > val_cutoff]["user_idx"])

all_user_idx = np.arange(n_users)
train_mask = torch.tensor([i in train_users for i in all_user_idx], dtype=torch.bool)
val_mask   = torch.tensor([i in val_users   for i in all_user_idx], dtype=torch.bool)
test_mask  = torch.tensor([i in test_users  for i in all_user_idx], dtype=torch.bool)

data["user"].train_mask = train_mask
data["user"].val_mask   = val_mask
data["user"].test_mask  = test_mask

print(f"Train users: {train_mask.sum():,}  "
      f"Val: {val_mask.sum():,}  "
      f"Test: {test_mask.sum():,}")
print(f"Train fraud rate: {y_user[train_mask].float().mean():.4f}")
print(f"Val   fraud rate: {y_user[val_mask].float().mean():.4f}")
print(f"Test  fraud rate: {y_user[test_mask].float().mean():.4f}")

# =========================================================
# 4. GRAPHSAGE MODEL
# =========================================================
print("\n" + "=" * 60)
print("STEP 4: Training GraphSAGE")
print("=" * 60)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels_dict, hidden, out_dim, n_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # Input projection per node type (handles different feature sizes)
        self.proj = nn.ModuleDict({
            ntype: nn.Linear(in_ch, hidden)
            for ntype, in_ch in in_channels_dict.items()
        })

        for _ in range(n_layers):
            conv = SAGEConv(hidden, hidden)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden))

        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x_dict, edge_index_dict):
        # Project all node types to same hidden dim
        h = {ntype: F.relu(self.proj[ntype](x))
             for ntype, x in x_dict.items()}

        # Aggregate over all edge types for user nodes
        # (Simplified: mean of neighbor messages)
        for conv, bn in zip(self.convs, self.bns):
            h_new = {}
            for ntype in h:
                messages = [h[ntype]]  # self
                for (src, rel, dst), edge_index in edge_index_dict.items():
                    if dst == ntype and src in h:
                        src_h  = h[src]
                        dst_h  = h[dst]
                        # SAGEConv expects (x_src, x_dst)
                        agg = conv((src_h, dst_h), edge_index)
                        messages.append(agg)
                h_new[ntype] = torch.stack(messages, dim=0).mean(0)
                if ntype == "user":
                    h_new[ntype] = bn(h_new[ntype])
                h_new[ntype] = F.relu(h_new[ntype])
                h_new[ntype] = F.dropout(h_new[ntype], p=0.3, training=self.training)
            h = h_new

        return self.head(h["user"]), h["user"]  # logits, embeddings

# Class weights for imbalance
fraud_rate = y_user.float().mean().item()
pos_weight = torch.tensor([(1 - fraud_rate) / fraud_rate]).to(DEVICE)

in_channels = {
    "user":   data["user"].x.shape[1],
    "device": data["device"].x.shape[1],
    "ip":     data["ip"].x.shape[1],
}

model = GraphSAGE(in_channels, SAGE_HIDDEN, EMB_DIM, SAGE_LAYERS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=SAGE_LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, SAGE_EPOCHS)

# Move data to device
x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
y = data["user"].y.to(DEVICE)
train_mask = data["user"].train_mask.to(DEVICE)
val_mask   = data["user"].val_mask.to(DEVICE)
test_mask  = data["user"].test_mask.to(DEVICE)

best_val_auc = 0
best_embeddings = None
patience = 10
patience_ctr = 0

for epoch in range(1, SAGE_EPOCHS + 1):
    model.train()
    optimizer.zero_grad()

    logits, _ = model(x_dict, edge_index_dict)
    loss = F.binary_cross_entropy_with_logits(
        logits[train_mask, 0],
        y[train_mask].float(),
        pos_weight=pos_weight
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            logits, emb = model(x_dict, edge_index_dict)
            probs = torch.sigmoid(logits[:, 0])

            val_auc = roc_auc_score(
                y[val_mask].cpu().numpy(),
                probs[val_mask].cpu().numpy()
            )
            val_ap = average_precision_score(
                y[val_mask].cpu().numpy(),
                probs[val_mask].cpu().numpy()
            )

        print(f"  Epoch {epoch:3d} | loss: {loss.item():.4f} | "
              f"val AUC: {val_auc:.4f} | val AP: {val_ap:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_embeddings = emb.detach().cpu().numpy()
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

print(f"\nBest val AUC (GraphSAGE alone): {best_val_auc:.4f}")

# =========================================================
# 5. HYBRID: EMBEDDINGS + TABULAR → LIGHTGBM
# =========================================================
print("\n" + "=" * 60)
print("STEP 5: Training LightGBM on embeddings + tabular")
print("=" * 60)

# Combine embeddings with tabular features
emb_df  = pd.DataFrame(best_embeddings, columns=[f"emb_{i}" for i in range(EMB_DIM)])
tab_df  = pd.DataFrame(user_tab_np, columns=TABULAR_FEATURES)
X_full  = np.hstack([best_embeddings, user_tab_np])
y_full  = y_user.numpy()

train_idx = np.where(train_mask.cpu().numpy())[0]
val_idx   = np.where(val_mask.cpu().numpy())[0]
test_idx  = np.where(test_mask.cpu().numpy())[0]

X_train, y_train = X_full[train_idx], y_full[train_idx]
X_val,   y_val   = X_full[val_idx],   y_full[val_idx]
X_test,  y_test  = X_full[test_idx],  y_full[test_idx]

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

lgb_model = lgb.LGBMClassifier(
    n_estimators      = LGBM_ROUNDS,
    learning_rate     = 0.05,
    num_leaves        = 63,
    max_depth         = -1,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = scale_pos_weight,
    random_state      = SEED,
    n_jobs            = -1,
    verbose           = -1
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False),
               lgb.log_evaluation(100)]
)

# =========================================================
# 6. EVALUATION
# =========================================================
print("\n" + "=" * 60)
print("STEP 6: Final Evaluation on Test Set")
print("=" * 60)

# GraphSAGE alone
model.eval()
with torch.no_grad():
    logits, _ = model(x_dict, edge_index_dict)
    sage_probs = torch.sigmoid(logits[:, 0]).cpu().numpy()

sage_test_auc = roc_auc_score(y_test, sage_probs[test_idx])
sage_test_ap  = average_precision_score(y_test, sage_probs[test_idx])

# Hybrid
hybrid_probs = lgb_model.predict_proba(X_test)[:, 1]
hybrid_auc   = roc_auc_score(y_test, hybrid_probs)
hybrid_ap    = average_precision_score(y_test, hybrid_probs)

# Threshold at 0.5
hybrid_preds = (hybrid_probs >= 0.5).astype(int)

print(f"\n{'Model':<25} {'AUC':>8} {'AP':>8}")
print("-" * 42)
print(f"{'GraphSAGE (alone)':<25} {sage_test_auc:>8.4f} {sage_test_ap:>8.4f}")
print(f"{'Hybrid (SAGE + LGBM)':<25} {hybrid_auc:>8.4f} {hybrid_ap:>8.4f}")

print("\nHybrid Classification Report:")
print(classification_report(y_test, hybrid_preds, target_names=["legit", "fraud"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, hybrid_preds)
print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
print(f"  Precision@0.5: {cm[1,1]/(cm[1,1]+cm[0,1]+1e-9):.4f}")
print(f"  Recall@0.5:    {cm[1,1]/(cm[1,1]+cm[1,0]+1e-9):.4f}")

# Feature importance (tabular part)
feature_names = [f"emb_{i}" for i in range(EMB_DIM)] + TABULAR_FEATURES
imp = pd.DataFrame({
    "feature":    feature_names,
    "importance": lgb_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nTop 15 features (LightGBM):")
print(imp.head(15).to_string(index=False))

# Save artifacts
torch.save(model.state_dict(), "sage_model.pt")
lgb_model.booster_.save_model("lgbm_model.txt")
np.save("user_embeddings.npy", best_embeddings)
print("\n✅ Saved: sage_model.pt, lgbm_model.txt, user_embeddings.npy")
