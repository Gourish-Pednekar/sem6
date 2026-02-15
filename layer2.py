# =========================================================
# layer2_entity_model.py
# Infrastructure Risk Model (Leakage-Safe)
# =========================================================

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
import joblib
from collections import Counter
from scipy.stats import entropy

SEED = 42
np.random.seed(SEED)

# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("fraud_transactions_v6.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

TRAIN_CUTOFF = "2024-10-01"

train_df = df[df["timestamp"] < TRAIN_CUTOFF].copy()
test_df  = df[df["timestamp"] >= TRAIN_CUTOFF].copy()

print("Train rows:", len(train_df))
print("Test rows :", len(test_df))

# =========================================================
# 2. BUILD GRAPH (TRAIN WINDOW ONLY)
# =========================================================

G = nx.Graph()

for _, r in train_df.iterrows():
    u = f"user:{r.user_id}"
    d = f"device:{r.device_id}"
    ip = f"ip:{r.ip_address}"

    G.add_edge(u, d)
    G.add_edge(u, ip)

print("Graph nodes:", G.number_of_nodes())
print("Graph edges:", G.number_of_edges())

# =========================================================
# 3. ENTITY FEATURE COMPUTATION
# =========================================================

def smoothed_rate(fraud, total, alpha=5):
    return (fraud + alpha) / (total + 2 * alpha)

def entity_features(entity_type):

    if entity_type == "ip":
        key = "ip_address"
        prefix = "ip:"
    else:
        key = "device_id"
        prefix = "device:"

    # Basic counts
    agg = train_df.groupby(key).agg(
        total_txn=("is_fraud", "count"),
        fraud_txn=("is_fraud", "sum"),
        unique_users=("user_id", "nunique")
    ).reset_index()

    agg["hist_fraud_rate"] = agg.apply(
        lambda r: smoothed_rate(r.fraud_txn, r.total_txn), axis=1
    )

    # Graph-based contamination
    one_hop_rates = []
    two_hop_rates = []
    entropies = []

    for entity in agg[key]:

        node = prefix + str(entity)

        if node not in G:
            one_hop_rates.append(0)
            two_hop_rates.append(0)
            entropies.append(0)
            continue

        neighbors = list(G.neighbors(node))

        # 1-hop fraud
        neighbor_fraud = []
        user_ids = []

        for n in neighbors:
            if n.startswith("user:"):
                uid = n.split(":")[1]
                user_ids.append(uid)

                user_txns = train_df[train_df["user_id"] == uid]
                if len(user_txns) > 0:
                    rate = user_txns["is_fraud"].mean()
                    neighbor_fraud.append(rate)

        if neighbor_fraud:
            one_hop = np.mean(neighbor_fraud)
        else:
            one_hop = 0

        # 2-hop fraud (clipped)
        second_neighbors = set()
        for n in neighbors:
            second_neighbors.update(G.neighbors(n))

        second_neighbors.discard(node)

        second_rates = []
        for sn in second_neighbors:
            if sn.startswith("device:") or sn.startswith("ip:"):
                ent = sn.split(":")[1]
                sub = train_df[train_df[key] == ent]
                if len(sub) > 0:
                    second_rates.append(sub["is_fraud"].mean())

        if second_rates:
            two_hop = min(np.mean(second_rates), 0.5)
        else:
            two_hop = 0

        # entropy of user distribution
        if user_ids:
            counts = Counter(user_ids)
            probs = np.array(list(counts.values())) / sum(counts.values())
            entropies.append(entropy(probs))
        else:
            entropies.append(0)

        one_hop_rates.append(one_hop)
        two_hop_rates.append(two_hop)

    agg["one_hop_rate"] = one_hop_rates
    agg["two_hop_rate"] = two_hop_rates
    agg["user_entropy"] = entropies

    return agg


print("Computing IP features...")
ip_features = entity_features("ip")

print("Computing Device features...")
device_features = entity_features("device")

# =========================================================
# 4. FUTURE-WINDOW LABELS
# =========================================================

def future_labels(entity_type):

    if entity_type == "ip":
        key = "ip_address"
    else:
        key = "device_id"

    future = test_df.groupby(key)["is_fraud"].sum().reset_index()
    future["label"] = (future["is_fraud"] > 0).astype(int)

    return future[[key, "label"]]

ip_labels = future_labels("ip")
device_labels = future_labels("device")

ip_data = ip_features.merge(ip_labels, on="ip_address", how="left")
ip_data["label"] = ip_data["label"].fillna(0)

device_data = device_features.merge(device_labels, on="device_id", how="left")
device_data["label"] = device_data["label"].fillna(0)

# =========================================================
# 5. TRAIN ENTITY MODELS
# =========================================================

FEATURES = [
    "total_txn",
    "unique_users",
    "hist_fraud_rate",
    "one_hop_rate",
    "two_hop_rate",
    "user_entropy"
]

def train_entity_model(data, name):

    X = data[FEATURES]
    y = data["label"]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        eval_metric="logloss"
    )

    model.fit(X, y)

    proba = model.predict_proba(X)[:, 1]
    auc_pr = average_precision_score(y, proba)

    print(f"{name} AUC-PR:", round(auc_pr, 4))

    return model

print("Training IP model...")
ip_model = train_entity_model(ip_data, "IP")

print("Training Device model...")
device_model = train_entity_model(device_data, "Device")

# =========================================================
# 6. SAVE
# =========================================================

joblib.dump(ip_model, "layer2_ip_model.joblib")
joblib.dump(device_model, "layer2_device_model.joblib")

print("Layer-2 models saved.")
