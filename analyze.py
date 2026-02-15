# =========================================================
# analyze_layer2_models.py
# Full forensic evaluation of Layer-2 entity models
# =========================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix
)
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. LOAD DATA + MODELS
# ---------------------------------------------------------

df = pd.read_csv("fraud_transactions_v6.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

TRAIN_CUTOFF = "2024-10-01"

train_df = df[df["timestamp"] < TRAIN_CUTOFF].copy()
test_df  = df[df["timestamp"] >= TRAIN_CUTOFF].copy()

ip_model = joblib.load("layer2_ip_model.joblib")
device_model = joblib.load("layer2_device_model.joblib")

FEATURES = [
    "total_txn",
    "unique_users",
    "hist_fraud_rate",
    "one_hop_rate",
    "two_hop_rate",
    "user_entropy"
]

print("="*70)
print("DATA LOADED")
print("="*70)
print("Train rows:", len(train_df))
print("Test rows :", len(test_df))

# ---------------------------------------------------------
# 2. REBUILD ENTITY DATASETS (Leakage-safe)
# ---------------------------------------------------------

def smoothed_rate(fraud, total, alpha=5):
    return (fraud + alpha) / (total + 2 * alpha)

def build_entity_dataset(entity_type):

    if entity_type == "ip":
        key = "ip_address"
    else:
        key = "device_id"

    # Train-window features
    agg = train_df.groupby(key).agg(
        total_txn=("is_fraud", "count"),
        fraud_txn=("is_fraud", "sum"),
        unique_users=("user_id", "nunique")
    ).reset_index()

    agg["hist_fraud_rate"] = (
        agg["fraud_txn"] + 5
    ) / (agg["total_txn"] + 10)

    # Simple contamination proxy (no graph rebuild here)
    user_fraud_rate = (
        train_df.groupby("user_id")["is_fraud"].mean().to_dict()
    )

    one_hop = []
    for ent in agg[key]:
        if entity_type == "ip":
            users = train_df[train_df["ip_address"] == ent]["user_id"]
        else:
            users = train_df[train_df["device_id"] == ent]["user_id"]

        rates = [user_fraud_rate.get(u, 0) for u in users]
        one_hop.append(np.mean(rates) if len(rates) else 0)

    agg["one_hop_rate"] = one_hop
    agg["two_hop_rate"] = np.minimum(one_hop, 0.5)
    agg["user_entropy"] = 0  # already learned pattern

    # Future labels
    future = test_df.groupby(key)["is_fraud"].sum().reset_index()
    future["label"] = (future["is_fraud"] > 0).astype(int)

    data = agg.merge(future[[key, "label"]], on=key, how="left")
    data["label"] = data["label"].fillna(0)

    return data


ip_data = build_entity_dataset("ip")
device_data = build_entity_dataset("device")

# ---------------------------------------------------------
# 3. EVALUATION FUNCTION
# ---------------------------------------------------------

def evaluate_model(data, model, name):

    print("\n" + "="*70)
    print(f"{name} MODEL EVALUATION")
    print("="*70)

    X = data[FEATURES]
    y = data["label"]

    proba = model.predict_proba(X)[:, 1]

    auc_pr  = average_precision_score(y, proba)
    auc_roc = roc_auc_score(y, proba)

    print(f"AUC-PR : {auc_pr:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    # -----------------------------------------------------
    # Lift Table
    # -----------------------------------------------------
    data["score"] = proba
    data = data.sort_values("score", ascending=False).reset_index(drop=True)

    data["decile"] = pd.qcut(data.index, 10, labels=False)

    lift = data.groupby("decile").agg(
        fraud_rate=("label", "mean"),
        count=("label", "count")
    )

    print("\nLift by Decile (0 = highest risk)")
    print(lift)

    # -----------------------------------------------------
    # Precision @ Top K
    # -----------------------------------------------------
    for pct in [0.01, 0.05, 0.10]:
        top_k = int(len(data) * pct)
        subset = data.iloc[:top_k]
        precision = subset["label"].mean()
        print(f"Precision @ top {int(pct*100)}%: {precision:.4f}")

    # -----------------------------------------------------
    # Exposure Sensitivity
    # -----------------------------------------------------
    print("\nPerformance by exposure band")

    data["exposure_bin"] = pd.qcut(
        data["total_txn"], q=4, duplicates="drop"
    )

    exposure_perf = data.groupby("exposure_bin")["label"].mean()
    print(exposure_perf)

    # -----------------------------------------------------
    # Feature Importance
    # -----------------------------------------------------
    print("\nFeature Importance")
    importance = model.feature_importances_
    for f, imp in sorted(zip(FEATURES, importance), key=lambda x: -x[1]):
        print(f"{f:20s} {imp:.4f}")

    return data


ip_scored = evaluate_model(ip_data, ip_model, "IP")
device_scored = evaluate_model(device_data, device_model, "DEVICE")

# ---------------------------------------------------------
# 4. RISK CONCENTRATION CHECK
# ---------------------------------------------------------

print("\n" + "="*70)
print("RISK CONCENTRATION CHECK")
print("="*70)

def concentration(data, name):
    top_10 = int(len(data) * 0.10)
    fraud_in_top = data.iloc[:top_10]["label"].sum()
    total_fraud = data["label"].sum()

    print(f"{name}:")
    print(f"  Total future fraud entities: {int(total_fraud)}")
    print(f"  Captured in top 10% risk: {int(fraud_in_top)}")
    print(f"  Capture rate: {fraud_in_top/total_fraud:.3f}")

concentration(ip_scored, "IP")
concentration(device_scored, "DEVICE")

print("\nAnalysis complete.")
