import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
from xgboost import XGBClassifier
import joblib

# ---------------------------------------------------------
# 1. LOAD & FEATURES
# ---------------------------------------------------------
df = pd.read_csv("indian_fraud_synthetic_v2.csv", parse_dates=["timestamp"])
df = df.copy()

# Feature engineering
df["hour"] = df["timestamp"].dt.hour
df["is_odd_hour"] = ((df["hour"] < 6) | (df["hour"] > 23)).astype(int)
df["log_amount"] = np.log1p(df["amount_inr"])
df["amount_bin"] = pd.qcut(df["amount_inr"], q=5, labels=False, duplicates='drop')
df["far_from_home"] = (df["distance_from_home_km"] > 500).astype(int)
df["log_distance"] = np.log1p(df["distance_from_home_km"])
df["high_velocity"] = (df["txn_count_1h"] >= 8).astype(int)
df["ultra_high_velocity"] = (df["txn_count_1h"] >= 12).astype(int)
df["suspicious_combo"] = ((df["far_from_home"] == 1) & (df["high_velocity"] == 1)).astype(int)
df["amount_velocity"] = df["amount_inr"] * df["txn_count_1h"]
df["distance_velocity"] = df["distance_from_home_km"] * df["txn_count_1h"]
df["speed"] = df["distance_from_home_km"] / (df["time_since_last_txn_min"] + 1)
df["risk_score"] = df["high_velocity"] * 2 + df["far_from_home"] * 1 + df["is_odd_hour"] * 1

FEATURES = [
    "log_amount", "amount_bin", "distance_from_home_km", "log_distance",
    "txn_count_1h", "time_since_last_txn_min", "is_odd_hour",
    "far_from_home", "high_velocity", "ultra_high_velocity",
    "suspicious_combo", "amount_velocity", "distance_velocity",
    "speed", "risk_score"
]

# Temporal split
train_df = df[df["timestamp"] < "2024-10-01"]
test_df = df[df["timestamp"] >= "2024-10-01"]

X_train = train_df[FEATURES]
y_train = train_df["is_fraud"]
X_test = test_df[FEATURES]
y_test = test_df["is_fraud"]

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# ---------------------------------------------------------
# 2. TRAIN MODEL
# ---------------------------------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.3,
    min_child_weight=3,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

y_proba = model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 3. BUSINESS STRATEGY SELECTION
# ---------------------------------------------------------
print("\n" + "="*70)
print("CHOOSE YOUR FRAUD DETECTION STRATEGY")
print("="*70)

strategies = {
    'Conservative (High Precision)': {
        'threshold': 0.70,
        'use_case': 'Automatic blocking - few false alarms',
        'target': 'Banks with low tolerance for customer complaints'
    },
    'Balanced (F1 Optimized)': {
        'threshold': 0.50,
        'use_case': 'Balanced detection - moderate review queue',
        'target': 'Most financial institutions'
    },
    'Aggressive (High Recall)': {
        'threshold': 0.40,
        'use_case': 'Flag for manual review - catch more fraud',
        'target': 'High-risk environments or new systems'
    }
}

results = {}
for name, config in strategies.items():
    threshold = config['threshold']
    pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred)
    
    rec = cm[1,1] / (cm[1,0] + cm[1,1])
    prec = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    fa_rate = cm[0,1] / (cm[0,0] + cm[0,1])
    
    results[name] = {
        'threshold': threshold,
        'recall': rec,
        'precision': prec,
        'f1': f1,
        'fp': cm[0,1],
        'tp': cm[1,1],
        'fn': cm[1,0],
        'false_alarm_rate': fa_rate
    }
    
    print(f"\n{name} (Threshold: {threshold:.2f})")
    print(f"  Use Case: {config['use_case']}")
    print(f"  Target: {config['target']}")
    print(f"  ─────────────────────────────────")
    print(f"  Fraud Caught:     {cm[1,1]:,} / {cm[1,0] + cm[1,1]:,} ({rec:.1%})")
    print(f"  Precision:        {prec:.1%}")
    print(f"  F1-Score:         {f1:.1%}")
    print(f"  False Alarms:     {cm[0,1]:,} ({fa_rate:.1%})")
    print(f"  Manual Reviews:   {cm[0,1] + cm[1,1]:,} transactions/day")

# ---------------------------------------------------------
# 4. FRAUD TYPE BREAKDOWN
# ---------------------------------------------------------
print("\n" + "="*70)
print("FRAUD TYPE DETECTION (at each threshold)")
print("="*70)

for strategy_name, result in results.items():
    threshold = result['threshold']
    pred = (y_proba >= threshold).astype(int)
    
    test_results = test_df.copy()
    test_results["pred"] = pred
    
    fraud_analysis = test_results[test_results["is_fraud"] == 1].groupby("fraud_type").agg({
        "pred": ["sum", "count"]
    })
    fraud_analysis.columns = ["caught", "total"]
    fraud_analysis["rate"] = fraud_analysis["caught"] / fraud_analysis["total"]
    
    print(f"\n{strategy_name} (threshold={threshold:.2f}):")
    print(fraud_analysis.sort_values("rate", ascending=False).to_string())

# ---------------------------------------------------------
# 5. RECOMMENDATION
# ---------------------------------------------------------
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

print("""
Based on your data patterns:

1. **VELOCITY & ATO FRAUD**: Your model catches these well (60-100%)
   → Use CONSERVATIVE threshold (0.70) for auto-blocking

2. **OTHER FRAUD TYPES**: Model struggles (<40% detection)
   → Need additional signals:
     • Device fingerprinting
     • Behavioral biometrics
     • Transaction network analysis
     • Manual review rules

3. **PRODUCTION STRATEGY**:
   → Multi-layered approach:
     ├─ Auto-block: Velocity fraud (threshold 0.70)
     ├─ Manual review: ATO/Merchant fraud (threshold 0.50)
     └─ Rule-based: Location/IP/Network fraud (heuristics)

4. **IMPROVE YOUR SYNTHETIC DATA**:
   → Make fraud types more distinct:
     • Location fraud: Add impossible travel speed
     • IP fraud: Add VPN/proxy indicators
     • Mule fraud: Add rapid money movement patterns
""")

# ---------------------------------------------------------
# 6. SAVE BOTH MODELS
# ---------------------------------------------------------
for strategy_name, result in results.items():
    filename = f"fraud_model_{strategy_name.split()[0].lower()}.joblib"
    
    model_package = {
        'model': model,
        'threshold': result['threshold'],
        'features': FEATURES,
        'strategy': strategy_name,
        'expected_performance': {
            'recall': result['recall'],
            'precision': result['precision'],
            'f1': result['f1']
        }
    }
    
    joblib.dump(model_package, filename)
    print(f"✓ Saved: {filename}")

print("\n" + "="*70)