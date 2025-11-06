import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_PATH = "scaled_combined.csv"
TARGET_COL = "readmission_30days"
OUTPUT_DIR = "model_comparison_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=[TARGET_COL])
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")


neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.2f}")


print("\nTraining XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    reg_lambda=2,
    min_child_weight=3,
    eval_metric="auc",
    random_state=42,
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]


print("\nTraining LightGBM...")
lgb_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42
)
lgb_model.fit(X_train, y_train)
y_pred_proba_lgbm = lgb_model.predict_proba(X_test)[:, 1]


print("\nCombining models (ensemble)...")
ensemble_proba = (y_pred_proba_xgb + y_pred_proba_lgbm) / 2
threshold = 0.10
y_pred_ensemble = (ensemble_proba >= threshold).astype(int)


def evaluate_model(name, y_true, y_proba, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    return [name, acc, prec, rec, f1, auc]

results = []
results.append(evaluate_model("XGBoost", y_test, y_pred_proba_xgb, (y_pred_proba_xgb >= threshold)))
results.append(evaluate_model("LightGBM", y_test, y_pred_proba_lgbm, (y_pred_proba_lgbm >= threshold)))
results.append(evaluate_model("Ensemble", y_test, ensemble_proba, y_pred_ensemble))

metrics_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"])
metrics_df.to_csv(f"{OUTPUT_DIR}/comparison_metrics.csv", index=False)
print("\nModel comparison metrics saved.")
print(metrics_df)


plt.figure(figsize=(8, 4))
metrics_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
sns.barplot(x="Metric", y="Value", hue="Model", data=metrics_melted)
plt.title("Model Comparison: XGBoost vs LightGBM vs Ensemble")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/model_comparison.png")
plt.close()

cm = confusion_matrix(y_test, y_pred_ensemble)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Ensemble Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/ensemble_confusion_matrix.png")
plt.close()


fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
fpr_lgbm, tpr_lgbm, _ = roc_curve(y_test, y_pred_proba_lgbm)
fpr_ens, tpr_ens, _ = roc_curve(y_test, ensemble_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={metrics_df.loc[0, 'ROC AUC']:.3f})")
plt.plot(fpr_lgbm, tpr_lgbm, label=f"LightGBM (AUC={metrics_df.loc[1, 'ROC AUC']:.3f})")
plt.plot(fpr_ens, tpr_ens, label=f"Ensemble (AUC={metrics_df.loc[2, 'ROC AUC']:.3f})", linestyle="--")
plt.plot([0, 1], [0, 1], "k--", alpha=0.7)
plt.title("ROC Curves Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_comparison.png")
plt.close()

print(f"\nAll comparison outputs saved in: {OUTPUT_DIR}")
