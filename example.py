import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

CSV_PATH = "fairness_income.csv"

if not os.path.exists(CSV_PATH):
    print("ERROR: fairness_income.csv file not found. Place the CSV in the same folder.")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)

expected_cols = {
    "age", "workclass", "education_num", "hours_per_week",
    "capital_gain", "capital_loss", "sex", "race", "income"
}
if not expected_cols.issubset(set(df.columns)):
    print("ERROR: CSV missing required columns:", expected_cols)
    print("Found columns:", list(df.columns))
    sys.exit(1)

sns.countplot(x="income", data=df)
plt.title("Income distribution")
plt.tight_layout()
plt.show()

df["income"] = df["income"].astype(str).str.strip()
df["target"] = df["income"].apply(lambda s: 1 if s.startswith(">50") else 0)

sensitive_cols = ["sex", "race"]
num_cols = ["age", "education_num", "hours_per_week", "capital_gain", "capital_loss"]
cat_cols = ["workclass"]

df[num_cols] = df[num_cols].fillna(0)
df[cat_cols + sensitive_cols] = df[cat_cols + sensitive_cols].fillna("Unknown")

X = df[num_cols + cat_cols].copy()
y = df["target"].copy()
df_sensitive = df[sensitive_cols].copy()

X_train, X_test, y_train, y_test, df_sensitive_train, df_sensitive_test = train_test_split(
    X, y, df_sensitive, test_size=0.2, random_state=42, stratify=y
)

numeric_transformer = Pipeline([("scaler", StandardScaler())])
categorical_transformer = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop",
)

clf = Pipeline([
    ("pre", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", auc)
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

def group_rates(df_sensitive_local, y_true_local, y_pred_local, group_col):
    results = {}
    groups = pd.Series(df_sensitive_local[group_col].values).unique()
    for g in groups:
        idx = (df_sensitive_local[group_col].values == g)
        y_true_grp = np.asarray(y_true_local)[idx]
        y_pred_grp = np.asarray(y_pred_local)[idx]
        cm = confusion_matrix(y_true_grp, y_pred_grp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        tpr = (tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        fpr = (fp / (fp + tn)) if (fp + tn) > 0 else float("nan")
        results[g] = {"TPR": tpr, "FPR": fpr, "n": int(idx.sum())}
    return pd.DataFrame(results).T

group_metrics_sex = group_rates(df_sensitive_test, y_test, y_pred, "sex")
group_metrics_race = group_rates(df_sensitive_test, y_test, y_pred, "race")

print("By sex:\n", group_metrics_sex)
print("\nBy race:\n", group_metrics_race)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
group_metrics_sex[["TPR", "FPR"]].plot.bar(ax=axes[0], title="Performance by sex")
group_metrics_race[["TPR", "FPR"]].plot.bar(ax=axes[1], title="Performance by race")
plt.tight_layout()
plt.show()