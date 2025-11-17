#=======================================================================================================================
# ENSEMBLE LEARNING PROJECT
# Human Activity Recognition with Smartphones (Kaggle)
#=======================================================================================================================

# ------- Runtime knobs -------
FAST_SEARCH = True   # True = faster hyperparam search; set False for deeper search

# 1. Importing all the important libraries
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import Perceptron, RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay

from sklearn.base import clone
from scipy.stats import loguniform, randint as sp_randint
from joblib import dump

# ------------------------- 1.1 Load dataset -------------------------
df = pd.read_csv("train.csv")
print("\n== Loading the Datasets ==")
print("\nDataset Shape(rows, columns): ", df.shape)
print("\nFirst 5 rows: \n", df.head(5))
print("\nColumn names: \n", df.columns.tolist())

# ------------------------- 2. Data Exploration -------------------------
print("\n== Basic info == ")
print(df.info())

total_missing = df.isna().sum().sum()
print("\nTotal number of missing values in the entire dataset: ", total_missing)

missing_per_column = df.isna().sum()
print("\nMissing values per column (only columns with missing values): ")
print(missing_per_column[missing_per_column > 0])

print("\n== Numerical Summary Statistics for the first 10 features ==")
print(df.describe().T.head(10))

print("\n== Activity Label Distribution ==")
activity_counts = df["Activity"].value_counts()
print(activity_counts)

plt.figure(figsize=(10, 10))
sns.countplot(x="Activity", data=df, order=activity_counts.index)
plt.title("Activity Label Distribution")
plt.tight_layout()
plt.show()

print("\n== Number of samples per subject ==")
subject_count = df["subject"].value_counts().sort_index()
print(subject_count.head(10))

plt.figure(figsize=(10, 10))
subject_count.plot(kind="bar")
plt.title("Number of Samples per Subject")
plt.xlabel("Subject ID")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# 2.5 Look for Outliers and key features
features_to_inspect = ["tBodyAcc-mean()-X", "tBodyAcc-mean()-Y", "tBodyAcc-std()-X", "tBodyAcc-std()-Y"]
print("\n== Example Features to inspect ==")
print(features_to_inspect)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for ax, col in zip(axes.ravel(), features_to_inspect):
    ax.hist(df[col], bins=30)
    ax.set_title(col)
fig.suptitle("Histograms of selected features", y=1.02)
fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=df[features_to_inspect], ax=ax)
ax.set_title("Boxplot of selected features (outlier inspection)")
ax.set_ylabel("Standardized measurement units")
ax.tick_params(axis='x', rotation=20)
fig.tight_layout()
plt.show()

corr_subset = df[features_to_inspect].corr()
print("\n== Correlation Matrix for the selected features==")
print(corr_subset)

fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corr_subset, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
ax.set_title("Correlation HeatMap for selected features")
fig.tight_layout()
plt.show()

# ------------------------- 3. Prepare Features / Split -------------------------
# Drop ID-like column to avoid leakage
X = df.drop(columns=['Activity', 'subject'])
y = df['Activity']
num_features = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\n== Train/Test Sizes==")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")

# Preprocessing
scaler = ColumnTransformer(transformers=[("num", StandardScaler(), num_features)], remainder="drop")

def scaled_pipeline(estimator):
    return Pipeline(steps=[("scaler", scaler), ("estimator", estimator)])

def passthrough_pipeline(estimator):
    return Pipeline(steps=[("clf", estimator)])

# ------------------------- 3.4 Models -------------------------
models = {
    "Perceptron": scaled_pipeline(Perceptron(random_state=42)),
    "RidgeClassifier (Adaline-like)": scaled_pipeline(RidgeClassifier(random_state=42)),
    "Logistic Regression": scaled_pipeline(LogisticRegression(max_iter=2000, random_state=42)),
    "SVM (RBF)": scaled_pipeline(SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
    "KNN (k=15)": scaled_pipeline(KNeighborsClassifier(n_neighbors=15, weights="distance")),
    "DecisionTree": passthrough_pipeline(DecisionTreeClassifier(max_depth=None, random_state=42)),
    "RandomForest": passthrough_pipeline(RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42)),
}

# ------------------------- 3.5 Train & Evaluate -------------------------
results = []
reports = {}

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    results.append({"model": name, "accuracy": acc, "f1_macro": f1m})
    reports[name] = classification_report(y_test, y_pred, digits=3)
    print(f"\n== {name} ==")
    print(f"Accuracy: {acc:.4f} | f1_macro: {f1m:.4f}")

results_df = pd.DataFrame(results).sort_values(by=["accuracy", "f1_macro"], ascending=False)
print("\n == Model Leaderboard (Test Set) ==")
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["model"]
print(f"\n== Best Model: {best_name} ==")
print(reports[best_name])

best_pipe = models[best_name]
y_pred_best = best_pipe.predict(X_test)
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_best, xticks_rotation=45, cmap="Blues", colorbar=False, ax=ax
)
ax.set_title(f"Confusion Matrix for {best_name}")
fig.tight_layout()
plt.show()

# ------------------------- 4. Hyperparameter Search -------------------------
cv_splits = 3 if FAST_SEARCH else 5
cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

def eval_and_log(name, estimator, Xtr=X_train, ytr=y_train, xte=X_test, yte=y_test):
    estimator = estimator.fit(Xtr, ytr)
    yhat = estimator.predict(xte)
    acc = accuracy_score(yte, yhat)
    f1m = f1_score(yte, yhat, average="macro")
    print(f"\n== {name} ==")
    print(f"Accuracy: {acc:.4f} | f1_macro: {f1m:.4f}")
    return {"model": name, "estimator": estimator, "accuracy": acc, "f1_macro": f1m}

# Parameter spaces (target the inner estimator in each pipeline)
search_specs = {
    "SVM (RBF)": {
        "pipe": models["SVM (RBF)"],
        "params": {
            "estimator__C": loguniform(1e-1, 1e2),   # 0.1–100
            "estimator__gamma": ["scale", "auto"]
        },
        "n_iter": 15 if FAST_SEARCH else 25
    },
    "RandomForest": {
        "pipe": models["RandomForest"],
        "params": {
            "clf__n_estimators": sp_randint(150, 401),  # 150–400 trees
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_split": sp_randint(2, 11),
            "clf__min_samples_leaf": sp_randint(1, 5),
            "clf__max_features": ["sqrt", "log2", None]
        },
        "n_iter": 15 if FAST_SEARCH else 30
    },
    "Logistic Regression": {
        "pipe": models["Logistic Regression"],
        "params": {
            "estimator__C": loguniform(1e-2, 1e2),
            "estimator__penalty": ["l2"],
            "estimator__solver": ["lbfgs", "saga"]
        },
        "n_iter": 15 if FAST_SEARCH else 25
    },
    "KNN (k=15)": {
        "pipe": models["KNN (k=15)"],
        "params": {
            "estimator__n_neighbors": sp_randint(5, 31),
            "estimator__weights": ["uniform", "distance"],
            "estimator__p": [1, 2]
        },
        "n_iter": 15 if FAST_SEARCH else 25
    },
}

print("\n == Hyperparameter search (RandomizedSearchCV) ==")
tuned_models = {}
for name, spec in search_specs.items():
    search = RandomizedSearchCV(
        estimator=spec["pipe"],
        param_distributions=spec["params"],
        n_iter=spec["n_iter"],
        cv=cv,
        scoring="f1_macro",
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True,
        error_score="raise"
    )
    search.fit(X_train, y_train)
    print(f"\n == Best params for {name}: {search.best_params_}")
    print(f"CV best f1_macro: {search.best_score_:.4f}")
    tuned_models[f"{name} (tuned)"] = search.best_estimator_

# Evaluate tuned models on the test set and extend the leaderboard
more_results = []
for nm, est in tuned_models.items():
    more_results.append(eval_and_log(nm, est))

if more_results:
    more_df = pd.DataFrame([{k: v for k, v in r.items() if k != "estimator"} for r in more_results])
    results_df = (
        pd.concat([results_df, more_df[["model", "accuracy", "f1_macro"]]], ignore_index=True)
          .sort_values(by=["accuracy", "f1_macro"], ascending=False)
    )
    print("\n == Updated Leaderboard (incl. tuned) ==")
    print(results_df.to_string(index=False))

# ------------------------- 5. True Ensembles: Voting + Stacking -------------------------
def ensure_svc_probability(pipe):
    p = clone(pipe)
    if "estimator" in p.named_steps and isinstance(p.named_steps["estimator"], SVC):
        if not p.named_steps["estimator"].probability:
            p.set_params(estimator__probability=True)
    return p

svm_base = tuned_models.get("SVM (RBF) (tuned)", models["SVM (RBF)"])
rf_base  = tuned_models.get("RandomForest (tuned)", models["RandomForest"])
lr_base  = tuned_models.get("Logistic Regression (tuned)", models["Logistic Regression"])
knn_base = tuned_models.get("KNN (k=15) (tuned)", models["KNN (k=15)"])

svm_soft = ensure_svc_probability(svm_base)

voting_hard = VotingClassifier(
    estimators=[
        ("perc", models["Perceptron"]),
        ("ridge", models["RidgeClassifier (Adaline-like)"]),
        ("lr",   lr_base),
        ("svm",  svm_base),
        ("dt",   models["DecisionTree"]),
        ("rf",   rf_base),
        ("knn",  knn_base),
    ],
    voting="hard",
    n_jobs=-1
)

voting_soft = VotingClassifier(
    estimators=[
        ("lr",  lr_base),
        ("svm", svm_soft),
        ("rf",  rf_base),
        ("knn", knn_base),
    ],
    voting="soft",
    n_jobs=-1
)

stacking = StackingClassifier(
    estimators=[
        ("lr",  lr_base),
        ("svm", svm_soft),
        ("rf",  rf_base),
        ("knn", knn_base),
    ],
    final_estimator=LogisticRegression(max_iter=5000, random_state=42),
    stack_method="predict_proba",
    cv=cv,
    n_jobs=-1,
    passthrough=False
)

ensemble_models = {
    "Voting (hard)": voting_hard,
    "Voting (soft)": voting_soft,
    "Stacking (LR meta)": stacking,
}

print("\n== Ensembles ==")
ensemble_results = []
for nm, est in ensemble_models.items():
    ensemble_results.append(eval_and_log(nm, est))

# ------------------------- 6. Consolidate and choose global best -------------------------
all_results = []
for r in results:
    all_results.append({"model": r["model"], "accuracy": r["accuracy"], "f1_macro": r["f1_macro"]})
for r in more_results:
    all_results.append({"model": r["model"], "accuracy": r["accuracy"], "f1_macro": r["f1_macro"]})
for r in ensemble_results:
    all_results.append({"model": r["model"], "accuracy": r["accuracy"], "f1_macro": r["f1_macro"]})

all_df = pd.DataFrame(all_results).sort_values(by=["accuracy", "f1_macro"], ascending=False)
print("\n == GRAND LEADERBOARD ==")
print(all_df.to_string(index=False))

best_overall_name = all_df.iloc[0]["model"]
print(f"\n== Best Overall: {best_overall_name} ==")

name_to_est = {**{k: v for k, v in models.items()}, **tuned_models, **ensemble_models}
best_overall_est = name_to_est[best_overall_name]
best_overall_est.fit(X_train, y_train)

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test, best_overall_est.predict(X_test), xticks_rotation=45, cmap="Blues", colorbar=False, ax=ax
)
ax.set_title(f"Confusion Matrix for {best_overall_name}")
fig.tight_layout()
plt.show()

# ------------------------- 7. Persist the final pipeline & inference helper -------------------------
MODEL_PATH = "best_har_model.joblib"
FEATURE_LIST_PATH = "feature_names.txt"

dump(best_overall_est, MODEL_PATH)
with open(FEATURE_LIST_PATH, "w") as f:
    for c in num_features:
        f.write(c + "\n")

print(f"\nSaved model to: {MODEL_PATH}")
print(f"Saved feature list to: {FEATURE_LIST_PATH}")

def predict_activities(new_data, model_path=MODEL_PATH, feature_list_path=FEATURE_LIST_PATH):
    """
    new_data: pandas DataFrame (same feature space) OR path to CSV with same columns.
    Returns: numpy array of predicted activity labels.
    """
    from joblib import load
    if isinstance(new_data, str):
        new_df = pd.read_csv(new_data)
    else:
        new_df = new_data.copy()

    with open(feature_list_path, "r") as f:
        feats = [ln.strip() for ln in f if ln.strip()]

    missing = [c for c in feats if c not in new_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    new_df = new_df.reindex(columns=feats)
    model = load(model_path)
    return model.predict(new_df)