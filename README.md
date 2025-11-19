# Human Activity Recognition with Smartphones – Ensemble Learning Project

**Author:** Shivesh (Ethan) Sahu  

This project builds and compares multiple classical machine-learning and ensemble models to classify human physical activities (e.g., WALKING, SITTING, STANDING) from smartphone sensor data. The final stacked ensemble reaches **~99.5% macro-F1** on a held-out test set.  [oai_citation:0‡Human Activity Recognition with Smartphones.pdf](sediment://file_0000000079a071f7916dd06e3f4bdb20)  

---

## 1. Dataset

- **Source:** Kaggle – *Human Activity Recognition with Smartphones* (`train.csv`)
- **Shape:** `7352 × 563`
  - **561** numeric feature columns (engineered time/frequency-domain features from accelerometer + gyroscope)
  - `subject` – integer subject ID
  - `Activity` – target label (string)
- **Classes & counts:**
  - LAYING – 1407  
  - STANDING – 1374  
  - SITTING – 1286  
  - WALKING – 1226  
  - WALKING_UPSTAIRS – 1073  
  - WALKING_DOWNSTAIRS – 986  
- No missing values in any column.  [oai_citation:1‡scratch_1.pdf](sediment://file_00000000d78071f59e773764ac002c4e)  

---

## 2. Exploratory Data Analysis (EDA)

Key checks and plots (see notebook):

- **Basic stats** for the first 10 features show values roughly in `[-1, 1]` with moderate variance – features are already well-scaled and numerically stable.  [oai_citation:2‡scratch_1.pdf](sediment://file_00000000d78071f59e773764ac002c4e)  
- **Activity distribution:** mild imbalance (more LAYING / STANDING than WALKING_DOWNSTAIRS), but every class has >900 samples – macro-F1 is appropriate.  [oai_citation:3‡Human Activity Recognition with Smartphones.pdf](sediment://file_0000000079a071f7916dd06e3f4bdb20)  
- **Subject distribution:** each subject contributes roughly 280–360 samples, giving good diversity and reducing overfitting to any single person.  [oai_citation:4‡scratch_1.pdf](sediment://file_00000000d78071f59e773764ac002c4e)  
- **Feature inspection** (histograms + boxplots) for e.g. `tBodyAcc-mean()-X`, `tBodyAcc-std()-X`, etc.:
  - Realistic outliers but no obvious data errors.
- **Correlation (subset):**
  - Very strong correlation between `tBodyAcc-std()-X` and `tBodyAcc-std()-Y` (~0.93).
  - Very weak correlation between mean vs std features, so they carry complementary information.  [oai_citation:5‡Human Activity Recognition with Smartphones.pdf](sediment://file_0000000079a071f7916dd06e3f4bdb20)  

---

## 3. Data Preparation

- Dropped `subject` to avoid ID leakage.
- **Features (`X`)**: all 561 numeric feature columns.  
- **Target (`y`)**: `Activity`.  [oai_citation:6‡scratch_1.pdf](sediment://file_00000000d78071f59e773764ac002c4e)  

### Train–test split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

	•	X_train: (5881, 561)
	•	X_test: (1471, 561)

### Scaling & pipelines

A reusable ColumnTransformer + Pipeline setup:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

num_features = X.columns.tolist()

scaler = ColumnTransformer(
    [("num", StandardScaler(), num_features)],
    remainder="drop",
)

def scaled_pipeline(estimator):
    return Pipeline([("scaler", scaler), ("estimator", estimator)])

def passthrough_pipeline(estimator):
    return Pipeline([("clf", estimator)])

	•	Linear and distance-based models → scaled_pipeline
	•	Tree-based / ensemble models → passthrough_pipeline (no scaling)

___

## 4. Baseline Models

Trained on the training set and evaluated on the held-out test set (accuracy & macro-F1):
Model                      Accuracy               F1 (macro)
SVM (RBF)                   0.9898                  0.9906
RandomForest                0.9878                  0.9879
Logistic Regression         0.9857                  0.9868
RidgeClassifier (Adaline)   0.9810                  0.9822
Perceptron                  0.9810                  0.9818
KNN (k=15, distance)        0.9565                  0.9578
DecisionTree                0.9388                  0.9380

Observations:
	•	Once scaled, linear models perform extremely well, confirming that the engineered HAR features are almost linearly separable.
	•	RBF-SVM already reaches ~0.99 accuracy; remaining headroom is small.
	•	The main residual confusion in the SVM confusion matrix is between SITTING and STANDING, which is intuitive given similar motion patterns

___

5. Hyperparameter Tuning

Used RandomizedSearchCV with StratifiedKFold (3 folds in FAST mode, scoring = f1_macro) for:
	•	SVM (RBF)
	•	RandomForest
	•	Logistic Regression
	•	KNN  ￼

Highlights:
	•	SVM (RBF tuned) – best C ≈ 31.4, gamma="auto", test metrics essentially match baseline (~0.9898 acc / 0.9906 macro-F1).
	•	RandomForest tuned – ~341 trees, max_depth=30, max_features="log2", slightly below the original RF on the test set (~0.985 acc).
	•	Logistic Regression tuned – C ≈ 0.53, penalty="l2", same performance as baseline (~0.986 F1).
	•	KNN tuned – improved to ~0.972 macro-F1 but still below linear / SVM.  ￼

⸻

6. Ensembles: Voting, Stacking, Bagging & Boosting

6.1 Voting & Stacking

Built using the tuned/baseline pipelines:
	•	Hard Voting: Perceptron, Ridge, Logistic Regression, SVM, DecisionTree, RandomForest, KNN.
	•	Soft Voting: Logistic Regression, probabilistic SVM, RandomForest, KNN.
	•	Stacking (final model):
	•	Base estimators: Logistic Regression, probabilistic SVM, RandomForest, KNN
	•	Meta-learner: Logistic Regression (max_iter=5000, stack_method="predict_proba").

Performance (test set):
Ensemble             Accuracy           F1 (macro)
Voting (hard)         0.9905              0.9912
Voting (soft)         0.9918              0.9925
Stacking (LR meta)    0.9946              0.9950

The stacked ensemble is the global best model.

Error analysis:
	•	Confusion matrix is almost perfectly diagonal.
	•	Remaining errors are almost entirely SITTING vs STANDING, reflecting truly subtle differences in the underlying sensor patterns.  ￼

6.2 Bagging & Boosting

Models kept in a separate registry:
	•	BaggingClassifier with DecisionTree base learners
	•	ExtraTreesClassifier
	•	AdaBoostClassifier (shallow DecisionTrees)
	•	HistGradientBoostingClassifier  ￼

Baseline performance:
Model                       Accuracy             F1 (macro)
HistGradientBoosting        0.9939                  0.9941
ExtraTrees                  0.9932                  0.9937
AdaBoost                    0.9864                  0.9869
Bagging (DT)                0.9721                  0.9722

These tree ensembles nearly match the stacked model, but Stacking (LR meta) still holds the top score and is chosen as the production model.

⸻

7. Saved Artifacts
	•	best_har_model.joblib – the full scikit-learn pipeline for the stacked model, including preprocessing (scaling) and all base/meta learners.
	•	feature_names.txt – ordered list of the 561 feature names used during training.

These are stored next to the notebook and can be committed to the repo.  ￼

⸻

8. Inference Helper

A small utility wraps consistency checks and prediction:

import pandas as pd
from joblib import load

MODEL_PATH = "best_har_model.joblib"
FEATURE_LIST_PATH = "feature_names.txt"

def predict_activities(new_data, model_path=MODEL_PATH, feature_list_path=FEATURE_LIST_PATH):
    """
    new_data: pandas DataFrame with HAR features
              OR path to a CSV file with the same columns.
    Returns: numpy array of predicted activity labels.
    """
    # Accept DataFrame or CSV path
    if isinstance(new_data, str):
        new_df = pd.read_csv(new_data)
    else:
        new_df = new_data.copy()

    # Load required feature names
    with open(feature_list_path, "r") as f:
        feats = [ln.strip() for ln in f if ln.strip()]

    # Check that all required columns exist
    missing = [c for c in feats if c not in new_df.columns]
    if missing:
        short = missing[:10]
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"Missing required columns: {short}{suffix}")

    # Align column order
    new_df = new_df.reindex(columns=feats)

    # Load pipeline & predict
    model = load(model_path)
    return model.predict(new_df)


⸻

9. How to Run This Project
	1.	Clone the repo & install deps
      git clone <this-repo-url>
      cd <this-repo-folder>
      pip install -r requirements.txt

	2.	Download the Kaggle HAR dataset and place train.csv in the project root (or update the path in the notebook/script).
	3.	Run the notebook
	•	Open scratch_1.ipynb in PyCharm (Jupyter support enabled) or VS Code / Jupyter Lab.
	•	Run all cells to reproduce:
	•	EDA
	•	Baseline models
	•	Hyperparameter tuning
	•	Voting / stacking ensembles
	•	Bagging / boosting experiments
	4.	Use the trained model for inference
	•	Make sure best_har_model.joblib and feature_names.txt are present.
	•	Import and call predict_activities from your own script or API.

⸻

10. Limitations & Future Work
	•	Remaining errors occur mainly between SITTING and STANDING; more explicit postural features or sequence models (LSTMs, HMMs) could help.  ￼
	•	Evaluation uses a random subject mix; a stricter protocol would train on some subjects and test on completely unseen individuals.
	•	Stacking is relatively heavy; for on-device deployment, a lighter model such as HistGradientBoosting or a single tuned SVM might be preferable.
___
