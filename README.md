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
