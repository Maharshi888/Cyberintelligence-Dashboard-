# Practical 10: Network Intrusion / Cyber Threat Classification using Data Mining
# Dataset : combinenew.csv (CIC-IDS2017 style network flow data, label = BENIGN / attacks)
# Steps   : Preprocessing -> Model building -> Hyperparameter tuning -> Evaluation

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------- settings you can change ----------------

BASE = Path(__file__).parent
CSV_PATH = BASE.parent / "raw" / "combinenew.csv"
TARGET = "Label"        # label column (leading spaces are removed automatically)
BINARY = True           # True  : BENIGN vs ATTACK (2 classes, like the sample practical)
                        # False : every attack type is its own class
MAX_PER_CLASS = 15000 if BINARY else 2000   # big files are sampled per class
MIN_CLASS_SIZE = 30     # attack types with fewer rows are removed (multi-class only)
CHUNK_SIZE = 200000     # file is read in pieces so it fits in memory
# ----------------------------------------------------------

print("=" * 50)
print("CYBER THREAT CLASSIFICATION")
print("=" * 50)

# ==================================================
# STEP 1: LOAD DATASET (in chunks, with per-class sampling)
# ==================================================
path = CSV_PATH
try:
    open(path, encoding="utf-8").read(2_000_000)
    encoding = "utf-8"
except UnicodeDecodeError:
    encoding = "latin1"          # some CIC-IDS files contain special characters

parts, total_rows = [], 0
for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, encoding=encoding, low_memory=False):
    chunk.columns = chunk.columns.str.strip()        # " Label" -> "Label"
    total_rows += len(chunk)
    chunk = chunk.replace([np.inf, -np.inf], np.nan)
    chunk = chunk.dropna(subset=[TARGET])
    chunk[TARGET] = chunk[TARGET].astype(str).str.strip()
    if BINARY:
        chunk[TARGET] = np.where(chunk[TARGET].str.upper() == "BENIGN", "BENIGN", "ATTACK")
    parts.append(chunk.groupby(TARGET, group_keys=False).head(MAX_PER_CLASS))

df = pd.concat(parts, ignore_index=True)

print("\nOriginal Rows in File:", total_rows)
print("\nDataset Shape (after loading):")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

# ==================================================
# STEP 2: DATA PREPROCESSING
# ==================================================
# Remove duplicate rows
before = len(df)
df = df.drop_duplicates()
print("\nDuplicate Rows Removed:", before - len(df))

# Remove attack types that are too rare to learn or split
counts = df[TARGET].value_counts()
rare = counts[counts < MIN_CLASS_SIZE].index.tolist()
if rare:
    print("Rare classes removed:", rare)
    df = df[~df[TARGET].isin(rare)]

# Keep at most MAX_PER_CLASS rows per class (keeps the data balanced enough)
df = df.groupby(TARGET, group_keys=False).sample(
    n=None, frac=1.0, random_state=42).groupby(TARGET, group_keys=False).head(MAX_PER_CLASS)

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

print("\nTarget Classes:")
print(df[TARGET].value_counts())

# Encode label to numbers (original names are kept for the reports)
encoder = LabelEncoder()
y = encoder.fit_transform(df[TARGET])
class_names = [str(c) for c in encoder.classes_]

# Features: all numeric columns except the label; remove constant columns
X = df.drop(columns=[TARGET]).apply(pd.to_numeric, errors="coerce")
X = X.loc[:, X.nunique(dropna=True) > 1]
print("\nNumber of Features:", X.shape[1])

# Train / test split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Samples:")
print(X_train.shape[0])
print("\nTesting Samples:")
print(X_test.shape[0])

# Missing values -> median, then feature scaling (fitted on training data only)
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X_train_p = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_p = scaler.transform(imputer.transform(X_test))

print("\nFeature scaling completed.")

average = "binary" if len(class_names) == 2 else "weighted"
pos_label = 1 if len(class_names) == 2 else None   # 1 = the second class (ATTACK)


def show_results(title, y_true, y_pred, prefix=""):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    acc = accuracy_score(y_true, y_pred)
    kw = dict(average=average, zero_division=0)
    if pos_label is not None:
        kw["pos_label"] = pos_label
    print(f"{prefix}Accuracy: {acc:.4f}")
    print(f"{prefix}Accuracy (%): {acc * 100:.2f}")
    print(f"{prefix}Precision: {precision_score(y_true, y_pred, **kw)}")
    print(f"{prefix}Recall: {recall_score(y_true, y_pred, **kw)}")
    print(f"{prefix}F1 Score: {f1_score(y_true, y_pred, **kw)}")
    print(f"\n{prefix}Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\n{prefix}Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))


# ==================================================
# STEP 3: BASELINE (always predicts the most common class)
# ==================================================
baseline = DummyClassifier(strategy="most_frequent").fit(X_train_p, y_train)
print("\nBaseline Accuracy (majority class):",
      round(accuracy_score(y_test, baseline.predict(X_test_p)), 4))

# ==================================================
# STEP 4: INITIAL MODEL BUILDING (Random Forest)
# ==================================================
model = RandomForestClassifier(random_state=42, n_jobs=-1)
model.fit(X_train_p, y_train)
print("\nInitial model training completed.")

show_results("INITIAL MODEL RESULTS", y_test, model.predict(X_test_p))

# ==================================================
# STEP 5: HYPERPARAMETER TUNING (Grid Search)
# ==================================================
print("\nPerforming hyperparameter tuning...")

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
)
grid.fit(X_train_p, y_train)
print("Hyperparameter tuning completed.")

print("\n" + "=" * 50)
print("BEST PARAMETERS")
print("=" * 50)
print(grid.best_params_)
print("Best Cross-Validation Accuracy:", round(grid.best_score_, 4))

# ==================================================
# STEP 6: FINAL MODEL
# ==================================================
best_model = grid.best_estimator_
show_results("FINAL MODEL RESULTS", y_test, best_model.predict(X_test_p), prefix="Final ")

# ==================================================
# STEP 7: MOST IMPORTANT FEATURES
# ==================================================
importance = pd.Series(best_model.feature_importances_, index=X.columns)
print("\nTop 10 Important Features:")
print(importance.sort_values(ascending=False).head(10).round(4))

print("\nProgram completed successfully.")