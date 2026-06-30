# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay
)
from imblearn.over_sampling import SMOTE

# Load Dataset
df = pd.read_csv("creditcard.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# Class Distribution
print("\nClass Distribution:")
print(df["Class"].value_counts())

df["Class"].value_counts().plot(kind="bar")
plt.title("Fraud vs Genuine Transactions")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Feature Scaling
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

# Drop Time Column
df.drop("Time", axis=1, inplace=True)

# Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_smote.value_counts())

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_smote, y_train_smote)

# Prediction
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# ROC Curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.show()

# Cross Validation
scores = cross_val_score(
    model,
    X_train_smote,
    y_train_smote,
    cv=5,
    scoring="roc_auc"
)

print("\nCross Validation ROC-AUC Scores:")
print(scores)

print("Average ROC-AUC:", scores.mean())

# Feature Importance
importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).head(10).plot(kind="bar")
plt.title("Top 10 Important Features")
plt.show()

# Test Sample Prediction
sample = X_test.iloc[[0]]
prediction = model.predict(sample)

if prediction[0] == 1:
    print("\nPrediction: Fraud Transaction")
else:
    print("\nPrediction: Genuine Transaction")