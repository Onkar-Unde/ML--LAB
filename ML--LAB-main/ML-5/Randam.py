from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

# ------------------------------
# Utility Functions
# ------------------------------

def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def evaluate_model(model, X_test, y_test, dataset_name="Dataset"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n--- {dataset_name} ---")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def predict_random_sample(model, X_test, y_test):
    index = np.random.randint(len(X_test))
    sample = X_test[index].reshape(1, -1)
    prediction = model.predict(sample)[0]
    print(f"\nSample #{index} - Actual: {y_test[index]}, Predicted: {prediction}")

# ------------------------------
# Part 1: Synthetic Dataset
# ------------------------------

# Generate synthetic classification dataset
X, y = make_classification(
    n_features=6,
    n_classes=3,
    n_samples=800,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

# Visualize first 2 features
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c=y, marker="*")
plt.title("Synthetic Data (2 Features)")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

# Split and standardize
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=125)
X_train_std, X_test_std = standardize_data(X_train, X_test)

# Train and evaluate
model = GaussianNB()
model.fit(X_train_std, y_train)
predict_random_sample(model, X_test_std, y_test)
evaluate_model(model, X_test_std, y_test, "Synthetic Data")

# ------------------------------
# Part 2: Used Cars Dataset
# ------------------------------

csv_path = 'C:/Users/Icon/Downloads/ML--LAB-main/ML-5/used_cars_data.csv'

if not os.path.exists(csv_path):
    print(f"\nFile not found: {csv_path}")
else:
    df = pd.read_csv(csv_path)
    print("\nDataset Loaded Successfully.")
    print("First 5 rows:")
    print(df.head())

    print("\nColumns in dataset:", df.columns.tolist())

    # Visualize purpose vs not.fully.paid
    if 'purpose' in df.columns and 'not.fully.paid' in df.columns:
        plt.figure(figsize=(10, 4))
        sns.countplot(data=df, x='purpose', hue='not.fully.paid')
        plt.xticks(rotation=45, ha='right')
        plt.title("Loan Purpose vs Not Fully Paid")
        plt.show()
    else:
        print("Required columns 'purpose' and/or 'not.fully.paid' not found.")

    # Handle missing values
    if df.isnull().sum().any():
        print("Missing values found. Filling with median values.")
        df.fillna(df.median(numeric_only=True), inplace=True)

    # One-hot encode 'purpose' if exists
    if 'purpose' in df.columns:
        df = pd.get_dummies(df, columns=['purpose'], drop_first=True)

    # Train model if target exists
    if 'not.fully.paid' in df.columns:
        X = df.drop('not.fully.paid', axis=1)
        y = df['not.fully.paid']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=125)
        X_train_std, X_test_std = standardize_data(X_train, X_test)

        model = GaussianNB()
        model.fit(X_train_std, y_train)
        predict_random_sample(model, X_test_std, y_test)
        evaluate_model(model, X_test_std, y_test, "Used Cars Data")
    else:
        print("Target column 'not.fully.paid' not found.")
