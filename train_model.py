# -*- coding: utf-8 -*-
"""
Cancer Care Detection - Model Training Script
Run this once to train and save the model before starting the Flask app.
"""
import pickle
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

print("=" * 60)
print("  Cancer Care Detection - Model Training")
print("=" * 60)

# ── 1. Load Dataset ──────────────────────────────────────────
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names.tolist()

print(f"\n[OK] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"     Malignant (0): {(y == 0).sum()}")
print(f"     Benign    (1): {(y == 1).sum()}")

# ── 2. Pre-process ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\n[OK] Train/Test split: {len(X_train)} / {len(X_test)}")

# ── 3. Train Random Forest ───────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

# ── 4. Evaluate ──────────────────────────────────────────────
y_pred = rf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\n[OK] Model accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── 5. Save artifacts ────────────────────────────────────────
os.makedirs("model", exist_ok=True)

with open("model/cancer_model.pkl", "wb") as f:
    pickle.dump(rf, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("\n[OK] Saved: model/cancer_model.pkl")
print("[OK] Saved: model/scaler.pkl")
print("[OK] Saved: model/feature_names.pkl")
print("\n  Training complete! Run  python app.py  to start the app.")
print("=" * 60)
