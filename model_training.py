# ==============================================================================
# Complete End-to-End MLOps Model Training & MLflow Tracking Script
# ==============================================================================
import time
import os
import subprocess
import urllib.request
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

print("--- Initializing Host-Allowed Ngrok & MLflow Setup ---")

# 1. Clean up existing background processes and stale database files
os.system("pkill -f mlflow")
os.system("pkill -f ngrok")
if os.path.exists("mlflow.db"):
    os.remove("mlflow.db")
    print("🗑️ Removed stale 'mlflow.db' to ensure clean database migration.")
time.sleep(2)

# 2. Configure local SQLite telemetry persistence
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Wellness_Tourism_Package_Prediction")

# 3. Authenticate and Initialize Ngrok Tunnel Safely using Environment Secret Token
try:
    from pyngrok import ngrok
    ngrok_token = os.getenv("NGROK_AUTH_TOKEN")
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        print("🔑 Ngrok authentication token successfully configured.")
    else:
        print("⚠️ No NGROK_AUTH_TOKEN found in environment variables. Tunnel may fail if unauthenticated.")
        
    public_url = ngrok.connect(5000).public_url
    print(f"✅ Ngrok Tunnel established. MLflow UI available at: {public_url}")
except Exception as e:
    print(f"⚠️ Pyngrok tunnel bypassed/skipped due to authentication/environment constraints: {e}")
    print("ℹ️ Continuing with robust local SQLite tracking (sqlite:///mlflow.db).")

print("\n--- Loading and Preprocessing Dataset ---")

# Robust multi-path dataset lookup for CI/CD runners
data_path = 'data/tourism.csv'

target_col = 'ProdTaken'
if target_col not in df.columns:
    target_col = df.columns[-1]

X = df.drop(columns=[target_col])
y = df[target_col]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\n--- Comprehensive Hyperparameter Tuning (Pipeline + XGBoost) ---")
with mlflow.start_run(run_name="Comprehensive_XGBoost_Optimization") as run:
    run_id = run.info.run_id
    
    param_grid = {
        'classifier__n_estimators': [50, 75, 100],
        'classifier__max_depth': [3, 4, 5],
        'classifier__learning_rate': [0.01, 0.05, 0.1]
    }
    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    
    grid_search = GridSearchCV(
        estimator=full_pipeline, 
        param_grid=param_grid, 
        cv=3, 
        scoring='roc_auc', 
        n_jobs=-1
    )
    
    print("Executing Comprehensive Grid Search... (This may take a moment)")
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    y_pred = best_pipeline.predict(X_test)
    y_prob = best_pipeline.predict_proba(X_test)[:, 1] if hasattr(best_pipeline, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n--- MODEL TRAINING & EVALUATION SUMMARY ---")
    print(f"MLflow Run ID: {run_id}")
    print(f"Best Hyperparameters: {best_params}")
    print(f"Evaluation Metrics:")
    print(f" - ACCURACY: {acc:.4f}")
    print(f" - PRECISION: {prec:.4f}")
    print(f" - RECALL: {rec:.4f}")
    print(f" - F1_SCORE: {f1:.4f}")
    print(f" - ROC_AUC: {roc_auc:.4f}")
    
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    
    mlflow.sklearn.log_model(
        sk_model=best_pipeline, 
        artifact_path="best_pipeline_model",
        serialization_format="cloudpickle"
    )
    
    os.makedirs('model', exist_ok=True)
    joblib.dump(best_pipeline, 'model/wellness_model.joblib')
    
    print("\n✅ Training complete. Model serialized to 'model/wellness_model.joblib'.")
