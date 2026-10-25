import os
import subprocess
import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from data_preprocessing import get_preprocessor

# 1. Purge stale SQLite tracking cache
db_path = "mlflow.db"
if os.path.exists(db_path):
    os.remove(db_path)

mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("wellness_tourism_prediction")

# 2. Optional: Initialize Ngrok Tunnel for Live MLflow UI Access
ngrok_token = os.environ.get("NGROK_AUTH_TOKEN")
if ngrok_token:
    try:
        from pyngrok import ngrok
        ngrok.set_auth_token(ngrok_token)
        # Start MLflow UI server in the background on port 5000
        subprocess.Popen([
            "mlflow", "ui", 
            "--host", "127.0.0.1", 
            "--port", "5000", 
            "--backend-store-uri", f"sqlite:///{db_path}"
        ])
        # Open Ngrok tunnel to port 5000
        public_url = ngrok.connect(5000)
        print(f"\n🚀 ==================================================")
        print(f"🌍 MLflow UI Live Ngrok Tunnel URL: {public_url}")
        print(f"🚀 ==================================================\n")
    except Exception as e:
        print(f"⚠️ Could not establish Ngrok tunnel: {e}")

# 3. Load Pre-split Artifacts from data_preparation.py
print("📂 Loading training and testing artifacts...")
try:
    X_train = pd.read_csv('artifacts/X_train.csv')
    X_test = pd.read_csv('artifacts/X_test.csv')
    y_train = pd.read_csv('artifacts/y_train.csv').values.ravel()
    y_test = pd.read_csv('artifacts/y_test.csv').values.ravel()
except FileNotFoundError as e:
    print(f"❌ Error loading artifacts: {e}. Please ensure data_preparation.py runs first.")
    raise e

# 4. Build Full Pipeline using centralized preprocessor
preprocessor = get_preprocessor()

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(eval_metric='logloss'))
])

# 5. Hyperparameter Tuning & MLflow Tracking
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5]
}

with mlflow.start_run():
    print("🚀 Starting GridSearchCV hyperparameter tuning...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # 6. Console & Automated Metric Logging
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n----------------------------------------")
    print(f"🎯 Best Hyperparameters Found: {grid_search.best_params_}")
    print(f"📊 Test Accuracy : {accuracy:.4f}")
    print(f"📈 Test ROC-AUC  : {roc_auc:.4f}")
    print("----------------------------------------\n")
    
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # 7. Safe Model Serialization for Streamlit App and MLflow
    trusted_types = ["numpy.dtype", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
    mlflow.sklearn.log_model(
        best_model, 
        artifact_path="model", 
        skops_trusted_types=trusted_types
    )
    
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, "model/wellness_model.joblib")
    print("✅ Model successfully trained, evaluated, and serialized as model/wellness_model.joblib")
