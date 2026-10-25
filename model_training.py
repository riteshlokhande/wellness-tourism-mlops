import sys
import os
import shutil
import subprocess
import time

# Ensure current working directory is in system path for robust module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_preprocessing import get_preprocessor

def train_and_track_model():
    print("--- Initiating Model Training & Tracking Lifecycle ---", flush=True)
    
    # 1. Purge stale SQLite databases or mlruns caching absolute /content paths from Colab
    if os.path.exists("mlflow.db"):
        os.remove("mlflow.db")
        print("🗑️ Cleared stale mlflow.db cache.", flush=True)
    if os.path.exists("mlruns"):
        shutil.rmtree("mlruns")
        print("🗑️ Cleared stale mlruns directory.", flush=True)

    # 2. Start MLflow Server locally
    print("Starting local MLflow server...", flush=True)
    os.system("pkill -f mlflow")
    mlflow_process = subprocess.Popen(
        ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--port", "5000", "--allowed-hosts", "*", "--cors-allowed-origins", "*"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(3)

    # 3. Attempt to establish Pyngrok Tunnel for live public URL
    public_url = None
    try:
        from pyngrok import ngrok
        auth_token = os.environ.get("NGROK_AUTH_TOKEN")
        if auth_token:
            ngrok.set_auth_token(auth_token)
        
        tunnel = ngrok.connect(5000)
        public_url = tunnel.public_url
        print(f"🚀 Live Ngrok MLflow UI Dashboard URL: {public_url}", flush=True)
    except Exception as e:
        print(f"⚠️ Pyngrok tunnel could not be established in this environment: {e}", flush=True)
        print("ℹ️ Continuing with robust local SQLite tracking (`sqlite:///mlflow.db`).", flush=True)

    # Set MLflow tracking URI
    if public_url:
        mlflow.set_tracking_uri(public_url)
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
    mlflow.set_experiment("Wellness_Tourism_Package_Prediction")

    # 4. Load Data Partitions
    X_train = pd.read_csv('artifacts/X_train.csv')
    y_train = pd.read_csv('artifacts/y_train.csv').values.ravel()
    X_test = pd.read_csv('artifacts/X_test.csv')
    y_test = pd.read_csv('artifacts/y_test.csv').values.ravel()

    # 5. Assemble Composite Pipeline
    preprocessor = get_preprocessor()
    xgb_estimator = XGBClassifier(eval_metric='logloss', random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb_estimator)])

    # 6. Hyperparameter Grid Configuration
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.05, 0.1]
    }

    # 7. MLflow Experiment Tracking & Training
    with mlflow.start_run() as run:
        print("Executing GridSearchCV Hyperparameter Tuning...", flush=True)
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        # 8. Evaluation Metrics Calculation
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        # Log parameters, metrics, and model artifact with cloudpickle
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            best_model, 
            name="composite_pipeline", 
            serialization_format="cloudpickle"
        )

        # 9. Print Explicit Summary for Console & Logs
        print("\n" + "="*50, flush=True)
        print("🎉 MODEL TRAINING & EVALUATION SUMMARY 🎉", flush=True)
        print("="*50, flush=True)
        print(f"MLflow Run ID: {run.info.run_id}", flush=True)
        if public_url:
            print(f"Live MLflow UI URL: {public_url}", flush=True)
        print(f"Tracking Database: sqlite:///mlflow.db", flush=True)
        print(f"\nBest Hyperparameters:", flush=True)
        for k, v in best_params.items():
            print(f"  - {k}: {v}", flush=True)
        print(f"\nEvaluation Metrics:", flush=True)
        for m_name, m_val in metrics.items():
            print(f"  - {m_name.upper()}: {m_val:.4f}", flush=True)
        print("="*50 + "\n", flush=True)

        # Local Serialization fallback
        os.makedirs('model', exist_ok=True)
        joblib.dump(best_model, 'model/wellness_model.joblib')
        print("✅ Training complete. Model serialized to 'model/wellness_model.joblib'", flush=True)

if __name__ == "__main__":
    train_and_track_model()
