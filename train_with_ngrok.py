import time
import os
import subprocess
import urllib.request
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from pyngrok import ngrok
import joblib

print("--- Initializing Robust Local SQLite Tracking & Ngrok Dashboard ---")

# 1. Terminate existing MLflow/Ngrok processes to prevent port conflicts
os.system("pkill -f mlflow")
os.system("pkill -f ngrok")
time.sleep(2)

# 2. Set tracking URI locally to SQLite database (Guarantees 100% reliable local logging)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Wellness_Tourism_Package_Prediction")

# 3. Start MLflow Server pointing to sqlite:///mlflow.db with host headers enabled
print("Starting local MLflow server...")
mlflow_process = subprocess.Popen(
    ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--host", "127.0.0.1", "--port", "5000", "--allowed-hosts", "*", "--cors-allowed-origins", "*"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

# 4. Active Port Polling: Wait until port 5000 is actively accepting connections
print("Waiting for MLflow server to become responsive...")
server_ready = False
for i in range(20):
    try:
        urllib.request.urlopen("http://127.0.0.1:5000", timeout=1)
        server_ready = True
        print(f"✅ MLflow server is active and online (attempt {i+1}).")
        break
    except Exception:
        time.sleep(1)

if not server_ready:
    raise RuntimeError("MLflow server failed to start on port 5000 within the timeout window.")

# 5. Create public tunnel to view the dashboard live
public_url = ngrok.connect("http://127.0.0.1:5000").public_url
print("\n" + "="*60)
print(f"🚀 Live Interactive MLflow UI Dashboard URL:")
print(f"👉 {public_url}")
print("="*60 + "\n")

print("--- Comprehensive Hyperparameter Tuning (Pipeline + XGBoost) ---")
with mlflow.start_run(run_name="Comprehensive_XGBoost_Optimization"):
    param_grid = {
        'classifier__n_estimators': [50, 75, 100],
        'classifier__max_depth': [2, 3, 4],
        'classifier__colsample_bytree': [0.4, 0.5, 0.6],
        'classifier__colsample_bylevel': [0.4, 0.5, 0.6],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__reg_lambda': [0.4, 0.5, 0.6]
    }
    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    
    grid_search = GridSearchCV(estimator=full_pipeline, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    
    print("Executing Comprehensive Grid Search... (This may take a moment)")
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    y_pred_tuned = best_pipeline.predict(X_test)
    tuned_acc = accuracy_score(y_test, y_pred_tuned)
    
    print(f"✅ Tuning Complete.")
    print(f"Best Parameters: {best_params}")
    print(f"Tuned Pipeline Accuracy: {tuned_acc:.4f}")
    
    print("\n--- Logging to MLflow & Packaging Model ---")
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", tuned_acc)
    
    mlflow.sklearn.log_model(
        best_pipeline, 
        "best_pipeline_model",
        serialization_format="cloudpickle"
    )
    
    os.makedirs('model', exist_ok=True)
    joblib.dump(best_pipeline, 'model/wellness_model.joblib')
    
    print("✅ Complete pipeline successfully logged to MLflow and saved as 'wellness_model.joblib'.")
