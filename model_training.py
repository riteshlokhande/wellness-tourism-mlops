import os
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

# 2. Load Pre-split Artifacts from data_preparation.py
print("Loading training and testing artifacts...")
try:
    X_train = pd.read_csv('artifacts/X_train.csv')
    X_test = pd.read_csv('artifacts/X_test.csv')
    # Load target variables safely as 1D arrays
    y_train = pd.read_csv('artifacts/y_train.csv').values.ravel()
    y_test = pd.read_csv('artifacts/y_test.csv').values.ravel()
except FileNotFoundError as e:
    print(f"❌ Error loading artifacts: {e}. Please ensure data_preparation.py runs first.")
    raise e

# 3. Build Full Pipeline using centralized preprocessor
preprocessor = get_preprocessor()

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(eval_metric='logloss'))
])

# 4. Hyperparameter Tuning & MLflow Tracking
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5]
}

with mlflow.start_run():
    print("Starting Grid Search training...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # 5. Automated Metric Logging
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)
    
    # 6. Safe Model Serialization for Streamlit App and MLflow
    trusted_types = ["numpy.dtype", "xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
    mlflow.sklearn.log_model(
        best_model, 
        artifact_path="model", 
        skops_trusted_types=trusted_types
    )
    
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, "model/wellness_model.joblib")
    print("✅ Model successfully trained, evaluated, and serialized as model/wellness_model.joblib")
