import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_preprocessing import get_preprocessor

def train_and_track_model():
    print("--- Initiating Model Training & Tracking Lifecycle ---")
    
    # 1. Load Data Partitions
    X_train = pd.read_csv('artifacts/X_train.csv')
    y_train = pd.read_csv('artifacts/y_train.csv').values.ravel()
    X_test = pd.read_csv('artifacts/X_test.csv')
    y_test = pd.read_csv('artifacts/y_test.csv').values.ravel()

    # 2. Assemble Composite Pipeline
    preprocessor = get_preprocessor()
    xgb_estimator = XGBClassifier(eval_metric='logloss', random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb_estimator)])

    # 3. Hyperparameter Grid Configuration
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.05, 0.1]
    }

    # 4. MLflow Experiment Tracking
    mlflow.set_experiment("Wellness_Tourism_Package_Prediction")
    with mlflow.start_run():
        print("Executing GridSearchCV Hyperparameter Tuning...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        
        # 5. Evaluation Metrics Calculation
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        # Log parameters, metrics, and model artifact using cloudpickle serialization format for XGBoost compatibility
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            best_model, 
            artifact_path="composite_pipeline", 
            serialization_format="cloudpickle"
        )

        # Local Serialization
        os.makedirs('model', exist_ok=True)
        joblib.dump(best_model, 'model/wellness_model.joblib')
        print("✅ Training complete. Model serialized to 'model/wellness_model.joblib'")

if __name__ == "__main__":
    train_and_track_model()
