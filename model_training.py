import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def train_and_track_model():
    """
    Loads pre-split data artifacts, constructs the preprocessing pipeline,
    performs hyperparameter tuning, logs telemetry to MLflow, and saves the final model.
    """
    print("--- Starting Model Training & MLflow Tracking Pipeline ---")
    
    # Load pre-split data artifacts
    X_train = pd.read_csv('artifacts/X_train.csv')
    X_test = pd.read_csv('artifacts/X_test.csv')
    y_train = pd.read_csv('artifacts/y_train.csv').squeeze()
    y_test = pd.read_csv('artifacts/y_test.csv').squeeze()
    print("✅ Training and testing data artifacts loaded successfully.")
    
    # Dynamically identify feature types
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Define ColumnTransformer preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Construct full Scikit-Learn pipeline with XGBoost classifier
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    
    # Configure combinatorial hyperparameter search grid
    param_grid = {
        'classifier__n_estimators': [50, 75, 100],
        'classifier__max_depth': [2, 3, 4],
        'classifier__colsample_bytree': [0.4, 0.5, 0.6],
        'classifier__colsample_bylevel': [0.4, 0.5, 0.6],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__reg_lambda': [0.4, 0.5, 0.6]
    }
    
    # Execute GridSearchCV
    grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    
    print("Executing hyperparameter optimization via GridSearchCV...")
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluate performance on test set
    y_pred = best_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Tuning Complete.")
    print(f"Best Parameters: {best_params}")
    print(f"Optimized Test Accuracy: {test_accuracy:.4f}")
    
    # Initialize MLflow experiment tracking and log parameters, metrics, and model
    mlflow.set_experiment("Wellness_Tourism_Package_Prediction")
    
    with mlflow.start_run(run_name="Standalone_Script_Optimization"):
        mlflow.log_params(best_params)
        mlflow.log_metric("accuracy", test_accuracy)
        
        mlflow.sklearn.log_model(
            best_pipeline,
            "best_pipeline_model",
            serialization_format="cloudpickle"
        )
        print("✅ Experiment metrics and model successfully recorded to MLflow.")
    
    # Serialize winning composite pipeline artifact locally
    os.makedirs('model', exist_ok=True)
    model_path = 'model/wellness_model.joblib'
    joblib.dump(best_pipeline, model_path)
    print(f"✅ Winning composite pipeline successfully serialized to '{model_path}'.")
    print("--- Model Training & Tracking Pipeline Completed Successfully ---")
    
    return best_pipeline

if __name__ == "__main__":
    train_and_track_model()
