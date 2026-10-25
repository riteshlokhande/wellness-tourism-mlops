import os
import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

# 1. Purge stale SQLite tracking cache to prevent database locking/corruption
db_path = "mlflow.db"
if os.path.exists(db_path):
    os.remove(db_path)

mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("wellness_tourism_prediction")

# 2. Robust Multi-Path Dataset Loading
data_paths = ["data/wellness_dataset.csv", "../data/wellness_dataset.csv", "wellness_dataset.csv"]
df = None
for path in data_paths:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Dataset loaded successfully from: {path}")
        break

if df is None:
    raise FileNotFoundError("wellness_dataset.csv not found in any standard path.")

# 3. Data Preparation
X = df.drop(columns=["Target_Package_Taken"])
y = df["Target_Package_Taken"]

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Hyperparameter Tuning & MLflow Tracking
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5]
}

with mlflow.start_run():
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
    
    mlflow.sklearn.log_model(best_model, "model")
    
    # 6. Safe Model Serialization for Streamlit App
    joblib.dump(best_model, "wellness_model.joblib")
    print("Model successfully serialized as wellness_model.joblib")
