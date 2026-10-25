import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

def prepare_data():
    print("--- Starting Pipeline Data Preparation ---")
    data_path = 'data/tourism.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    
    # 1. Clean Anomalies
    df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
    df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')
    df['Occupation'] = df['Occupation'].replace('Free Lancer', 'Freelancer')
    
    # 2. Drop Identifiers
    df_cleaned = df.drop(columns=['Unnamed: 0', 'CustomerID'])
    
    # 3. Stratified Split on Raw Data
    X = df_cleaned.drop('ProdTaken', axis=1)
    y = df_cleaned['ProdTaken']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Save Artifacts as 4 separate files
    os.makedirs('artifacts', exist_ok=True)
    
    X_train.to_csv('artifacts/X_train.csv', index=False)
    X_test.to_csv('artifacts/X_test.csv', index=False)
    y_train.to_csv('artifacts/y_train.csv', index=False)
    y_test.to_csv('artifacts/y_test.csv', index=False)
    
    print("✅ Pipeline data preparation complete. 4 raw artifact files saved to 'artifacts/' folder.")

if __name__ == "__main__":
    prepare_data()
