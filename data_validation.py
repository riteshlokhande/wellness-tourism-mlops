import pandas as pd
import os
import sys

DATA_PATH = 'data/tourism.csv'

expected_columns = [
    'Unnamed: 0', 'CustomerID', 'ProdTaken', 'Age', 'TypeofContact', 'CityTier',
    'DurationOfPitch', 'Occupation', 'Gender', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'ProductPitched', 'PreferredPropertyStar',
    'MaritalStatus', 'NumberOfTrips', 'Passport', 'PitchSatisfactionScore',
    'OwnCar', 'NumberOfChildrenVisiting', 'Designation', 'MonthlyIncome'
]

def validate_dataset():
    print(f"--- Validating Dataset at {DATA_PATH} ---")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Dataset not found at {DATA_PATH}")
        sys.exit(1) # Fail the GitHub Action job
    
    df = pd.read_csv(DATA_PATH)
    
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"❌ Validation Failed: Missing columns: {missing_cols}")
        sys.exit(1)
        
    print("✅ All expected columns are present.")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

if __name__ == "__main__":
    validate_dataset()
