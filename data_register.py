import pandas as pd
import os

DATA_PATH = "Tourism.csv"

def register_data():
    """Loads and validates the raw wellness tourism dataset."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at expected path: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Validate that expected columns are present
    expected_columns = [
        "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", 
        "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting", 
        "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar", 
        "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore", 
        "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
    ]
    
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    print("✅ Dataset registered and validated successfully.")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    print("Target distribution ('ProdTaken'):")
    print(df["ProdTaken"].value_counts(normalize=True))

if __name__ == "__main__":
    register_data()
