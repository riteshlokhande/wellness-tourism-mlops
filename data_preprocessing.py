from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def get_preprocessor():
    """Returns the composite ColumnTransformer preprocessing pipeline."""
    numeric_features = ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups', 
                        'PreferredPropertyStar', 'NumberOfTrips', 'PitchSatisfactionScore', 
                        'NumberOfChildrenVisiting', 'MonthlyIncome']
    
    categorical_features = ['TypeofContact', 'CityTier', 'Occupation', 'Gender', 'ProductPitched', 
                            'MaritalStatus', 'Passport', 'OwnCar', 'Designation']

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
    
    return preprocessor
