import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump

# Load data
df = pd.read_csv('student_performance_prediction.csv')

# Drop Student ID (not useful for prediction)
df = df.drop('Student ID', axis=1)

# Target variable: drop rows where 'Passed' is nan, and encode Yes/No to 1/0
df = df[df['Passed'].isin(['Yes', 'No'])]
df['Passed'] = df['Passed'].map({'Yes': 1, 'No': 0})

# Features and target
X = df.drop('Passed', axis=1)
y = df['Passed']

# Identify categorical and numerical columns
categorical_cols = ['Participation in Extracurricular Activities', 'Parent Education Level']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Preprocessing pipelines
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', LabelEncoder())  # Will handle in a loop since LabelEncoder doesn't work with DataFrame
])
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean'))
])

# Impute and encode categoricals manually
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])
    X[col] = LabelEncoder().fit_transform(X[col])

# Impute numericals
for col in numerical_cols:
    X[col] = X[col].astype(float)
    X[col] = X[col].fillna(X[col].mean())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model and columns
model_bundle = {
    'model': model,
    'columns': X.columns.tolist(),
    'categorical_cols': categorical_cols,
    'label_encoders': {col: LabelEncoder().fit(df[col].fillna(df[col].mode()[0])) for col in categorical_cols}
}
dump(model_bundle, 'student_performance_model.joblib')

print('Model trained and saved as student_performance_model.joblib') 