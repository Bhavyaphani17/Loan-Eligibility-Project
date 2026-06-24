import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Added LabelEncoder here
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
print("Dataset Preview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Feature Engineering
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['LoanToIncomeRatio'] = df['LoanAmount'] / df['TotalIncome'].replace(0, 1e-10)
df['LogApplicantIncome'] = np.log1p(df['ApplicantIncome'])
df['IncomeCreditInteraction'] = df['ApplicantIncome'] * df['Credit_History'].fillna(0)

# Handle outliers with capping (99th percentile)
for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome']:
    upper_limit = df[col].quantile(0.99)
    df[col] = df[col].clip(upper=upper_limit)

# Handle missing values
for column in df.columns:
    if df[column].dtype != 'object':
        df[column] = df[column].fillna(df[column].median())
    else:
        df[column] = df[column].fillna(df[column].mode()[0])

# Encode categorical variables with separate LabelEncoders
encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    if column != 'Loan_ID' and column != 'Loan_Status':
        encoders[column] = LabelEncoder()
        unique_values = df[column].unique()
        encoders[column].fit(unique_values)
        df[column] = encoders[column].transform(df[column])
# Encode target separately
le_target = LabelEncoder()
df['Loan_Status'] = le_target.fit_transform(df['Loan_Status'])

# Drop low-impact features and Loan_ID, then separate features and target
X = df.drop(['Loan_ID', 'Loan_Status', 'Gender', 'Self_Employed', 'Education', 'Loan_Amount_Term'], axis=1)
y = df['Loan_Status']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to handle imbalance
smote = SMOTE(random_state=42)
X_scaled_res, y_res = smote.fit_resample(X_scaled, y)
print("\nClass Distribution After SMOTE (Training Data):")
print(pd.Series(y_res).value_counts())

# Split the resampled data with stratification
X_train, X_test, y_train, y_test = train_test_split(X_scaled_res, y_res, test_size=0.3, random_state=42, stratify=y_res)

# GridSearchCV for hyperparameter tuning with StratifiedKFold
param_grid = {
    'max_depth': [15, 20], 
    'min_samples_split': [5, 10],
    'min_samples_leaf': [4, 6],
    'n_estimators': [200, 300]
}
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
print("\nStarting GridSearchCV... This may take a few minutes. Please do not interrupt the process.")
grid_search.fit(X_train, y_train)
print(f"\nBest Parameters: {grid_search.best_params_}")

# Best model
best_model = grid_search.best_estimator_
print(f"\nCross-Validation Accuracy (Training): {grid_search.best_score_:.3f} ± {grid_search.cv_results_['std_test_score'][grid_search.best_index_]:.3f}")

# Train and evaluate on full training set
best_model.fit(X_train, y_train)
y_train_pred = best_model.predict(X_train)
print("\nTraining Set Performance:")
print(f"Accuracy (Training): {accuracy_score(y_train, y_train_pred):.3f}")
print("Classification Report (Training):")
print(classification_report(y_train, y_train_pred))

# Evaluate on test set
y_test_pred = best_model.predict(X_test)
print("\nTest Set Performance:")
print(f"Accuracy (Test): {accuracy_score(y_test, y_test_pred):.3f}")
print("Classification Report (Test):")
print(classification_report(y_test, y_test_pred))
print(f"ROC-AUC Score (Test): {roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]):.3f}")

# Feature Importance
importances = best_model.feature_importances_
print("\nFeature Importances:")
for name, importance in zip(X.columns, importances):
    print(f"{name}: {importance:.4f}")

# Save the model, scaler, feature names, feature encoders, and target encoder
joblib.dump(best_model, 'loan_eligibility_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl')
joblib.dump(encoders, 'feature_encoders.pkl')
joblib.dump(le_target, 'label_encoder.pkl')
print("\nModel, scaler, feature names, feature encoders, and target encoder saved successfully.")