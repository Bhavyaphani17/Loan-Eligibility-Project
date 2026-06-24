import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

try:
    # 1. Load the Test Dataset
    test_df = pd.read_csv('C:/Users/vadla/Documents/loans_eligibility_project/test_Y3wMUE5_7gLdaTN.csv')
    print("Test Dataset Preview:")
    print(test_df.head())
    print("\nTest Dataset Info:")
    print(test_df.info())

    # 2. Load the Saved Model, Scaler, Feature Names, Feature Encoders, and Label Encoder
    try:
        model = joblib.load('C:/Users/vadla/Documents/loans_eligibility_project/loan_eligibility_model.pkl')
        scaler = joblib.load('C:/Users/vadla/Documents/loans_eligibility_project/scaler.pkl')
        feature_names = joblib.load('C:/Users/vadla/Documents/loans_eligibility_project/feature_names.pkl')
        encoders = joblib.load('C:/Users/vadla/Documents/loans_eligibility_project/feature_encoders.pkl')
        le = joblib.load('C:/Users/vadla/Documents/loans_eligibility_project/label_encoder.pkl')
        print("\nLoaded LabelEncoder classes:", le.classes_ if hasattr(le, 'classes_') and le.classes_ is not None else "Invalid or None")
    except FileNotFoundError as e:
        print(f"Error: Could not load saved files. Ensure 'loan_prediction.py' has been run to generate the necessary files: {e}")
        raise

    # 3. Feature Engineering (same as training)
    test_df['TotalIncome'] = test_df['ApplicantIncome'] + test_df['CoapplicantIncome']
    test_df['LoanToIncomeRatio'] = test_df['LoanAmount'] / test_df['TotalIncome'].replace(0, 1e-10)
    test_df['LogApplicantIncome'] = np.log1p(test_df['ApplicantIncome'])
    test_df['IncomeCreditInteraction'] = test_df['ApplicantIncome'] * test_df['Credit_History'].fillna(0)

    # Handle outliers with capping (99th percentile)
    for col in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome']:
        upper_limit = test_df[col].quantile(0.99)
        test_df[col] = test_df[col].clip(upper=upper_limit)

    # 4. Preprocess the Test Data
    # Load training data to get mode for consistency
    train_df = pd.read_csv('C:/Users/vadla/Documents/loans_eligibility_project/train_u6lujuX_CVtuZ9i.csv')
    for column in test_df.select_dtypes(include=['object']).columns:
        if column != 'Loan_ID':
            mode_value = train_df[column].mode()[0]
            test_df[column] = test_df[column].fillna(mode_value)
            test_df[column] = encoders[column].transform(test_df[column])

    # Drop low-impact features and Loan_ID, then reindex
    test_df = test_df.drop(['Loan_ID', 'Gender', 'Self_Employed', 'Education', 'Loan_Amount_Term'], axis=1, errors='ignore')

    # Ensure test data has the same features as training data
    test_df = test_df.reindex(columns=feature_names, fill_value=0)
    if set(test_df.columns) != set(feature_names):
        print("Warning: Feature names mismatch. Reindexing may have altered data.")

    # Scale the features
    test_scaled = scaler.transform(test_df)

    # 5. Make Predictions with Threshold Tuning
    probabilities = model.predict_proba(test_scaled)[:, 1]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    for thresh in thresholds:
        predictions = ['Y' if p >= thresh else 'N' for p in probabilities]
        results_df = pd.DataFrame({
            'Loan_ID': pd.read_csv('C:/Users/vadla/Documents/loans_eligibility_project/test_Y3wMUE5_7gLdaTN.csv')['Loan_ID'],
            'Predicted_Loan_Status': predictions
        })
        results_df.to_csv(f'C:/Users/vadla/Documents/loans_eligibility_project/test_predictions_thresh_{thresh}.csv', index=False)
        print(f"\nThreshold {thresh} Distribution:")
        print(results_df['Predicted_Loan_Status'].value_counts())

except Exception as e:
    print(f"An error occurred while running the prediction script: {e}")
    raise