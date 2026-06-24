import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import sqlite3

# Load model and preprocessing artifacts
model = joblib.load('loan_eligibility_model.pkl')
encoders = joblib.load('feature_encoders.pkl')
feature_names = joblib.load('feature_names.pkl')
scaler = joblib.load('scaler.pkl')

# Hardcoded credentials (replace with secure auth in production)
USERS = {"user1": "pass123", "admin1": "admin123"}
role = st.session_state.get('role', None)

# Login page
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state['logged_in'] = True
            st.session_state['role'] = 'user' if username.startswith('user') else 'admin'
            st.rerun()
        else:
            st.error("Invalid username or password")
else:
    role = st.session_state['role']

# User view
if role == 'user':
    st.title("Loan Eligibility Predictor - User")
    with st.form("loan_form"):
        gender = st.selectbox("Gender", options=["Male", "Female", "Other"])
        married = st.selectbox("Married", options=["Yes", "No"])
        dependents = st.selectbox("Dependents", options=["0", "1", "2", "3+"])
        education = st.selectbox("Education", options=["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", options=["Yes", "No"])
        property_area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])
        credit_history = st.selectbox("Credit History", options=["1", "0"])

        applicant_income = st.number_input("Applicant Income ($)", min_value=0.0, value=5000.0, step=100.0)
        coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0.0, value=0.0, step=100.0)
        loan_amount = st.number_input("Loan Amount ($ thousands)", min_value=0.0, value=150.0, step=1.0)
        loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=0.0, value=360.0, step=12.0)

        submitted = st.form_submit_button("Predict Eligibility")

    if submitted:
        if applicant_income <= 0 or loan_amount <= 0 or loan_amount_term <= 0:
            st.error("Applicant Income, Loan Amount, and Loan Amount Term must be positive values.")
        else:
            input_data = pd.DataFrame({
                'Gender': [gender], 'Married': [married], 'Dependents': [dependents],
                'Education': [education], 'Self_Employed': [self_employed],
                'ApplicantIncome': [applicant_income], 'CoapplicantIncome': [coapplicant_income],
                'LoanAmount': [loan_amount], 'Loan_Amount_Term': [loan_amount_term],
                'Credit_History': [float(credit_history)], 'Property_Area': [property_area]
            })

            numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
            categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
            input_data[numerical_cols] = input_data[numerical_cols].fillna(input_data[numerical_cols].median())
            input_data[categorical_cols] = input_data[categorical_cols].fillna(input_data[categorical_cols].mode().iloc[0])

            for col in categorical_cols:
                input_data[col] = input_data[col].map(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
                input_data[col] = encoders[col].transform(input_data[col])

            input_data['TotalIncome'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
            input_data['LoanToIncomeRatio'] = input_data['LoanAmount'] / input_data['TotalIncome']
            input_data['LogApplicantIncome'] = np.log1p(input_data['ApplicantIncome'])
            input_data['IncomeCreditInteraction'] = input_data['TotalIncome'] * input_data['Credit_History']

            input_data = input_data.reindex(columns=feature_names, fill_value=0)
            input_data_scaled = scaler.transform(input_data)

            prediction = model.predict(input_data_scaled)[0]
            probability = model.predict_proba(input_data_scaled)[0][1]

            st.header("Prediction Result")
            if prediction == 1:
                st.success(f"Eligible for Loan (Probability: {probability:.2%})")
            else:
                st.error(f"Not Eligible for Loan (Probability: {probability:.2%})")

            # Use with statement for database connection
            with sqlite3.connect('loan_applications.db') as conn:
                c = conn.cursor()
                c.execute('''CREATE TABLE IF NOT EXISTS applications
                             (id INTEGER PRIMARY KEY AUTOINCREMENT,
                              gender TEXT, married TEXT, dependents TEXT, education TEXT,
                              self_employed TEXT, applicant_income REAL, coapplicant_income REAL,
                              loan_amount REAL, loan_amount_term REAL, credit_history REAL,
                              property_area TEXT, total_income REAL, loan_to_income_ratio REAL,
                              log_applicant_income REAL, income_credit_interaction REAL,
                              prediction INTEGER, probability REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
                c.execute('''INSERT INTO applications (gender, married, dependents, education, self_employed,
                             applicant_income, coapplicant_income, loan_amount, loan_amount_term, credit_history,
                             property_area, total_income, loan_to_income_ratio, log_applicant_income,
                             income_credit_interaction, prediction, probability)
                             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                          (gender, married, dependents, education, self_employed,
                           applicant_income, coapplicant_income, loan_amount, loan_amount_term, float(credit_history),
                           property_area, input_data['TotalIncome'].iloc[0], input_data['LoanToIncomeRatio'].iloc[0],
                           input_data['LogApplicantIncome'].iloc[0], input_data['IncomeCreditInteraction'].iloc[0],
                           prediction, probability))
                conn.commit()
            st.success("Application saved successfully!")

# Admin view
elif role == 'admin':
    st.title("Admin Dashboard")
    st.header("Loan Applications")
    with sqlite3.connect('loan_applications.db') as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM applications")
        applications = c.fetchall()
        if applications:
            columns = ['id', 'gender', 'married', 'dependents', 'education', 'self_employed', 'applicant_income',
                       'coapplicant_income', 'loan_amount', 'loan_amount_term', 'credit_history', 'property_area',
                       'total_income', 'loan_to_income_ratio', 'log_applicant_income', 'income_credit_interaction',
                       'prediction', 'probability', 'timestamp']
            df = pd.DataFrame(applications, columns=columns)
            st.dataframe(df)
        else:
            st.write("No applications yet.")

    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['role'] = None
        st.rerun()