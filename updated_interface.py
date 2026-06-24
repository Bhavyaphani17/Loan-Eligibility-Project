import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Loan Eligibility", page_icon="💰")
st.title("Loan Eligibility Prediction System")

# ------------------ Load Artifacts ------------------
model = joblib.load('loan_eligibility_model.pkl')
encoders = joblib.load('feature_encoders.pkl')
feature_names = joblib.load('feature_names.pkl')
scaler = joblib.load('scaler.pkl')

# ------------------ SQLite Setup ------------------
with sqlite3.connect('users.db') as conn:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
    ''')
    conn.execute("INSERT OR IGNORE INTO users (username, password, role) VALUES (?, ?, ?)",
                 ("Bhavya_Vadlamudi", "Admin_Password", "admin"))
    conn.commit()

# ------------------ Role Selection ------------------
role_choice = st.sidebar.selectbox("Select Role", ["User", "Admin"])

# ------------------ USER INTERFACE ------------------
if role_choice == "User":
    st.header("Welcome to the Loan Eligibility System")

    if 'user_logged_in' not in st.session_state:
        st.session_state['user_logged_in'] = False
        st.session_state['username'] = ""

    if not st.session_state['user_logged_in']:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                st.session_state['auth_mode'] = 'login'
        with col2:
            if st.button("Sign Up"):
                st.session_state['auth_mode'] = 'signup'

        auth_mode = st.session_state.get('auth_mode')

        if auth_mode == 'login':
            st.subheader("User Login")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")

            if st.button("Submit Login"):
                with sqlite3.connect('users.db') as conn:
                    user = conn.execute("SELECT role FROM users WHERE username=? AND password=?", (username, password)).fetchone()
                    if user and user[0] == "user":
                        st.session_state['user_logged_in'] = True
                        st.session_state['username'] = username
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        elif auth_mode == 'signup':
            st.subheader("Create a New Account")
            new_username = st.text_input("New Username", key="signup_user")
            new_password = st.text_input("New Password", type="password", key="signup_pass")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")

            if st.button("Submit Sign Up"):
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(new_username) < 3 or len(new_password) < 4:
                    st.error("Username or password too short.")
                else:
                    try:
                        with sqlite3.connect('users.db') as conn:
                            conn.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", (new_username, new_password, "user"))
                            conn.commit()
                            st.success("Account created successfully! Please login.")
                            st.session_state['auth_mode'] = 'login'
                    except sqlite3.IntegrityError:
                        st.error("Username already exists.")

    elif st.session_state['user_logged_in']:
        st.subheader("Loan Prediction Form")
        with st.form("loan_form"):
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            credit_history = st.selectbox("Credit History", ["1", "0"])
            applicant_income = st.number_input("Applicant Income ($)", min_value=0.0, value=5000.0, step=100.0)
            coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0.0, value=0.0, step=100.0)
            loan_amount = st.number_input("Loan Amount ($ thousands)", min_value=0.0, value=150.0, step=1.0)
            loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=0.0, value=360.0, step=12.0)
            submit = st.form_submit_button("Predict")

        if submit:
            # Save raw values
            raw_inputs = {
                'username': st.session_state['username'],
                'gender': gender,
                'married': married,
                'dependents': dependents,
                'education': education,
                'self_employed': self_employed,
                'applicant_income': applicant_income,
                'coapplicant_income': coapplicant_income,
                'loan_amount': loan_amount,
                'loan_amount_term': loan_amount_term,
                'credit_history': float(credit_history),
                'property_area': property_area
            }

            input_data = pd.DataFrame({
                'Gender': [gender], 'Married': [married], 'Dependents': [dependents],
                'Education': [education], 'Self_Employed': [self_employed],
                'ApplicantIncome': [applicant_income], 'CoapplicantIncome': [coapplicant_income],
                'LoanAmount': [loan_amount], 'Loan_Amount_Term': [loan_amount_term],
                'Credit_History': [float(credit_history)], 'Property_Area': [property_area]
            })

            for col in encoders:
                input_data[col] = input_data[col].map(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
                input_data[col] = encoders[col].transform(input_data[col])

            input_data['TotalIncome'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
            input_data['LoanToIncomeRatio'] = input_data['LoanAmount'] / input_data['TotalIncome'].replace(0, 1e-10)
            input_data['LogApplicantIncome'] = np.log1p(input_data['ApplicantIncome'])
            input_data['IncomeCreditInteraction'] = input_data['TotalIncome'] * input_data['Credit_History']

            input_data = input_data.reindex(columns=feature_names, fill_value=0)
            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            st.markdown("---")
            st.subheader("Prediction Result")
            if prediction == 1:
                st.success(f"✅ Eligible for Loan (Probability: {probability:.2%})")
            else:
                st.error(f"❌ Not Eligible for Loan (Probability: {probability:.2%})")

            with sqlite3.connect('loan_applications.db') as conn:
                conn.execute('''CREATE TABLE IF NOT EXISTS applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT, gender TEXT, married TEXT, dependents TEXT, education TEXT,
                    self_employed TEXT, applicant_income REAL, coapplicant_income REAL,
                    loan_amount REAL, loan_amount_term REAL, credit_history REAL,
                    property_area TEXT, total_income REAL, loan_to_income_ratio REAL,
                    log_applicant_income REAL, income_credit_interaction REAL,
                    prediction INTEGER, probability REAL, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

                conn.execute('''INSERT INTO applications (
                    username, gender, married, dependents, education, self_employed,
                    applicant_income, coapplicant_income, loan_amount, loan_amount_term,
                    credit_history, property_area, total_income, loan_to_income_ratio,
                    log_applicant_income, income_credit_interaction, prediction, probability)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
                    raw_inputs['username'], raw_inputs['gender'], raw_inputs['married'], raw_inputs['dependents'],
                    raw_inputs['education'], raw_inputs['self_employed'], raw_inputs['applicant_income'],
                    raw_inputs['coapplicant_income'], raw_inputs['loan_amount'], raw_inputs['loan_amount_term'],
                    raw_inputs['credit_history'], raw_inputs['property_area'],
                    input_data['TotalIncome'].iloc[0], input_data['LoanToIncomeRatio'].iloc[0],
                    input_data['LogApplicantIncome'].iloc[0], input_data['IncomeCreditInteraction'].iloc[0],
                    int(prediction), float(probability)))
                conn.commit()

            st.success("Application recorded successfully!")

        if st.button("Logout"):
            st.session_state['user_logged_in'] = False
            st.session_state['username'] = ""
            st.rerun()

# ------------------ ADMIN INTERFACE ------------------
elif role_choice == "Admin":
    st.header("Admin Login")

    if 'admin_logged_in' not in st.session_state:
        st.session_state['admin_logged_in'] = False

    if not st.session_state['admin_logged_in']:
        admin_username = st.text_input("Admin Username")
        admin_password = st.text_input("Admin Password", type="password")
        if st.button("Login as Admin"):
            with sqlite3.connect('users.db') as conn:
                user = conn.execute("SELECT role FROM users WHERE username=? AND password=?", (admin_username, admin_password)).fetchone()
                if user and user[0] == "admin":
                    st.session_state['admin_logged_in'] = True
                    st.success("Admin login successful!")
                    st.rerun()
                else:
                    st.error("Invalid admin credentials.")

    else:
        st.subheader("User Registrations")
        with sqlite3.connect('users.db') as conn:
            df_users = pd.read_sql_query("SELECT id, username, role FROM users", conn)
            st.dataframe(df_users)

        st.subheader("Loan Predictions History")
        with sqlite3.connect('loan_applications.db') as conn:
            df_apps = pd.read_sql_query("SELECT * FROM applications", conn)
            if not df_apps.empty:
                st.dataframe(df_apps)
            else:
                st.info("No applications submitted yet.")

        if st.button("Logout Admin"):
            st.session_state['admin_logged_in'] = False
            st.rerun()
