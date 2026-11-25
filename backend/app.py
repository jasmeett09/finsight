import re
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # Allow React frontend requests

# Load trained model bundles
score_bundle = joblib.load('credit_score_bundle.pkl')
loan_bundle = joblib.load('loan_eligibility_bundle.pkl')


def parse_history_months(val):
  """Converts '8 Years 4 Months' or '100' into total months."""
  if isinstance(val, (int, float)):
    return float(val)
  text = str(val)
  years = re.search(r'(\d+)\s*year', text, re.I)
  months = re.search(r'(\d+)\s*month', text, re.I)
  if years or months:
    y = int(years.group(1)) if years else 0
    m = int(months.group(1)) if months else 0
    return float(y * 12 + m)
  nums = re.findall(r'\d+\.?\d*', text)
  return float(nums[0]) if nums else 60.0


def safe_encode(encoder, value, fallback_idx=0):
  classes = list(encoder.classes_)
  if value in classes:
    return int(encoder.transform([value])[0])
  return fallback_idx


@app.route('/api/predict-credit', methods=['POST'])
def predict_credit():
  data = request.json or {}

  # Map frontend dropdown labels to Score.csv categories
  mix_map = {
      'Good': 'Good',
      'Standard': 'Standard',
      'Poor': 'Bad',
      'Balanced': 'Good',
      'Varied': 'Standard',
  }
  min_pay_map = {'Yes': 'Yes', 'No': 'No', 'Usually': 'NM'}
  behav_map = {
      'High spending': 'High_spent_Medium_value_payments',
      'Low spending': 'Low_spent_Small_value_payments',
      'Medium spending': 'Low_spent_Medium_value_payments',
      'On-time payments': 'High_spent_Large_value_payments',
      'Delayed payments': 'Low_spent_Small_value_payments',
  }

  annual_income = float(data.get('Annual income', 50000) or 50000)
  outstanding_debt = float(data.get('Outstanding debt', 1000) or 0)

  row = {
      'Delay_from_due_date': float(data.get('Delay from due date', 5) or 0),
      'Num_of_Delayed_Payment': float(
          data.get('Number of delayed payments', 2) or 0
      ),
      'Num_Credit_Inquiries': float(
          data.get('Number of credit inquiries', 2) or 0
      ),
      'Credit_Utilization_Ratio': float(
          data.get('Credit utilization ratio', 30) or 0
      ),
      'Credit_History_Age': parse_history_months(
          data.get('Credit history age', 60)
      ),
      'Payment_of_Min_Amount': safe_encode(
          score_bundle['encoders']['Payment_of_Min_Amount'],
          min_pay_map.get(data.get('Payment of minimum amount'), 'No'),
      ),
      'Amount_invested_monthly': float(
          data.get('Amount invested monthly', 100) or 0
      ),
      'Monthly_Balance': float(data.get('Monthly balance', 300) or 0),
      'Credit_Mix': safe_encode(
          score_bundle['encoders']['Credit_Mix'],
          mix_map.get(data.get('Credit mix'), 'Standard'),
      ),
      'Payment_Behaviour': safe_encode(
          score_bundle['encoders']['Payment_Behaviour'],
          behav_map.get(
              data.get('Payment behaviour'),
              'High_spent_Medium_value_payments',
          ),
      ),
      'Age': float(data.get('Age', 30) or 30),
      'Annual_Income': annual_income,
      'Num_Bank_Accounts': float(data.get('Number of bank accounts', 3) or 1),
      'Num_Credit_Card': float(data.get('Number of credit cards', 2) or 1),
      'Interest_Rate': float(data.get('Interest rate', 10) or 10),
      'Num_of_Loan': float(data.get('Number of loans', 1) or 0),
      'Monthly_Inhand_Salary': float(
          data.get('Monthly in-hand salary', annual_income / 12)
          or (annual_income / 12)
      ),
      'Changed_Credit_Limit': float(data.get('Changed credit limit', 5) or 0),
      'Outstanding_Debt': outstanding_debt,
      'Total_EMI_per_month': float(data.get('Total EMI per month', 100) or 0),
      'Debt_to_Income_Ratio': outstanding_debt / max(annual_income, 1.0),
  }

  df_input = pd.DataFrame([row])[score_bundle['features']]
  pred_idx = int(score_bundle['model'].predict(df_input)[0])
  pred_label = score_bundle['encoders']['Credit_Score'].inverse_transform(
      [pred_idx]
  )[0]
  probs = score_bundle['model'].predict_proba(df_input)[0]
  confidence = round(float(np.max(probs)) * 100, 1)

  tone = (
      'good'
      if pred_label == 'Good'
      else ('warn' if pred_label == 'Standard' else 'bad')
  )
  return jsonify(
      {'result': pred_label, 'tone': tone, 'confidence': confidence}
  )


@app.route('/api/predict-loan', methods=['POST'])
def predict_loan():
  data = request.json or {}

  # Map frontend dropdowns to dataset categories
  gender_map = {
      'Female': 'Female',
      'Male': 'Male',
      'Non-binary': 'Male',
      'Prefer not to say': 'Female',
  }
  marital_map = {'Single': 'Single', 'Married': 'Married', 'Other': 'Divorced'}
  edu_map = {
      'Graduate': 'Graduate',
      'Postgraduate': 'Postgraduate',
      'High school': 'High School',
      'Other': 'Graduate',
  }
  emp_map = {
      'Salaried': 'Employed',
      'Self-employed': 'Self-Employed',
      'Contract': 'Self-Employed',
      'Unemployed': 'Unemployed',
  }
  occ_map = {
      'Professional': 'Professional',
      'Business': 'Business',
      'Service': 'Salaried',
      'Skilled worker': 'Freelancer',
      'Other': 'Salaried',
  }
  res_map = {
      'Owned': 'Own',
      'Rented': 'Rent',
      'Family owned': 'Own',
      'Other': 'Other',
  }
  purpose_map = {
      'Home': 'Home',
      'Vehicle': 'Vehicle',
      'Education': 'Education',
      'Personal': 'Personal',
      'Business': 'Personal',
      'Other': 'Personal',
  }
  type_map = {
      'Secured': 'Secured',
      'Unsecured': 'Unsecured',
      'Fixed rate': 'Secured',
      'Variable rate': 'Unsecured',
  }
  history_map = {
      'No previous loans': 0,
      'Good': 1,
      'Some missed payments': 0,
      'Defaults': 0,
  }
  freq_map = {'Low': 8, 'Moderate': 16, 'High': 25}
  risk_map = {'Low': 0.18, 'Medium': 0.50, 'High': 0.85}
  stab_map = {'Stable': 0.80, 'Variable': 0.50, 'Seasonal': 0.35}

  city_raw = str(data.get('City / town', 'Urban')).strip().title()
  city_val = (
      city_raw if city_raw in ['Urban', 'Suburban', 'Rural'] else 'Urban'
  )

  annual_income = float(data.get('Annual income', 75000) or 75000)
  monthly_income = float(
      data.get('Monthly income', annual_income / 12) or (annual_income / 12)
  )
  monthly_expenses = float(data.get('Monthly expenses', 2000) or 0)
  existing_monthly_debt = float(
      data.get('Existing monthly debt payment', 500) or 0
  )
  loan_req = float(data.get('Loan amount requested', 20000) or 1000)
  loan_term = max(float(data.get('Loan term', 36) or 36), 1.0)
  rate = float(data.get('Interest rate', 8.5) or 8.5)

  # Calculate Proposed EMI & Post-Loan Affordability Ratios automatically
  r_monthly = (rate / 100.0) / 12.0
  if r_monthly > 0:
    proposed_emi = (
        loan_req
        * r_monthly
        * ((1 + r_monthly) ** loan_term)
        / (((1 + r_monthly) ** loan_term) - 1)
    )
  else:
    proposed_emi = loan_req / loan_term

  total_monthly_debt_after = existing_monthly_debt + proposed_emi
  current_dti = existing_monthly_debt / max(monthly_income, 1.0)
  post_loan_dti = total_monthly_debt_after / max(monthly_income, 1.0)
  residual_income = monthly_income - monthly_expenses - existing_monthly_debt
  residual_after_loan = residual_income - proposed_emi
  collateral = float(data.get('Collateral value', 0) or 0)
  ltv = (
      (loan_req / collateral)
      if collateral > 0
      else float(data.get('Loan-to-value', 0) or 0) / 100.0
  )

  enc = loan_bundle['encoders']
  row = {
      'Gender': safe_encode(
          enc['Gender'], gender_map.get(data.get('Gender'), 'Male')
      ),
      'Age': float(data.get('Age', 32) or 32),
      'Marital_Status': safe_encode(
          enc['Marital_Status'],
          marital_map.get(data.get('Marital status'), 'Married'),
      ),
      'Dependents': float(data.get('Dependents', 0) or 0),
      'Education': safe_encode(
          enc['Education'], edu_map.get(data.get('Education'), 'Graduate')
      ),
      'Employment_Status': safe_encode(
          enc['Employment_Status'],
          emp_map.get(data.get('Employment status'), 'Employed'),
      ),
      'Occupation_Type': safe_encode(
          enc['Occupation_Type'],
          occ_map.get(data.get('Occupation type'), 'Salaried'),
      ),
      'Residential_Status': safe_encode(
          enc['Residential_Status'],
          res_map.get(data.get('Residential status'), 'Own'),
      ),
      'City_Town': safe_encode(enc['City_Town'], city_val),
      'Annual_Income': annual_income,
      'Monthly_Expenses': monthly_expenses,
      'Credit_Score': float(data.get('Credit score', 720) or 720),
      'Existing_Loans': float(data.get('Existing loans', 1) or 0),
      'Total_Existing_Loan_Amount': float(
          data.get('Total existing loan amount', 5000) or 0
      ),
      'Outstanding_Debt': float(data.get('Outstanding debt', 5000) or 0),
      'Loan_History': history_map.get(data.get('Loan history'), 1),
      'Loan_Amount_Requested': loan_req,
      'Loan_Term': loan_term,
      'Loan_Purpose': safe_encode(
          enc['Loan_Purpose'], purpose_map.get(data.get('Loan purpose'), 'Home')
      ),
      'Interest_Rate': rate,
      'Loan_Type': safe_encode(
          enc['Loan_Type'], type_map.get(data.get('Loan type'), 'Secured')
      ),
      'Co_Applicant': safe_encode(
          enc['Co_Applicant'], data.get('Co-applicant', 'No')
      ),
      'Bank_Account_History': float(data.get('Bank account history', 5) or 1),
      'Transaction_Frequency': freq_map.get(
          data.get('Transaction frequency'), 16
      ),
      'Default_Risk': risk_map.get(data.get('Default risk'), 0.25),
      'Monthly_Income': monthly_income,
      'Existing_Debt_to_Income': (
          float(data.get('Existing debt-to-income ratio', 15) or 0) / 100.0
      ),
      'Existing_Monthly_Debt_Payment': existing_monthly_debt,
      'Proposed_EMI': proposed_emi,
      'Total_Monthly_Debt_After_Loan': total_monthly_debt_after,
      'Current_DTI': current_dti,
      'Post_Loan_DTI': post_loan_dti,
      'Residual_Income': residual_income,
      'Residual_Income_After_Loan': residual_after_loan,
      'Loan_to_Annual_Income': loan_req / max(annual_income, 1.0),
      'Income_Volatility_6M': (
          float(data.get('Income volatility (6 months)', 5) or 0) / 100.0
      ),
      'Income_Growth_Rate_6M': (
          float(data.get('Income growth rate (6 months)', 3) or 0) / 100.0
      ),
      'Credit_Utilization': (
          float(data.get('Credit utilization', 25) or 0) / 100.0
      ),
      'Credit_Inquiries_6M': float(
          data.get('Credit inquiries (6 months)', 1) or 0
      ),
      'Delinquencies_12M': float(
          data.get('Delinquencies (12 months)', 0) or 0
      ),
      'Credit_History_Length_Years': float(
          data.get('Credit history length', 5) or 1
      ),
      'Savings_Balance': float(data.get('Savings balance', 10000) or 0),
      'Cash_Flow_Stability': stab_map.get(
          data.get('Cash flow stability'), 0.70
      ),
      'Collateral_Value': collateral,
      'Loan_to_Value': ltv,
      'Employment_Duration_Years': float(
          data.get('Employment duration', 3) or 1
      ),
  }

  df_input = pd.DataFrame([row])[loan_bundle['features']]
  pred = int(loan_bundle['model'].predict(df_input)[0])
  probs = loan_bundle['model'].predict_proba(df_input)[0]
  confidence = round(float(np.max(probs)) * 100, 1)

  result_label = 'Eligible' if pred == 1 else 'Not eligible'
  tone = 'good' if pred == 1 else 'bad'
  return jsonify({
      'result': result_label,
      'tone': tone,
      'confidence': confidence,
      'proposed_emi': round(proposed_emi, 2),
  })


if __name__ == '__main__':
  app.run(port=5000, debug=True)