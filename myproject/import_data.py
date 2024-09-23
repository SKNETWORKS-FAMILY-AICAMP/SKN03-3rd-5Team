# import_data.py
import os
import django
import pandas as pd

# Django 환경 설정
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')
django.setup()

from blog.models import Customer

# CSV 파일 읽기
# 절대 경로를 사용할 경우, 다음과 같이 파일 경로를 지정합니다.
csv_file_path = r'C:\Users\owner\Desktop\miniproject\myproject\output.csv'

df = pd.read_csv(csv_file_path)

# 데이터 저장
for _, row in df.iterrows():
    customer = Customer(
        customerID=row['customerID'],
        gender=bool(row['gender']),
        senior_citizen=bool(row['SeniorCitizen']),
        partner=bool(row['Partner']),
        dependents=bool(row['Dependents']),
        tenure=row['tenure'],
        phone_service=bool(row['PhoneService']),
        paperless_billing=bool(row['PaperlessBilling']),
        monthly_charges=row['MonthlyCharges'],
        total_charges=row['TotalCharges'],
        churn=bool(row['Churn']),
        streaming_movies_no=row['StreamingMovies_No'],
        streaming_movies_no_internet=row['StreamingMovies_No internet service'],
        streaming_movies_yes=row['StreamingMovies_Yes'],
        contract_month_to_month=row['Contract_Month-to-month'],
        contract_one_year=row['Contract_One year'],
        contract_two_year=row['Contract_Two year'],
        payment_bank_transfer=row['PaymentMethod_Bank transfer (automatic)'],
        payment_credit_card=row['PaymentMethod_Credit card (automatic)'],
        payment_electronic_check=row['PaymentMethod_Electronic check'],
        payment_mailed_check=row['PaymentMethod_Mailed check']
    )
    customer.save()
