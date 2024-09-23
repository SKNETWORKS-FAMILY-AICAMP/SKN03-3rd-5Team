# models.py
from django.db import models

class Customer(models.Model):
    customerID = models.CharField(max_length=20, unique=True)
    gender = models.BooleanField()  # 0: Female, 1: Male
    senior_citizen = models.BooleanField()  # 0: No, 1: Yes
    partner = models.BooleanField()
    dependents = models.BooleanField()
    tenure = models.IntegerField()
    phone_service = models.BooleanField()
    paperless_billing = models.BooleanField()
    monthly_charges = models.FloatField()
    total_charges = models.FloatField()
    churn = models.BooleanField()

    # StreamingMovies 관련 One-Hot Encoding 필드들
    streaming_movies_no = models.BooleanField()
    streaming_movies_no_internet = models.BooleanField()
    streaming_movies_yes = models.BooleanField()

    # Contract 관련 One-Hot Encoding 필드들
    contract_month_to_month = models.BooleanField()
    contract_one_year = models.BooleanField()
    contract_two_year = models.BooleanField()

    # PaymentMethod 관련 One-Hot Encoding 필드들
    payment_bank_transfer = models.BooleanField()
    payment_credit_card = models.BooleanField()
    payment_electronic_check = models.BooleanField()
    payment_mailed_check = models.BooleanField()

    def __str__(self):
        return self.customerID
