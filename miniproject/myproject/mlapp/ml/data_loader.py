# mlapp/ml/data_loader.py

import pandas as pd
from blog.models import Customer

def load_customer_data():
    """
    Customer 데이터를 데이터베이스에서 불러와 Pandas DataFrame으로 반환합니다.
    """
    customers = Customer.objects.all()
    data = pd.DataFrame(list(customers.values()))
    return data
