# mlapp/ml/preprocessor.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    데이터 전처리 수행: 특성 선택, 결측치 처리, 인코딩, 스케일링 등을 수행합니다.
    """
    # customerID와 타겟 변수를 제외한 모든 특성 사용
    X = data.drop(['customerID', 'churn'], axis=1)
    y = data['churn']

    # 문자열 특성 인코딩
    X = pd.get_dummies(X, drop_first=True)

    # 결측치 처리
    X = X.fillna(0)

    # 데이터 표준화 (스케일링)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y
