# mlapp/ml/preprocessor.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    데이터 전처리 수행: 특성 선택, 결측치 처리, 인코딩, 스케일링 등을 수행합니다.
    """
    # customerID와 타겟 변수를 제외한 모든 특성 사용
    data = data.drop(['customerID'], axis=1, errors='ignore')  # customerID 제거, 존재하지 않을 경우 무시
    X = data.drop(['churn'], axis=1)  # 타겟 변수 제거
    y = data['churn']

    # 문자열 특성 인코딩
    X = pd.get_dummies(X, drop_first=True)  # 문자열 컬럼을 숫자로 인코딩

    # 결측치 처리
    X = X.fillna(0)

    feature_columns = X.columns.tolist()  # feature_columns 추출

    return X, y, feature_columns
