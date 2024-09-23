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
    X_scaled = scaler.fit_transform(X)  # 스케일링한 후 X_scaled에 저장

    # DataFrame으로 변환하여 feature_columns 유지
    X = pd.DataFrame(X_scaled, columns=X.columns)

    # 변환된 열 이름을 반환하기 위해 feature_columns 생성
    feature_columns = X.columns.tolist()

    return X, y, feature_columns  # feature_columns 반환
