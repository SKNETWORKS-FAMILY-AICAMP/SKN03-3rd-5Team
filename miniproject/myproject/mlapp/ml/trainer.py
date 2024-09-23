# mlapp/ml/trainer.py

from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
    """
    로지스틱 회귀 모델을 사용하여 학습을 수행합니다.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model
