# mlapp/ml/evaluator.py

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
# mlapp/ml/evaluator.py
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # 백엔드 설정을 Agg로 변경

import io
import base64
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    모델의 성능을 평가하고 정확도와 혼동 행렬을 반환합니다.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    cm = confusion_matrix(y_test, predictions)
    return accuracy, cm

def adjust_threshold(model, X_test, threshold=0.5):
    """
    모델의 예측 확률에 임계값을 적용하여 새로운 예측 결과를 반환합니다.
    """
    # 예측 확률 구하기
    probabilities = model.predict_proba(X_test)[:, 1]  # 양성 클래스(1)에 대한 확률만 사용

    # 임계값에 따라 1 또는 0으로 변환
    adjusted_predictions = np.where(probabilities >= threshold, 1, 0)
    return adjusted_predictions

def plot_confusion_matrix(cm):
    """
    혼동 행렬을 플롯으로 생성하고 이를 Base64 인코딩된 이미지로 반환합니다.
    """
    fig, ax = plt.subplots()
    cm_display = ConfusionMatrixDisplay(cm)
    cm_display.plot(ax=ax)
    plt.title('Confusion Matrix')
    plt.close(fig)

    # 이미지를 메모리에 저장
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return image_base64

def evaluate_with_threshold(model, X_test, y_test, threshold):
    """
    임계값을 조정하여 모델의 성능(정확도, 정밀도, 재현율)을 평가합니다.
    """
    # 임계값 조정된 예측값
    adjusted_predictions = adjust_threshold(model, X_test, threshold)
    
    # 성능 평가
    accuracy = accuracy_score(y_test, adjusted_predictions)
    precision = precision_score(y_test, adjusted_predictions)
    recall = recall_score(y_test, adjusted_predictions)
    cm = confusion_matrix(y_test, adjusted_predictions)

    return accuracy, precision, recall, cm
