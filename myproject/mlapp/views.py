# mlapp/views.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from django.conf import settings
from django.shortcuts import render, redirect
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from .ml.trainer import train_random_forest_model
from .ml.data_loader import load_customer_data
from .ml.preprocessor import preprocess_data
from .ml.evaluator import evaluate_model, evaluate_with_threshold, adjust_threshold, plot_confusion_matrix
from .models import ModelInfo


# 기존 모델 학습 뷰
# mlapp/views.py

def train_model_view(request):
    # 1. 데이터 로드
    data = load_customer_data()

    # 2. 데이터 전처리
    X, y, feature_columns = preprocess_data(data)

    # 3. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 4. 모델 학습
    model = LogisticRegression(max_iter=1000)
    X_train_df = pd.DataFrame(X_train, columns=feature_columns)  # X_train을 DataFrame으로 변환
    model.fit(X_train_df, y_train)

    # 5. 모델 파일로 저장 (필요할 경우)
    model_path = os.path.join(settings.BASE_DIR, 'mlapp', 'model', 'logistic_model.pkl')
    joblib.dump(model, model_path)

    # 6. 모델 기본 평가
    accuracy, cm = evaluate_model(model, X_test, y_test)

    # 7. 임계값 가져오기 (GET 또는 POST 요청에 따라 다르게 처리)
    if request.method == 'POST':
        # POST 요청으로부터 임계값 가져오기
        threshold = float(request.POST.get('threshold', 0.5))
    else:
        # GET 요청으로부터 임계값 가져오기
        threshold = float(request.GET.get('threshold', 0.5))

    # 8. 임계값 조정 평가
    adjusted_accuracy, precision, recall, adjusted_cm = evaluate_with_threshold(model, X, y, threshold)

    # 9. 혼동 행렬의 각 요소 계산
    # adjusted_cm이 2x2 행렬일 경우, 이를 ravel()로 풀어서 TN, FP, FN, TP 값을 가져옵니다.
    tn, fp, fn, tp = adjusted_cm.ravel()

    # 10. 혼동 행렬 이미지 생성 (필요한 경우)
    confusion_matrix_image = plot_confusion_matrix(adjusted_cm)

    # 11. 예측 결과를 numpy array로 저장
    predictions = adjust_threshold(model, X_test, threshold)
    request.session['predictions'] = predictions.tolist()  # 리스트로 변환하여 세션에 저장
    request.session['threshold'] = threshold
    request.session['accuracy'] = adjusted_accuracy  # 필요 시 정확도 저장

    # 12. 모델 정보를 데이터베이스에 저장 (POST 요청일 경우)
    if request.method == 'POST' and 'save_model' in request.POST:
        model_info = ModelInfo.objects.create(
            model_name="Logistic Regression",
            accuracy=accuracy,
            adjusted_accuracy=adjusted_accuracy,
            precision=precision,
            recall=recall,
            threshold=threshold,
            n_train=len(X_train),
            n_test=len(X_test),
            cm_true_negative=tn,  # TN 값 저장
            cm_false_positive=fp,  # FP 값 저장
            cm_false_negative=fn,  # FN 값 저장
            cm_true_positive=tp   # TP 값 저장
        )
        model_info.save()
        return redirect('model_info_list')  # 저장 후 모델 정보 리스트 페이지로 이동

    # 13. 결과 반환
    context = {
        'accuracy': accuracy,
        'adjusted_accuracy': adjusted_accuracy,
        'precision': precision,
        'recall': recall,
        'threshold': threshold,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'confusion_matrix_image': confusion_matrix_image,
        'cm_true_negative': tn,
        'cm_false_positive': fp,
        'cm_false_negative': fn,
        'cm_true_positive': tp
    }
    return render(request, 'mlapp/train_model.html', context)

def retrain_random_forest_view(request):
    # 1. 전체 데이터 불러오기
    data = load_customer_data()
    X, _, feature_columns = preprocess_data(data)  # 타겟 값과 feature_columns도 반환

    # 2. 세션에서 예측 모델 및 예측 데이터 불러오기
    model_path = os.path.join(settings.BASE_DIR, 'mlapp', 'model', 'logistic_model.pkl')
    model = joblib.load(model_path)  # 저장된 모델 불러오기
    threshold = request.session.get('threshold', 0.5)

    # 3. 전체 데이터에 대해 예측 수행
    predictions = adjust_threshold(model, X, threshold)

    # 4. 예측 결과로 타겟 값 변경
    if len(predictions) == len(data):
        # 변경된 데이터를 식별
        data['old_churn'] = data['churn']
        data['churn'] = predictions
        
        # 변경된 데이터 필터링
        changed_data = data[data['old_churn'] != data['churn']]
        
        # FP와 FN 구분
        fp_data = changed_data[(changed_data['old_churn'] == 0) & (changed_data['churn'] == 1)]
        fn_data = changed_data[(changed_data['old_churn'] == 1) & (changed_data['churn'] == 0)]

        # X를 DataFrame으로 변환
        if feature_columns is None:
            raise ValueError("feature_columns이 제공되지 않았습니다.")
        
        if len(feature_columns) != X.shape[1]:
            raise ValueError(f"열 개수 불일치: X has {X.shape[1]} columns, but feature_columns has {len(feature_columns)} columns")

        X = pd.DataFrame(X, columns=feature_columns)  # 데이터프레임으로 변환하고, 적절한 컬럼 이름 지정

        # FP 데이터 중에서 모델 예측 결과가 1인 데이터만 필터링
        fp_X = X.loc[fp_data.index]  # fp_data의 인덱스에 해당하는 특징 데이터
        fp_predictions = model.predict(fp_X)  # FP 데이터에 대한 모델 예측 수행
        fp_data_filtered = fp_data[fp_predictions == 1]  # 예측 결과가 1인 데이터만 필터링

    else:
        return redirect('train_model')  # 예측 결과 길이가 데이터 길이와 맞지 않으면 다시 학습 페이지로 리다이렉트

    # 5. 사용자가 입력한 max_depth 값 가져오기 (GET 또는 POST 요청에 따라 다르게 처리)
    if request.method == 'POST':
        max_depth = int(request.POST.get('max_depth', 3))  # POST 요청으로부터 max_depth 값 가져오기
    else:
        max_depth = 3  # 기본값 설정

    # 6. 랜덤 포레스트 모델 학습
    X_modified, _, _ = preprocess_data(data)  # 타겟 값은 교체되었으므로 X_modified만 필요

    rf_accuracy, tree_image_path = train_random_forest_model(data, X_modified, predictions, max_depth=max_depth)

    # 7. 결과 반환
    context = {
        'rf_accuracy': rf_accuracy,
        'n_train': len(X),
        'n_test': len(X),
        'tree_image_path': tree_image_path,
        'max_depth': max_depth,  # max_depth 값을 템플릿으로 전달
        'fp_data': fp_data.to_dict(orient='records'),  # 변경된 FP 데이터를 템플릿으로 전달
        'fn_data': fn_data.to_dict(orient='records'),  # 변경된 FN 데이터를 템플릿으로 전달
        'fp_data_filtered': fp_data_filtered.to_dict(orient='records')  # 예측 결과가 1인 FP 데이터만 전달
    }
    return render(request, 'mlapp/random_forest_result.html', context)


def predict_result_view(request):
    # 세션에서 예측 결과와 임계값 불러오기
    predictions = request.session.get('predictions', None)
    threshold = request.session.get('threshold', 0.5)
    accuracy = request.session.get('accuracy', None)

    if predictions is not None:
        # 예측 결과를 리스트로부터 numpy array로 변환
        predictions = np.array(predictions)

        # 결과 반환
        context = {
            'predictions': predictions,
            'threshold': threshold,
            'accuracy': accuracy
        }
        return render(request, 'mlapp/predict_result.html', context)
    else:
        # 예측 결과가 세션에 없을 경우 학습 페이지로 리다이렉트
        return redirect('train_model')

def model_info_list_view(request):
    model_infos = ModelInfo.objects.all().order_by('-created_at')
    return render(request, 'mlapp/model_info_list.html', {'model_infos': model_infos})
