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

    original_data = data.copy()  # 전처리 전에 원본 데이터 복사

    # 2. 데이터 전처리
    X, _, feature_columns = preprocess_data(data)

    # 3. 세션에서 예측 모델 및 예측 데이터 불러오기
    logistic_model_path = os.path.join(settings.BASE_DIR, 'mlapp', 'model', 'logistic_model.pkl')
    logistic_model = joblib.load(logistic_model_path)  # 저장된 로지스틱 회귀 모델 불러오기
    threshold = request.session.get('threshold', 0.5)

    # 4. 전체 데이터에 대해 로지스틱 회귀 모델로 예측 수행
    predictions = adjust_threshold(logistic_model, X, threshold)

    # 5. 예측 결과로 타겟 값 변경
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

    else:
        return redirect('train_model')  # 예측 결과 길이가 데이터 길이와 맞지 않으면 다시 학습 페이지로 리다이렉트

    # 6. 사용자가 입력한 max_depth 값 가져오기 (GET 또는 POST 요청에 따라 다르게 처리)
    if request.method == 'POST':
        max_depth = int(request.POST.get('max_depth', 3))  # POST 요청으로부터 max_depth 값 가져오기
    else:
        max_depth = 3  # 기본값 설정

    # 7. 랜덤 포레스트 모델 학습
    rf_model, rf_accuracy, tree_image_path = train_random_forest_model(data, X, data['churn'], max_depth=max_depth)

    

    # 8. FP 데이터 중에서 랜덤 포레스트 모델을 사용하여 예측 수행
    fp_X = X.loc[fp_data.index]  # fp_data의 인덱스에 해당하는 특징 데이터
    fp_predictions = rf_model.predict(fp_X)  # 랜덤 포레스트 모델로 FP 데이터 예측 수행
    fp_data_filtered = fp_data[fp_predictions == 1]  # 예측 결과가 1인 데이터만 필터링

    # Access the first tree in the Random Forest model
    tree = rf_model.estimators_[0]

    # Prepare to collect flip results
    flip_results = []

    # Features DataFrame (ensure indices align)
    X_fp_filtered = X.loc[fp_data_filtered.index]

    # retrain_random_forest_view 함수 내에서
    for idx, X_sample in X_fp_filtered.iterrows():
        X_sample_reshaped = X_sample.values.reshape(1, -1)
        sample_flip_results = flip_decisions_for_sample(
            tree,
            X_sample_reshaped,
            idx,
            feature_names=X.columns,
            original_data=original_data  # 원본 데이터프레임 전달
        )
        flip_results.extend(sample_flip_results)

    # 10. Results return
    context = {
        'rf_accuracy': rf_accuracy,
        'n_train': len(X),
        'n_test': len(X),
        'tree_image_path': tree_image_path,
        'max_depth': max_depth,
        'fp_data': fp_data.to_dict(orient='records'),
        'fn_data': fn_data.to_dict(orient='records'),
        'fp_data_filtered': fp_data_filtered.to_dict(orient='records'),
        'flip_results': flip_results  # Include the flip results in the context
    }
    return render(request, 'mlapp/random_forest_result.html', context)


def flip_decisions_for_sample(tree, X_sample, sample_id, feature_names, original_data):
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    node_value = tree.tree_.value
    impurity = tree.tree_.impurity  # 노드의 지니 불순도 값
    n_node_samples = tree.tree_.n_node_samples  # 각 노드의 샘플 개수

    X_sample_values = X_sample.flatten()

    # 원래의 결정 경로와 예측 가져오기
    node_indicator = tree.decision_path(X_sample)
    node_index = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]]
    leaf_node = node_index[-1]
    original_prediction = node_value[leaf_node].argmax()

    # 원본 데이터에서 해당 샘플의 특징 값 가져오기
    original_sample = original_data.loc[sample_id]

    # 원래 경로의 각 노드에 대해 반복 (리프 노드 제외)
    flip_results = []
    for current_node in node_index[:-1]:
        original_feature = feature[current_node]
        if original_feature == -2:
            continue  # 리프 노드일 경우 건너뜁니다.

        original_threshold = threshold[current_node]

        # 노드 분기 조건 생성
        feature_name = feature_names[original_feature]
        threshold_value = original_threshold

        # 노드의 지니 불순도 값 및 샘플 개수 가져오기
        gini_value = impurity[current_node]
        n_samples_at_node = n_node_samples[current_node]  # 해당 노드의 샘플 개수

        # 샘플이 원래 어떤 방향으로 갔는지 결정
        if X_sample_values[original_feature] <= original_threshold:
            original_direction = 'left'
            alternative_node = tree.tree_.children_right[current_node]
            alternative_direction = 'right'
            node_condition = f"{feature_name} ≤ {threshold_value:.3f}"
        else:
            original_direction = 'right'
            alternative_node = tree.tree_.children_left[current_node]
            alternative_direction = 'left'
            node_condition = f"{feature_name} > {threshold_value:.3f}"

        # 대체 경로가 있을 경우
        if alternative_node != -1:
            alternative_prediction = traverse_alternative_path(tree, X_sample_values, alternative_node)
            if alternative_prediction != original_prediction:
                flip_results.append({
                    'sample_id': sample_id,
                    'feature_name': feature_name,
                    'feature_value': original_sample[feature_name],
                    'original_prediction': original_prediction,
                    'alternative_prediction': alternative_prediction,
                    'node_condition': node_condition,
                    'gini_value': gini_value,
                    'node_index': current_node,  # 노드 인덱스 추가
                    'n_samples_at_node': n_samples_at_node  # 샘플 개수 추가
                })
    return flip_results




def traverse_alternative_path(tree, X_sample_values, start_node):
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    node_value = tree.tree_.value

    current_node = start_node
    while tree.tree_.children_left[current_node] != -1:
        feature_index = feature[current_node]
        threshold_value = threshold[current_node]
        if X_sample_values[feature_index] <= threshold_value:
            current_node = tree.tree_.children_left[current_node]
        else:
            current_node = tree.tree_.children_right[current_node]
    leaf_prediction = node_value[current_node].argmax()
    return leaf_prediction



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