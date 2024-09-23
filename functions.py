import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pydotplus

def visualize_individual_tree(X_train, X_test, y_train, y_test, tree_index=0, n_estimators=100, max_depth=None, random_state=42, output_file="tree.png"):
    """
    랜덤 포레스트의 개별 결정 트리를 시각화하는 함수.

    Parameters:
    X (pd.DataFrame or np.ndarray): 입력 데이터 (특성 변수)
    y (pd.Series or np.ndarray): 타겟 변수
    tree_index (int): 시각화할 트리의 인덱스 (기본값은 0)
    n_estimators (int): 트리의 개수 (기본값은 100)
    max_depth (int or None): 트리의 최대 깊이 (기본값은 None)
    random_state (int): 랜덤 시드 값 (기본값은 42)

    Returns:
    None
    """
    # 랜덤 포레스트 모델 학습
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    rf.fit(X_train, y_train)

    # 특정 트리 선택
    tree = rf.estimators_[tree_index]

    # 트리를 Graphviz 형식으로 내보내기
    dot_data = export_graphviz(
        tree,
        out_file=None,
        feature_names=X_train.columns if isinstance(X_train, pd.DataFrame) else [f'Feature {i}' for i in range(X_train.shape[1])],
        class_names=True,
        filled=True,
        rounded=True,
        special_characters=True
    )

    # Graphviz 시각화
    graph = pydotplus.graph_from_dot_data(dot_data)
    # 그래프를 파일로 저장 (png 형식)
    graph.write_png(output_file)


def change_target(model, threshold, x, y):
  y_pred_proba = model.predict_proba(x)[:, 1]

  y_pred = (y_pred_proba >= threshold).astype(int)

  fasle_negatives = (y == 1) & (y_pred ==0)
  fasle_positives = (y == 0) & (y_pred ==1)

  y_corrected = y.copy()

  y_corrected[fasle_negatives] = 0
  y_corrected[fasle_positives] = 1

  return y_corrected

def evaluate_model_with_threshold(model, X, y, threshold=0.5):
    # 테스트 세트에 대한 예측 확률 수행
    y_pred_proba = model.predict_proba(X)[:, 1]  # 양성 클래스(1)의 확률

    # 임계값을 기준으로 예측 클래스 결정
    y_pred_threshold = (y_pred_proba >= threshold).astype(int)

    view_confusion_matrix(model, y, y_pred_threshold)

def view_confusion_matrix(model, y_target, y_pred):
    # 혼동 행렬 계산
    cm = confusion_matrix(y_target, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 혼동 행렬 시각화
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


def logistic(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=3000, random_state=42)

    # 모델 학습
    model.fit(X_train, y_train)

    # 테스트 세트에 대한 예측 수행
    y_pred = model.predict(X_test)

    return y_pred, model


def data_split3(X, y):
    # 데이터 분리 (훈련 세트와 테스트 세트)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def data_split2(df, target_column):
    # Features (X)와 Target (y) 분리
    X = df.drop(columns=[target_column])  # 'Churn' 컬럼을 제외한 모든 컬럼을 Features로 설정
    y = df[target_column]  # 'Churn' 컬럼을 Target으로 설정

    return X, y

def data_split(df, target_column):
    # Features (X)와 Target (y) 분리
    X = df.drop(columns=[target_column])  # 'Churn' 컬럼을 제외한 모든 컬럼을 Features로 설정
    y = df[target_column]  # 'Churn' 컬럼을 Target으로 설정

    # 데이터 분리 (훈련 세트와 테스트 세트)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test

def check_class_percentages(encoded_df):
    # 클래스 분포 확인
    class_counts = encoded_df['Churn'].value_counts()
    class_percentages = encoded_df['Churn'].value_counts(normalize=True) * 100

    print("전체 데이터셋에서의 클래스 분포:")
    print(class_counts)
    print("\n전체 데이터셋에서의 클래스 비율 (%):")
    print(class_percentages)

def csv_to_dataframe(p_file_path):

  # 예를 들어, CSV 파일이 'My Drive' 폴더에 있다면:
  file_path = p_file_path

  # CSV 파일 읽기
  df = pd.read_csv(file_path)

  return df

def drop_column(df, p_column_list):
    """
    데이터프레임에서 지정된 컬럼들을 삭제합니다.

    Parameters:
    df (pd.DataFrame): 컬럼을 삭제할 데이터프레임
    p_column_list (list): 삭제할 컬럼 이름들의 리스트

    Returns:
    pd.DataFrame: 지정된 컬럼들이 삭제된 새로운 데이터프레임
    """
    # 지정된 컬럼 리스트를 데이터프레임에서 삭제
    df_dropped = df.drop(columns=p_column_list)

    return df_dropped

def transform_columns(df):
    # 이진 변수 변환 (Churn 제외)
    binary_columns = ['gender', 'Partner', 'Dependents', 'PhoneService',
                      'PaperlessBilling']
    for col in binary_columns:
        df[col] = df[col].apply(lambda x: 1 if x in ['Yes', 'Male'] else 0)

    # TotalCharges 컬럼의 공백을 0으로 대체하고 숫자형으로 변환
    df['TotalCharges'] = df['TotalCharges'].replace(" ", 0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    # 원-핫 인코딩
    multi_category_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                              'OnlineBackup', 'DeviceProtection', 'TechSupport',
                              'StreamingTV', 'StreamingMovies', 'Contract',
                              'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_category_columns)

    # Churn 컬럼 변환
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    return df