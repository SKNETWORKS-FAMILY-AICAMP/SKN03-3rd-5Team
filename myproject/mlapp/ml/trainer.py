# mlapp/ml/trainer.py

import os

import matplotlib.pyplot as plt

from django.conf import settings

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score


def train_model(X_train, y_train):
    """
    로지스틱 회귀 모델을 사용하여 학습을 수행합니다.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest_model(data, X, y, n_estimators = 100, random_state = 42, max_depth = 10, bootstrap=False):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth, bootstrap=bootstrap)
    rf_model.fit(X, y)

    rf_predictions = rf_model.predict(X)
    rf_accuracy = accuracy_score(rf_predictions, y)

    tree_image_path = image_tree(rf_model, data)

    return rf_accuracy, tree_image_path


def image_tree(rf_model, data):
    tree_image_path = os.path.join(settings.BASE_DIR, 'mlapp', 'static', 'tree_image.png')
    plt.figure(figsize=(20, 10))
    plot_tree(rf_model.estimators_[0], feature_names=data.columns, filled=True)
    plt.savefig(tree_image_path)
    plt.close()

    return tree_image_path