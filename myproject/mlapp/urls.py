# mlapp/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('train/', views.train_model_view, name='train_model'),
    path('retrain/', views.retrain_random_forest_view, name='retrain_random_forest'),
    path('predict/', views.predict_result_view, name='predict_result'),
    path('model-info-list/', views.model_info_list_view, name='model_info_list'),  # URL 패턴 추가
]
