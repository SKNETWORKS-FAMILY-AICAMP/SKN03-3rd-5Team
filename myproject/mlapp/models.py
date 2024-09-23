# mlapp/models.py

from django.db import models

class ModelInfo(models.Model):
    model_name = models.CharField(max_length=100)
    accuracy = models.FloatField()
    adjusted_accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    threshold = models.FloatField()
    n_train = models.IntegerField()
    n_test = models.IntegerField()
    cm_true_negative = models.IntegerField(null=True, blank=True)  # TN 값 저장
    cm_false_positive = models.IntegerField(null=True, blank=True)  # FP 값 저장
    cm_false_negative = models.IntegerField(null=True, blank=True)  # FN 값 저장
    cm_true_positive = models.IntegerField(null=True, blank=True)  # TP 값 저장
    created_at = models.DateTimeField(auto_now_add=True)
