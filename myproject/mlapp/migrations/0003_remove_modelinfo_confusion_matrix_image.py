# Generated by Django 5.1.1 on 2024-09-23 00:42

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('mlapp', '0002_modelinfo_confusion_matrix_image'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='modelinfo',
            name='confusion_matrix_image',
        ),
    ]
