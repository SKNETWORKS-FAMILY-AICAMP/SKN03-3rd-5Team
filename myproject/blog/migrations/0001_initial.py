# Generated by Django 5.1.1 on 2024-09-22 09:39

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Customer",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("customerID", models.CharField(max_length=20, unique=True)),
                ("gender", models.BooleanField()),
                ("senior_citizen", models.BooleanField()),
                ("partner", models.BooleanField()),
                ("dependents", models.BooleanField()),
                ("tenure", models.IntegerField()),
                ("phone_service", models.BooleanField()),
                ("paperless_billing", models.BooleanField()),
                ("monthly_charges", models.FloatField()),
                ("total_charges", models.FloatField()),
                ("churn", models.BooleanField()),
                ("streaming_movies_no", models.BooleanField()),
                ("streaming_movies_no_internet", models.BooleanField()),
                ("streaming_movies_yes", models.BooleanField()),
                ("contract_month_to_month", models.BooleanField()),
                ("contract_one_year", models.BooleanField()),
                ("contract_two_year", models.BooleanField()),
                ("payment_bank_transfer", models.BooleanField()),
                ("payment_credit_card", models.BooleanField()),
                ("payment_electronic_check", models.BooleanField()),
                ("payment_mailed_check", models.BooleanField()),
            ],
        ),
    ]
