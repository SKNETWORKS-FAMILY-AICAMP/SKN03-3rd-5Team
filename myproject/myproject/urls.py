# myproject/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('mlapp/', include('mlapp.urls')),  # mlapp의 URL 포함
]
