from django.urls import path
from .views import go_homepages

# localhost:8000/ -> select or insert
# localhost:8000/update/<str:task> -> update  <str:task> <- 여기에 ID를 받아옴
# localhost:8000/delete/<str:task> -> delete  <str:task> <- 여기에 ID를 받아옴
urlpatterns = [
    path('', go_homepages, name="user-index")
]