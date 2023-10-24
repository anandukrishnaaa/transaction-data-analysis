from django.urls import path, include
from .views import home_with_data

urlpatterns = [path("", home_with_data, name="home")]
