# main / urls.py

from django.urls import path
from . import views

urlpatterns = [
    path("signup/", views.signup, name="signup"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("upload/", views.upload_file, name="upload"),
    path("dashboard/", views.dashboard, name="dashboard"),
]
