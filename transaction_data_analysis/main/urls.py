# main / urls.py

from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("register/", views.register, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("upload/", views.upload_file, name="upload"),
    path("replace/<int:file_id>/", views.replace_file, name="replace_file"),
    path("dashboard/<int:file_id>/", views.dashboard, name="dashboard"),
]
