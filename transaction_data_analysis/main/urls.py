# main / urls.py

from django.urls import path
from . import views


urlpatterns = [
    path("", views.home, name="home"),
    path("register/", views.register, name="register"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("upload/", views.upload_file, name="upload"),
    path("dashboard/<int:file_id>/", views.dashboard, name="dashboard"),
    path("get_report/<int:report_id>/", views.get_report, name="get_report"),
    path(
        "train_main_model/<int:report_id>/",
        views.train_main_model,
        name="train_main_model",
    ),
    path(
        "batch_run_model/<int:report_id>/",
        views.batch_run_model,
        name="batch_run_model",
    ),
    path(
        "single_run_model/<int:report_id>/",
        views.single_run_model,
        name="single_run_model",
    ),
]
