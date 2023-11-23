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
        "get_report_pdf/<int:report_id>/",
        views.get_report_pdf,
        name="get_report_pdf",
    ),
]
