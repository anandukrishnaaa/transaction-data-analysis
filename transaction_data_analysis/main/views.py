from django.shortcuts import render

# Create your views here.

from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from .models import FileUpload
from .forms import UploadFileForm, CustomUserCreationForm
from .utils.data_drill import perform_analysis
from datetime import datetime


def register(request):
    if request.method == "POST":
        register_form = CustomUserCreationForm(request.POST)
        if register_form.is_valid():
            user = register_form.save(commit=False)
            user.name = register_form.cleaned_data["name"]
            user.email = register_form.cleaned_data["email"]
            user.save()
            return redirect("login")
    else:
        register_form = CustomUserCreationForm()
    template_name = "main/register.html"
    context = {"register_form": register_form}
    return render(request, template_name, context)


def login_view(request):
    if request.method == "POST":
        login_form = AuthenticationForm(data=request.POST)
        if login_form.is_valid():
            user = login_form.get_user()
            login(request, user)
            return redirect("upload")
    else:
        login_form = AuthenticationForm()
    template_name = "main/login.html"
    context = {"login_form": login_form}
    return render(request, template_name, context)


@login_required
def upload_file(request):
    if request.method == "POST":
        file_upload_form = UploadFileForm(request.POST, request.FILES)
        if file_upload_form.is_valid():
            file = request.FILES["file"]
            if not file.name.endswith(".csv"):
                context = {
                    "file_upload_form": file_upload_form,
                    "error_message": "File is not CSV type",
                }
                template_name = "main/upload.html"
                return render(request, template_name, context)
            FileUpload.objects.create(
                user=request.user, file_id=file.name, file_path=file
            )
            return redirect("dashboard")
    else:
        file_upload_form = UploadFileForm()
    template_name = "main/upload.html"
    context = {"file_upload_form": file_upload_form}
    return render(request, template_name, context)


@login_required
def dashboard(request):
    file_uploads = FileUpload.objects.filter(user=request.user)
    data = []
    for file_upload in file_uploads:
        data.append(file_upload.file_path)
    template_name = "main/dashboard.html"
    context = {"data": data}
    return render(request, template_name, context)


def logout_view(request):
    logout(request)
    return redirect("login")


def home(request):
    template_name = "main/home.html"
    return render(request, template_name)
