from django.shortcuts import render

# Create your views here.

from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from .models import FileUpload
from .forms import UploadFileForm, CustomUserCreationForm
from .utils import data_drill


def signup(request):
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.name = form.cleaned_data["name"]
            user.email = form.cleaned_data["email"]
            user.save()
            return redirect("login")
    else:
        form = CustomUserCreationForm()
    return render(request, "main/signup.html", {"form": form})


def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect("dashboard")
    else:
        form = AuthenticationForm()
    return render(request, "main/login.html", {"form": form})


@login_required
def upload_file(request):
    if request.method == "POST":
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES["file"]
            if not file.name.endswith(".csv"):
                return render(
                    request,
                    "main/upload.html",
                    {"form": form, "error_message": "File is not CSV type"},
                )
            FileUpload.objects.create(
                user=request.user, file_id=file.name, file_path=file
            )
            return redirect("dashboard")
    else:
        form = UploadFileForm()
    return render(request, "main/upload.html", {"form": form})


@login_required
def dashboard(request):
    file_uploads = FileUpload.objects.filter(user=request.user)
    data = []
    for file_upload in file_uploads:
        data.append(data_drill(file_upload.file_path))
    return render(request, "main/dashboard.html", {"data": data})


def logout_view(request):
    logout(request)
    return redirect("login")
