from django.shortcuts import render

# Create your views here.

from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from .models import FileUpload
from .forms import UploadFileForm, CustomUserCreationForm, ReplaceFileForm
from django.http import JsonResponse
import pandas as pd


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
            uploaded_file = request.FILES["file"]
            file_uploaded = FileUpload.objects.create(
                user=request.user, file_id=uploaded_file.name, file_path=uploaded_file
            )
            df = pd.read_csv(file_uploaded.file_path)
            file_details = df.head(10).to_html() + df.describe().to_html()
            file_replace_form = ReplaceFileForm()
            return render(
                request,
                "main/upload.html",
                {
                    "file_details": file_details,
                    "file_id": file_uploaded.id,
                    "file_replace_form": file_replace_form,
                    "file_upload_form": file_upload_form,
                },
            )
    else:
        file_upload_form = UploadFileForm()
        file_replace_form = ReplaceFileForm()  # Create an instance of ReplaceFileForm
    return render(
        request,
        "main/upload.html",
        {
            "file_upload_form": file_upload_form,
            "file_replace_form": file_replace_form,
            "file_id": None,
        },
    )


@login_required
def replace_file(request, file_id):
    file_upload = FileUpload.objects.get(id=file_id)
    if request.method == "POST":
        file_replace_form = ReplaceFileForm(request.POST, request.FILES)
        if file_replace_form.is_valid():
            new_uploaded_file = file_replace_form.cleaned_data["new_file"]
            file_upload.file_path = new_uploaded_file
            file_upload.save()
            df = pd.read_csv(file_upload.file_path)
            file_details = df.head(10).to_html() + df.describe().to_html()
            return render(
                request,
                "main/upload.html",
                {
                    "file_replace_form": file_replace_form,
                    "file_details": file_details,
                    "file_id": file_id,
                },
            )

    else:
        file_replace_form = ReplaceFileForm()
    return redirect("upload")


@login_required
def dashboard(request, file_id):
    # Add your data analysis logic here
    file_upload = FileUpload.objects.get(id=file_id)
    df = pd.read_csv(file_upload.file_path)
    # Perform data analysis on 'df' as needed
    return render(request, "main/dashboard.html", {"file_id": file_id})


def logout_view(request):
    logout(request)
    return redirect("login")


def home(request):
    template_name = "main/home.html"
    return render(request, template_name)
