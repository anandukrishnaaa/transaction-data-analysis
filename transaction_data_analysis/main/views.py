from django.shortcuts import render

# Create your views here.

from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from .models import FileUpload
from .forms import UploadFileForm, CustomUserCreationForm
from .utils import extra
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
    template_name = "register.html"
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
    template_name = "login.html"
    context = {"login_form": login_form}
    return render(request, template_name, context)


@login_required
def upload_file(request):
    if request.method == "POST":
        file_upload_form = UploadFileForm(request.POST, request.FILES)
        if file_upload_form.is_valid():
            uploaded_file = request.FILES["file"]
            size = uploaded_file.size
            file_uploaded = FileUpload.objects.create(
                user=request.user, file_id=uploaded_file.name, file_path=uploaded_file
            )
            uploaded_file_details = extra.file_preview(file_uploaded, size)
            context = {
                "uploaded_file_details": uploaded_file_details,
            }
            template = "upload.html"
            return render(
                request,
                template,
                context,
            )
    else:
        file_upload_form = UploadFileForm()
        return render(
            request,
            "upload.html",
            {
                "file_upload_form": file_upload_form,
                "uploaded_file_details": None,
            },
        )


@login_required
def dashboard(request, file_id):
    # Add your data analysis logic here
    file_upload = FileUpload.objects.get(id=file_id)
    df = pd.read_csv(file_upload.file_path)
    # Perform data analysis on 'df' as needed
    return render(request, "dashboard.html", {"file_id": file_id})


def logout_view(request):
    logout(request)
    return redirect("login")


def home(request):
    template_name = "home.html"
    return render(request, template_name)
