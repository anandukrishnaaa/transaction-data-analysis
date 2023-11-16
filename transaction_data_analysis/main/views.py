from django.shortcuts import render

# Create your views here.

from django.shortcuts import render, redirect
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from .models import FileUpload, Report
from .forms import UploadFileForm, CustomUserCreationForm
from .utils import file_handling

from .utils.logger_config import set_logger

ic = set_logger(
    print_to_console=False
)  # Set print_to_console = True for console outputs


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
            uploaded_file_details = {"file_id": file_uploaded.id, "file_size": size}
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


# @login_required
def dashboard(request, file_id):
    # Add your data analysis logic here
    # file_upload = FileUpload.objects.get(id=file_id)
    # da = file_handling.DataAnalysis(file_upload.file_path)
    uploaded_file_path = r"C:\Users\AK\Projects\code\coding-projects\transaction-data-analysis\dataset\sample.csv"
    da = file_handling.DataAnalysis(uploaded_file_path)
    exploratory_analysis = da.exploratory_analysis()
    univariate_analysis = da.univariate_data_visualization()
    bivariate_analysis = da.bivariate_data_visualization()
    analysis_result = {
        "exploratory_analysis": exploratory_analysis,
        "univariate_analysis": univariate_analysis,
        "bivariate_analysis": bivariate_analysis,
    }
    # processed_data = file_handling.file_preview(file_upload)
    # Convert the dictionary to a JSON string
    # report = Report.objects.create(
    #     user=request.user, file_id=file_id, report_json=report_json
    # )
    return render(
        request,
        "dashboard.html",
        context=analysis_result,
    )


def logout_view(request):
    logout(request)
    return redirect("login")


def home(request):
    template_name = "home.html"
    return render(request, template_name)
