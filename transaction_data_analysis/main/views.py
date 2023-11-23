# Create your views here.

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from .models import FileUpload, Report
from .forms import UploadFileForm, CustomUserCreationForm
from .utils import file_handler
import pickle


from .utils.logger_config import set_logger

ic = set_logger(
    print_to_console=False
)  # Set print_to_console = True for console outputs


def register(request):
    if request.method == "POST":
        register_form = CustomUserCreationForm(request.POST)
        if register_form.is_valid():
            user = register_form.save()
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
                "user_reports": Report.objects.filter(user=request.user),
            }

            template = "upload.html"
            return render(
                request,
                template,
                context,
            )
    else:
        file_upload_form = UploadFileForm()
        # Fetch reports associated with current user
        user_reports = Report.objects.filter(user=request.user)
        context = {
            "file_upload_form": file_upload_form,
            "uploaded_file_details": None,
            "user_reports": user_reports,
        }
        return render(
            request,
            "upload.html",
            context=context,
        )


@login_required
def dashboard(request, file_id):
    # Add your data analysis logic here
    file_upload = FileUpload.objects.get(id=file_id)
    da = file_handler.DataAnalysis(file_upload.file_path)
    exploratory_analysis = da.exploratory_analysis()
    univariate_analysis = da.univariate_data_visualization()
    bivariate_analysis = da.bivariate_data_visualization()
    multivariate_analysis = da.multivariate_data_visualization()
    analysis_result = {
        "exploratory_analysis": exploratory_analysis,
        "univariate_analysis": univariate_analysis,
        "bivariate_analysis": bivariate_analysis,
        "multivariate_analysis": multivariate_analysis,
    }
    # Store Report
    report = Report.objects.create(
        user=request.user, file_id=file_upload, report=pickle.dumps(analysis_result)
    )
    ic(f"Report for uploaded file create at {report.id}")
    context = {"report_id": report.id, "analysis_result": analysis_result}
    return render(
        request,
        "dashboard.html",
        context=context,
    )


@login_required
def get_report(request, report_id):
    report = get_object_or_404(Report, id=report_id, user=request.user)
    report_dict = pickle.loads(report.report)
    analysis_result = {
        "exploratory_analysis": report_dict.get("exploratory_analysis"),
        "univariate_analysis": report_dict.get("univariate_analysis"),
        "bivariate_analysis": report_dict.get("bivariate_analysis"),
        "multivariate_analysis": report_dict.get("multivariate_analysis"),
    }
    print(report.id)
    context = {"report_id": report.id, "analysis_result": analysis_result}

    return render(
        request,
        "dashboard.html",
        context=context,
    )


def get_report_pdf(request, report_id):
    pass


def logout_view(request):
    logout(request)
    return redirect("login")


def home(request):
    template_name = "home.html"
    return render(request, template_name)


"""
Used `pickle` to load and unload the complex data analysis report to and from the database.
"""
