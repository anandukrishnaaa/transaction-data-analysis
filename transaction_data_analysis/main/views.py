# Create your views here.

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from .models import FileUpload, Report
from .forms import UploadFileForm, CustomUserCreationForm
from .utils import file_handler, ml
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
        "exploratory_analysis": ic(report_dict.get("exploratory_analysis")),
        "univariate_analysis": ic(report_dict.get("univariate_analysis")),
        "bivariate_analysis": ic(report_dict.get("bivariate_analysis")),
        "multivariate_analysis": ic(report_dict.get("multivariate_analysis")),
    }
    print(report.id)
    context = {"report_id": report.id, "analysis_result": analysis_result}

    return render(
        request,
        "dashboard.html",
        context=context,
    )


def train_main_model(request, report_id):
    # Get the Report object based on the report_id
    report = get_object_or_404(Report, id=report_id)

    # Get the file path from the Report object
    file_path = report.file_id.file_path.path

    train_main_model_result = ml.train_main_model(file_path)
    # Prepare data to display in the template
    context = {"train_main_model_result": train_main_model_result}

    # Render the ml.html template with the prepared data
    return render(request, "ml.html", context)


def batch_run_model(request, report_id):
    # Get the Report object based on the report_id
    report = get_object_or_404(Report, id=report_id)

    # Get the file path from the Report object
    file_path = report.file_id.file_path.path

    # Check if custom model exists
    model = ml.custom_model_check(file_path)
    # Preprocess and load dataframe
    df = ml.load_and_prep_df(file_path)
    # Train and run sub model
    model, xtest, ytest = ml.train_sub_model(df, model)
    # Fetch customer ids
    customer_ids_list = ml.get_customer_ids(df)
    # Fetch most fraudulent customer ids
    most_fraudulent = ml.find_most_fraudulent_amount(df)
    # Top fraud prone x number of customers
    top_x = 10
    predict_fraud_prone_customers = ml.predict_fraud_prone_customers(model, df, top_x)
    # Top fraud prone x number of customers
    bottom_x = 10
    predict_least_fraud_prone_customers = ml.predict_least_fraud_prone_customers(
        model, df, bottom_x
    )
    # Probability of being frauded
    compute_fraud_probabilities = ml.compute_fraud_probabilities(model, df)

    batch_run_model_result = {
        "report_id": ic(report_id),
        "customer_ids_list": ic(customer_ids_list),
        "most_fraudulent": ic(most_fraudulent),
        "predict_fraud_prone_customers": ic(predict_fraud_prone_customers),
        "predict_least_fraud_prone_customers": ic(predict_least_fraud_prone_customers),
        "compute_fraud_probabilities": ic(compute_fraud_probabilities),
    }
    # Prepare data to display in the template
    context = {"batch_run_model_result": batch_run_model_result}

    # Render the ml.html template with the prepared data
    return render(request, "ml.html", context)


def single_run_model(request, report_id):
    # Get the Report object based on the report_id
    report = get_object_or_404(Report, id=report_id)
    customer_id = ic(request.POST.get("nameOrig"))

    # Get the file path from the Report object
    file_path = report.file_id.file_path.path
    # Check if custom model exists
    model = ml.custom_model_check(file_path)
    # Preprocess and load dataframe
    df = ml.load_and_prep_df(file_path)
    # Train and run sub model
    model, xtest, ytest = ml.train_sub_model(df, model)
    # Anomalies for selected customer
    anomalies_and_patterns_for_customer = ml.get_anomalies_and_patterns_for_customer(
        df, model, customer_id
    ).to_html(classes="table table-bordered table-striped", escape=False, index=False)
    # Various probability metrics as a dictionary
    customer_probabilities = ml.get_customer_probabilities(model, df, customer_id)

    single_run_model_result = {
        "report_id": report_id,
        "customer_id": customer_id,
        "anomalies_and_patterns_for_customer": anomalies_and_patterns_for_customer,
        "customer_probabilities": customer_probabilities,
    }

    # Prepare data to display in the template
    context = {"single_run_model_result": single_run_model_result}

    # Render the ml.html template with the prepared data
    return render(request, "ml.html", context)


def logout_view(request):
    logout(request)
    return redirect("login")


def home(request):
    template_name = "home.html"
    return render(request, template_name)


"""
Used `pickle` to load and unload the complex data analysis report to and from the database.
"""
