from django.shortcuts import render
from .utils.data_processor import say_hello

# Create your views here.
# Templates @ transaction_analysis/home.html


def home_with_data(request):
    context = {"first_name": "AK"}
    template_name = "transaction_analysis/home.html"
    return render(request, template_name, context)
