from django.shortcuts import render

# Create your views here.
# Templates @ transaction_analysis/home.html


def home_with_data(request):
    context = {
        "first_name": "Anjaneyulu",
        "last_name": "Batta",
        "address": "Hyderabad, India",
    }
    template_name = "user_template.html"
    return render(request, "transaction_analysis/home.html", context)
