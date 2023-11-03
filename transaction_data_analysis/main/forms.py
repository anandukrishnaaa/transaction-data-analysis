# main / forms.py

from django import forms
from django.core.validators import FileExtensionValidator
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class UploadFileForm(forms.Form):
    file = forms.FileField(validators=[FileExtensionValidator(["csv"])])


class CustomUserCreationForm(UserCreationForm):
    name = forms.CharField(max_length=255)
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ["name", "email", "username", "password1", "password2"]


class ReplaceFileForm(forms.Form):
    new_file = forms.FileField(validators=[FileExtensionValidator(["csv"])])
