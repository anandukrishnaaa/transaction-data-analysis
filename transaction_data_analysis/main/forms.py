# main / forms.py

from django import forms
from django.core.validators import FileExtensionValidator
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User


class UploadFileForm(forms.Form):
    file = forms.FileField(validators=[FileExtensionValidator(["csv"])], label="")


class CustomUserCreationForm(UserCreationForm):
    name = forms.CharField(max_length=255)
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ["name", "email", "username", "password1", "password2"]

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]  # Include the email field
        name = self.cleaned_data["name"]

        # Split the name into first name and last name
        user.first_name, user.last_name = (
            name.split(maxsplit=1) if " " in name else (name, "")
        )

        if commit:
            user.save()
        return user
