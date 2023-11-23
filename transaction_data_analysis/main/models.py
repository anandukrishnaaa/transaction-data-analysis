from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator

# Create your models here.


class FileUpload(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_id = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    file_path = models.FileField(
        upload_to="uploads/", validators=[FileExtensionValidator(["csv"])]
    )


class Report(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file_id = models.ForeignKey(FileUpload, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    report = models.BinaryField()
