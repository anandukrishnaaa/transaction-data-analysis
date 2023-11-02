from django.contrib import admin
from .models import FileUpload

# Register your models here.


@admin.register(FileUpload)
class FileUploadAdmin(admin.ModelAdmin):
    list_display = ("user", "file_id")
    readonly_fields = ("created_at", "updated_at")
