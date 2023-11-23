from django.contrib import admin
from .models import FileUpload, Report

# Register your models here.


@admin.register(FileUpload)
class FileUploadAdmin(admin.ModelAdmin):
    list_display = ("user", "file_id")
    readonly_fields = ("created_at", "updated_at")


@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    list_display = ("user", "file_id", "report")
    readonly_fields = ("created_at", "updated_at")
