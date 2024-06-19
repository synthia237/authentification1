from django.contrib import admin
from .models import CustomUser,Code


# Register your models here.

@admin.register(CustomUser)
class UserAdmin(admin.ModelAdmin):
    list_display = ("id","email", "password","image")

@admin.register(Code)
class CodeAdmin(admin.ModelAdmin):
    list_display = ("id", "user","email",'code')

