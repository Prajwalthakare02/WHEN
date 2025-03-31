from django.contrib import admin
from .models import UserProfile, Company, Job, Application

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'phone_number', 'created_at')
    search_fields = ('user__username', 'user__email', 'phone_number')

@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ('name', 'website', 'created_at')
    search_fields = ('name', 'description')

@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ('title', 'company', 'location', 'employment_type', 'is_active', 'created_at')
    list_filter = ('is_active', 'employment_type', 'company')
    search_fields = ('title', 'description', 'requirements', 'location')

@admin.register(Application)
class ApplicationAdmin(admin.ModelAdmin):
    list_display = ('job', 'applicant', 'status', 'applied_date')
    list_filter = ('status', 'applied_date')
    search_fields = ('job__title', 'applicant__username', 'applicant__email')
