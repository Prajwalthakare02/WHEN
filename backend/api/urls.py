from django.urls import path, include
from . import views

urlpatterns = [
    # API test endpoint
    path('hello/', views.hello_world, name='hello_world'),
    
    # User API endpoints
    path('v1/user/register', views.register_user, name='register_user'),
    path('v1/user/login', views.login_user, name='login_user'),
    path('v1/user/logout', views.logout_user, name='logout_user'),
    path('v1/user/profile/update', views.update_profile, name='update_profile'),
    
    # Job API endpoints
    path('v1/job/get', views.get_jobs, name='get_jobs'),
    
    # Company API endpoints
    path('v1/company/get', views.get_companies, name='get_companies'),
    
    # Application API endpoints
    path('v1/application/get', views.get_applications, name='get_applications'),
] 