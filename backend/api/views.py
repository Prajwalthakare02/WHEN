from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework import status
import json
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from .models import UserProfile, Company, Job, Application
from django.db.models import Q

# Create your views here.

@api_view(['GET'])
@permission_classes([AllowAny])
def hello_world(request):
    return Response({"message": "Hello, World!"})

# User API endpoints
@api_view(['POST'])
@permission_classes([AllowAny])
def register_user(request):
    try:
        data = request.data
        print(f"Received registration data: {data}")
        
        # Check if required fields are present
        required_fields = ['email', 'password', 'fullname']
        for field in required_fields:
            if not data.get(field):
                return Response({
                    'success': False,
                    'message': f'Field {field} is required'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check if user already exists
        if User.objects.filter(email=data.get('email')).exists():
            return Response({
                'success': False,
                'message': 'User with this email already exists'
            }, status=status.HTTP_400_BAD_REQUEST)
            
        # Create user
        user = User.objects.create_user(
            username=data.get('email'),  # Using email as username
            email=data.get('email'),
            password=data.get('password'),
            first_name=data.get('fullname', '').split(' ')[0],
            last_name=' '.join(data.get('fullname', '').split(' ')[1:]) if len(data.get('fullname', '').split(' ')) > 1 else ''
        )
        
        # Create user profile
        profile = UserProfile.objects.create(
            user=user,
            phone_number=data.get('phoneNumber', '')
        )
        
        # Format user data for response
        user_profile = {
            'id': user.id,
            'fullname': data.get('fullname', ''),
            'email': user.email,
            'phoneNumber': profile.phone_number,
            'profile': {
                'bio': profile.bio or '',
                'skills': profile.skills or [],
                'resume': profile.resume.url if profile.resume else None,
                'resumeOriginalName': profile.resume_original_name
            }
        }
        
        return Response({
            'success': True,
            'message': 'User registered successfully',
            'user': user_profile
        }, status=status.HTTP_201_CREATED)
    
    except Exception as e:
        import traceback
        print(f"Error during user registration: {str(e)}")
        print(traceback.format_exc())
        return Response({
            'success': False,
            'message': f'Server error: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([AllowAny])
def login_user(request):
    try:
        data = request.data
        print(f"Received login data: {data}")
        
        # Check required fields
        if not data.get('email') or not data.get('password'):
            return Response({
                'success': False,
                'message': 'Email and password are required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Authenticate user
        user = authenticate(
            username=data.get('email'),  # Using email as username
            password=data.get('password')
        )
        
        if not user:
            return Response({
                'success': False,
                'message': 'Invalid credentials'
            }, status=status.HTTP_401_UNAUTHORIZED)
        
        # Get or create user profile
        profile, created = UserProfile.objects.get_or_create(user=user)
        
        # Format user data for response
        user_profile = {
            'id': user.id,
            'fullname': f'{user.first_name} {user.last_name}',
            'email': user.email,
            'phoneNumber': profile.phone_number or '',
            'profile': {
                'bio': profile.bio or '',
                'skills': profile.skills or [],
                'resume': profile.resume.url if profile.resume else None,
                'resumeOriginalName': profile.resume_original_name
            }
        }
        
        return Response({
            'success': True,
            'message': 'Login successful',
            'user': user_profile
        })
    
    except Exception as e:
        import traceback
        print(f"Error during user login: {str(e)}")
        print(traceback.format_exc())
        return Response({
            'success': False,
            'message': f'Server error: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Job API endpoints
@api_view(['GET'])
@permission_classes([AllowAny])
def get_jobs(request):
    try:
        keyword = request.query_params.get('keyword', '')
        
        # Query jobs from database
        if keyword:
            jobs_query = Job.objects.filter(
                Q(title__icontains=keyword) | 
                Q(description__icontains=keyword) |
                Q(location__icontains=keyword) |
                Q(company__name__icontains=keyword)
            ).filter(is_active=True)
        else:
            jobs_query = Job.objects.filter(is_active=True)
            
        # Format job data for response
        jobs = []
        for job in jobs_query:
            jobs.append({
                "id": job.id,
                "title": job.title,
                "description": job.description,
                "location": job.location,
                "salary": f"${job.salary_min} - ${job.salary_max}" if job.salary_min and job.salary_max else "Not specified",
                "company": {
                    "id": job.company.id,
                    "name": job.company.name,
                    "logo": job.company.logo.url if job.company.logo else "https://placehold.co/100"
                }
            })
        
        return Response({
            "success": True,
            "jobs": jobs
        })
    except Exception as e:
        return Response({
            "success": False,
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Company API endpoints
@api_view(['GET'])
@permission_classes([AllowAny])
def get_companies(request):
    try:
        # Query companies from database
        companies_query = Company.objects.all()
        
        # Format company data for response
        companies = []
        for company in companies_query:
            companies.append({
                "id": company.id,
                "name": company.name,
                "description": company.description,
                "logo": company.logo.url if company.logo else "https://placehold.co/100"
            })
        
        return Response({
            "success": True,
            "companies": companies
        })
    except Exception as e:
        return Response({
            "success": False,
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Application API endpoints
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_applications(request):
    try:
        # Get current user
        user = request.user
        
        # Query applications for current user
        applications_query = Application.objects.filter(applicant=user)
        
        # Format application data for response
        applications = []
        for application in applications_query:
            applications.append({
                "id": application.id,
                "job": {
                    "id": application.job.id,
                    "title": application.job.title,
                    "company": {
                        "name": application.job.company.name,
                        "logo": application.job.company.logo.url if application.job.company.logo else "https://placehold.co/100"
                    }
                },
                "status": application.status,
                "appliedDate": application.applied_date.strftime("%Y-%m-%d")
            })
        
        return Response({
            "success": True,
            "application": applications
        })
    except Exception as e:
        return Response({
            "success": False,
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([AllowAny])
def logout_user(request):
    try:
        # Since we're using session authentication, we need to clear the session
        request.session.flush()
        return Response({
            'success': True,
            'message': 'Logged out successfully'
        })
    except Exception as e:
        import traceback
        print(f"Error during user logout: {str(e)}")
        print(traceback.format_exc())
        return Response({
            'success': False,
            'message': f'Server error: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_profile(request):
    try:
        user = request.user
        data = request.data
        print(f"Received profile update data: {data}")
        
        # Get the user profile
        profile = UserProfile.objects.get(user=user)
        
        # Update user data
        if 'fullname' in data:
            fullname = data.get('fullname', '')
            user.first_name = fullname.split(' ')[0]
            user.last_name = ' '.join(fullname.split(' ')[1:]) if len(fullname.split(' ')) > 1 else ''
            
        if 'email' in data:
            # Check if new email already exists for another user
            if User.objects.exclude(id=user.id).filter(email=data.get('email')).exists():
                return Response({
                    'success': False,
                    'message': 'Email already in use by another account'
                }, status=status.HTTP_400_BAD_REQUEST)
            user.email = data.get('email')
            user.username = data.get('email')  # Since email is username
            
        # Save user model changes
        user.save()
        
        # Update profile data
        if 'phoneNumber' in data:
            profile.phone_number = data.get('phoneNumber', '')
            
        if 'bio' in data:
            profile.bio = data.get('bio', '')
            
        if 'skills' in data and isinstance(data.get('skills'), list):
            profile.skills = data.get('skills')
            
        # Handle resume file upload
        if 'file' in request.FILES:
            file = request.FILES['file']
            profile.resume = file
            profile.resume_original_name = file.name
            
        # Save profile changes
        profile.save()
        
        # Format user data for response
        user_profile = {
            'id': user.id,
            'fullname': f'{user.first_name} {user.last_name}',
            'email': user.email,
            'phoneNumber': profile.phone_number or '',
            'profile': {
                'bio': profile.bio or '',
                'skills': profile.skills or [],
                'resume': profile.resume.url if profile.resume else None,
                'resumeOriginalName': profile.resume_original_name
            }
        }
        
        return Response({
            'success': True,
            'message': 'Profile updated successfully',
            'user': user_profile
        })
        
    except Exception as e:
        import traceback
        print(f"Error during profile update: {str(e)}")
        print(traceback.format_exc())
        return Response({
            'success': False,
            'message': f'Server error: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
