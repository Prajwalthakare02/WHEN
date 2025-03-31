from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework import status
import json
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from .models import UserProfile, Company, Job, Application, PlacementPrediction
from django.db.models import Q
from datetime import datetime
from .predictor import predict_placement

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
            'role': data.get('role') or 'student',  # Set default role to student if not provided
            'profile': {
                'bio': profile.bio or '',
                'skills': profile.skills or [],
                'resume': profile.resume.url if profile.resume else None,
                'resumeOriginalName': profile.resume_original_name,
                'profile_picture': profile.profile_picture.url if profile.profile_picture else None
            }
        }
        
        return Response({
            'success': True,
            'message': 'User registered successfully',
            'user': user_profile
        }, status=status.HTTP_201_CREATED)
    
    except Exception as e:
        return Response({
            'success': False,
            'message': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([AllowAny])
def login_user(request):
    try:
        data = request.data
        
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
            'role': data.get('role') or 'student',  # Set default role to student if not provided
            'profile': {
                'bio': profile.bio or '',
                'skills': profile.skills or [],
                'resume': profile.resume.url if profile.resume else None,
                'resumeOriginalName': profile.resume_original_name,
                'profile_picture': profile.profile_picture.url if profile.profile_picture else None
            }
        }
        
        return Response({
            'success': True,
            'message': 'Login successful',
            'user': user_profile
        })
    
    except Exception as e:
        return Response({
            'success': False,
            'message': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def logout_user(request):
    """
    Logout user by clearing the session
    """
    try:
        # For token-based auth, you could blacklist the token here
        # For session-based auth, just return a success response
        
        return Response({
            'success': True,
            'message': 'Logged out successfully'
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
        return Response({
            'success': False,
            'message': str(e)
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

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_profile(request):
    """
    Update user profile
    """
    try:
        user = request.user
        # Get or create user profile
        profile, created = UserProfile.objects.get_or_create(user=user)
        
        # Update user fields
        if 'fullname' in request.data:
            name_parts = request.data['fullname'].split(' ', 1)
            user.first_name = name_parts[0]
            user.last_name = name_parts[1] if len(name_parts) > 1 else ""
            
        if 'email' in request.data:
            user.email = request.data['email']
            
        user.save()
        
        # Update profile fields
        if 'phoneNumber' in request.data:
            profile.phone_number = request.data['phoneNumber']
            
        if 'bio' in request.data:
            profile.bio = request.data['bio']
            
        if 'skills' in request.data and request.data['skills']:
            # Convert comma-separated string to list
            skills_str = request.data['skills']
            if isinstance(skills_str, str):
                profile.skills = [skill.strip() for skill in skills_str.split(',')]
            
        # Handle resume upload
        if 'file' in request.FILES:
            file = request.FILES['file']
            profile.resume = file
            profile.resume_original_name = file.name
            
        # Handle profile picture upload
        if 'profilePicture' in request.FILES:
            profile_picture = request.FILES['profilePicture']
            profile.profile_picture = profile_picture
            
        profile.save()
        
        # Prepare response
        user_profile = {
            'id': user.id,
            'username': user.username,
            'fullname': f"{user.first_name} {user.last_name}".strip(),
            'email': user.email,
            'phoneNumber': profile.phone_number or '',
            'role': request.data.get('role') or getattr(user, 'role', 'student'),  # Preserve role if it exists
            'profile': {
                'bio': profile.bio or '',
                'skills': profile.skills or [],
                'resume': profile.resume.url if profile.resume else None,
                'resumeOriginalName': profile.resume_original_name or 'Resume',
                'profile_picture': profile.profile_picture.url if profile.profile_picture else None
            }
        }
        
        return Response({
            'success': True,
            'message': 'Profile updated successfully',
            'user': user_profile
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        print(f"Error updating profile: {str(e)}")
        return Response({'success': False, 'message': f'Error updating profile: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def predict_placement_api(request):
    """
    Predict placement based on student data
    """
    try:
        # Get student data from request
        student_data = request.data
        
        # Call the predict_placement function from predictor.py
        result, probability = predict_placement(student_data)
        
        # Save the prediction result
        user = request.user
        prediction = PlacementPrediction.objects.create(
            user=user,
            result=result,
            probability=probability,
            cgpa=student_data.get('cgpa'),
            soft_skills_score=student_data.get('soft_skills_score'),
            technical_skills=student_data.get('technical_skills'),
            leadership_score=student_data.get('leadership_score'),
            experience_years=student_data.get('experience_years'),
            live_backlogs=student_data.get('live_backlogs'),
            internships=student_data.get('internships'),
            projects=student_data.get('projects'),
            certifications=student_data.get('certifications'),
            programming_language=student_data.get('programming_language'),
            branch=student_data.get('branch'),
            year_of_passing=student_data.get('year_of_passing'),
            gender=student_data.get('gender')
        )
        
        # Format response
        prediction_data = {
            "result": result,
            "probability": probability,
            "date": prediction.date
        }
        
        return Response({
            "success": True,
            "result": prediction_data
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
        print(f"Error in predict_placement_api: {str(e)}")
        return Response({
            "success": False,
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_placement_prediction(request):
    """
    Get the latest placement prediction for the current user
    """
    try:
        user = request.user
        
        # Try to get the latest prediction from the database
        try:
            latest_prediction = PlacementPrediction.objects.filter(user=user).latest('date')
            prediction = {
                "result": latest_prediction.result,
                "probability": latest_prediction.probability,
                "date": latest_prediction.date
            }
            
            # Get stored profile data
            profile_data = {
                "cgpa": latest_prediction.cgpa,
                "soft_skills_score": latest_prediction.soft_skills_score,
                "technical_skills": latest_prediction.technical_skills,
                "leadership_score": latest_prediction.leadership_score,
                "experience_years": latest_prediction.experience_years,
                "live_backlogs": latest_prediction.live_backlogs,
                "internships": latest_prediction.internships,
                "projects": latest_prediction.projects,
                "certifications": latest_prediction.certifications,
                "programming_language": latest_prediction.programming_language,
                "branch": latest_prediction.branch,
                "year_of_passing": latest_prediction.year_of_passing,
                "gender": latest_prediction.gender
            }
        except PlacementPrediction.DoesNotExist:
            # No prediction exists, return default values
            prediction = None
            profile_data = {
                "cgpa": "8.5",
                "soft_skills_score": "8",
                "technical_skills": "9",
                "leadership_score": "7",
                "experience_years": "1",
                "live_backlogs": "0",
                "internships": "2",
                "projects": "3",
                "certifications": "2",
                "programming_language": "Python",
                "branch": "Computer Science",
                "year_of_passing": "2025",
                "gender": "Male"
            }
        
        return Response({
            "success": True,
            "prediction": prediction,
            "profile_data": profile_data
        }, status=status.HTTP_200_OK)
    
    except Exception as e:
        print(f"Error in get_placement_prediction: {str(e)}")
        return Response({
            "success": False,
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
