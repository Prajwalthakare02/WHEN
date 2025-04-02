from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework import status
import json
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from .models import UserProfile, Company, Job, Application, PlacementPrediction
from django.db.models import Q
from datetime import datetime
from .predictor import predict_placement
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.authtoken.models import Token
import logging
from django.conf import settings
import os
import base64
from django.core.files.base import ContentFile
import re
import random
import string
import uuid
from django.contrib.auth import get_user_model
import requests
from functools import wraps
from django.middleware.csrf import get_token

# Set up logger
logger = logging.getLogger(__name__)

User = get_user_model()

# Modified authentication decorator to handle both cookie and token auth
def token_auth_required(view_func):
    """Custom decorator that validates token from cookies or Authorization header"""
    @wraps(view_func)
    def _wrapped_view(request, *args, **kwargs):
        # Log auth headers for debugging
        logger.info(f"Auth debug - Headers: {request.headers.get('Authorization')}")
        
        # First check if user is already authenticated via session
        if request.user and request.user.is_authenticated:
            return view_func(request, *args, **kwargs)
            
        # Then check for token in Authorization header
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token_key = auth_header.split(' ')[1]
            try:
                token = Token.objects.get(key=token_key)
                request.user = token.user
                return view_func(request, *args, **kwargs)
            except Token.DoesNotExist:
                logger.warning(f"Invalid token: {token_key}")
                return Response({"success": False, "message": "Invalid token"}, status=401)
                
        # Finally, return unauthorized if no valid auth method
        return Response({"success": False, "message": "Authentication required"}, status=401)
    
    return _wrapped_view

# Create your views here.

@api_view(['GET'])
@permission_classes([AllowAny])
def hello_world(request):
    return Response({"message": "Hello, World!"})

# User API endpoints
@api_view(['POST'])
@permission_classes([AllowAny])
@csrf_exempt  # Temporarily exempt CSRF for testing
def register_user(request):
    try:
        data = request.data
        
        # Log the incoming request data for debugging
        logger.info(f"Registration attempt for email: {data.get('email')}")
        
        # Check if user already exists
        if User.objects.filter(email=data.get('email')).exists():
            logger.warning(f"Registration failed: Email already exists - {data.get('email')}")
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
        
        # Create user profile with role
        profile = UserProfile.objects.create(
            user=user,
            phone_number=data.get('phoneNumber', ''),
            role=data.get('role', 'student')  # Set role with default as student
        )
        
        # Create token for the new user
        token = Token.objects.create(user=user)
        
        # Format user data for response
        user_profile = {
            'id': user.id,
            'fullname': data.get('fullname', ''),
            'email': user.email,
            'phoneNumber': profile.phone_number,
            'role': profile.role,
            'profile': {
                'bio': profile.bio or '',
                'skills': profile.skills or [],
                'resume': profile.resume.url if profile.resume else None,
                'resumeOriginalName': profile.resume_original_name,
                'profile_picture': profile.profile_picture.url if profile.profile_picture else None
            }
        }
        
        # Log in the user
        login(request, user)
        
        response = Response({
            'success': True,
            'message': 'User registered successfully',
            'user': user_profile,
            'token': token.key
        }, status=status.HTTP_201_CREATED)
        
        # Set token in cookie
        response.set_cookie(
            'auth_token',
            token.key,
            httponly=True,
            samesite='Lax',
            max_age=86400  # 24 hours
        )
        
        logger.info(f"User registered successfully: {user.email}")
        return response
    
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return Response({
            'success': False,
            'message': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([AllowAny])
@csrf_exempt  # Temporarily exempt CSRF for testing
def login_user(request):
    try:
        data = request.data
        username = data.get('email', '').lower()
        password = data.get('password', '')
        role = data.get('role', 'student')
        
        # Log the incoming request data for debugging
        logger.info(f"Login attempt for user: {username}")
        
        # Authenticate user
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            # Set session
            login(request, user)
            
            # Delete any existing tokens for this user
            Token.objects.filter(user=user).delete()
            
            # Create a new token
            token = Token.objects.create(user=user)
            
            # Get or create profile with role
            profile, created = UserProfile.objects.get_or_create(
                user=user,
                defaults={'role': role}
            )
            
            # Update role if it's different
            if profile.role != role:
                profile.role = role
                profile.save()
            
            user_data = {
                'id': user.id,
                'username': user.username,
                'fullname': f"{user.first_name} {user.last_name}".strip(),
                'email': user.email,
                'role': profile.role,
                'phoneNumber': profile.phone_number or '',
                'profile': {
                    'bio': profile.bio or '',
                    'skills': profile.skills or [],
                    'resume': profile.resume.url if profile.resume else None,
                    'resumeOriginalName': profile.resume_original_name or 'Resume',
                    'profile_picture': profile.profile_picture.url if profile.profile_picture else None
                }
            }
            
            response = Response({
                'success': True,
                'message': 'Login successful',
                'user': user_data,
                'token': token.key
            })
            
            # Set token in cookie
            response.set_cookie(
                'auth_token',
                token.key,
                httponly=True,
                samesite='Lax',
                max_age=86400  # 24 hours
            )
            
            return response
        else:
            logger.warning(f"Failed login attempt for user: {username}")
            return Response({
                'success': False,
                'message': 'Invalid credentials'
            }, status=status.HTTP_401_UNAUTHORIZED)
    
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
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
@token_auth_required  # Use our custom auth decorator
def get_applications(request):
    try:
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
        logger.error(f"Error in get_applications: {str(e)}")
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
def predict_placement_api(request):
    try:
        user = get_user_from_token(request)
        if not user:
            logger.error(f"Authentication failed in predict_placement_api: {request.META.get('HTTP_AUTHORIZATION', 'No auth header')}")
            return Response({
                "success": False,
                "message": "Authentication required"
            }, status=401)
            
        # Check if user profile exists
        profile = UserProfile.objects.filter(user=user).first()
        if not profile:
            return Response({
                "success": False,
                "message": "User profile not found"
            }, status=404)
        
        # Extract data from request
        data = request.data
        logger.info(f"Received prediction data: {data}")
        
        # Call the function to get prediction
        placement_result = predict_placement(data)
        
        # Save prediction to database
        prediction = PlacementPrediction.objects.create(
            user=profile,
            ssc_percentage=data.get('ssc_percentage'),
            hsc_percentage=data.get('hsc_percentage'),
            degree_percentage=data.get('degree_percentage'),
            work_experience=data.get('work_experience'),
            etest_percentage=data.get('etest_percentage'),
            mba_percentage=data.get('mba_percentage'),
            gender=data.get('gender'),
            specialisation=data.get('specialisation'),
            placement_prediction=placement_result
        )
        
        return Response({
            "success": True,
            "message": "Prediction created successfully",
            "placement": placement_result,
            "prediction_id": prediction.id
        }, status=201)
    
    except Exception as e:
        logger.error(f"Error in predict_placement_api: {str(e)}")
        return Response({
            "success": False,
            "message": str(e)
        }, status=500)

@api_view(['GET'])
@token_auth_required  # Use our custom auth decorator
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
        logger.error(f"Error in get_placement_prediction: {str(e)}")
        return Response({
            "success": False,
            "message": str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def get_user_from_token(request):
    """Extract user from token in Authorization header or cookies"""
    auth_header = request.META.get('HTTP_AUTHORIZATION', '')
    
    if auth_header.startswith('Bearer '):
        token_key = auth_header.split(' ')[1]
        try:
            token = Token.objects.get(key=token_key)
            return token.user
        except Token.DoesNotExist:
            return None
    
    return request.user

@api_view(['GET'])
def get_applications(request):
    try:
        user = get_user_from_token(request)
        if not user:
            logger.error(f"Authentication failed for request: {request.META.get('HTTP_AUTHORIZATION', 'No auth header')}")
            return Response({
                "success": False,
                "message": "Authentication required"
            }, status=401)
            
        logger.info(f"User authenticated: {user.username}")
        
        profile = UserProfile.objects.filter(user=user).first()
        
        if profile and profile.role == "student":
            applications = Application.objects.filter(applicant__user=user)
            serialized_data = []
            
            for application in applications:
                job = application.job
                company = job.company
                
                serialized_data.append({
                    "id": application.id,
                    "status": application.status,
                    "application_date": application.application_date,
                    "job": {
                        "id": job.id,
                        "title": job.title,
                        "salary": job.salary,
                        "location": job.location,
                        "description": job.description,
                        "created_at": job.created_at,
                        "company": {
                            "id": company.id,
                            "name": company.name,
                            "email": company.email,
                            "logo": company.logo.url if company.logo else None
                        }
                    }
                })
            
            return Response({
                "success": True,
                "data": serialized_data,
                "message": "Applications fetched"
            }, status=200)
        
        else:
            return Response({
                "success": False,
                "message": "Only students can view their applications"
            }, status=400)
    
    except Exception as e:
        logger.error(f"Error in get_applications: {str(e)}")
        return Response({
            "success": False,
            "message": str(e)
        }, status=500)

@api_view(['POST'])
@csrf_exempt
@token_auth_required
def chatbot_api(request):
    """
    API endpoint for student queries using Hugging Face models
    """
    try:
        # Get the user query from the request
        user_query = request.data.get('query', '')
        chat_history = request.data.get('history', [])
        
        if not user_query:
            return Response({"error": "Query is required"}, status=400)
        
        # Hugging Face API endpoint for inference
        API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
        headers = {
            "Authorization": "Bearer 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",
            "Content-Type": "application/json"
        }
        
        # Format the input for the model with enhanced system context
        conversation = [
            {
                "role": "system", 
                "content": """You are an advanced placement assistant for college students. Your expertise includes:
                1. Resume Building: Help create impactful resumes, optimize content, and highlight key achievements
                2. Interview Preparation: Guide on technical and behavioral interviews, common questions, and best practices
                3. Career Development: Provide insights on industry trends, skill development, and career paths
                4. Placement Process: Explain placement procedures, company expectations, and selection criteria
                5. Technical Skills: Advise on programming languages, tools, and technologies in demand
                6. Soft Skills: Guide on communication, leadership, and teamwork development
                7. Profile Optimization: Help improve academic and professional profiles
                8. Placement Prediction: Analyze placement chances based on academic performance and skills
                
                Provide specific, actionable advice while maintaining a supportive and encouraging tone. Focus on practical tips and real-world applications."""
            }
        ]
        
        # Add chat history
        for msg in chat_history:
            role = "user" if msg.get("isUser") else "assistant"
            conversation.append({"role": role, "content": msg.get("message", "")})
        
        # Add the current query
        conversation.append({"role": "user", "content": user_query})
        
        payload = {
            "inputs": conversation,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "repetition_penalty": 1.2
            }
        }
        
        # Make request to Hugging Face API
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            # Extract the generated response from the model's output
            bot_response = result[0]["generated_text"]
            
            # For LLaMA models, extract just the assistant's last response
            if isinstance(bot_response, str):
                # Try to extract just the assistant's response
                assistant_prefix = "assistant:"
                if assistant_prefix in bot_response.lower():
                    parts = bot_response.lower().split(assistant_prefix)
                    if len(parts) > 1:
                        bot_response = parts[-1].strip()
            
            return Response({
                "response": bot_response,
                "success": True
            })
        else:
            # Enhanced fallback responses
            fallback_responses = [
                "I can help you optimize your resume, prepare for interviews, and develop your career path. What specific guidance do you need?",
                "For placement success, focus on both technical excellence and soft skills. Which area would you like to improve first?",
                "Your profile shows potential! Let's enhance it further by highlighting your achievements and adding relevant projects.",
                "For technical interviews, practice coding problems and system design. Would you like specific resources or practice problems?",
                "I can help you understand placement procedures and company expectations. What would you like to know more about?",
                "Let's work on developing your skills. Which technical or soft skill would you like to focus on first?"
            ]
            return Response({
                "response": random.choice(fallback_responses),
                "success": True,
                "is_fallback": True
            })
            
    except Exception as e:
        logger.error(f"Error in chatbot_api: {str(e)}")
        return Response({
            "error": "An error occurred while processing your request",
            "success": False
        }, status=500)

# Add this function for CSRF token
@api_view(['GET'])
@permission_classes([AllowAny])
def get_csrf_token(request):
    """
    Get CSRF token for frontend to use in requests
    """
    csrf_token = get_token(request)
    return JsonResponse({'csrfToken': csrf_token})
