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
import time
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
                'resumeOriginalName': profile.resume_original_name or '',
                'profile_picture': profile.profile_picture.url if profile.profile_picture else None
            }
        }
        
        # Log in the user
        login(request, user)
        
        # Create response with CORS headers
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
            samesite='Lax',
            max_age=86400,  # 24 hours
            httponly=True,
            secure=False  # Set to True in production with HTTPS
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
            
            # Create response with CORS headers
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
                samesite='Lax',
                max_age=86400,  # 24 hours
                httponly=True,
                secure=False  # Set to True in production with HTTPS
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
@permission_classes([AllowAny])
def chatbot_api(request):
    try:
        data = request.data
        query = data.get('query', '').lower().strip()
        
        # Log the incoming request
        print(f"Chatbot received query: {query}")
        
        # Handle basic greetings and casual conversation
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you']
        if query in greetings:
            return Response({
                'response': "Hello! I'm WiseBot, your placement assistant. I can help you with resume tips, interview preparation, career guidance, and more. What would you like to know about?",
                'suggestions': [
                    "How to improve my resume?",
                    "What are common interview questions?",
                    "How to prepare for placement season?",
                    "Tips for technical interviews"
                ],
                'source': 'rule-based'
            })
        
        # Handle "thank you" and similar expressions
        if query in ['thanks', 'thank you', 'thank you so much']:
            return Response({
                'response': "You're welcome! Is there anything else I can help you with?",
                'suggestions': [
                    "How to improve my resume?",
                    "What are common interview questions?",
                    "How to prepare for placement season?",
                    "Tips for technical interviews"
                ],
                'source': 'rule-based'
            })
        
        # Handle "bye" and similar expressions
        if query in ['bye', 'goodbye', 'see you', 'see you later']:
            return Response({
                'response': "Goodbye! Good luck with your placement preparation. Feel free to come back if you have more questions!",
                'suggestions': [],
                'source': 'rule-based'
            })
        
        # Handle "help" requests
        if query in ['help', 'what can you do', 'what do you do', 'what are you']:
            return Response({
                'response': "I'm WiseBot, your placement assistant! I can help you with:\n\n" +
                           "• Resume building and improvement\n" +
                           "• Interview preparation and common questions\n" +
                           "• Technical interview tips\n" +
                           "• Placement season strategies\n" +
                           "• Job search techniques\n" +
                           "• Offer negotiation advice\n\n" +
                           "What would you like to know about?",
                'suggestions': [
                    "How to improve my resume?",
                    "What are common interview questions?",
                    "How to prepare for placement season?",
                    "Tips for technical interviews"
                ],
                'source': 'rule-based'
            })
        
        # Predefined responses for common placement-related queries
        responses = {
            'resume': {
                'keywords': ['resume', 'cv', 'curriculum vitae', 'bio data'],
                'response': "To improve your resume:\n\n" +
                           "1. Tailor it to each job application\n" +
                           "2. Include relevant skills and experiences\n" +
                           "3. Use quantifiable achievements (e.g., 'increased efficiency by 20%')\n" +
                           "4. Keep it concise (1-2 pages)\n" +
                           "5. Use a clean, professional format\n" +
                           "6. Proofread carefully\n" +
                           "7. Include relevant keywords from the job description\n" +
                           "8. Add a professional summary at the top",
                'suggestions': [
                    "What should I include in my resume?",
                    "How to format my resume?",
                    "Common resume mistakes to avoid"
                ]
            },
            'interview': {
                'keywords': ['interview', 'interview preparation', 'interview questions'],
                'response': "Interview preparation tips:\n\n" +
                           "1. Research the company thoroughly\n" +
                           "2. Practice common interview questions\n" +
                           "3. Prepare your STAR method responses\n" +
                           "4. Dress professionally\n" +
                           "5. Arrive early\n" +
                           "6. Bring copies of your resume\n" +
                           "7. Prepare questions to ask the interviewer\n" +
                           "8. Follow up with a thank-you email",
                'suggestions': [
                    "What are common interview questions?",
                    "How to answer behavioral questions?",
                    "Tips for technical interviews"
                ]
            },
            'technical': {
                'keywords': ['technical interview', 'coding interview', 'programming', 'coding', 'dsa', 'data structures', 'algorithms'],
                'response': "Technical interview preparation:\n\n" +
                           "1. Practice coding problems regularly\n" +
                           "2. Review data structures and algorithms\n" +
                           "3. Practice explaining your thought process\n" +
                           "4. Work on time and space complexity analysis\n" +
                           "5. Practice system design questions\n" +
                           "6. Review your projects thoroughly\n" +
                           "7. Be ready to write clean, efficient code\n" +
                           "8. Practice on a whiteboard or paper",
                'suggestions': [
                    "Common DSA interview questions",
                    "How to approach system design questions?",
                    "Tips for coding interviews"
                ]
            },
            'placement': {
                'keywords': ['placement', 'placement season', 'campus placement', 'job search', 'job hunt'],
                'response': "Placement season preparation:\n\n" +
                           "1. Start early - at least 6 months before\n" +
                           "2. Build a strong resume and portfolio\n" +
                           "3. Practice coding and problem-solving\n" +
                           "4. Prepare for aptitude tests\n" +
                           "5. Practice group discussions and presentations\n" +
                           "6. Network with seniors and alumni\n" +
                           "7. Research companies visiting your campus\n" +
                           "8. Maintain a good CGPA",
                'suggestions': [
                    "How to prepare for aptitude tests?",
                    "Tips for group discussions",
                    "How to research companies?"
                ]
            },
            'job_search': {
                'keywords': ['job search', 'find job', 'job hunt', 'job application', 'apply job'],
                'response': "Effective job search strategies:\n\n" +
                           "1. Use multiple job portals (LinkedIn, Indeed, etc.)\n" +
                           "2. Network with professionals in your field\n" +
                           "3. Attend job fairs and career events\n" +
                           "4. Follow companies you're interested in\n" +
                           "5. Customize your resume for each application\n" +
                           "6. Write compelling cover letters\n" +
                           "7. Follow up on applications\n" +
                           "8. Prepare for interviews",
                'suggestions': [
                    "How to write a good cover letter?",
                    "Which job portals should I use?",
                    "How to network effectively?"
                ]
            },
            'negotiation': {
                'keywords': ['negotiate', 'salary', 'offer', 'package', 'compensation'],
                'response': "Salary negotiation tips:\n\n" +
                           "1. Research market rates for your role\n" +
                           "2. Know your worth and minimum acceptable salary\n" +
                           "3. Wait for the employer to mention numbers first\n" +
                           "4. Highlight your value and achievements\n" +
                           "5. Be prepared to walk away if needed\n" +
                           "6. Consider the entire package, not just salary\n" +
                           "7. Practice your negotiation pitch\n" +
                           "8. Get offers in writing",
                'suggestions': [
                    "How to research salary ranges?",
                    "What to do if the offer is too low?",
                    "How to negotiate benefits?"
                ]
            }
        }
        
        # Check if the query matches any of our predefined responses
        for category, data in responses.items():
            if any(keyword in query for keyword in data['keywords']):
                try:
                    # Add a small delay to simulate thinking time
                    time.sleep(1)
                    return Response({
                        'response': data['response'],
                        'suggestions': data['suggestions'],
                        'source': 'rule-based'
                    })
                except Exception as e:
                    print(f"Error in rule-based response: {str(e)}")
                    # Continue execution in case of error
        
        # If no match found, try using the Hugging Face API
        try:
            # Get the API key from settings
            api_key = getattr(settings, 'HUGGINGFACE_API_KEY', None)
            
            if not api_key:
                raise ValueError("No Hugging Face API key configured")
                
            # Call the Hugging Face API
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": f"As a placement assistant, answer this question: {query}",
                "parameters": {
                    "max_length": 500,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True
                }
            }
            
            response = requests.post(
                "https://api-inference.huggingface.co/models/google/flan-t5-large",
                headers=headers,
                json=payload,
                timeout=10  # Add timeout to prevent hanging
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    # Add a small delay to simulate thinking time
                    time.sleep(1)
                    return Response({
                        'response': result[0].get('generated_text', 'I apologize, but I could not generate a response for that query.'),
                        'suggestions': [
                            "How to improve my resume?",
                            "What are common interview questions?",
                            "How to prepare for placement season?",
                            "Tips for technical interviews"
                        ],
                        'source': 'huggingface'
                    })
            
            # If API call fails or returns unexpected format, log the error
            print(f"Hugging Face API error: {response.text}")
            raise ValueError(f"API error: {response.text}")
            
        except Exception as e:
            # Log the error but don't crash
            print(f"Hugging Face API error: {str(e)}")
        
        # If we get here, either no match was found or the API call failed
        # Return a helpful default response
        return Response({
            'response': "I'm not sure I understand your question completely. Could you try rephrasing it or ask about resume building, interview preparation, or placement strategies?",
            'suggestions': [
                "How to improve my resume?",
                "What are common interview questions?",
                "How to prepare for placement season?",
                "Tips for technical interviews"
            ],
            'source': 'default'
        })
        
    except Exception as e:
        print(f"Chatbot error: {str(e)}")
        # Ensure we always return a response, even in case of error
        return Response({
            'response': "I'm having some technical difficulties at the moment. Please try again later or ask a different question.",
            'suggestions': [
                "How to improve my resume?",
                "What are common interview questions?",
                "How to prepare for placement season?",
                "Tips for technical interviews"
            ],
            'source': 'error'
        }, status=200)  # Return 200 even for errors to prevent frontend crashes

# Add this function for CSRF token
@api_view(['GET'])
@permission_classes([AllowAny])
def get_csrf_token(request):
    """Get CSRF token for frontend."""
    return JsonResponse({'csrfToken': get_token(request)})
