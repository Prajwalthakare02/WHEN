from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

class UserProfile(models.Model):
    ROLE_CHOICES = [
        ('student', 'Student'),
        ('recruiter', 'Recruiter'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='user_profile')
    bio = models.TextField(blank=True, null=True)
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    skills = models.JSONField(default=list, blank=True, null=True)
    profile_picture = models.ImageField(upload_to='profile_pictures/', blank=True, null=True)
    resume = models.FileField(upload_to='resumes/', blank=True, null=True)
    resume_original_name = models.CharField(max_length=255, blank=True, null=True)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='student')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s profile"

class Company(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    website = models.URLField(blank=True, null=True)
    logo = models.ImageField(upload_to='company_logos/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return self.name
    
    class Meta:
        verbose_name_plural = "Companies"

class Job(models.Model):
    EMPLOYMENT_TYPE_CHOICES = [
        ('full_time', 'Full Time'),
        ('part_time', 'Part Time'),
        ('contract', 'Contract'),
        ('freelance', 'Freelance'),
        ('internship', 'Internship'),
    ]
    
    title = models.CharField(max_length=100)
    company = models.ForeignKey(Company, on_delete=models.CASCADE, related_name='jobs')
    description = models.TextField()
    requirements = models.TextField()
    location = models.CharField(max_length=100)
    employment_type = models.CharField(max_length=20, choices=EMPLOYMENT_TYPE_CHOICES, default='full_time')
    salary_min = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    salary_max = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.title} at {self.company.name}"

class Application(models.Model):
    STATUS_CHOICES = [
        ('applied', 'Applied'),
        ('under_review', 'Under Review'),
        ('interview', 'Interview'),
        ('offer', 'Offer'),
        ('hired', 'Hired'),
        ('rejected', 'Rejected'),
    ]
    
    job = models.ForeignKey(Job, on_delete=models.CASCADE, related_name='applications')
    applicant = models.ForeignKey(User, on_delete=models.CASCADE, related_name='applications')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='applied')
    cover_letter = models.TextField(blank=True, null=True)
    resume = models.FileField(upload_to='application_resumes/', blank=True, null=True)
    applied_date = models.DateTimeField(default=timezone.now)
    
    def __str__(self):
        return f"{self.applicant.username}'s application for {self.job.title}"

# Add the missing PlacementPrediction model
class PlacementPrediction(models.Model):
    """
    Model for storing placement predictions
    """
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='predictions')
    ssc_percentage = models.FloatField()
    hsc_percentage = models.FloatField()
    degree_percentage = models.FloatField()
    etest_percentage = models.FloatField()
    mba_percentage = models.FloatField()
    work_experience = models.IntegerField()
    gender = models.CharField(max_length=1)
    specialisation = models.CharField(max_length=20)
    placement_prediction = models.BooleanField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Prediction for {self.user} - {'Placed' if self.placement_prediction else 'Not Placed'}"
