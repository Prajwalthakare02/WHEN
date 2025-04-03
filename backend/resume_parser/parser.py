import os
import re
import logging
import PyPDF2
import docx
from typing import Dict, List, Union, Optional

# Set up logger
logger = logging.getLogger(__name__)

class ResumeParser:
    """A class to parse resumes and extract relevant information."""
    
    def __init__(self):
        self.tech_skills = [
            'python', 'java', 'javascript', 'react', 'node', 'django', 'flask',
            'html', 'css', 'sql', 'mongodb', 'aws', 'docker', 'kubernetes',
            'machine learning', 'ai', 'data science', 'angular', 'vue',
            'spring', 'hibernate', 'rest api', 'graphql', 'php', 'laravel',
            'ruby', 'rails', 'git', 'jenkins', 'ci/cd', 'agile', 'scrum'
        ]
        
        self.programming_languages = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php',
            'swift', 'kotlin', 'go', 'rust'
        ]
        
        self.branches = {
            'computer science': 'Computer Science',
            'information technology': 'Information Technology',
            'electronics': 'Electronics & Communication',
            'electrical': 'Electrical Engineering',
            'mechanical': 'Mechanical Engineering',
            'civil': 'Civil Engineering'
        }
        
        self.leadership_keywords = [
            'leader', 'managed', 'coordinated', 'organized', 'led',
            'team lead', 'supervisor', 'head', 'captain', 'president',
            'secretary', 'chair', 'director'
        ]
        
        self.soft_skills_keywords = [
            'communication', 'teamwork', 'leadership', 'problem solving',
            'analytical', 'interpersonal', 'presentation', 'public speaking',
            'time management', 'organization', 'adaptability', 'flexibility'
        ]

    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF or DOCX file."""
        file_extension = file_path.split('.')[-1].lower()
        text = ""
        
        try:
            if file_extension == 'pdf':
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                logger.info(f"Extracted {len(text)} characters from PDF")
                
            elif file_extension == 'docx':
                doc = docx.Document(file_path)
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                logger.info(f"Extracted {len(text)} characters from DOCX")
                
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_extension.upper()}: {str(e)}")
            raise
            
        return text

    def extract_email(self, text: str) -> Optional[str]:
        """Extract email address from text."""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(pattern, text)
        return emails[0] if emails else None

    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number from text."""
        pattern = r'\b(?:\+?\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(pattern, text)
        return phones[0] if phones else None

    def extract_skills(self, text: str) -> List[str]:
        """Extract technical skills from text."""
        found_skills = []
        for skill in self.tech_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text.lower()):
                found_skills.append(skill)
        return found_skills

    def count_projects(self, text: str) -> int:
        """Count number of projects mentioned in text."""
        project_indicators = [
            r'\bproject[s]?\b', r'\bimplemented\b', r'\bdeveloped\b',
            r'\bcreated\b', r'\bbuilt\b'
        ]
        
        project_count = 0
        for indicator in project_indicators:
            project_count += len(re.findall(indicator, text.lower()))
        return min(max(project_count // 2, 0), 10)  # Normalize between 0 and 10

    def count_certifications(self, text: str) -> int:
        """Count number of certifications mentioned in text."""
        cert_indicators = [
            r'\bcertification[s]?\b', r'\bcertified\b', r'\bcertificate[s]?\b'
        ]
        
        cert_count = 0
        for indicator in cert_indicators:
            cert_count += len(re.findall(indicator, text.lower()))
        return min(max(cert_count, 0), 10)  # Normalize between 0 and 10

    def extract_experience(self, text: str) -> int:
        """Extract years of experience from text."""
        exp_patterns = [
            r'(\d+)\+?\s*(?:year[s]?|yr[s]?)\s*(?:of)?\s*experience',
            r'experience\s*:?\s*(\d+)\+?\s*(?:year[s]?|yr[s]?)',
        ]
        
        for pattern in exp_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                try:
                    return int(matches[0])
                except ValueError:
                    continue
        return 0

    def extract_programming_languages(self, text: str) -> List[str]:
        """Extract programming languages mentioned in text."""
        return [
            lang for lang in self.programming_languages
            if re.search(r'\b' + re.escape(lang) + r'\b', text.lower())
        ]

    def extract_cgpa(self, text: str) -> float:
        """Extract CGPA from text."""
        pattern = r'(?:CGPA|GPA)\s*:?\s*(\d+\.?\d*)'
        matches = re.findall(pattern, text)
        return float(matches[0]) if matches else 8.5  # Default if not found

    def count_internships(self, text: str) -> int:
        """Count number of internships mentioned in text."""
        pattern = r'\binternship[s]?\b'
        return len(re.findall(pattern, text.lower()))

    def determine_branch(self, text: str) -> str:
        """Determine educational branch/major from text."""
        for key, value in self.branches.items():
            if re.search(r'\b' + re.escape(key) + r'\b', text.lower()):
                return value
        return 'Computer Science'  # Default

    def extract_year_of_passing(self, text: str) -> str:
        """Extract year of passing from text."""
        pattern = r'(?:20[1-2][0-9])'
        years = re.findall(pattern, text)
        return max(years) if years else '2024'  # Default to 2024 if not found

    def calculate_leadership_score(self, text: str) -> int:
        """Calculate leadership score based on keywords."""
        score = 0
        for keyword in self.leadership_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
                score += 1
        return min(score, 10)  # Cap at 10

    def calculate_soft_skills_score(self, text: str) -> int:
        """Calculate soft skills score based on keywords."""
        score = 0
        for keyword in self.soft_skills_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text.lower()):
                score += 1
        return min(score, 10)  # Cap at 10

    def parse(self, file_path: str) -> Dict[str, Union[str, int, float, List[str]]]:
        """
        Parse resume and extract all relevant information.
        
        Args:
            file_path: Path to the resume file (PDF or DOCX)
            
        Returns:
            Dictionary containing parsed information and prediction data
        """
        try:
            # Extract text from document
            text = self.extract_text(file_path)
            
            if not text:
                raise ValueError("Could not extract text from the document")
            
            # Extract all information
            skills = self.extract_skills(text)
            tech_skills_score = min(len(skills) / 3, 10)  # 3 skills = 1 point, max 10
            programming_langs = self.extract_programming_languages(text)
            
            parsed_data = {
                'email': self.extract_email(text),
                'phone': self.extract_phone(text),
                'skills': skills,
                'projects': self.count_projects(text),
                'certifications': self.count_certifications(text),
                'experience_years': self.extract_experience(text),
                'cgpa': self.extract_cgpa(text),
                'internships': self.count_internships(text),
                'branch': self.determine_branch(text),
                'year_of_passing': self.extract_year_of_passing(text),
                'leadership_score': self.calculate_leadership_score(text),
                'soft_skills_score': self.calculate_soft_skills_score(text),
                'technical_skills_score': tech_skills_score,
                'programming_languages': programming_langs
            }
            
            # Format data for placement prediction
            prediction_data = {
                'cgpa': parsed_data['cgpa'],
                'soft_skills_score': parsed_data['soft_skills_score'],
                'technical_skills': parsed_data['technical_skills_score'],
                'leadership_score': parsed_data['leadership_score'],
                'experience_years': parsed_data['experience_years'],
                'live_backlogs': 0,  # Default assumption
                'programming_language': programming_langs[0] if programming_langs else 'Python',
                'internships': max(parsed_data['internships'], 1),  # At least 1
                'projects': max(parsed_data['projects'], 2),  # At least 2
                'certifications': max(parsed_data['certifications'], 1),  # At least 1
                'branch': parsed_data['branch'],
                'year_of_passing': parsed_data['year_of_passing'],
                'gender': 'M'  # Default assumption
            }
            
            return {
                'parsed_data': parsed_data,
                'prediction_data': prediction_data
            }
            
        except Exception as e:
            logger.error(f"Error parsing resume: {str(e)}")
            raise 