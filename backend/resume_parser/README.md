# Resume Parser

A Python module for parsing resumes in PDF and DOCX formats, extracting relevant information for placement prediction.

## Features

- Supports PDF and DOCX file formats
- Extracts:
  - Contact information (email, phone)
  - Technical skills
  - Programming languages
  - Education details (CGPA, branch, year of passing)
  - Experience (years, internships)
  - Projects and certifications count
  - Leadership and soft skills scores

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from resume_parser.parser import ResumeParser

# Create parser instance
parser = ResumeParser()

# Parse a resume file
result = parser.parse('path/to/resume.pdf')

# Access parsed data
parsed_data = result['parsed_data']
prediction_data = result['prediction_data']
```

## Data Format

The parser returns a dictionary with two main sections:

1. `parsed_data`: Raw extracted information
   - email: str
   - phone: str
   - skills: List[str]
   - projects: int
   - certifications: int
   - experience_years: int
   - cgpa: float
   - internships: int
   - branch: str
   - year_of_passing: str
   - leadership_score: int
   - soft_skills_score: int
   - technical_skills_score: float
   - programming_languages: List[str]

2. `prediction_data`: Formatted data for placement prediction
   - cgpa: float
   - soft_skills_score: int
   - technical_skills: float
   - leadership_score: int
   - experience_years: int
   - live_backlogs: int
   - programming_language: str
   - internships: int
   - projects: int
   - certifications: int
   - branch: str
   - year_of_passing: str
   - gender: str

## Error Handling

The parser includes comprehensive error handling and logging. All errors are logged using Python's logging module. 