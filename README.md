# Job Portal Application

A full-stack job portal application with Django REST Framework backend and React frontend.

## Features

- User authentication (register, login, logout)
- Profile management
- Job listings and search
- Company profiles
- Job application tracking

## Tech Stack

### Backend
- Django 5.1.2
- Django REST Framework
- SQLite database

### Frontend
- React
- Vite
- Redux for state management
- Tailwind CSS for styling
- Radix UI components

## Setup and Running

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run migrations:
   ```
   python manage.py migrate
   ```

5. Start the development server:
   ```
   python manage.py runserver
   ```
   The backend server will run at http://localhost:8000/

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm run dev
   ```
   The frontend development server will run at http://localhost:5173/ or another available port

## API Endpoints

### User API
- `POST /api/v1/user/register` - Register a new user
- `POST /api/v1/user/login` - User login
- `GET /api/v1/user/logout` - User logout
- `POST /api/v1/user/profile/update` - Update user profile

### Job API
- `GET /api/v1/job/get` - Get job listings with optional search

### Company API
- `GET /api/v1/company/get` - Get company listings

### Application API
- `GET /api/v1/application/get` - Get current user's job applications 