"""
WSGI config for churn_project project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/wsgi/
"""

# import os

# from django.core.wsgi import get_wsgi_application

# # os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'churn_project.settings')
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'churn_project.settings.production')  # or .local
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'churn_project.settings.production')
# application = get_wsgi_application()
import os
import sys
from django.core.wsgi import get_wsgi_application

# Add project directory to Python path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Set default settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'churn_project.settings')

application = get_wsgi_application()

# Windows-specific Waitress server (for local development only)
if os.name == 'nt' and 'runserver' in sys.argv:
    from waitress import serve
    print("\n" + "="*50)
    print("Running Waitress server on http://localhost:8000")
    print("="*50 + "\n")
    serve(application, port=8000)