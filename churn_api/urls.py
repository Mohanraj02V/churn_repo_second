from django.urls import path
from .views import PredictChurnAPIView
from django.http import JsonResponse

def health_check(request):
    return JsonResponse({"status": "ok"})
urlpatterns = [
    path('predict/', PredictChurnAPIView.as_view(), name='predict-churn'),
]
