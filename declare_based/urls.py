from django.urls import path
from .views import Recommendation

urlpatterns = [
    path('recommendations/', Recommendation.recommend),
]
