from django.urls import path
from .views import DeclareTemplates, Recommendation

urlpatterns = [
    path('declare-templates/', DeclareTemplates.generate),
    path('recommendations/', Recommendation.recommend),
]
