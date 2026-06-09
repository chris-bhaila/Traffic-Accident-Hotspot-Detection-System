from django.urls import path
from . import views
from predictions.views import debug_centroids

urlpatterns = [
    path("predict/", views.predict_risk, name="predict_risk"),
    path("route-risk/", views.route_risk, name="route_risk"),
    path('debug/', views.debug_centroids),
]
