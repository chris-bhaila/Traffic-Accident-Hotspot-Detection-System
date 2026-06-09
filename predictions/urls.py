from django.urls import path
from . import views

urlpatterns = [
    path("predict/", views.predict_risk, name="predict_risk"),
    path("route-risk/", views.route_risk, name="route_risk"),
]
