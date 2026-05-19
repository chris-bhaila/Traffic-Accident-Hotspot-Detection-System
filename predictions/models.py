from django.db import models


class PredictionLog(models.Model):
    latitude = models.FloatField()
    longitude = models.FloatField()
    input_time = models.CharField(max_length=20)
    input_day = models.CharField(max_length=20)
    input_weather = models.CharField(max_length=50)
    input_road_type = models.CharField(max_length=50)
    predicted_risk = models.CharField(max_length=20)
    confidence_score = models.FloatField()
    decision_path = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.predicted_risk} prediction at ({self.latitude}, {self.longitude})"