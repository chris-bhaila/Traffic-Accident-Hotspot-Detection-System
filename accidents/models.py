from django.db import models


class AccidentRecord(models.Model):
    latitude = models.FloatField()
    longitude = models.FloatField()
    date = models.DateField()
    time = models.TimeField(null=True, blank=True)
    day_of_week = models.CharField(max_length=20)
    weather_condition = models.CharField(max_length=50)
    road_type = models.CharField(max_length=50)
    light_condition = models.CharField(max_length=50)
    speed_limit = models.IntegerField(null=True, blank=True)
    junction_type = models.CharField(max_length=50, null=True, blank=True)
    severity = models.CharField(max_length=20)
    number_of_vehicles = models.IntegerField(default=1)
    number_of_casualties = models.IntegerField(default=0)
    number_of_deaths = models.IntegerField(default=0)
    source = models.CharField(
        max_length=20,
        choices=[
            ("UK_STATS19", "UK STATS19"),
            ("KTM_SYNTHETIC", "Kathmandu Synthetic"),
            ("KTM_SCRAPED", "Kathmandu Scraped"),
        ],
    )
    # New fields
    vehicle_type = models.CharField(max_length=50, null=True, blank=True)
    accident_type = models.CharField(max_length=100, null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    source_url = models.URLField(null=True, blank=True)
    location_name = models.CharField(max_length=100, null=True, blank=True)

    def __str__(self):
        return f"{self.severity} accident at ({self.latitude}, {self.longitude}) on {self.date}"

    class Meta:
        ordering = ["-date"]


class HotspotCluster(models.Model):
    centroid_latitude = models.FloatField()
    centroid_longitude = models.FloatField()
    accident_count = models.IntegerField()
    radius = models.FloatField()
    average_severity = models.FloatField()
    peak_time = models.CharField(max_length=20, null=True, blank=True)
    peak_day = models.CharField(max_length=20, null=True, blank=True)
    dominant_weather = models.CharField(max_length=50, null=True, blank=True)
    risk_level = models.CharField(
        max_length=20,
        choices=[
            ("LOW", "Low"),
            ("MEDIUM", "Medium"),
            ("HIGH", "High"),
            ("CRITICAL", "Critical"),
        ],
    )
    district = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return f"{self.risk_level} cluster at ({self.centroid_latitude}, {self.centroid_longitude})"


class DataUpload(models.Model):
    filename = models.CharField(max_length=255)
    uploaded_by = models.ForeignKey("auth.User", on_delete=models.SET_NULL, null=True)
    upload_date = models.DateTimeField(auto_now_add=True)
    record_count = models.IntegerField(default=0)
    status = models.CharField(
        max_length=20,
        choices=[
            ("PROCESSING", "Processing"),
            ("COMPLETE", "Complete"),
            ("FAILED", "Failed"),
        ],
        default="PROCESSING",
    )

    def __str__(self):
        return f"{self.filename} - {self.status}"
