from django.contrib import admin
from .models import AccidentRecord, HotspotCluster, DataUpload

admin.site.register(AccidentRecord)
admin.site.register(HotspotCluster)
admin.site.register(DataUpload)