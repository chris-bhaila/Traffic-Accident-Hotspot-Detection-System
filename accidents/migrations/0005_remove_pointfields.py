from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("accidents", "0004_accidentrecord_number_of_deaths"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="accidentrecord",
            name="location",
        ),
        migrations.RemoveField(
            model_name="hotspotcluster",
            name="centroid_location",
        ),
    ]
