from django.db import models

# Create your models here.
class UserSettings(models.Model):
    frequency = models.CharField(max_length=20, choices=(
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('fortnightly', 'Fortnightly'),
        ('monthly', 'Monthly'),
    ))
    calling_time = models.TimeField()