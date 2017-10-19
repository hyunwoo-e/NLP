from django.db import models

class Dialog(models.Model):
    text = models.CharField(max_length=1024)