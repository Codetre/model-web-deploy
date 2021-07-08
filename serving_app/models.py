from django.db import models


# Create your models here.
class Image(models.Model):
    file = models.ImageField(upload_to='images/')


# Don't migrate it yet.
# class Prediction(models.Model):
#     file = models.ForeignKey()
#     bboxes = models.CharField()
#     scores = models.CharField()
#     labels = models.CharField()