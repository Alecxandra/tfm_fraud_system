from django.db import models
from django.contrib.postgres import fields as pgfields

from . import constants
import uuid


class IAModel(models.Model):
    TYPE_CHOICES = (
        (constants.IAModel.Type.NNW_CLASSIFIER, 'Modelo Red Neuronal para clasificación'),
        (constants.IAModel.Type.SVM_CLASSIFIER, 'Modelo Support Vector Machine para clasificación'),
    )

    ENVIROMENT_CHOICES = (
        (constants.IAModel.Enviroment.TESTING, 'Testing'),
        (constants.IAModel.Enviroment.PRODUCTION, 'Production'),
        (constants.IAModel.Enviroment.TRAINING, 'Training'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=30)
    type = models.CharField(choices=TYPE_CHOICES, max_length=100)
    enviroment = models.CharField(choices=ENVIROMENT_CHOICES, max_length=100)
    settings = pgfields.JSONField(blank=True, default='{}')
    architecture_model = pgfields.JSONField(blank=True, default='{}')

    class Meta:
        verbose_name = 'IAModel'



class IATraining(models.Model):
    STATUS_CHOICES = (
        (constants.IATraining.Status.SUCCEEDED, 'Exitoso'),
        (constants.IATraining.Status.FAILED, 'Fallido'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status = models.CharField(choices=STATUS_CHOICES, max_length=100)
    weights = pgfields.JSONField(blank=True, default='{}')
    model = models.ForeignKey(IAModel, on_delete=models.CASCADE)

    class Meta:
        verbose_name = 'IATraining'






