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
    environment = models.CharField(choices=ENVIROMENT_CHOICES, max_length=100)
    settings = pgfields.JSONField(blank=True, default='{}')
    architecture_model = pgfields.JSONField(blank=True, default='{}')

    class Meta:
        verbose_name = 'IAModel'

    @classmethod
    def create(cls, data, **kwargs):
        "Método para crear un modelo de inteligencia artificial"

        try:

            ia_model = IAModel(
                name=data.get('name'),
                type=data.get('type'),
                environment=data.get('environment'),
                settings=data.get('settings'),
                architecture_model=data.get('architecture_model')
            )

            ia_model.save()

            return ia_model
        except Exception as error:
            print("[models][IAModel] Ocurrió un error al salvar el modelo")
            prnit(error)



class IATraining(models.Model):
    STATUS_CHOICES = (
        (constants.IATraining.Status.SUCCEEDED, 'Exitoso'),
        (constants.IATraining.Status.FAILED, 'Fallido'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    status = models.CharField(choices=STATUS_CHOICES, max_length=100)
    weights_url = models.CharField(max_length=300)
    model = models.ForeignKey(IAModel, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'IATraining'

    @classmethod
    def create(cls, data, **kwargs):
        try:
            ia_training = IATraining(
                status=data.get('status'),
                weights_url=data.get('weights_url'),
                model_id=data.get('model_id')
            )
            ia_training.save()

        except Exception as error:
            print("[models][IATraining] Ocurrió un error al salvar el modelo")
            print(error)









