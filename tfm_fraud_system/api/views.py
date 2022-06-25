from rest_framework.generics import ListAPIView, ListCreateAPIView, RetrieveAPIView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from datetime import datetime

from ..data_processor.tasks import generate_test_data
from ..ml_models.tasks import svm_classifier_training, neural_network_classifier_training
from ..data_processor.models import Customer, Transaction
from ..ml_models.models import IAModel
from ..ml_models import constants
from . import serializers


class CustomerListApiView(ListAPIView):
    serializer_class = serializers.CustomerSerializer

    def get_queryset(self):
        queryset = Customer.objects.all()
        env = self.request.query_params.get('env', None)

        if env is not None:
            queryset = queryset.filter(environment=env)

        return queryset


class TransactionListApiView(ListAPIView):
    serializer_class = serializers.TransactionSerializer

    def get_queryset(self):
        queryset = Transaction.objects.all()
        env = self.request.query_params.get('env', None)

        if env is not None:
            queryset = queryset.filter(environment=env)

        return queryset


class GenerateDataApiView(APIView):
    serializer_class = serializers.GenerateDataSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Se forma un paquete para enviar al proceso asíncrono
        packet = {
            "customer": serializer.validated_data.get('customers'),
            "start_date": serializer.validated_data.get('start_date').strftime("%m-%d-%Y"),
            "end_date": serializer.validated_data.get('end_date').strftime("%m-%d-%Y")
        }

        # se envía el paquete a celery
        generate_test_data.apply_async(args=[packet])

        return Response(
            status=status.HTTP_200_OK,
            data={
                'message': 'Se esta realizando la generación de datos'
            }
        )

class IAModelListApiView(ListCreateAPIView):
    serializer_class = serializers.IAModelSerializer

    def get_queryset(self):
        queryset = IAModel.objects.all()

        return queryset


class IAModelDetailApiView(RetrieveAPIView):
    serializer_class = serializers.IAModelSerializer

    def get_queryset(self):
        queryset = IAModel.objects.all()

        return queryset

class IATrainingApiView(APIView):
    serializer_class = serializers.IAModelTrainingSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Get IA model
        ia_model = IAModel.objects.get(id=serializer.validated_data.get('model_id'))

        if ia_model.type == constants.IAModel.Type.NNW_CLASSIFIER:

            settings = ia_model.settings
            settings['compilation'] = serializer.validated_data.get('compilation')
            settings['training'] = serializer.validated_data.get('training')

            # Save training data
            ia_model.settings = settings
            ia_model.save()

            # Se forma el paquete para enviar a entenamiento
            packet = {
                "model_id": ia_model.id,
                "model_name": ia_model.name,
                "environment": ia_model.environment,
                "layers": ia_model.settings.get('layers'),
                "compilation": ia_model.settings.get('compilation'),
                "training": ia_model.settings.get('training')
            }

            # se envía el paquete a celery
            neural_network_classifier_training.apply_async(args=[packet])



        if ia_model.type == constants.IAModel.Type.SVM_CLASSIFIER:

            packet = {
                "model_id": ia_model.id,
                "model_name": ia_model.name,
                "environment": ia_model.environment,
                "kernel": ia_model.settings.kernel
            }

            # se envía el paquete a celery
            svm_classifier_training.apply_async(args=[packet])


        return Response(
            status=status.HTTP_200_OK,
            data={
                'message': 'Se esta realizando el entrenamiento'
            }
        )
