from rest_framework.generics import ListAPIView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from datetime import datetime

from ..data_processor.tasks import generate_test_data
from ..data_processor.models import Customer, Transaction
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

