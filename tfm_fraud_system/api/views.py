from rest_framework.generics import ListAPIView, ListCreateAPIView, RetrieveAPIView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ..data_processor.tasks import generate_test_data
from ..ml_models.tasks import svm_classifier_training, neural_network_classifier_training
from ..data_processor.models import Customer, Transaction
from ..ml_models.models import IAModel, IATrainingResults, IATraining
from ..ml_models import constants
from tfm_fraud_system.ml_models.ia_models.neuralnetwork_classifier_model import NeuralNetworkClassifierModel
from tfm_fraud_system.ml_models.ia_models.svm_classifier_model import SVMClassifierModel
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

            settings = ia_model.settings
            settings['kernel'] = serializer.validated_data.get('compilation').get('kernel')

            # Save training data
            ia_model.settings = settings
            ia_model.save()

            packet = {
                "model_id": ia_model.id,
                "model_name": ia_model.name,
                "environment": ia_model.environment,
                "kernel": ia_model.settings.get('kernel')
            }

            # se envía el paquete a celery
            svm_classifier_training.apply_async(args=[packet])


        return Response(
            status=status.HTTP_200_OK,
            data={
                'message': 'Se esta realizando el entrenamiento'
            }
        )


class ResultsDataApiView(APIView):

    def get(self, request, model_id):

        # Get last Training
        a_training = IATraining.objects.filter(model=model_id).order_by('-created_at').first()

        # Get results
        ia_results = IATrainingResults.objects.filter(
            training_model=a_training.id).order_by('-created_at').first()

        return Response(
            status=status.HTTP_200_OK,
            data={
                'accuracy': ia_results.accuracy,
                'auc': ia_results.auc,
                'loss': ia_results.loss,
                'val_loss': ia_results.val_loss,
                'precision': ia_results.precision,
                'recall': ia_results.recall,
                'settings': ia_results.settings
            }
        )


class SetProductionModelApiView(APIView):
    serializer_class = serializers.SetProductionSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)

        model = IAModel.objects.get(id=serializer.validated_data.get("model_id"))
        model.environment = constants.IAModel.Enviroment.PRODUCTION
        model.save()

        return Response(status=status.HTTP_200_OK, data={})


class PredictFraudTransactionApiView(APIView):
    serializer_class = serializers.TransactionSerializer

    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)

        print(serializer.validated_data)

        customer = Customer.objects.filter(identifier=serializer.validated_data.get("customer").get("identifier")).first()

        if not customer:
            customer = Customer.objects.create(
                identifier=serializer.validated_data.get("customer").get("identifier"),
                cc_number=serializer.validated_data.get("customer").get("cc_number"),
                account_number=serializer.validated_data.get("customer").get("account_number"),
                first_name=serializer.validated_data.get("customer").get("first_name"),
                last_name=serializer.validated_data.get("customer").get("last_name"),
                gender=serializer.validated_data.get("customer").get("gender"),
                street=serializer.validated_data.get("customer").get("street"),
                city=serializer.validated_data.get("customer").get("city"),
                state=serializer.validated_data.get("customer").get("state"),
                lon=serializer.validated_data.get("customer").get("lon"),
                lat=serializer.validated_data.get("customer").get("lat"),
                job=serializer.validated_data.get("customer").get("job"),
            )

        transaction = Transaction.objects.create(
            transaction_number=serializer.validated_data.get("transaction_number"),
            transaction_date=serializer.validated_data.get("transaction_date"),
            unix_time=serializer.validated_data.get("unix_time"),
            category=serializer.validated_data.get("category"),
            amt=serializer.validated_data.get("amt"),
            merchant=serializer.validated_data.get("merchant"),
            merch_lat=serializer.validated_data.get("merch_lat"),
            merch_long=serializer.validated_data.get("merch_long"),
            environment=constants.IAModel.Enviroment.PRODUCTION,
            customer=customer
        )

        # Get production model
        is_fraud = False
        action = "ALLOW"

        model = IAModel.objects.get(environment=constants.IAModel.Enviroment.PRODUCTION)

        packet = {"model_id": model.id}

        # Data
        dataframe = pd.DataFrame([transaction.to_representation()])
        Y_col = 'is_fraud'
        X_cols = dataframe.loc[:, dataframe.columns != Y_col].columns

        encoder = LabelEncoder()
        encoder.classes_ = np.load('classes.npy', allow_pickle=True)
        x_dataframe = dataframe[X_cols].apply(encoder.fit_transform)

        standar_scaler = StandardScaler()
        x_normalize = standar_scaler.fit_transform(x_dataframe)

        if model.type == constants.IAModel.Type.NNW_CLASSIFIER:
            print("[API][neural_network_classifier_predict] Carga modelo")
            nn_model = NeuralNetworkClassifierModel(packet)
            result = nn_model.predict(x_normalize)
            print("result")
            print(result)
            if result.argmax(axis=-1)[0] == 0:
                is_fraud = False
                action = "ALLOW"
            else:
                is_fraud = True
                action = "PREVENT"

        if model.type == constants.IAModel.Type.SVM_CLASSIFIER:
            print("[API][neural_network_classifier_predict] Carga modelo")
            svm_model = SVMClassifierModel(packet)
            result = svm_model.predict(x_normalize)
            print("result")
            print(result)

            if result[0]:
                is_fraud = True
                action = "PREVENT"
            else:
                is_fraud = False
                action = "ALLOW"

        return Response(
            status=status.HTTP_200_OK,
            data={
                'is_fraud': is_fraud,
                'action': action
            }
        )


class UploadTransactionDataApiView(APIView):
    serializer_class = serializers.FileUploadDataSerializer

    def post(self, request, *args, **kwargs):
        file = request.FILES.get('file')
        # serializer = self.serializer_class(data=request.data)
        # serializer.is_valid(raise_exception=True)
        # file = serializer.validated_data['file']
        print("ANTES DE LEER FILE")
        reader = pd.read_excel(file)
        for _, row in reader.iterrows():

            customer = Customer.objects.create(
                identifier=row['customer_identifier'],
                cc_number=row['cc_number'],
                account_number=row['account_number'],
                first_name=row['first_name'],
                last_name=row['last_name'],
                gender=row['gender'],
                street=row['street'],
                city=row['city'],
                state=row['state'],
                lon=row['lon'],
                lat=row['lat'],
                job=row['job'],
                row_number=-1
            )

            Transaction.objects.create(
                customer=customer,
                transaction_number=row['transaction_number'],
                unix_time=row['unix_time'],
                category=row['category'],
                amt=row['amt'],
                merchant=row['merchant'],
                merch_lat=row['merch_lat'],
                merch_long=row['merch_long'],
                transaction_date=row['transaction_date']
            )

        return Response({"status": "success"}, status.HTTP_201_CREATED)


class GetCurrentModel(APIView):

    def get(self, request):

        # Get last Training
        ia_model = IAModel.objects.filter(environment=constants.IAModel.Enviroment.PRODUCTION).first()

        return Response(
            status=status.HTTP_200_OK,
            data={
                'id': ia_model.id,
                'name': ia_model.name,
                'type': ia_model.type
            }
        )
