from rest_framework import serializers
from ..data_processor.models import Customer, Transaction
from ..ml_models.models import IAModel


class CustomerSerializer(serializers.ModelSerializer):

    class Meta:
        model = Customer
        fields = '__all__'


class TransactionSerializer(serializers.ModelSerializer):

    class Meta:
        model = Transaction
        fields = '__all__'


class GenerateDataSerializer(serializers.Serializer):
    customers = serializers.IntegerField(required=True)
    start_date = serializers.DateField(required=True, format="%m-%d-%Y", input_formats=['%m-%d-%Y'])
    end_date = serializers.DateField(required=True, format="%m-%d-%Y", input_formats=['%m-%d-%Y'])


class IAModelSerializer(serializers.ModelSerializer):

    class Meta:
        model = IAModel
        fields = '__all__'


class IAModelTrainingSerializer(serializers.Serializer):
    compilation = serializers.JSONField(required=True)
    training = serializers.JSONField(required=True)
    model_id = serializers.CharField(required=True)


class ResultsDataSerializer(serializers.Serializer):
    model_id = serializers.CharField(required=True)