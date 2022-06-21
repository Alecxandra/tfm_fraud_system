from rest_framework import serializers
from ..data_processor.models import Customer, Transaction


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