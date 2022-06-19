from django.db import models
from . import constants

import uuid

class Customer(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    identifier = models.CharField(max_length=30)
    cc_number = models.CharField(max_length=20)
    account_number = models.CharField(max_length=30)
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    gender = models.CharField(max_length=1)
    row_number = models.IntegerField()
    street = models.CharField(max_length=255)
    city = models.CharField(max_length=255)
    state = models.CharField(max_length=255)
    lon = models.FloatField(default=0)
    lat = models.FloatField(default=0)
    job = models.CharField(max_length=255)
    profile = models.CharField(max_length=255)


class Transaction(models.Model):
    STATUS_CHOICES = (
        (constants.Transaction.Environment.TESTING, 'Pruebas'),
        (constants.Transaction.Environment.PRODUCTON, 'Producción'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    transaction_number = models.CharField(max_length=255)
    transaction_date = models.DateTimeField()
    unix_time = models.IntegerField()
    category = models.CharField(max_length=255)
    amt = models.CharField(max_length=255)
    is_fraud = models.BooleanField(default=False)
    merchant = models.CharField(max_length=255)
    merch_lat = models.FloatField(default=0)
    merch_long = models.FloatField(default=0)
    environment = models.CharField(max_length=255)
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)





