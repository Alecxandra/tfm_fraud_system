from django.db import models
from . import constants

import uuid

class Customer(models.Model):
    STATUS_CHOICES = (
        (constants.Transaction.Environment.TESTING, 'Pruebas'),
        (constants.Transaction.Environment.PRODUCTON, 'Producción'),
    )

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
    environment = models.CharField(max_length=255, default=constants.Transaction.Environment.TESTING)


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

    def to_representation(self):
        return {
            'transaction_number': self.transaction_number,
            'transaction_date': self.transaction_date,
            'unix_time': self.unix_time,
            'category': self.category,
            'amt': self.amt,
            'is_fraud': self.is_fraud,
            'merchant': self.merchant,
            'merch_lat': self.merch_lat,
            'merch_long': self.merch_long,
            'customer_identifier': self.customer.identifier,
            'cc_number': self.customer.cc_number,
            'account_number': self.customer.account_number,
            'first_name': self.customer.first_name,
            'last_name': self.customer.last_name,
            'gender': self.customer.gender,
            'street': self.customer.street,
            'city': self.customer.city,
            'state': self.customer.state,
            'lon': self.customer.lon,
            'lat': self.customer.lat,
            'job': self.customer.job,

        }





