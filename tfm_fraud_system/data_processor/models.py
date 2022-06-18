from django.db import models

import uuid

class Customer(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    cc_number = models.CharField(max_length=20)
    first_name = models.CharField(max_length=255)
    last_name = models.CharField(max_length=255)
    gender = models.CharField(max_length=1)
    row_number = models.IntegerField()
    street = models.CharField(max_length=255)
    city = models.CharField(max_length=255)
    # TODO LAT Y LONG
    job = models.CharField(max_length=255)
    profile = models.CharField(max_length=255)


class Transaction(models.Model):
    pass


