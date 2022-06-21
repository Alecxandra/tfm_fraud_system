# -*- coding: utf-8 -*-

from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^customers/$', view=views.CustomerListApiView.as_view(), name='customer_list'),
    url(r'^transactions/$', view=views.TransactionListApiView.as_view(), name='transactions_list'),
    url(r'^data/generate/$', view=views.GenerateDataApiView.as_view(), name='generate_data'),
]
