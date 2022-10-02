# -*- coding: utf-8 -*-

from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^customers/$', view=views.CustomerListApiView.as_view(), name='customer_list'),
    url(r'^transactions/$', view=views.TransactionListApiView.as_view(), name='transactions_list'),
    url(r'^data/generate/$', view=views.GenerateDataApiView.as_view(), name='generate_data'),
    url(r'^models/$', view=views.IAModelListApiView.as_view(), name='model_list'),
    url(r'^models/(?P<pk>[\w\-]+)/$', view=views.IAModelDetailApiView.as_view(), name='model_detail'),
    url(r'^production/$', view=views.SetProductionModelApiView.as_view(), name='model_production'),
    url(r'^training/$', view=views.IATrainingApiView.as_view(), name='model_training'),
    url(r'^results/(?P<model_id>[\w\-]+)/$', view=views.ResultsDataApiView.as_view(), name='model_results'),
    url(r'^predict/$', view=views.PredictFraudTransactionApiView.as_view(), name='model_predict'),
    url(r'^upload-data/$', view=views.UploadTransactionDataApiView.as_view(), name='upload_data'),
    url(r'^current-model/$', view=views.GetCurrentModel.as_view(), name='current_model')
]
