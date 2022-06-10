from config import celery_app

import os
import pandas as pd
import numpy as np
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from tfm_fraud_system.ml_models.models.neuralnetwork_classifier_model import NeuralNetworkClassifierModel
from tfm_fraud_system.ml_models.models.svm_classifier_model import SVMClassifierModel

@celery_app.task(soft_time_limit=600, time_limit=700)
def neural_network_classifier_training(packet):

    try:
        # TODO resolver lo del dataset (por ahora quemado)
        file_path = os.path.join(settings.STATIC_ROOT, 'testing_data/creditcard.csv')
        dataframe = pd.read_csv(file_path)
        print(dataframe.head())

        # Training and test datasets
        features = dataframe.iloc[:, 0:30]
        target = dataframe.iloc[:, 30]

        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3)

        # Configuración modelo

        print("[tasks][neural_network_classifier_training] Creación del modelo")
        nn_model = NeuralNetworkClassifierModel(packet)

        print("[tasks][neural_network_classifier_training] Configuración del modelo")
        nn_model.config_model()

        print("[tasks][neural_network_classifier_training] Compilcación del modelo")
        nn_model.compilation()

        print("[tasks][neural_network_classifier_training] Inicio del entrenamiento")
        nn_model.training(x_train, y_train)

        print("[tasks][neural_network_classifier_training] Finalización del entrenamiento")

        print("[tasks][neural_network_classifier_training] Predicción y resultados")
        target_predicted = nn_model.predict(x_test)

        fpr, tpr, thresholds = roc_curve(y_test, target_predicted)
        auc_result = auc(fpr, tpr)

        print(f"[tasks][neural_network_classifier_training] Valor AUC: {auc_result}")

    except Exception as error:
        print(f"[tasks][neural_network_classifier_training] Ocurrió un error en el proceso {str(error)}")


@celery_app.task(soft_time_limit=600, time_limit=700)
def svm_classifier_training(packet):

    try:
        # TODO resolver lo del dataset (por ahora quemado)
        file_path = os.path.join(settings.STATIC_ROOT, 'testing_data/creditcard.csv')
        dataframe = pd.read_csv(file_path)
        print(dataframe.head())

        # Training and test datasets
        dataset = dataframe.reset_index()

        x = np.nan_to_num(dataset.drop('Class', axis=1))
        y = np.nan_to_num(dataset['Class'])

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        # Configuración modelo

        print("[tasks][svm_classifier_training] Creación del modelo")
        svm_model = SVMClassifierModel(packet)

        print("[tasks][svm_classifier_training] Configuración del modelo")
        svm_model.config_model()

        print("[tasks][svm_classifier_training] Inicio del entrenamiento")
        svm_model.training(x_train, y_train)

        print("[tasks][svm_classifier_training] Finalización del entrenamiento")

        print("[tasks][svm_classifier_training] Predicción y resultados")
        target_predicted = svm_model.predict(x_test)

        fpr, tpr, thresholds = roc_curve(y_test, target_predicted)
        auc_result = auc(fpr, tpr)

        print(f"[tasks][svm_classifier_training] Valor AUC: {auc_result}")

    except Exception as error:
        print(f"[tasks][svm_classifier_training] Ocurrió un error en el proceso {str(error)}")



