from config import celery_app

import os
import pickle
import uuid
import pandas as pd
import numpy as np
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from tfm_fraud_system.ml_models.ia_models.neuralnetwork_classifier_model import NeuralNetworkClassifierModel
from tfm_fraud_system.ml_models.ia_models.svm_classifier_model import SVMClassifierModel
from . import models
from . import constants
@celery_app.task(soft_time_limit=7200, time_limit=7200)
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

        # Saving model changes

        if not nn_model.get_init_presaved_model():
            data = {
                'name': packet.get('model_name'),
                'type': constants.IAModel.Type.NNW_CLASSIFIER,
                'environment': packet.get('environment'),
                'settings': packet,
                'architecture_model': nn_model.get_model().to_json()
            }
            ia_model = models.IAModel.create(data)

            nn_model.set_db_model(ia_model)

            print(f"[tasks][neural_network_classifier_training] Modelo almacenado: {ia_model.id}")

        # Save training

        training_model = nn_model.get_model()
        db_model = nn_model.get_db_model()

        weights_url = os.path.join(settings.STATIC_ROOT, f"results/nnc/{uuid.uuid1()}.h5")
        training_model.save_weights(weights_url)


        training_data = {
            'status': constants.IATraining.Status.SUCCEEDED,
            'weights_url': weights_url,
            'model_id': db_model.id
        }

        models.IATraining.create(training_data)


    except Exception as error:
        print(f"[tasks][neural_network_classifier_training] Ocurrió un error en el proceso {str(error)}")


@celery_app.task(soft_time_limit=7200, time_limit=7200)
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


        # Saving model changes
        if not svm_model.get_db_model():
            data = {
                'name': packet.get('model_name'),
                'type': constants.IAModel.Type.SVM_CLASSIFIER,
                'environment': packet.get('environment'),
                'settings': packet,
                'architecture_model': {}
            }

            ia_model = models.IAModel.create(data)
            svm_model.set_db_model(ia_model)

            print(f"[tasks][svm_classifier_training] Modelo almacenado: {ia_model.id}")

        # Saving training
        training_model = svm_model.get_model()
        db_model = svm_model.get_db_model()

        weights_url = os.path.join(settings.STATIC_ROOT, f"results/svm/{uuid.uuid1()}.sav")
        pickle.dump(training_model, open(weights_url, 'wb'))


        training_data = {
            'status': constants.IATraining.Status.SUCCEEDED,
            'weights_url': weights_url,
            'model_id': db_model.id
        }

        models.IATraining.create(training_data)

    except Exception as error:
        print(f"[tasks][svm_classifier_training] Ocurrió un error en el proceso {str(error)}")



