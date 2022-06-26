from config import celery_app

import json
import os
import pickle
import uuid
import pandas as pd
import numpy as np
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score

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
        nn_model.training(x_train, y_train, x_test, y_test)

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

        training_obj = models.IATraining.create(training_data)

        # Save training results
        model_history = nn_model.get_history()

        # ROC curve
        fp_rate, tp_rate, thresholds = roc_curve(y_test, target_predicted)

        fp_rate_list = fp_rate.tolist()
        tp_rate_list = tp_rate.tolist()


        training_results_data = {
            'accuracy': model_history.history['accuracy'][0],
            'auc': model_history.history['auc'][0],
            'loss': model_history.history['loss'][0],
            'val_loss': model_history.history['val_loss'][0],
            'precision': model_history.history['precision'][0],
            'recall': model_history.history['recall'][0],
            'settings': {
                'roc_curve': {
                    'fp_rate': json.dumps(fp_rate_list),
                    'tp_rate': json.dumps(tp_rate_list)
                }
            },
            'training_model_id': training_obj.id
        }

        models.IATrainingResults.create(training_results_data)
        print(f"[tasks][neural_network_classifier_training] Se guardan datos de entrenamiento")


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

        training_obj = models.IATraining.create(training_data)

        # ROC curve
        fp_rate, tp_rate, thresholds = roc_curve(y_test, target_predicted)

        fp_rate_list = fp_rate.tolist()
        tp_rate_list = tp_rate.tolist()

        accuracy_value = accuracy_score(y_test, target_predicted)
        recall_score_value = recall_score(y_test, target_predicted)
        precision_score_value = precision_score(y_test, target_predicted)

        training_results_data = {
            'accuracy': accuracy_value,
            'auc': auc_result,
            'precision': precision_score_value,
            'recall': recall_score_value,
            'settings': {
                'roc_curve': {
                    'fp_rate': json.dumps(fp_rate_list),
                    'tp_rate': json.dumps(tp_rate_list)
                }
            },
            'training_model_id': training_obj.id
        }

        models.IATrainingResults.create(training_results_data)
        print(f"[tasks][svm_classifier_training] Se guardan datos de entrenamiento")

    except Exception as error:
        print(f"[tasks][svm_classifier_training] Ocurrió un error en el proceso {str(error)}")

