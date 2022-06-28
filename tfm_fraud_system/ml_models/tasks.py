from config import celery_app

import json
import os
import pickle
import uuid
import pandas as pd
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tfm_fraud_system.ml_models.ia_models.neuralnetwork_classifier_model import NeuralNetworkClassifierModel
from tfm_fraud_system.ml_models.ia_models.svm_classifier_model import SVMClassifierModel
from . import models
from . import constants
from ..data_processor.models import Transaction



@celery_app.task(soft_time_limit=7200, time_limit=7200)
def neural_network_classifier_training(packet):

    try:

        transactions = Transaction.objects.select_related('customer').all()

        dataset = [transaction.to_representation() for transaction in transactions]
        dataframe = pd.DataFrame(dataset)

        Y_col = 'is_fraud'
        X_cols = dataframe.loc[:, dataframe.columns != Y_col].columns

        # Encode string variables
        x_dataframe = dataframe[X_cols].apply(LabelEncoder().fit_transform)
        print(x_dataframe.head())

        # Data normalization
        standarScaler = StandardScaler()
        x_normalize = standarScaler.fit_transform(x_dataframe)

        # TODO SACAR TEST SIZE DEL PAQUETE
        x_train, x_test, y_train, y_test = train_test_split(x_normalize, dataframe[Y_col], test_size=0.5)

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

        # loss, accuracy, f1_score, precision, recall = nn_model.get_model().evaluate(x_test, y_test, verbose=0)

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

        transactions = Transaction.objects.select_related('customer').all()

        dataset = [ transaction.to_representation() for transaction in transactions]
        dataframe = pd.DataFrame(dataset)


        Y_col = 'is_fraud'
        X_cols = dataframe.loc[:, dataframe.columns != Y_col].columns

        # Encode string variables

        # TODO revisar que hace si se le envian datos numéricos
        x_dataframe = dataframe[X_cols].apply(LabelEncoder().fit_transform)

        # Data normalization
        standarScaler = StandardScaler()
        x_normalize = standarScaler.fit_transform(x_dataframe)

        # TODO SACAR TEST SIZE DEL PAQUETE
        x_train, x_test, y_train, y_test = train_test_split(x_normalize, dataframe[Y_col], test_size=0.3)

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
        print(error)

