import pickle

from sklearn import svm
from .. import models


class SVMClassifierModel:
    """
    Clase para crear un modelo de support vector machine con parámetros variables utilizando sklearn

    Estructura del diccionario settings:
        {
           C=1.0,                          # The regularization parameter
           kernel='rbf',                   # The kernel type used
           degree=3,                       # Degree of polynomial function
           gamma='scale',                  # The kernel coefficient
           coef0=0.0,                      # If kernel = 'poly'/'sigmoid'
           shrinking=True,                 # To use shrinking heuristic
           probability=False,              # Enable probability estimates
           tol=0.001,                      # Stopping crierion
           cache_size=200,                 # Size of kernel cache
           class_weight=None,              # The weight of each class
           verbose=False,                  # Enable verbose output
           max_iter=- 1,                   # Hard limit on iterations
           decision_function_shape='ovr',  # One-vs-rest or one-vs-one
           break_ties=False,               # How to handle breaking ties
           random_state=None
        }
    """

    def __init__(self, settings):
        self.model = None
        self.db_model = None
        self.init_presaved_model = False
        self.settings = settings
        self.init_model()

    def get_model(self):
        return self.model

    def config_model(self):
        """
         Función que realiza la configuración de los parámetros del modelo
        """

        # TODO constans
        # TODO métricas de evaluación

        if not self.init_presaved_model:

            self.model = svm.SVC(
                C=self.settings.get('C',1.0),  # The regularization parameter
                kernel=self.settings.get('kernel','rbf'),  # The kernel type used
                degree=self.settings.get('degree', 3),  # Degree of polynomial function
                gamma=self.settings.get('gamma', 'scale'),  # The kernel coefficient
                coef0=self.settings.get('coef0',0.0),  # If kernel = 'poly'/'sigmoid'
                shrinking=self.settings.get('shrinking', True),  # To use shrinking heuristic
                probability=self.settings.get('probability', False),  # Enable probability estimates
                tol=self.settings.get('tol', 0.001),  # Stopping crierion
                cache_size=self.settings.get('cache_size', 200),  # Size of kernel cache
                class_weight=self.settings.get('class_weight', None),  # The weight of each class
                verbose=self.settings.get('verbose', True),  # Enable verbose output
                max_iter=self.settings.get('max_iter', - 1),  # Hard limit on iterations
                decision_function_shape=self.settings.get('decision_function_shape', 'ovr'),  # One-vs-rest or one-vs-one
                break_ties=self.settings.get('break_ties', False),  # How to handle breaking ties
                random_state=self.settings.get('random_state', None),
            )


    def training(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)

    def get_init_presaved_model(self):
        return self.init_presaved_model

    def set_db_model(self, model):
        self.db_model = model

    def get_db_model(self):
        return self.db_model


    def init_model(self):
        # Se verifica si ya existe un modelo previamente almacenado
        try:
            if self.settings.get('model_id', None):
                model_db = models.IAModel.objects.get(id=self.settings.get('model_id'))

                if model_db:

                    self.db_model = model_db
                    weights = self.load_weights()

                    if weights:
                        self.model = pickle.load(open(weights.weights_url, 'rb'))
                        self.init_presaved_model = True
                        print("[ml_models][SVMClassifierModel] Modelo cargado")
                    else:
                        self.model = None
                else:
                    self.model = None
            else:
                self.model = None

        except Exception as error:
            print("[ml_models][SVMClassifierModel] Ocurrió un error al cargar el modelo")
            print(error)


    def load_weights(self):
        try:
            last_training = models.IATraining.objects \
                .filter(model_id=self.db_model.id) \
                .order_by('created_at') \
                .latest('created_at')

            return last_training

        except Exception as error:
            return None
