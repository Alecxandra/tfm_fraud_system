from sklearn import svm


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
        self.settings = settings

    def get_model(self):
        return self.model

    def config_model(self):
        """
         Función que realiza la configuración de los parámetros del modelo
        """

        # TODO constans
        # TODO métricas de evaluación

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
            verbose=self.settings.get('verbose', False),  # Enable verbose output
            max_iter=self.settings.get('max_iter', - 1),  # Hard limit on iterations
            decision_function_shape=self.settings.get('decision_function_shape', 'ovr'),  # One-vs-rest or one-vs-one
            break_ties=self.settings.get('break_ties', False),  # How to handle breaking ties
            random_state=self.settings.get('random_state', None)
        )


    def training(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
