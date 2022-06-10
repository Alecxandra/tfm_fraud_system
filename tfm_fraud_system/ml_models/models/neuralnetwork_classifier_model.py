from keras import Sequential
from keras.layers import Dense, Dropout


class NeuralNetworkClassifierModel:
    """
    Clase para crear un modelo de Red Neuronal con parámetros variables utilizando Keras

    Estructura del diccionario settings:
        {
            layers: {
                input_layer: {
                    units: 1,
                    activation: "relu",
                    kernel: "random_normal",
                    input_dim: 30,
                    kernel_regularizer: 'L1',
                },
                output_layer: {
                    units: 1,
                    activation: "relu",
                    kernel: "random_normal",
                    kernel_regularizer: 'L1'
                },
                hidden_layers: [
                    {
                        type: 'DENSE'
                        units: 1,
                        activation: "relu",
                        kernel: "random_normal",
                        kernel_regularizer: 'L1',
                    },
                     {
                        type: 'DROPOUT'
                        rate: 0.5,
                    },
                ]
            },
            compilation: {
                optimizer: 'adam',
                loss: 'binary_crossentropy'
            }
        }
    """

    def __init__(self, settings):
        self.model = Sequential()
        print(settings)
        self.settings = settings

    def get_model(self):
        return self.model

    def config_model(self):
        """
         Función que realiza la configuración de las capas ocultas como de los parámetros de la red neuronal
        """

        # TODO constans
        # TODO métricas de evaluación

        layers = self.settings.get('layers', {})

        # Input layer
        input_layer = layers.get('input_layer', {})

        self.model.add(Dense(
            input_layer.get('units'),
            activation=input_layer.get('activation', None),
            kernel_initializer=input_layer.get('kernel', 'glorot_uniform'),
            kernel_regularizer=input_layer.get('kernel_regularizer', None),
            input_dim=input_layer.get('input_dim'))
        )

        # Hidden layers
        for layer in layers.get('hidden_layers', []):

            if layer.get('type', '') == 'DENSE':
                self.model.add(Dense(
                    layer.get('units'),
                    activation=layer.get('activation', None),
                    kernel_initializer=layer.get('kernel', None),
                    kernel_regularizer=layer.get('kernel_regularizer', None))
                )

            if layer.get('type', '') == 'DROPOUT':
                self.model.add(
                    Dropout(layer.get('rate'))
                )

        # Output layer
        output_layer = layers.get('output_layer', {})

        self.model.add(Dense(
            output_layer.get('units'),
            activation=output_layer.get('activation', None),
            kernel_initializer=output_layer.get('kernel', None),
            kernel_regularizer=output_layer.get('kernel_regularizer', None)),
        )

    def compilation(self):
        compilation = self.settings.get('compilation', {})

        self.model.compile(
            optimizer=compilation.get('optimizer', 'adam'),
            loss=compilation.get('loss', 'binary_crossentropy'),
            metrics=['accuracy'])


    def training(self, x_train, y_train):
        self.model.fit(
            x_train,
            y_train,
            batch_size=self.settings.get('training', {}).get('batch_size', 10),
            epochs=self.settings.get('training', {}).get('epochs', 10)
        )

    def predict(self, x_test):
        return self.model.predict(x_test)

