class IAModel:

    class Type:
        SVM_CLASSIFIER = 'SVM_CLASSIFIER'
        NNW_CLASSIFIER = 'NNW_CLASSIFIER'

    class Enviroment:
        TESTING = 'TESTING'
        TRAINING = 'TRAINING'
        PRODUCTION = 'PRODUCTION'


class IATraining:

    class Status:
        SUCCEEDED = 'SUCCEEDED'
        FAILED = 'FAILED'
