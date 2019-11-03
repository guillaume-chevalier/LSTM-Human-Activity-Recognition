from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import MiniBatchSequentialPipeline, Joiner
from steps.lstm_rnn_tensorflow_model import LSTMRNNTensorflowModel
from steps.lstm_rnn_tensorflow_model_wrapper import LSTMRNNTensorflowModelTrainingWrapper
from steps.one_hot_encoder import OneHotEncoder
from steps.transform_expected_output_wrapper import TransformExpectedOutputWrapper

N_HIDDEN = 32
N_STEPS = 128
N_INPUTS = 9
LAMBDA_LOSS_AMOUNT = 0.0015
LEARNING_RATE = 0.0025
N_CLASSES = 6
BATCH_SIZE = 1500

LSTM_RNN_MODEL_HYPERPARAMS = {
    'n_steps': N_STEPS,  # 128 timesteps per series
    'n_inputs': N_INPUTS,  # 9 input parameters per timestep
    'n_hidden': N_HIDDEN,  # Hidden layer num of features
    'n_classes': N_CLASSES  # Total classes (should go up, or should go down)
}

LSTM_RNN_WRAPPER_HYPERPARAMS = {
    'n_classes': N_CLASSES,  # Total classes (should go up, or should go down)
    'learning_rate': LEARNING_RATE,
    'lambda_loss_amount': LAMBDA_LOSS_AMOUNT
}


class HumanActivityRecognitionPipeline(MiniBatchSequentialPipeline):
    def __init__(self):
        MiniBatchSequentialPipeline.__init__(self, [
            TransformExpectedOutputWrapper(
                OneHotEncoder(
                    no_columns=LSTM_RNN_WRAPPER_HYPERPARAMS['n_classes'],
                    name='one_hot_encoded_label'
                )
            ),
            LSTMRNNTensorflowModelTrainingWrapper(
                tensorflow_model=LSTMRNNTensorflowModel(
                    hyperparams=HyperparameterSamples(LSTM_RNN_MODEL_HYPERPARAMS)
                ),
                hyperparams=HyperparameterSamples(LSTM_RNN_WRAPPER_HYPERPARAMS)
            ),
            Joiner(batch_size=BATCH_SIZE)
        ])
