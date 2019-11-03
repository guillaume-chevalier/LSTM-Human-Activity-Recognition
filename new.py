# Those are separate normalised input features for the neural network
import math
import os

import numpy as np

from data_reading import DATASET_PATH, TRAIN, TEST, X_train_signals_paths, X_test_signals_paths, load_X, load_y
from neuraxle.api.flask import FlaskRestApiWrapper
from neuraxle.base import ExecutionContext, DEFAULT_CACHE_FOLDER, ExecutionMode
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import MiniBatchSequentialPipeline, Joiner
from steps.custom_json_decoder_for_2darray import CustomJSONDecoderFor2DArray
from steps.custom_json_encoder_of_outputs import CustomJSONEncoderOfOutputs
from steps.lstm_rnn_tensorflow_model import LSTMRNNTensorflowModel
from steps.lstm_rnn_tensorflow_model_wrapper import LSTMRNNTensorflowModelTrainingWrapper
from steps.one_hot_encoder import OneHotEncoder
from steps.transform_expected_output_wrapper import TransformExpectedOutputWrapper

TEST_FILE_NAME = "y_test.txt"
TRAIN_FILE_NAME = "y_train.txt"

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


def main():
    # Load "X" (the neural network's training and testing inputs)

    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    # Load "y" (the neural network's training and testing outputs)

    y_train_path = os.path.join(DATASET_PATH, TRAIN, TRAIN_FILE_NAME)
    y_test_path = os.path.join(DATASET_PATH, TEST, TEST_FILE_NAME)

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("(X shape, y shape, every X's mean, every X's standard deviation)")
    print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
    print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

    training_data_count = len(X_train)
    training_iters = training_data_count * 300

    pipeline = HumanActivityRecognitionPipeline()

    no_iter = int(math.floor(training_iters / BATCH_SIZE))
    for _ in range(no_iter):
        pipeline = pipeline.fit(X_train, y_train)

    pipeline.save(
        ExecutionContext.create_from_root(
            pipeline,
            ExecutionMode.FIT,
            DEFAULT_CACHE_FOLDER
        )
    )


def serve_rest_api():
    pipeline = HumanActivityRecognitionPipeline()

    pipeline = pipeline.load(
        ExecutionContext.create_from_root(
            pipeline,
            ExecutionMode.FIT,
            DEFAULT_CACHE_FOLDER
        )
    )

    # Easy REST API deployment.
    app = FlaskRestApiWrapper(
        json_decoder=CustomJSONDecoderFor2DArray(),
        wrapped=pipeline,
        json_encoder=CustomJSONEncoderOfOutputs()
    ).get_app()

    app.run(debug=False, port=5000)
