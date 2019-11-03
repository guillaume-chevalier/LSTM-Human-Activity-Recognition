# Those are separate normalised input features for the neural network
import math
import os

import numpy as np

from data_reading import DATASET_PATH, TRAIN, TEST, X_train_signals_paths, X_test_signals_paths, load_X, load_y, \
    TRAIN_FILE_NAME, TEST_FILE_NAME
from neuraxle.api.flask import FlaskRestApiWrapper
from neuraxle.base import ExecutionContext, DEFAULT_CACHE_FOLDER, ExecutionMode
from pipeline import HumanActivityRecognitionPipeline, BATCH_SIZE
from steps.custom_json_decoder_for_2darray import CustomJSONDecoderFor2DArray
from steps.custom_json_encoder_of_outputs import CustomJSONEncoderOfOutputs


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
        pipeline, outputs = pipeline.fit_transform(X_train, y_train)

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


if __name__ == '__main__':
    main()
