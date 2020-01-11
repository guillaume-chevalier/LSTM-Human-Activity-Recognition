import os

import numpy as np

INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

DATA_PATH = "data/"
DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"

TRAIN = "train/"
TEST = "test/"

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]

X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

TEST_FILE_NAME = "y_test.txt"
TRAIN_FILE_NAME = "y_train.txt"


def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'r')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


def load_y(y_path):
    file = open(y_path, 'r')
    # Read dataset from disk, dealing with text file's syntax
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()

    # Substract 1 to each output class for friendly 0-based indexing
    return y_ - 1


def load_data():
    # Load "X" (the neural network's training and testing inputs)

    X_train = load_X(X_train_signals_paths)
    # X_test = load_X(X_test_signals_paths)

    # Load "y" (the neural network's training and testing outputs)

    y_train_path = os.path.join(DATASET_PATH, TRAIN, TRAIN_FILE_NAME)
    # y_test_path = os.path.join(DATASET_PATH, TEST, TEST_FILE_NAME)

    y_train = load_y(y_train_path)
    # y_test = load_y(y_test_path)

    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("(data_inputs shape, expected_outputs shape, every data input mean, every data input standard deviation)")
    print(X_train.shape, y_train.shape, np.mean(X_train), np.std(X_train))
    print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

    return X_train, y_train
