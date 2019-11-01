import numpy as np
import tensorflow as tf

from steps.one_hot_encoder import OneHotEncoder

# Those are separate normalised input features for the neural network
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

# Output classes to learn how to classify
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


def main():
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


    def LSTM_RNN(_X, _weights, _biases):
        # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
        # Moreover, two LSTM cells are stacked which adds deepness to the neural network.
        # Note, some code of this notebook is inspired from an slightly different
        # RNN architecture used on another dataset, some of the credits goes to
        # "aymericdamien" under the MIT license.

        # (NOTE: This step could be greatly optimised by shaping the dataset once
        # input shape: (batch_size, n_steps, n_input)
        _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
        # Reshape to prepare input to hidden activation
        _X = tf.reshape(_X, [-1, n_input])
        # new shape: (n_steps*batch_size, n_input)

        # ReLU activation, thanks to Yu Zhao for adding this improvement here:
        _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(_X, n_steps, 0)
        # new shape: n_steps * (batch_size, n_hidden)

        # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
        # Get LSTM cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

        # Get last time step's output feature for a "many-to-one" style classifier,
        # as in the image describing RNNs at the top of this page
        lstm_last_output = outputs[-1]

        # Linear activation
        return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


    def extract_batch_size(_train, step, batch_size):
        # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

        shape = list(_train.shape)
        shape[0] = batch_size
        batch_s = np.empty(shape)

        for i in range(batch_size):
            # Loop index
            index = ((step - 1) * batch_size + i) % len(_train)
            batch_s[i] = _train[index]

        return batch_s

    # Load "X" (the neural network's training and testing inputs)

    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    # Load "y" (the neural network's training and testing outputs)

    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"

    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    # Input Data

    training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
    test_data_count = len(X_test)  # 2947 testing series
    n_steps = len(X_train[0])  # 128 timesteps per series
    n_input = len(X_train[0][0])  # 9 input parameters per timestep

    # LSTM Neural Network's internal structure

    n_hidden = 32  # Hidden layer num of features
    n_classes = 6  # Total classes (should go up, or should go down)

    # Training

    learning_rate = 0.0025
    lambda_loss_amount = 0.0015
    training_iters = training_data_count * 300  # Loop 300 times on the dataset
    batch_size = 1500
    display_iter = 30000  # To show test set accuracy during training

    # Some debugging info

    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("(X shape, y shape, every X's mean, every X's standard deviation)")
    print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
    print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Graph weights
    weights = {
        'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),  # Hidden layer weights
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
    }
    biases = {
        'hidden': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = LSTM_RNN(x, weights, biases)

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )  # L2 loss prevents this overkill neural network to overfit the data
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred)) + l2  # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)  # Adam Optimizer

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # To keep track of training's performance
    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []

    # Launch the graph
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    init = tf.global_variables_initializer()
    sess.run(init)

    # Perform Training steps with "batch_size" amount of example data at each loop
    step = 1
    while step * batch_size <= training_iters:
        batch_xs = extract_batch_size(X_train, step, batch_size)
        batch_size_train = extract_batch_size(y_train, step, batch_size)

        batch_ys = OneHotEncoder(
            no_columns=n_classes,
            name='batch_ys'
        ).transform(batch_size_train)

        # Fit training using batch data
        _, loss, acc = sess.run(
            [optimizer, cost, accuracy],
            feed_dict={
                x: batch_xs,
                y: batch_ys
            }
        )
        train_losses.append(loss)
        train_accuracies.append(acc)

        # Evaluate network only at some steps for faster training:
        if (step * batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
            # To not spam console, show training accuracy/loss in this "if"
            print("Training iter #" + str(step * batch_size) + \
                  ":   Batch Loss = " + "{:.6f}".format(loss) + \
                  ", Accuracy = {}".format(acc))

        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        one_hot_encoded_y_test = OneHotEncoder(
            no_columns=n_classes,
            name='one_hot_encoded_y_test'
        ).transform(y_test)

        loss, acc = sess.run(
            [cost, accuracy],
            feed_dict={
                x: X_test,
                y: one_hot_encoded_y_test
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

        step += 1

    print("Optimization Finished!")

    # Accuracy for test data

    one_host_encoded_y_test = OneHotEncoder(
        no_columns=n_classes,
        name='one_hot_predictions'
    ).transform(y_test)

    one_hot_predictions, accuracy, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={
            x: X_test,
            y: one_host_encoded_y_test
        }
    )

    test_losses.append(final_loss)
    test_accuracies.append(accuracy)

    print("FINAL RESULT: " + \
          "Batch Loss = {}".format(final_loss) + \
          ", Accuracy = {}".format(accuracy))



