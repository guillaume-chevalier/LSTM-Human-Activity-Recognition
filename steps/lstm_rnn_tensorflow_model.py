import tensorflow as tf

from neuraxle.base import BaseStep
from neuraxle.hyperparams.space import HyperparameterSamples
from savers.tensorflow1_step_saver import Tensorflow1StepSaver


class LSTMRNNTensorflowModel(BaseStep):
    HYPERPARAMS = HyperparameterSamples({
        'n_steps': 128,
        'n_input': 9,
        'n_hidden': 32,
        'n_classes': 6
    })

    def __init__(self, name=None, hyperparams=HYPERPARAMS):
        BaseStep.__init__(
            self,
            name=name,
            hyperparams=hyperparams,
            savers=[Tensorflow1StepSaver()]
        )

        self.x = None
        self.y = None
        self.weights = None
        self.biases = None
        self.model = None

    def get_x_placeholder(self):
        return self.x

    def get_y_placeholder(self):
        return self.y

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    def get_model(self):
        return self.model

    def setup(self):
        # Graph input/output
        self.x = tf.placeholder(tf.float32, [None, self.hyperparams['n_steps'], self.hyperparams['n_input']])
        self.y = tf.placeholder(tf.float32, [None, self.hyperparams['n_classes']])

        # Graph weights
        self.weights = {
            'hidden': tf.Variable(
                tf.random_normal([self.hyperparams['n_input'], self.hyperparams['n_hidden']])
            ),  # Hidden layer weights
            'out': tf.Variable(
                tf.random_normal([self.hyperparams['n_hidden'], self.hyperparams['n_classes']], mean=1.0)
            )
        }

        self.biases = {
            'hidden': tf.Variable(
                tf.random_normal([self.hyperparams['n_hidden']])
            ),
            'out': tf.Variable(
                tf.random_normal([self.hyperparams['n_classes']])
            )
        }

        self.model = self._create_lstm_rnn_model(self.x)

    def _create_lstm_rnn_model(self, data_inputs):
        # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
        # Moreover, two LSTM cells are stacked which adds deepness to the neural network.
        # Note, some code of this notebook is inspired from an slightly different
        # RNN architecture used on another dataset, some of the credits goes to
        # "aymericdamien" under the MIT license.
        # (NOTE: This step could be greatly optimised by shaping the dataset once
        # input shape: (batch_size, n_steps, n_input)

        data_inputs = tf.transpose(
            data_inputs,
            [1, 0, 2])  # permute n_steps and batch_size

        # Reshape to prepare input to hidden activation
        data_inputs = tf.reshape(data_inputs, [-1, self.hyperparams['n_input']])
        # new shape: (n_steps*batch_size, n_input)

        # ReLU activation, thanks to Yu Zhao for adding this improvement here:
        _X = tf.nn.relu(
            tf.matmul(data_inputs, self.weights['hidden']) + self.biases['hidden']
        )

        # Split data because rnn cell needs a list of inputs for the RNN inner loop
        _X = tf.split(_X, self.hyperparams['n_steps'], 0)
        # new shape: n_steps * (batch_size, n_hidden)

        # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
        lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.hyperparams['n_hidden'], forget_bias=1.0, state_is_tuple=True)
        lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(self.hyperparams['n_hidden'], forget_bias=1.0, state_is_tuple=True)
        lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)

        # Get LSTM cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

        # Get last time step's output feature for a "many-to-one" style classifier,
        # as in the image describing RNNs at the top of this page
        lstm_last_output = outputs[-1]

        # Linear activation
        return tf.matmul(lstm_last_output, self.weights['out']) + self.biases['out']
