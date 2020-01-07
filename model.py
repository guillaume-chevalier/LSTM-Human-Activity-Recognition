import numpy as np
from neuraxle.base import BaseStep, ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.hyperparams.space import HyperparameterSamples
import tensorflow as tf

from neuraxle_tensorflow.tensorflow_v1 import TensorflowV1ModelStep

N_HIDDEN = 32
N_STEPS = 128
N_INPUTS = 9
LAMBDA_LOSS_AMOUNT = 0.0015
LEARNING_RATE = 0.0025
N_CLASSES = 6
BATCH_SIZE = 1500


def create_graph(step: TensorflowV1ModelStep):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network.
    # Note, some code of this notebook is inspired from an slightly different
    # RNN architecture used on another dataset, some of the credits goes to
    # "aymericdamien" under the MIT license.
    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)

    # Graph input/output
    data_inputs = tf.placeholder(tf.float32, [None, step.hyperparams['n_steps'], step.hyperparams['n_inputs']],
                                 name='data_inputs')
    expected_outputs = tf.placeholder(tf.float32, [None, step.hyperparams['n_classes']], name='expected_outputs')

    # Graph weights
    weights = {
        'hidden': tf.Variable(
            tf.random_normal([step.hyperparams['n_inputs'], step.hyperparams['n_hidden']])
        ),  # Hidden layer weights
        'out': tf.Variable(
            tf.random_normal([step.hyperparams['n_hidden'], step.hyperparams['n_classes']], mean=1.0)
        )
    }

    biases = {
        'hidden': tf.Variable(
            tf.random_normal([step.hyperparams['n_hidden']])
        ),
        'out': tf.Variable(
            tf.random_normal([step.hyperparams['n_classes']])
        )
    }

    data_inputs = tf.transpose(
        data_inputs,
        [1, 0, 2])  # permute n_steps and batch_size

    # Reshape to prepare input to hidden activation
    data_inputs = tf.reshape(data_inputs, [-1, step.hyperparams['n_inputs']])
    # new shape: (n_steps*batch_size, n_input)

    # ReLU activation, thanks to Yu Zhao for adding this improvement here:
    _X = tf.nn.relu(
        tf.matmul(data_inputs, weights['hidden']) + biases['hidden']
    )

    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, step.hyperparams['n_steps'], 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(step.hyperparams['n_hidden'], forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(step.hyperparams['n_hidden'], forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)

    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many-to-one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, weights['out']) + biases['out']


def create_optimizer(step: TensorflowV1ModelStep):
    return tf.train.AdamOptimizer(learning_rate=step.hyperparams['learning_rate'])


def create_loss(step: TensorflowV1ModelStep):
    # Loss, optimizer and evaluation
    # L2 loss prevents this overkill neural network to overfit the data
    l2 = step.hyperparams['lambda_loss_amount'] * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

    # Softmax loss
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=step['expected_outputs'],
            logits=step['output']
        )
    ) + l2


class ClassificationRNNTensorFlowModel(TensorflowV1ModelStep):
    def setup(self) -> BaseStep:
        TensorflowV1ModelStep.setup(self)

        self.losses = []
        self.accuracies = []

        return self

    def _will_process(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        if not isinstance(data_container.data_inputs, np.ndarray):
            data_container.data_inputs = np.array(data_container.data_inputs)

        if data_container.expected_outputs is not None:
            if not isinstance(data_container.expected_outputs, np.ndarray):
                data_container.expected_outputs = np.array(data_container.expected_outputs)

            if data_container.expected_outputs.shape != (len(data_container.data_inputs), self.hyperparams['n_classes']):
                data_container.expected_outputs = np.reshape(data_container.expected_outputs, (len(data_container.data_inputs), self.hyperparams['n_classes']))

        return data_container, context

    def _did_fit_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        accuracy = np.mean(np.argmax(data_container.data_inputs, axis=1) == np.argmax(data_container.expected_outputs, axis=1))

        self.accuracies.append(accuracy)
        self.losses.append(self.loss)

        print("Batch Loss = " + "{:.6f}".format(self.losses[-1]) + ", Accuracy = {}".format(self.accuracies[-1]))

        return data_container


model_step = ClassificationRNNTensorFlowModel(
    create_graph=create_graph,
    create_loss=create_loss,
    create_optimizer=create_optimizer
).set_hyperparams(
    HyperparameterSamples({
        'n_steps': N_STEPS,  # 128 timesteps per series
        'n_inputs': N_INPUTS,  # 9 input parameters per timestep
        'n_hidden': N_HIDDEN,  # Hidden layer num of features
        'n_classes': N_CLASSES,  # Total classes (should go up, or should go down)
        'learning_rate': LEARNING_RATE,
        'lambda_loss_amount': LAMBDA_LOSS_AMOUNT,
        'batch_size': BATCH_SIZE
    })
)
