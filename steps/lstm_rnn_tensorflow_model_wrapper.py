import tensorflow as tf

from neuraxle.base import BaseStep
from neuraxle.hyperparams.space import HyperparameterSamples
from savers.tensorflow1_step_saver import TensorflowV1StepSaver
from steps.lstm_rnn_tensorflow_model import tf_model_forward
from steps.one_hot_encoder import OneHotEncoder

N_HIDDEN = 32
N_STEPS = 128
N_INPUTS = 9
LAMBDA_LOSS_AMOUNT = 0.0015
LEARNING_RATE = 0.0025
N_CLASSES = 6
BATCH_SIZE = 1500


class ClassificationRNNTensorFlowModel(BaseStep):
    HYPERPARAMS = HyperparameterSamples({
        'n_steps': N_STEPS,  # 128 timesteps per series
        'n_inputs': N_INPUTS,  # 9 input parameters per timestep
        'n_hidden': N_HIDDEN,  # Hidden layer num of features
        'n_classes': N_CLASSES,  # Total classes (should go up, or should go down)
        'learning_rate': LEARNING_RATE,
        'lambda_loss_amount': LAMBDA_LOSS_AMOUNT
    })

    def __init__(
            self,
            # TODO: replace with issue 174
            #
            X_test=None,
            y_test=None
    ):
        BaseStep.__init__(
            self,
            hyperparams=ClassificationRNNTensorFlowModel.HYPERPARAMS,
            savers=[TensorflowV1StepSaver()]
        )

        # TODO: replace with issue 174
        #
        self.y_test = y_test
        self.X_test = X_test

        self.l2 = None
        self.cost = None
        self.optimizer = None
        self.correct_pred = None
        self.accuracy = None
        self.test_losses = None
        self.test_accuracies = None
        self.train_losses = None
        self.train_accuracies = None

    def setup(self) -> BaseStep:
        # TODO: call with tf variable scope to put in a sub graph ????
        # with a tensorflow graph or variable scope

        # Launch the graph
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Loss, optimizer and evaluation
        # L2 loss prevents this overkill neural network to overfit the data
        self.x_name = ''
        self.y_name = ''

        # TODO: name pred (wrap identity ?)
        self.pred_name = ''
        pred = tf_model_forward(self.pred_name, self.x_name, self.y_name, self.hyperparams)

        l2 = self.hyperparams['lambda_loss_amount'] * sum(
            tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
        )

        # Softmax loss
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.get_variable(self.y_name),
                logits=pred
            )
        ) + self.l2

        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.hyperparams['learning_rate']
        ).minimize(self.cost)

        self.correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(tf.get_variable(self.y_name), 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # To keep track of training's performance
        self.test_losses = []
        self.test_accuracies = []
        self.train_losses = []
        self.train_accuracies = []

        self.is_initialized = True

        return self

    def teardown(self):
        self.sess.close()

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        _, loss, acc = self.sess.run(
            [self.optimizer, self.cost, self.accuracy],
            feed_dict={
                tf.get_variable(self.x_name): data_inputs,
                tf.get_variable(self.y_name): expected_outputs
            }
        )
        self.train_losses.append(loss)
        self.train_accuracies.append(acc)

        return self

    def transform(self, data_inputs):
        # TODO: pass individual element instead of list to sess run ??
        return self.sess.run(
            [tf.get_variable(self.pred_name)],
            feed_dict={
                self.x_name: data_inputs
            }
        )[0]

    def _evaluate_on_test_set(self):
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        one_hot_encoded_y_test = OneHotEncoder(
            nb_columns=self.hyperparams['n_classes'],
            name='one_hot_encoded_y_test'
        ).transform(self.y_test)

        loss, acc = self.sess.run(
            [self.cost, self.accuracy],
            feed_dict={
                tf.get_variable(self.x_name): self.X_test,
                tf.get_variable(self.y_name): one_hot_encoded_y_test
            }
        )

        self.test_losses.append(loss)
        self.test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))
