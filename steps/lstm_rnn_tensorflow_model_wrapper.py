import tensorflow as tf

from neuraxle.base import MetaStepMixin, BaseStep
from neuraxle.hyperparams.space import HyperparameterSamples
from steps.lstm_rnn_tensorflow_model import LSTMRNNTensorflowModel
from steps.one_hot_encoder import OneHotEncoder


class LSTMRNNTensorflowModelTrainingWrapper(MetaStepMixin, BaseStep):
    HYPERPARAMS = HyperparameterSamples({
        'n_classes': 6,
        'learning_rate': 0.0025,
        'lambda_loss_amount': 0.0015,
    })

    def __init__(
            self,
            tensorflow_model: LSTMRNNTensorflowModel,
            hyperparams=None,
            X_test=None,
            y_test=None
    ):
        if hyperparams is None:
            BaseStep.__init__(self, hyperparams=self.HYPERPARAMS)
        else:
            BaseStep.__init__(self, hyperparams=hyperparams)

        MetaStepMixin.__init__(self, wrapped=tensorflow_model)

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

    def setup(self):
        # Loss, optimizer and evaluation
        # L2 loss prevents this overkill neural network to overfit the data
        model: LSTMRNNTensorflowModel = self.wrapped

        self.l2 = self.hyperparams['lambda_loss_amount'] * sum(
            tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
        )

        # Softmax loss
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=model.get_y_placeholder(),
                logits=model
            )
        ) + self.l2

        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.hyperparams['learning_rate']
        ).minimize(self.cost)

        self.correct_pred = tf.equal(tf.argmax(self.wrapped, 1), tf.argmax(model.get_y_placeholder(), 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        # To keep track of training's performance
        self.test_losses = []
        self.test_accuracies = []
        self.train_losses = []
        self.train_accuracies = []

        # Launch the graph
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def teardown(self):
        self.sess.close()

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        model: LSTMRNNTensorflowModel = self.wrapped

        _, loss, acc = self.sess.run(
            [self.optimizer, self.cost, self.accuracy],
            feed_dict={
                model.get_x_placeholder(): data_inputs,
                model.get_y_placeholder(): expected_outputs
            }
        )
        self.train_losses.append(loss)
        self.train_accuracies.append(acc)

        return self

    def transform(self, data_inputs):
        pass

    def _evaluate_on_test_set(self):
        model: LSTMRNNTensorflowModel = self.wrapped

        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        one_hot_encoded_y_test = OneHotEncoder(
            no_columns=self.hyperparams['n_classes'],
            name='one_hot_encoded_y_test'
        ).transform(self.y_test)

        loss, acc = self.sess.run(
            [self.cost, self.accuracy],
            feed_dict={
                model.get_x_placeholder(): self.X_test,
                model.get_y_placeholder(): one_hot_encoded_y_test
            }
        )

        self.test_losses.append(loss)
        self.test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))
