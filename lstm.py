
# Thanks to Zhao Yu for converting the .ipynb notebook to
# this simplified Python script that I edited a little.

# Note that the dataset must be already downloaded for this script to work, do:
#     $ cd data/
#     $ python download_dataset.py

import tensorflow as tf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics

import os


# Load "X" (the neural network's training and testing inputs)

def load_X(X_attribute):
    """Given attribute(train or test) of feature, and read all 9 features into a ndarray,
    shape is [sample_num,time_steps,feature_num]
        argument: X_path str attribute of feature: train or test
        return:  ndarray tensor of features
    """
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
    X_path = './data/UCI HAR Dataset/' + X_attribute + '/Inertial Signals/'
    X = []  # define a list to store the final features tensor
    for name in INPUT_SIGNAL_TYPES:
        absolute_name = X_path + name + X_attribute + '.txt'
        f = open(absolute_name, 'rb')
        # each_x shape is [sample_num,each_steps]
        each_X = [np.array(serie, dtype=np.float32) for serie in [
            row.replace("  ", " ").strip().split(" ") for row in f]]
        # add all feature into X, X shape [feature_num, sample_num, time_steps]
        X.append(each_X)
        f.close()
    # trans X from [feature_num, sample_num, time_steps] to [sample_num,
    # time_steps,feature_num]
    X = np.transpose(np.array(X), (1, 2, 0))
    # print X.shape
    return X


def load_Y(Y_attribute):
    """ read Y file and return Y 
        argument: Y_attribute str attibute of Y('train' or 'test')
        return: Y ndarray the labels of each sample,range [0,5]
    """
    LABELS = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]
    Y_path = './data/UCI HAR Dataset/' + Y_attribute + '/y_' + Y_attribute + '.txt'
    f = open(Y_path)
    # create Y, type is ndarray, range [0,5]
    Y = np.array([int(row) for row in f], dtype=np.int32) - 1
    f.close()
    return Y

class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.time_steps = len(X_train[0])  # 128 time_steps per series

        # Trainging
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 300
        self.batch_size = 300


        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count is of 9: three 3D sensors features over time
        self.n_hidden = 32  # nb of neurons inside the neural network
        self.n_classes = 6  # Final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }


def LSTM_Network(X, config):
    """model a LSTM Network,
      it stacks 2 LSTM layers, each layer has n_hidden=32 cells
       and 1 output layer, it is a full connet layer
      argument:
        X: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
    # Exchange dim 1 and dim 0
    X = tf.transpose(X, [1, 0, 2])
    # New X's shape: [time_steps, batch_size, n_inputs]

    # Temporarily crush the X's dimensions
    X = tf.reshape(X, [-1, config.n_inputs])
    # New X's shape: [time_steps*batch_size, n_inputs]

    # Linear activation, reshaping inputs to the LSTM's number of hidden:
    X = tf.matmul(
        X, config.W['hidden']
    ) + config.biases['hidden']
    # New X's shape: [time_steps*batch_size, n_hidden]

    # Split the series because the rnn cell needs time_steps features, each of shape:
    X = tf.split(0, config.time_steps, X)
    # New X's shape: a list of lenght "time_step" containing tensors of shape [batch_size, n_hidden]

    # Define LSTM cell of first hidden layer:
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0)

    # Stack two LSTM layers, both layers has the same shape
    lsmt_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)

    # Get LSTM outputs, the states are internal to the LSTM cells,they are not our attention here
    outputs, _ = tf.nn.rnn(lsmt_layers, X, dtype=tf.float32)
    # outputs' shape: a list of lenght "time_step" containing tensors of shape [batch_size, n_classes]

    # Linear activation
    # Get the last output tensor of the inner loop output series, of shape [batch_size, n_classes]
    return tf.matmul(outputs[-1], config.W['output']) + config.biases['output']


def one_hot(Y):
    """convert label from dense to one hot
      argument:
        Y: ndarray dense Y ,shape: [sample_num,1]
      return:
        _: ndarray  one hot, shape: [sample_num,n_class]
    """
    return np.eye(6)[np.array(Y)]

if __name__ == "__main__":

    #-----------------------------
    # step1: load and prepare data
    #-----------------------------
    # Those are separate normalised input features for the neural network
    # shape [sample_num,time_steps,feature_num]=[7352,128,9]
    X_train = load_X('train')
    # shape [sample_num,time_steps,feature_num]=[1947,128,9]
    X_test = load_X('test')
    Y_train = load_Y('train')  # shape [sample_num,]=[7352,]
    Y_test = load_Y('test')  # shape [sample_num,]=[2947]
    Y_train = one_hot(Y_train)
    Y_test = one_hot(Y_test)
    # print X_train.shape,X_test.shape,Y_train.shape,Y_test.shape
    # Output classes to learn how to classify

    #-----------------------------------
    # step2: define parameters for model
    #-----------------------------------
    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation:")
    print("features shape, labels shape, each features mean, each features standard deviation")
    print(X_test.shape, Y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected.")

    #------------------------------------------------------
    # step3: Let's get serious and build the neural network
    #------------------------------------------------------
    X = tf.placeholder(tf.float32, [None, config.time_steps, config.n_inputs])
    Y = tf.placeholder(tf.float32, [None, config.n_classes])

    pred_Y = LSTM_Network(X, config)

    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y)) + l2
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    #--------------------------------------------
    # step4: Hooray, now train the neural network
    #--------------------------------------------
    # Note that log_device_placement can be turned of for less console spam.
    sess=tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    tf.initialize_all_variables().run()

    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    for i in range(config.training_epochs):
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            _,acc_train,loss_train=sess.run([optimizer,accuracy,cost], feed_dict={X: X_train[start:end],
                                           Y: Y_train[start:end]})

        # Test completely at every epoch: calculate accuracy
        pred_out, acc_test, loss_test = sess.run([pred_Y, accuracy, cost], feed_dict={
                                                X: X_test, Y: Y_test})
        print("traing iter: {},".format(i)+\
              " train accuracy: {},".format(acc_train)+\
              " train_loss: {},".format(loss_train)+\
              " test accuracy : {},".format(acc_test)+\
              " test loss : {}".format(loss_test))
        best_accuracy = max(best_accuracy, acc_test)

    print("")
    print("final test accuracy: {}".format(acc_test))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")

    #------------------------------------------------------------------
    # step5: Training is good, but having visual insight is even better
    #------------------------------------------------------------------
    # The code is in the .ipynb

    #------------------------------------------------------------------
    # step6: And finally, the multi-class confusion matrix and metrics!
    #------------------------------------------------------------------
    # The code is in the .ipynb
