import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics
import os


# def load_X(X_signal_path):
#     """load feature data and return a matrix 
#     argument:
#             X_signal_path: the path where store feature data
#     return:
#             : ndarray  feature data , shape: [sample_num,time_steps,n_inputs]
#     """

#     file_list = os.listdir(X_signal_path)
#     print file_list
#     # read data from each file
#     X_signals = []  # create a list to store data
#     for file_path in file_list:
#         # add absolute path to file
#         file_path = os.path.join(X_signal_path, file_path)
#         file = open(file_path, 'rb')
#         clear_data_list = [row.replace(
#             "  ", " ").strip().split(' ') for row in file]

#         # add each feature into X_singals, shape:
#         # [n_inputs,sample_num,time_steps]
#         X_signals.append([np.array(series, dtype=np.float32)
#                           for series in clear_data_list])
#         file.close()
#     # get correct shape of X_signals: [sample_num,time_steps,n_inputs]
#     return np.transpose(np.array(X_signals), (1, 2, 0))


# def load_Y(Y_signal_path):
#     """load label data and return a matrix
#       argument:
#         Y_signal_path: str the path of label file
#       return: 
#                       : ndarray a matrix of label ,shape: [sample_num,1]
#     """
#     file = open(Y_signal_path)
#     clear_data_list = [raw.replace(
#         '  ', ' ').strip().split(' ') for raw in file]
#     file.close()
#     # cast variable type and make it index from 0
#     Y_mat = np.array(clear_data_list, dtype=np.int32) - 1
#     return Y_mat

# Load "X" (the neural network's training and testing inputs)

def load_X(X_signals_paths):
    X_signals = []

    for signal_type_path in X_signals_paths:
        file = open(signal_type_path, 'rb')
        # Read dataset from disk, dealing with text files' syntax
        X_signals.append(
            [np.array(serie, dtype=np.float32) for serie in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]]
        )
        file.close()

    return np.transpose(np.array(X_signals), (1, 2, 0))


# Load "y" (the neural network's training and testing outputs)

def load_y(y_path):
    file = open(y_path, 'rb')
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


class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing
    """

    def __init__(self, X_train, X_test):
        # input data
        self.train_count = len(
            X_train)  # 7352 training series
        self.test_data_count = len(X_test)  # 2947 testing series
        self.n_steps = len(X_train[0])  # 128 time_steps per series

        # trainging
        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epoch = 300
        self.batch_size = 1500
        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # feature num is 9
        self.n_hidden = 32
        self.n_classes = 6  # the final output classes
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }


def LSTM_Network(feature_mat, config):
    """model a LSTM Network, 
      it stacks 2 LSTM layers, each layer has n_hidden=32 cells
       and 1 output layer, it is a full connet layer
      argument:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
    # exchange dim 1 and dim 0,result:feature_mat
    # shape=[time_steps,batch_size,n_inputs]
    feature_mat = tf.transpose(feature_mat, [1, 0, 2])
    # pad feature_mat, result: feature_mat
    # shape=[tiem_steps*batch_size,n_inputs]
    feature_mat = tf.reshape(feature_mat, [-1, config.n_inputs])
    # linear activation,result:feature_mat
    # shape=[time_steps*batch_size,n_hidden]
    feature_mat = tf.matmul(feature_mat, config.W[
                            'hidden']) + config.biases['hidden']
    # split matrix because rnn cell need time_steps series, each series shape
    # [batch_size,n_hidden]
    feature_mat = tf.split(0, config.n_steps, feature_mat)

    # define LSTM cell of first hidden layer
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(config.n_hidden, forget_bias=1.0)
    # stack two LSTM layers, and both layers are the same
    lsmt_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)
    # get LSTM outputs, the states are mid sates of networks,they are not our attention here
    # the outputs shape:  time_steps series, each series shape
    # [batch_size,n_classes]
    outputs, _ = tf.nn.rnn(lsmt_layers, feature_mat, dtype=tf.float32)

    # linear activation
    # get inner loop last output, shape [batch_size,n_classes]
    return tf.matmul(outputs[-1], config.W['output']) + config.biases['output']


def one_hot(label):
    """convert label from dense to one hot
      argument:
        label: ndarray dense label ,shape: [sample_num,1]
      return:
        one_hot_label: ndarray  one hot, shape: [sample_num,n_class]
    """
    label_num = len(label)
    new_label = label.reshape(label_num)  # shape : [sample_num]
    # because max is 5, and we will create 6 columns
    n_values = np.max(new_label) + 1
    return np.eye(n_values)[np.array(new_label, dtype=np.int32)]


if __name__ == "__main__":

    #-----------------------------
    # step1: load and prepare data
    #-----------------------------
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
    print("\n" + "Dataset is now located at: " + DATASET_PATH)

    TRAIN = "train/"
    TEST = "test/"


    X_train_signals_paths = [
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
    ]

    X_train = load_X(X_train_signals_paths)
    X_test = load_X(X_test_signals_paths)

    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"
    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)
    
    y_test=one_hot(y_test)
    y_train=one_hot(y_train)


    #-----------------------------------
    #step2: define parameters for model
    #-----------------------------------
    config = Config(X_train, X_test)
    print("Some useful info to get an insight on dataset's shape and normalisation")
    print("feature shape, label shape, each feature mean, each feature standard deviation")
    print(X_test.shape, y_test.shape,
          np.mean(X_test), np.std(X_test))
    print("the dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")
    #------------------------------------------------------
    # step3: Let's get serious and build the neural network
    #------------------------------------------------------
    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    Y = tf.placeholder(tf.float32, [None, config.n_classes])
    pred_Y = LSTM_Network(X, config)
    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # softmax loss and l2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(pred_Y, Y)) + l2
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))
    #--------------------------------------------
    # step4: Hooray, now train the neural network
    #--------------------------------------------
    sess=tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
    tf.initialize_all_variables().run()
    # start training for each batch and loop epochs
    for i in range(config.training_epoch):
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            sess.run(optimizer, feed_dict={X: X_train[start:end],
                                           Y: y_train[start:end]})
        # start testing,calculate accuracy
        pred_out, accuracy_out, loss_out = sess.run([pred_Y, accuracy, cost], feed_dict={
                                                X: X_test, Y: y_test})
        print("traing iter: {}".format(i)+\
              "  accuracy : {}".format(accuracy_out)+\
              "  loss : {}".format(loss_out))
    #------------------------------------------------------------------
    # step5: Training is good, but having visual insight is even better
    #------------------------------------------------------------------
    #------------------------------------------------------------------
    # step6: And finally, the multi-class confusion matrix and metrics!
    #------------------------------------------------------------------


