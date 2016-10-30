
# LSTM for Human Activity Recognition

Human activity recognition using smartphones dataset and an LSTM RNN. Classifying the type of movement amongst six categories:
- WALKING,
- WALKING_UPSTAIRS,
- WALKING_DOWNSTAIRS,
- SITTING,
- STANDING,
- LAYING.


## Video dataset overview

Follow this link to see a video of the 6 activities recorded in the experiment with one of the participants:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=XOEN9W05_4A
" target="_blank"><img src="http://img.youtube.com/vi/XOEN9W05_4A/0.jpg" 
alt="Video of the experiment" width="400" height="300" border="10" /></a>
<a href="https://youtu.be/XOEN9W05_4A"><center>[Watch video]</center></a>

## Details

I will be using an LSTM on the data to learn (as a cellphone attached on the waist) to recognise the type of activity that the user is doing. The dataset's description goes like this:

> The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. 

That said, I will use the almost raw data: only the gravity effect has been filtered out of the accelerometer  as a preprocessing step for another 3D feature as an input to help learning. 


## Results 

Scroll on! Nice visuals awaits. 


```python
# All Includes

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version r0.10
from sklearn import metrics

import os
```


```python
# Useful Constants

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

```

## Let's start by downloading the data: 


```python
# Note: Linux bash commands start with a "!" inside those "ipython notebook" cells

DATA_PATH = "data/"

!pwd && ls
os.chdir(DATA_PATH)
!pwd && ls

!python download_dataset.py

!pwd && ls
os.chdir("..")
!pwd && ls

DATASET_PATH = DATA_PATH + "UCI HAR Dataset/"
print("\n" + "Dataset is now located at: " + DATASET_PATH)

```

    /home/gui/Documents/GIT/LSTM-Human-Activity-Recognition
    data  LSTM.ipynb
    /home/gui/Documents/GIT/LSTM-Human-Activity-Recognition/data
    download_dataset.py  __MACOSX  source.txt  UCI HAR Dataset  UCI HAR Dataset.zip
    
    Downloading...
    Dataset already downloaded. Did not download twice.
    
    Extracting...
    Dataset already extracted. Did not extract twice.
    
    /home/gui/Documents/GIT/LSTM-Human-Activity-Recognition/data
    download_dataset.py  __MACOSX  source.txt  UCI HAR Dataset  UCI HAR Dataset.zip
    /home/gui/Documents/GIT/LSTM-Human-Activity-Recognition
    data  LSTM.ipynb
    
    Dataset is now located at: data/UCI HAR Dataset/


## Preparing dataset:


```python
TRAIN = "train/"
TEST = "test/"


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

X_train_signals_paths = [
    DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
X_test_signals_paths = [
    DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

X_train = load_X(X_train_signals_paths)
X_test = load_X(X_test_signals_paths)


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

y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
y_test_path = DATASET_PATH + TEST + "y_test.txt"

y_train = load_y(y_train_path)
y_test = load_y(y_test_path)

```

## Additionnal Parameters:

Here are some core parameter definitions for the training. 

The whole neural network's structure could be summarised by enumerating those parameters and the fact an LSTM is used. 


```python

# Input Data 

training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep


# LSTM Neural Network's internal structure

n_hidden = 32 # Hidden layer num of features
n_classes = 6 # Total classes (should go up, or should go down)


# Training 

learning_rate = 0.0025
lambda_loss_amount = 0.0015
training_iters = training_data_count * 300  # Loop 300 times on the dataset
batch_size = 1500
display_iter = 15000  # To show test set accuracy during training


# Some debugging info

print "Some useful info to get an insight on dataset's shape and normalisation:"
print "(X shape, y shape, every X's mean, every X's standard deviation)"
print (X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print "The dataset is therefore properly normalised, as expected, but not yet one-hot encoded."

```

    Some useful info to get an insight on dataset's shape and normalisation:
    (X shape, y shape, every X's mean, every X's standard deviation)
    ((2947, 128, 9), (2947, 1), 0.099147044, 0.39534995)
    The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.


## Utility functions for training:


```python
def LSTM_RNN(_X, _weights, _biases):
    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters. 
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network. 
    # Note, some code of this notebook is inspired from an slightly different 
    # RNN architecture used on another dataset: 
    # https://tensorhub.com/aymericdamien/tensorflow-rnn

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) 
    # new shape: (n_steps*batch_size, n_input)
    
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) 
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.nn.rnn(lstm_cells, _X, dtype=tf.float32)

    # Linear activation
    # Get inner loop last output
    return tf.matmul(outputs[-1], _weights['out']) + _biases['out']


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data. 
    
    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step-1)*batch_size + i) % len(_train)
        batch_s[i] = _train[index] 

    return batch_s


def one_hot(y_):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

```

## Let's get serious and build the neural network:


```python

# Graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = LSTM_RNN(x, weights, biases)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
) # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) + l2 # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

```

## Hooray, now train the neural network:


```python
# To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.initialize_all_variables()
sess.run(init)

# Perform Training steps with "batch_size" iterations at each loop
step = 1
while step * batch_size <= training_iters:
    batch_xs =         extract_batch_size(X_train, step, batch_size)
    batch_ys = one_hot(extract_batch_size(y_train, step, batch_size))

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
    if (step*batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        
        # To not spam console, show training accuracy/loss in this "if"
        print "Training iter #" + str(step*batch_size) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc)
        
        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy], 
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print "PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc)

    step += 1

print "Optimization Finished!"

# Accuracy for test data

one_hot_predictions, accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: one_hot(y_test)
    }
)

test_losses.append(final_loss)
test_accuracies.append(accuracy)

print "FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy)

```

    Training iter #1500:   Batch Loss = 4.575454, Accuracy = 0.1426666677
    PERFORMANCE ON TEST SET: Batch Loss = 3.11178827286, Accuracy = 0.083814047277
    Training iter #15000:   Batch Loss = 1.805075, Accuracy = 0.512000024319
    PERFORMANCE ON TEST SET: Batch Loss = 1.83639490604, Accuracy = 0.459450274706
    Training iter #30000:   Batch Loss = 1.524733, Accuracy = 0.592666685581
    PERFORMANCE ON TEST SET: Batch Loss = 1.59639000893, Accuracy = 0.540889024734
    Training iter #45000:   Batch Loss = 1.266281, Accuracy = 0.660000026226
    PERFORMANCE ON TEST SET: Batch Loss = 1.42640054226, Accuracy = 0.594502866268
    Training iter #60000:   Batch Loss = 1.165662, Accuracy = 0.719333350658
    PERFORMANCE ON TEST SET: Batch Loss = 1.26158189774, Accuracy = 0.68917542696
    Training iter #75000:   Batch Loss = 1.090536, Accuracy = 0.754666686058
    PERFORMANCE ON TEST SET: Batch Loss = 1.22805702686, Accuracy = 0.715982377529
    Training iter #90000:   Batch Loss = 0.962699, Accuracy = 0.842666685581
    PERFORMANCE ON TEST SET: Batch Loss = 1.19088602066, Accuracy = 0.752629816532
    Training iter #105000:   Batch Loss = 0.845098, Accuracy = 0.88466668129
    PERFORMANCE ON TEST SET: Batch Loss = 1.17878103256, Accuracy = 0.768238902092
    Training iter #120000:   Batch Loss = 0.752325, Accuracy = 0.922666668892
    PERFORMANCE ON TEST SET: Batch Loss = 1.10739731789, Accuracy = 0.814726829529
    Training iter #135000:   Batch Loss = 0.719420, Accuracy = 0.930666685104
    PERFORMANCE ON TEST SET: Batch Loss = 0.983196675777, Accuracy = 0.846284329891
    Training iter #150000:   Batch Loss = 0.648962, Accuracy = 0.963333308697
    PERFORMANCE ON TEST SET: Batch Loss = 0.952148973942, Accuracy = 0.851374268532
    Training iter #165000:   Batch Loss = 0.746680, Accuracy = 0.899999976158
    PERFORMANCE ON TEST SET: Batch Loss = 0.903841257095, Accuracy = 0.863250792027
    Training iter #180000:   Batch Loss = 0.742886, Accuracy = 0.906666696072
    PERFORMANCE ON TEST SET: Batch Loss = 0.840512394905, Accuracy = 0.875127255917
    Training iter #195000:   Batch Loss = 0.720926, Accuracy = 0.911333322525
    PERFORMANCE ON TEST SET: Batch Loss = 0.872816443443, Accuracy = 0.871394634247
    Training iter #210000:   Batch Loss = 0.700941, Accuracy = 0.917999982834
    PERFORMANCE ON TEST SET: Batch Loss = 0.854153513908, Accuracy = 0.882253110409
    Training iter #225000:   Batch Loss = 0.669675, Accuracy = 0.924000024796
    PERFORMANCE ON TEST SET: Batch Loss = 0.802200436592, Accuracy = 0.887003719807
    Training iter #240000:   Batch Loss = 0.568134, Accuracy = 0.973999977112
    PERFORMANCE ON TEST SET: Batch Loss = 0.82608127594, Accuracy = 0.879877865314
    Training iter #255000:   Batch Loss = 0.549412, Accuracy = 0.980666637421
    PERFORMANCE ON TEST SET: Batch Loss = 0.811351120472, Accuracy = 0.880217194557
    Training iter #270000:   Batch Loss = 0.542750, Accuracy = 0.981999993324
    PERFORMANCE ON TEST SET: Batch Loss = 0.792157769203, Accuracy = 0.883949756622
    Training iter #285000:   Batch Loss = 0.530857, Accuracy = 0.984666645527
    PERFORMANCE ON TEST SET: Batch Loss = 0.796725153923, Accuracy = 0.885307073593
    Training iter #300000:   Batch Loss = 0.519957, Accuracy = 0.98400002718
    PERFORMANCE ON TEST SET: Batch Loss = 0.784544229507, Accuracy = 0.887343049049
    Training iter #315000:   Batch Loss = 0.541124, Accuracy = 0.972666680813
    PERFORMANCE ON TEST SET: Batch Loss = 0.780015945435, Accuracy = 0.879538536072
    Training iter #330000:   Batch Loss = 0.553707, Accuracy = 0.9646666646
    PERFORMANCE ON TEST SET: Batch Loss = 0.835344195366, Accuracy = 0.872073292732
    Training iter #345000:   Batch Loss = 0.589612, Accuracy = 0.949333310127
    PERFORMANCE ON TEST SET: Batch Loss = 0.802719056606, Accuracy = 0.881235182285
    Training iter #360000:   Batch Loss = 0.573641, Accuracy = 0.94866669178
    PERFORMANCE ON TEST SET: Batch Loss = 0.795131444931, Accuracy = 0.874448597431
    Training iter #375000:   Batch Loss = 0.598676, Accuracy = 0.938666641712
    PERFORMANCE ON TEST SET: Batch Loss = 0.830257892609, Accuracy = 0.864608049393
    Training iter #390000:   Batch Loss = 0.576627, Accuracy = 0.938000023365
    PERFORMANCE ON TEST SET: Batch Loss = 0.799493312836, Accuracy = 0.879199206829
    Training iter #405000:   Batch Loss = 0.526197, Accuracy = 0.953333318233
    PERFORMANCE ON TEST SET: Batch Loss = 0.729372382164, Accuracy = 0.891414999962
    Training iter #420000:   Batch Loss = 0.536317, Accuracy = 0.955333352089
    PERFORMANCE ON TEST SET: Batch Loss = 0.717933833599, Accuracy = 0.89616560936
    Training iter #435000:   Batch Loss = 0.514075, Accuracy = 0.959999978542
    PERFORMANCE ON TEST SET: Batch Loss = 0.74327981472, Accuracy = 0.882253110409
    Training iter #450000:   Batch Loss = 0.511001, Accuracy = 0.949333310127
    PERFORMANCE ON TEST SET: Batch Loss = 0.725793123245, Accuracy = 0.893790304661
    Training iter #465000:   Batch Loss = 0.548203, Accuracy = 0.939999997616
    PERFORMANCE ON TEST SET: Batch Loss = 0.829982995987, Accuracy = 0.866644024849
    Training iter #480000:   Batch Loss = 0.531429, Accuracy = 0.927999973297
    PERFORMANCE ON TEST SET: Batch Loss = 0.739032506943, Accuracy = 0.881574511528
    Training iter #495000:   Batch Loss = 0.489275, Accuracy = 0.955333352089
    PERFORMANCE ON TEST SET: Batch Loss = 0.84563010931, Accuracy = 0.865286707878
    Training iter #510000:   Batch Loss = 0.490075, Accuracy = 0.954666674137
    PERFORMANCE ON TEST SET: Batch Loss = 0.728687524796, Accuracy = 0.895486950874
    Training iter #525000:   Batch Loss = 0.451927, Accuracy = 0.974666655064
    PERFORMANCE ON TEST SET: Batch Loss = 0.717314362526, Accuracy = 0.890736341476
    Training iter #540000:   Batch Loss = 0.522697, Accuracy = 0.931999981403
    PERFORMANCE ON TEST SET: Batch Loss = 0.677192807198, Accuracy = 0.895826280117
    Training iter #555000:   Batch Loss = 0.521561, Accuracy = 0.931999981403
    PERFORMANCE ON TEST SET: Batch Loss = 0.656917333603, Accuracy = 0.894808292389
    Training iter #570000:   Batch Loss = 0.505313, Accuracy = 0.934000015259
    PERFORMANCE ON TEST SET: Batch Loss = 0.664022982121, Accuracy = 0.899219572544
    Training iter #585000:   Batch Loss = 0.509840, Accuracy = 0.934666693211
    PERFORMANCE ON TEST SET: Batch Loss = 0.68600410223, Accuracy = 0.887343049049
    Training iter #600000:   Batch Loss = 0.495022, Accuracy = 0.938000023365
    PERFORMANCE ON TEST SET: Batch Loss = 0.718553900719, Accuracy = 0.882253110409
    Training iter #615000:   Batch Loss = 0.436445, Accuracy = 0.977333307266
    PERFORMANCE ON TEST SET: Batch Loss = 0.717109918594, Accuracy = 0.885307073593
    Training iter #630000:   Batch Loss = 0.431761, Accuracy = 0.976666688919
    PERFORMANCE ON TEST SET: Batch Loss = 1.10843300819, Accuracy = 0.80556499958
    Training iter #645000:   Batch Loss = 0.605546, Accuracy = 0.906666696072
    PERFORMANCE ON TEST SET: Batch Loss = 0.739134371281, Accuracy = 0.840855121613
    Training iter #660000:   Batch Loss = 0.462404, Accuracy = 0.968666672707
    PERFORMANCE ON TEST SET: Batch Loss = 0.704330086708, Accuracy = 0.862911462784
    Training iter #675000:   Batch Loss = 0.453931, Accuracy = 0.973333358765
    PERFORMANCE ON TEST SET: Batch Loss = 0.690061211586, Accuracy = 0.879199206829
    Training iter #690000:   Batch Loss = 0.392534, Accuracy = 0.996666669846
    PERFORMANCE ON TEST SET: Batch Loss = 0.654640078545, Accuracy = 0.894129633904
    Training iter #705000:   Batch Loss = 0.454180, Accuracy = 0.973999977112
    PERFORMANCE ON TEST SET: Batch Loss = 0.603749155998, Accuracy = 0.90430945158
    Training iter #720000:   Batch Loss = 0.462097, Accuracy = 0.953333318233
    PERFORMANCE ON TEST SET: Batch Loss = 0.611262202263, Accuracy = 0.902612805367
    Training iter #735000:   Batch Loss = 0.457372, Accuracy = 0.949999988079
    PERFORMANCE ON TEST SET: Batch Loss = 0.600763738155, Accuracy = 0.906345427036
    Training iter #750000:   Batch Loss = 0.469722, Accuracy = 0.939999997616
    PERFORMANCE ON TEST SET: Batch Loss = 0.596443295479, Accuracy = 0.907702744007
    Training iter #765000:   Batch Loss = 0.473977, Accuracy = 0.940666675568
    PERFORMANCE ON TEST SET: Batch Loss = 0.609652996063, Accuracy = 0.901934146881
    Training iter #780000:   Batch Loss = 0.395754, Accuracy = 0.965333342552
    PERFORMANCE ON TEST SET: Batch Loss = 0.600235044956, Accuracy = 0.907024085522
    Training iter #795000:   Batch Loss = 0.394890, Accuracy = 0.959999978542
    PERFORMANCE ON TEST SET: Batch Loss = 0.58463537693, Accuracy = 0.915167987347
    Training iter #810000:   Batch Loss = 0.396357, Accuracy = 0.955999970436
    PERFORMANCE ON TEST SET: Batch Loss = 0.587031722069, Accuracy = 0.91313201189
    Training iter #825000:   Batch Loss = 0.402448, Accuracy = 0.953999996185
    PERFORMANCE ON TEST SET: Batch Loss = 0.588794231415, Accuracy = 0.91177469492
    Training iter #840000:   Batch Loss = 0.417953, Accuracy = 0.944000005722
    PERFORMANCE ON TEST SET: Batch Loss = 0.595950126648, Accuracy = 0.908381402493
    Training iter #855000:   Batch Loss = 0.425846, Accuracy = 0.941333353519
    PERFORMANCE ON TEST SET: Batch Loss = 0.604835391045, Accuracy = 0.904988110065
    Training iter #870000:   Batch Loss = 0.395381, Accuracy = 0.966000020504
    PERFORMANCE ON TEST SET: Batch Loss = 0.62311989069, Accuracy = 0.902612805367
    Training iter #885000:   Batch Loss = 0.379031, Accuracy = 0.970666646957
    PERFORMANCE ON TEST SET: Batch Loss = 0.556019067764, Accuracy = 0.908720731735
    Training iter #900000:   Batch Loss = 0.366955, Accuracy = 0.97866666317
    PERFORMANCE ON TEST SET: Batch Loss = 0.560609638691, Accuracy = 0.909399390221
    Training iter #915000:   Batch Loss = 0.422597, Accuracy = 0.939999997616
    PERFORMANCE ON TEST SET: Batch Loss = 0.551591157913, Accuracy = 0.912114024162
    Training iter #930000:   Batch Loss = 0.459800, Accuracy = 0.926666676998
    PERFORMANCE ON TEST SET: Batch Loss = 0.567300021648, Accuracy = 0.908381402493
    Training iter #945000:   Batch Loss = 0.427131, Accuracy = 0.935333311558
    PERFORMANCE ON TEST SET: Batch Loss = 0.570689320564, Accuracy = 0.902952134609
    Training iter #960000:   Batch Loss = 0.444879, Accuracy = 0.934000015259
    PERFORMANCE ON TEST SET: Batch Loss = 0.53837120533, Accuracy = 0.91177469492
    Training iter #975000:   Batch Loss = 0.407420, Accuracy = 0.947333335876
    PERFORMANCE ON TEST SET: Batch Loss = 0.576001763344, Accuracy = 0.897183597088
    Training iter #990000:   Batch Loss = 0.358638, Accuracy = 0.977333307266
    PERFORMANCE ON TEST SET: Batch Loss = 0.566704928875, Accuracy = 0.904988110065
    Training iter #1005000:   Batch Loss = 0.344259, Accuracy = 0.986000001431
    PERFORMANCE ON TEST SET: Batch Loss = 0.706675469875, Accuracy = 0.869019329548
    Training iter #1020000:   Batch Loss = 0.358466, Accuracy = 0.975333333015
    PERFORMANCE ON TEST SET: Batch Loss = 0.61932772398, Accuracy = 0.891754329205
    Training iter #1035000:   Batch Loss = 0.334988, Accuracy = 0.984666645527
    PERFORMANCE ON TEST SET: Batch Loss = 0.632777690887, Accuracy = 0.878520548344
    Training iter #1050000:   Batch Loss = 0.340259, Accuracy = 0.977999985218
    PERFORMANCE ON TEST SET: Batch Loss = 0.614942312241, Accuracy = 0.892772316933
    Training iter #1065000:   Batch Loss = 0.332005, Accuracy = 0.995333313942
    PERFORMANCE ON TEST SET: Batch Loss = 0.572589993477, Accuracy = 0.90430945158
    Training iter #1080000:   Batch Loss = 0.367203, Accuracy = 0.974666655064
    PERFORMANCE ON TEST SET: Batch Loss = 0.553013861179, Accuracy = 0.897183597088
    Training iter #1095000:   Batch Loss = 0.384246, Accuracy = 0.954666674137
    PERFORMANCE ON TEST SET: Batch Loss = 0.616080880165, Accuracy = 0.88361042738
    Training iter #1110000:   Batch Loss = 0.401190, Accuracy = 0.941999971867
    PERFORMANCE ON TEST SET: Batch Loss = 0.583316922188, Accuracy = 0.889379024506
    Training iter #1125000:   Batch Loss = 0.407854, Accuracy = 0.941999971867
    PERFORMANCE ON TEST SET: Batch Loss = 0.54475069046, Accuracy = 0.901594817638
    Training iter #1140000:   Batch Loss = 0.388828, Accuracy = 0.945333361626
    PERFORMANCE ON TEST SET: Batch Loss = 0.562619686127, Accuracy = 0.892772316933
    Training iter #1155000:   Batch Loss = 0.325556, Accuracy = 0.959333360195
    PERFORMANCE ON TEST SET: Batch Loss = 0.624062478542, Accuracy = 0.881235182285
    Training iter #1170000:   Batch Loss = 0.336760, Accuracy = 0.959333360195
    PERFORMANCE ON TEST SET: Batch Loss = 0.567292392254, Accuracy = 0.896844267845
    Training iter #1185000:   Batch Loss = 0.323525, Accuracy = 0.96266669035
    PERFORMANCE ON TEST SET: Batch Loss = 0.554633200169, Accuracy = 0.900916159153
    Training iter #1200000:   Batch Loss = 0.332793, Accuracy = 0.955999970436
    PERFORMANCE ON TEST SET: Batch Loss = 0.572944641113, Accuracy = 0.893111646175
    Training iter #1215000:   Batch Loss = 0.354524, Accuracy = 0.940666675568
    PERFORMANCE ON TEST SET: Batch Loss = 0.583880543709, Accuracy = 0.897183597088
    Training iter #1230000:   Batch Loss = 0.340542, Accuracy = 0.942666649818
    PERFORMANCE ON TEST SET: Batch Loss = 0.58384168148, Accuracy = 0.895826280117
    Training iter #1245000:   Batch Loss = 0.318747, Accuracy = 0.970666646957
    PERFORMANCE ON TEST SET: Batch Loss = 0.591021180153, Accuracy = 0.894808292389
    Training iter #1260000:   Batch Loss = 0.331109, Accuracy = 0.959999978542
    PERFORMANCE ON TEST SET: Batch Loss = 0.610709190369, Accuracy = 0.882931768894
    Training iter #1275000:   Batch Loss = 0.360516, Accuracy = 0.939333319664
    PERFORMANCE ON TEST SET: Batch Loss = 0.721515417099, Accuracy = 0.871055305004
    Training iter #1290000:   Batch Loss = 0.363310, Accuracy = 0.935333311558
    PERFORMANCE ON TEST SET: Batch Loss = 0.583494842052, Accuracy = 0.885646402836
    Training iter #1305000:   Batch Loss = 0.468750, Accuracy = 0.917333304882
    PERFORMANCE ON TEST SET: Batch Loss = 0.616247475147, Accuracy = 0.889379024506
    Training iter #1320000:   Batch Loss = 0.396604, Accuracy = 0.928666651249
    PERFORMANCE ON TEST SET: Batch Loss = 0.559162676334, Accuracy = 0.885646402836
    Training iter #1335000:   Batch Loss = 0.375801, Accuracy = 0.930666685104
    PERFORMANCE ON TEST SET: Batch Loss = 0.526473701, Accuracy = 0.895147621632
    Training iter #1350000:   Batch Loss = 0.330984, Accuracy = 0.966666638851
    PERFORMANCE ON TEST SET: Batch Loss = 0.504767656326, Accuracy = 0.90057682991
    Training iter #1365000:   Batch Loss = 0.303565, Accuracy = 0.973333358765
    PERFORMANCE ON TEST SET: Batch Loss = 0.545929431915, Accuracy = 0.889039695263
    Training iter #1380000:   Batch Loss = 0.307729, Accuracy = 0.972000002861
    PERFORMANCE ON TEST SET: Batch Loss = 0.558818459511, Accuracy = 0.898880243301
    Training iter #1395000:   Batch Loss = 0.293135, Accuracy = 0.981333315372
    PERFORMANCE ON TEST SET: Batch Loss = 0.496408224106, Accuracy = 0.90804207325
    Training iter #1410000:   Batch Loss = 0.272638, Accuracy = 0.987999975681
    PERFORMANCE ON TEST SET: Batch Loss = 0.531621038914, Accuracy = 0.892093658447
    Training iter #1425000:   Batch Loss = 0.279873, Accuracy = 0.987999975681
    PERFORMANCE ON TEST SET: Batch Loss = 0.577055752277, Accuracy = 0.878859877586
    Training iter #1440000:   Batch Loss = 0.322893, Accuracy = 0.9646666646
    PERFORMANCE ON TEST SET: Batch Loss = 0.532765746117, Accuracy = 0.903630793095
    Training iter #1455000:   Batch Loss = 0.339059, Accuracy = 0.953999996185
    PERFORMANCE ON TEST SET: Batch Loss = 0.504770159721, Accuracy = 0.904988110065
    Training iter #1470000:   Batch Loss = 0.334370, Accuracy = 0.954666674137
    PERFORMANCE ON TEST SET: Batch Loss = 0.516438186169, Accuracy = 0.895147621632
    Training iter #1485000:   Batch Loss = 0.337459, Accuracy = 0.942666649818
    PERFORMANCE ON TEST SET: Batch Loss = 0.47318738699, Accuracy = 0.909060060978
    Training iter #1500000:   Batch Loss = 0.334233, Accuracy = 0.940666675568
    PERFORMANCE ON TEST SET: Batch Loss = 0.489954531193, Accuracy = 0.903291463852
    Training iter #1515000:   Batch Loss = 0.294658, Accuracy = 0.953999996185
    PERFORMANCE ON TEST SET: Batch Loss = 0.476269245148, Accuracy = 0.910756707191
    Training iter #1530000:   Batch Loss = 0.284108, Accuracy = 0.956666648388
    PERFORMANCE ON TEST SET: Batch Loss = 0.539360523224, Accuracy = 0.891414999962
    Training iter #1545000:   Batch Loss = 0.350899, Accuracy = 0.946666657925
    PERFORMANCE ON TEST SET: Batch Loss = 0.529010236263, Accuracy = 0.891075670719
    Training iter #1560000:   Batch Loss = 0.299959, Accuracy = 0.950666666031
    PERFORMANCE ON TEST SET: Batch Loss = 0.511167883873, Accuracy = 0.900237500668
    Training iter #1575000:   Batch Loss = 0.295145, Accuracy = 0.953999996185
    PERFORMANCE ON TEST SET: Batch Loss = 0.47209841013, Accuracy = 0.904988110065
    Training iter #1590000:   Batch Loss = 0.363128, Accuracy = 0.925999999046
    PERFORMANCE ON TEST SET: Batch Loss = 0.484584033489, Accuracy = 0.898880243301
    Training iter #1605000:   Batch Loss = 0.361442, Accuracy = 0.941999971867
    PERFORMANCE ON TEST SET: Batch Loss = 0.52424800396, Accuracy = 0.888021707535
    Training iter #1620000:   Batch Loss = 0.293465, Accuracy = 0.965333342552
    PERFORMANCE ON TEST SET: Batch Loss = 0.544751405716, Accuracy = 0.892772316933
    Training iter #1635000:   Batch Loss = 0.270814, Accuracy = 0.97866666317
    PERFORMANCE ON TEST SET: Batch Loss = 0.504271149635, Accuracy = 0.898201584816
    Training iter #1650000:   Batch Loss = 0.320242, Accuracy = 0.931333363056
    PERFORMANCE ON TEST SET: Batch Loss = 0.705718159676, Accuracy = 0.84865963459
    Training iter #1665000:   Batch Loss = 0.793835, Accuracy = 0.796666681767
    PERFORMANCE ON TEST SET: Batch Loss = 0.677424192429, Accuracy = 0.847641646862
    Training iter #1680000:   Batch Loss = 0.473026, Accuracy = 0.917333304882
    PERFORMANCE ON TEST SET: Batch Loss = 0.643717765808, Accuracy = 0.870715975761
    Training iter #1695000:   Batch Loss = 0.499333, Accuracy = 0.887333333492
    PERFORMANCE ON TEST SET: Batch Loss = 0.604045450687, Accuracy = 0.878859877586
    Training iter #1710000:   Batch Loss = 0.464397, Accuracy = 0.880666673183
    PERFORMANCE ON TEST SET: Batch Loss = 0.662315666676, Accuracy = 0.872751951218
    Training iter #1725000:   Batch Loss = 0.434838, Accuracy = 0.94866669178
    PERFORMANCE ON TEST SET: Batch Loss = 0.707927942276, Accuracy = 0.858839511871
    Training iter #1740000:   Batch Loss = 0.338807, Accuracy = 0.936666667461
    PERFORMANCE ON TEST SET: Batch Loss = 0.656027317047, Accuracy = 0.872073292732
    Training iter #1755000:   Batch Loss = 0.291782, Accuracy = 0.984666645527
    PERFORMANCE ON TEST SET: Batch Loss = 0.628328800201, Accuracy = 0.8805565238
    Training iter #1770000:   Batch Loss = 0.287209, Accuracy = 0.982666671276
    PERFORMANCE ON TEST SET: Batch Loss = 0.61780667305, Accuracy = 0.883949756622
    Training iter #1785000:   Batch Loss = 0.277615, Accuracy = 0.986000001431
    PERFORMANCE ON TEST SET: Batch Loss = 0.604023754597, Accuracy = 0.88870036602
    Training iter #1800000:   Batch Loss = 0.271597, Accuracy = 0.987333357334
    PERFORMANCE ON TEST SET: Batch Loss = 0.604183316231, Accuracy = 0.8805565238
    Training iter #1815000:   Batch Loss = 0.293323, Accuracy = 0.98400002718
    PERFORMANCE ON TEST SET: Batch Loss = 0.601334929466, Accuracy = 0.88361042738
    Training iter #1830000:   Batch Loss = 0.333876, Accuracy = 0.953999996185
    PERFORMANCE ON TEST SET: Batch Loss = 0.604100167751, Accuracy = 0.881913781166
    Training iter #1845000:   Batch Loss = 0.331412, Accuracy = 0.952000021935
    PERFORMANCE ON TEST SET: Batch Loss = 0.58773380518, Accuracy = 0.887003719807
    Training iter #1860000:   Batch Loss = 0.344151, Accuracy = 0.941333353519
    PERFORMANCE ON TEST SET: Batch Loss = 0.59240424633, Accuracy = 0.882931768894
    Training iter #1875000:   Batch Loss = 0.339558, Accuracy = 0.940666675568
    PERFORMANCE ON TEST SET: Batch Loss = 0.612593412399, Accuracy = 0.878859877586
    Training iter #1890000:   Batch Loss = 0.284066, Accuracy = 0.956666648388
    PERFORMANCE ON TEST SET: Batch Loss = 0.614794969559, Accuracy = 0.885307073593
    Training iter #1905000:   Batch Loss = 0.241255, Accuracy = 0.97866666317
    PERFORMANCE ON TEST SET: Batch Loss = 0.629106223583, Accuracy = 0.8805565238
    Training iter #1920000:   Batch Loss = 0.298859, Accuracy = 0.958666682243
    PERFORMANCE ON TEST SET: Batch Loss = 0.603316783905, Accuracy = 0.888361036777
    Training iter #1935000:   Batch Loss = 0.294400, Accuracy = 0.953333318233
    PERFORMANCE ON TEST SET: Batch Loss = 0.587940692902, Accuracy = 0.89243298769
    Training iter #1950000:   Batch Loss = 0.284691, Accuracy = 0.95733332634
    PERFORMANCE ON TEST SET: Batch Loss = 0.580442249775, Accuracy = 0.890397012234
    Training iter #1965000:   Batch Loss = 0.295974, Accuracy = 0.945333361626
    PERFORMANCE ON TEST SET: Batch Loss = 0.605518162251, Accuracy = 0.884628415108
    Training iter #1980000:   Batch Loss = 0.287975, Accuracy = 0.948000013828
    PERFORMANCE ON TEST SET: Batch Loss = 0.609712958336, Accuracy = 0.88870036602
    Training iter #1995000:   Batch Loss = 0.289593, Accuracy = 0.96266669035
    PERFORMANCE ON TEST SET: Batch Loss = 0.582665205002, Accuracy = 0.895826280117
    Training iter #2010000:   Batch Loss = 0.251413, Accuracy = 0.982666671276
    PERFORMANCE ON TEST SET: Batch Loss = 0.591309666634, Accuracy = 0.887343049049
    Training iter #2025000:   Batch Loss = 0.317092, Accuracy = 0.931333363056
    PERFORMANCE ON TEST SET: Batch Loss = 0.591833949089, Accuracy = 0.88496774435
    Training iter #2040000:   Batch Loss = 0.324491, Accuracy = 0.934000015259
    PERFORMANCE ON TEST SET: Batch Loss = 0.569988429546, Accuracy = 0.89989823103
    Training iter #2055000:   Batch Loss = 0.309216, Accuracy = 0.936666667461
    PERFORMANCE ON TEST SET: Batch Loss = 0.577533602715, Accuracy = 0.881913781166
    Training iter #2070000:   Batch Loss = 0.322018, Accuracy = 0.934666693211
    PERFORMANCE ON TEST SET: Batch Loss = 0.58077198267, Accuracy = 0.883949756622
    Training iter #2085000:   Batch Loss = 0.322328, Accuracy = 0.941999971867
    PERFORMANCE ON TEST SET: Batch Loss = 0.565859556198, Accuracy = 0.887343049049
    Training iter #2100000:   Batch Loss = 0.250848, Accuracy = 0.989333331585
    PERFORMANCE ON TEST SET: Batch Loss = 0.568353116512, Accuracy = 0.884628415108
    Training iter #2115000:   Batch Loss = 0.259181, Accuracy = 0.966000020504
    PERFORMANCE ON TEST SET: Batch Loss = 0.525161743164, Accuracy = 0.897183597088
    Training iter #2130000:   Batch Loss = 0.243086, Accuracy = 0.987999975681
    PERFORMANCE ON TEST SET: Batch Loss = 0.539084136486, Accuracy = 0.891754329205
    Training iter #2145000:   Batch Loss = 0.236770, Accuracy = 0.987999975681
    PERFORMANCE ON TEST SET: Batch Loss = 0.527505517006, Accuracy = 0.892093658447
    Training iter #2160000:   Batch Loss = 0.229853, Accuracy = 0.987999975681
    PERFORMANCE ON TEST SET: Batch Loss = 0.51251077652, Accuracy = 0.902952134609
    Training iter #2175000:   Batch Loss = 0.256231, Accuracy = 0.980666637421
    PERFORMANCE ON TEST SET: Batch Loss = 0.51353931427, Accuracy = 0.900916159153
    Training iter #2190000:   Batch Loss = 0.278643, Accuracy = 0.977999985218
    PERFORMANCE ON TEST SET: Batch Loss = 0.533600509167, Accuracy = 0.896504938602
    Training iter #2205000:   Batch Loss = 0.327923, Accuracy = 0.944000005722
    PERFORMANCE ON TEST SET: Batch Loss = 0.530030608177, Accuracy = 0.907702744007
    Optimization Finished!
    FINAL RESULT: Batch Loss = 0.530030608177, Accuracy = 0.907702744007


## Training is good, but having visual insight is even better:

Okay, let's do it simply in the notebook for now


```python
# (Inline plots: )
%matplotlib inline

font = {
    'family' : 'Bitstream Vera Sans',
    'weight' : 'bold',
    'size'   : 18
}
matplotlib.rc('font', **font)

width = 12
height = 12
plt.figure(figsize=(width, height))

indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
plt.plot(indep_train_axis, np.array(train_losses),     "b--", label="Train losses")
plt.plot(indep_train_axis, np.array(train_accuracies), "g--", label="Train accuracies")

indep_test_axis = np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1] + [training_iters])
plt.plot(indep_test_axis, np.array(test_losses),     "b-", label="Test losses")
plt.plot(indep_test_axis, np.array(test_accuracies), "g-", label="Test accuracies")

plt.title("Training session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss or Accuracy values)')
plt.xlabel('Training iteration')

plt.show()
```


![png](LSTM_files/LSTM_16_0.png)


## And finally, the multi-class confusion matrix and metrics!


```python
# Results

predictions = one_hot_predictions.argmax(1)

print "Testing Accuracy: {}%".format(100*accuracy)

print ""
print "Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted"))
print "Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted"))
print "f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted"))

print ""
print "Confusion Matrix:"
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
print confusion_matrix
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print ""
print "Confusion matrix (normalised to % of total test data):"
print normalised_confusion_matrix
print ("Note: training and testing data is not equally distributed amongst classes, "
       "so it is normal that more than a 6th of the data is correctly classifier in the last category.")

# Plot Results: 
width = 12
height = 12
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix, 
    interpolation='nearest', 
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
```

    Testing Accuracy: 90.7702744007%
    
    Precision: 90.8521998563%
    Recall: 90.7702748558%
    f1_score: 90.7685497939%
    
    Confusion Matrix:
    [[450   2  15  12  17   0]
     [ 30 427  14   0   0   0]
     [  5   0 415   0   0   0]
     [  0  22   3 402  64   0]
     [  1   6   0  54 471   0]
     [  0  27   0   0   0 510]]
    
    Confusion matrix (normalised to % of total test data):
    [[ 15.2697649    0.06786563   0.5089922    0.40719375   0.57685781   0.        ]
     [  1.01798439  14.48931122   0.47505939   0.           0.           0.        ]
     [  0.16966406   0.          14.08211708   0.           0.           0.        ]
     [  0.           0.74652189   0.10179844  13.64099121   2.1717       0.        ]
     [  0.03393281   0.20359688   0.           1.83237195  15.98235512   0.        ]
     [  0.           0.91618598   0.           0.           0.          17.30573463]]
    Note: training and testing data is not equally distributed amongst classes, so it is normal that more than a 6th of the data is correctly classifier in the last category.



![png](LSTM_files/LSTM_18_1.png)



```python
sess.close()
```

## Conclusion

Outstandingly, the accuracy is of 90.77%! 

This means that the neural networks is almost always able to correctly identify the movement type! Remember, the phone is attached on the waist and each series to classify has just a 128 sample window of two internal sensors (a.k.a. 2.56 seconds at 50 FPS), so those predictions are extremely accurate.

I specially did not expect such good results for guessing between "WALKING" "WALKING_UPSTAIRS" and "WALKING_DOWNSTAIRS" as a cellphone. Tought, it is still possible to see a little cluster on the matrix between those 3 classes. This is great.

It is also possible to see that it was hard to do the difference between "SITTING" and "STANDING". Those are seemingly almost the same thing from the point of view of a device placed at waist level. 


## References

The [dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) can be found on the UCI Machine Learning Repository. 

> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.

## Connect with me

- https://ca.linkedin.com/in/chevalierg 
- https://twitter.com/guillaume_che
- https://github.com/guillaume-chevalier/


```python
# Let's convert this notebook to a README as the GitHub project's title page:
!jupyter nbconvert --to markdown LSTM.ipynb
!mv LSTM.md README.md
```

    [NbConvertApp] Converting notebook LSTM.ipynb to markdown
    [NbConvertApp] Support files will be in LSTM_files/
    [NbConvertApp] Making directory LSTM_files
    [NbConvertApp] Making directory LSTM_files
    [NbConvertApp] Writing 41669 bytes to LSTM.md

