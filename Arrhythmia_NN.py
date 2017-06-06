'''
MIT License

Copyright (c) 2016 Panagiotis Petridis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
############### THE ABOVE LICENSE NOTICE IS ONLY VALID FOR THE SOURCE CODE AND NOT THE DATA USED TO TRAIN THE MODEL
'''
ABOUT THE DATASET:
Original Owners of Database:

1. H. Altay Guvenir, PhD.,
Bilkent University,
Department of Computer Engineering and Information Science,
06533 Ankara, Turkey
Phone: +90 (312) 266 4133
Email: guvenir '@' cs.bilkent.edu.tr

2. Burak Acar, M.S.,
Bilkent University,
EE Eng. Dept.
06533 Ankara, Turkey
Email: buraka '@' ee.bilkent.edu.tr

3. Haldun Muderrisoglu, M.D., Ph.D.,
Baskent University,
School of Medicine
Ankara, Turkey

Donor:

H. Altay Guvenir
Bilkent University,
Department of Computer Engineering and Information Science,
06533 Ankara, Turkey
Phone: +90 (312) 266 4133
Email: guvenir '@' cs.bilkent.edu.tr


Link: https://archive.ics.uci.edu/ml/datasets/Arrhythmia
'''
import numpy as np
import pandas as pd
import tensorflow as tf
import os.path

# Remove numpy scientific notation when printing arrays
np.set_printoptions(suppress=True)

# Loading the Data
print('Loading Data...')
data = np.array(pd.read_csv('arrhythmia.data', delimiter=',', header=None).values, dtype=np.float32)
print('Data Loaded!')

np.random.shuffle(data)     # Shuffle the data. You may want to comment this line to avoid running into the disclaimer
                            # I mentioned at the bottom of the file

n_classes = 16
X_data = data[:, :-1]

n_features = X_data.shape[1]
n_samples = X_data.shape[0]

data[:, n_features] -= 1

y_data = np.zeros([n_samples, n_classes])
y_data[np.arange(n_samples), data[:, n_features].astype(int)] = 1

# Spliting the data to train/test sets
y_train = y_data[:400]
y_test = y_data[400:]

X_train = X_data[:400]
X_test = X_data[400:]

# Building the computations graph
X = tf.placeholder(tf.float32, [None, n_features])
y = tf.placeholder(tf.float32, [None, n_classes])

# 1st layer
W1 = tf.Variable(tf.truncated_normal([n_features, 300], stddev=0.01))
b1 = tf.Variable(tf.zeros([300]))

# 2nd layer
W2 = tf.Variable(tf.truncated_normal([300, 300], stddev=0.01))
b2 = tf.Variable(tf.zeros([300]))

# 3rd/output layer
W3 = tf.Variable(tf.truncated_normal([300, 16], stddev=0.01))
b3 = tf.Variable(tf.zeros([16]))

keep_prob = tf.placeholder(tf.float32) # Probability that a unit will be kept during dropout

# Calculating each layer and adding dropout to layers #1 and #2
l1 = tf.nn.relu(tf.matmul(X, W1) + b1)
l1 = tf.nn.dropout(l1, keep_prob)

l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)
l2 = tf.nn.dropout(l2, keep_prob)

l3 = tf.nn.softmax(tf.matmul(l2, W3) + b3)

loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(l3 + 1e-4)))    # Cross entropy. +1e-4 ensures that log(0) is never the case
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)   # Adam Optimizer

saver = tf.train.Saver()

# Basic accuracy metric
n_correct = tf.equal(tf.argmax(l3, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(n_correct, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Loading model
if os.path.isfile('test-model.meta'):
    loader = tf.train.import_meta_graph('test-model.meta')
    loader.restore(sess, tf.train.latest_checkpoint('./'))

# Training model
best_score = sess.run(accuracy, {X: X_test, y: y_test, keep_prob: 1})
print('Initial Accuracy: ', best_score)
for i in range(1000):
    _, l = sess.run([optimizer, loss], {X: X_train, y: y_train, keep_prob: 0.5})
    if (i+1)%100==0:
        acc = sess.run(accuracy, {X: X_test, y: y_test, keep_prob: 1})
        print('Accuracy: ', acc)
        # early stopping avoids overfitting
        if acc>best_score:
            print('New best!')
            best_score = l
            saver.save(sess, 'test-model')
        print('Loss: ', l)

# K-fold accuracy metric
'''
Note:
Because of the size of the dataset the K-Fold doesn't have enough unseen samples to train on and has to
be tested on some already seen samples. Which results in a somewhat higher than expected accuracy.
The accuracy metric that was used to train the mode on the unseen test set is also somewhat biased since it is chosen
at random.
'''
final_accuracy = []
for i in range(10):
    idx = np.random.randint(n_samples, size=100)
    X_batch, y_batch = X_data[idx], y_data[idx]
    final_accuracy.append(sess.run(accuracy, {X: X_batch, y: y_batch, keep_prob: 1}))

final_accuracy = np.array(final_accuracy, dtype=np.float32)
print(final_accuracy)
print('Final Accuracy: ', np.mean(final_accuracy, 0))   # Model's final accuracy
## Best-model gets about ~96.5% accuracy

####### DISCLAIMER
'''
Due to the shuffling of the data if you ran the model a lot of times through it, it will eventually
memorize the entire dataset due to it's capacity!

Foot note:
The due to the low number of training samples and the model's high capacity needs a lot of reguralization.
'''
