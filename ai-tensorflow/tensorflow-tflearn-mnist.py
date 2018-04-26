from __future__ import division, print_function, absolute_import

import tflearn
import numpy as np
import scipy.ndimage
# Data load and preprocessing
import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=True)

input_layer = tflearn.input_data(shape=[None, 784]) # 28 x 28

hidden_layer1 = tflearn.fully_connected(input_layer, 128, activation='relu', regularizer='L2', weight_decay=0.001) # regularizer sinir ağında bazı yollara daha çok ağırlık verilmesini engellemek için kullanılır, bir nevi cezalandırma
dropout1 = tflearn.dropout(hidden_layer1, 0.8) # 0.6 ve 0.8 arasında olması uygun

hidden_layer2 = tflearn.fully_connected(dropout1, 128, activation='relu', regularizer='L2', weight_decay=0.001)
dropout2 = tflearn.dropout(hidden_layer2, 0.8)

softmax = tflearn.fully_connected(dropout2, 10, activation='softmax')

sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=1000)
top_k = tflearn.metrics.Top_k(3)
neunet = tflearn.regression(softmax, optimizer=sgd, metric=top_k, loss='categorical_crossentropy')

model = tflearn.DNN(neunet, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=10, validation_set=(testX, testY), show_metric=True, run_id='dense-model')



draw = np.vectorize(lambda x:255 - x)(np.ndarray.flatten(scipy.ndimage.imread("test-draw.png", flatten=True)))
draw = np.array(draw).reshape(1, 784)

print(model.predict(draw))