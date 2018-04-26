# MNIST Simple Example
# Neural Network

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import numpy as np
import scipy.ndimage

# 55000 adet veri setinden öğrenme yapılacak

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10])) # tf.zeros tüm değerleri 0 yapar. 784 e 10 luk matris oluşturduk, bu matristen 55000 adet arka arkaya düşünerek eğitimi anlayabiliri. 0 - 9 arasındaki tüm sayıları 1 e 784 lük matristen oluşan değerlerini çizdiğimiz sayı değerine göre karşılaştırarak tahmin yapıyor diyebiliriz.
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b) # ilk olarak x ile W, tf.matmul ifadesi yardımı ile çarpılır - mathematical multiply

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) # düzensizliği minimum değere indirgemek için yapılan işlemler, y_ gerçek değerin log-y çarpımı ile olan toplamı düşür ve bu değerlerin ortlamasını düşür. hepsi sırası ile

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # tensorflowu dereceli düşürme ile eğitiyoruz (minimum düzensizlik bazında)

sess = tf.InteractiveSession() # modelimizi çalıştırıyoruz
tf.global_variables_initializer().run() # oluşturuğumuz değişkenleri başlatıyoruz

for _ in range(1000): # eğitmeye başlıyoruz
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# test #
# Test çizimlerimizi okuyup el ile yazılan rakamı bulmaya çalışıyoruz. #
# Doğruluk payı %90 civarında buluyor ancak çizimlerde belirlenen piksellerin dışına çıktığım için hatalı sonuçlar verebiliyor.

draw = np.vectorize(lambda x: 255 - x)(np.ndarray.flatten(scipy.ndimage.imread("test-draw4.png", flatten=True)))

output = sess.run(tf.argmax(y, 1), feed_dict={x: [draw]})

print(output)
