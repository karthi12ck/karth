!pip install tf-nightly-2.0-preview


import tensorflow as tf
tf.__version__

import os
import datetime
from tensorflow.keras import layers














def create_model():
  return tf.keras.Sequential([ layers.Dense(64,activation=tf.nn.relu,input_shape=(32,)),
                             layers.Dense(64,activation=tf.nn.relu),layers.Dense
                             (10,activation=tf.nn.softmax)])

import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))


def train_model():
  

  model=create_model()
  model.compile(optimizer=tf.keras.optimizers.Adam(0.1),loss=tf.keras.losses.CategoricalCrossentropy(),metrics=[tf.keras.metrics.Accuracy()])
  
  logdir=os.path.join("logt",datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
  tensorboard_callback=tf.keras.callbacks.TensorBoard(logdir,histogram_freq=1)
  model.fit(data,labels,epochs=10,batch_size=32,callbacks=[tensorboard_callback])
  



train_model()

%load_ext tensorboard

%tensorboard --logdir logt

  