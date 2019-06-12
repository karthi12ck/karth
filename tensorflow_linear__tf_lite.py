import tensorflow as tf
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.contrib import lite

model=keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
xs=np.array([-1.0,0.0,1.0,2.0,3.0,4.0])
ys=np.array([-3.0,-1.0,0.0,3.0,5.0,7.0])
model.fit(xs,ys,epochs=500)

print(model.predict([10]))


keras_file="linear.h5"
keras.models.save_model(model,keras_file)
converter=lite.TocoConverter.from_keras_model_file(keras_file)
tflite_model=converter.convert()
open("linear.tflite","wb").write(tflite_model)