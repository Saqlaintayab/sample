import numpy as np
import tensorflow as tf
x_train=np.array([0,1,2,3,4],dtype=float).reshape(-1,1)
y_train=np.array([1,3,5,7,9],dtype=float)
model=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)])
model.compile(optimizer='sgd',loss='mean_squared_error')
model.fit(x_train, y_train, epochs=500, verbose=0)
print(model.predict(np.array([[12.0]])))