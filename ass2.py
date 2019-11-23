import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celcius_q = np.array([11, 9, 14, 16, 14, 14, 12, 16, 10, 12], dtype = float)
fahrenheit_a = np.array([51.8, 48.2, 57.2, 60.8, 57.2, 57.2, 53.6, 60.8, 50.0, 53.6], dtype = float)

l0 = tf.keras.layers.Dense(units=1,input_shape=[1])

model = tf.keras.Sequential([l0])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(celcius_q,fahrenheit_a,epochs=1500,verbose=False)
print("Finished training the model")

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

print(model.predict([9]))
print(model.predict([10]))
print(model.predict([15]))
print(model.predict([20]))


