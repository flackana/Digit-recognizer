'''In this notebook we train and evaluate simple sequential neural network with two hidden layers. 
It scores 0.97 in the keaggle challenge.'''
#%%
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%% Import and split data
x_train_0 = pd.read_csv('../../Data/Data/Clean/x_train_clean.csv')
x_test_0 = pd.read_csv('../../Data/Data/Clean/x_test_clean.csv')
y_train_0 = pd.read_csv('../../Data/Data/Clean/Y_train_clean.csv').values.ravel()

x_train, x_val, y_train, y_val = train_test_split(
    x_train_0, y_train_0, test_size=0.1, stratify=y_train_0, random_state=42)

#%%
#******************************************************************************************
# Scaling and one-hot encoding
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float64))
x_val_scaled = scaler.transform(x_val)
x_test_sclaed = scaler.transform(x_test_0)
# One hot encoding y
encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_val = encoder.transform(y_val.reshape(-1, 1))
y_train_0 = encoder.transform(y_train_0.reshape(-1, 1))

x_train_0_s = scaler.fit_transform(x_train_0)
x_test_0_s = scaler.transform(x_test_0)

# %% Building a sequential model
# **************************************************************************************
model = keras.models.Sequential()
model.add(keras.layers.Dense(512, input_shape=(784,), activation="relu"))
model.add(keras.layers.Dense(512, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy",
optimizer="sgd",
metrics=["accuracy"])

history = model.fit(x_train_scaled, y_train, epochs=30,
validation_data=(x_val_scaled, y_val))
# %% Ploting the metrics
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('../../Figures/SNN_metrics.png')
# %% Train the final model
hi = model.fit(x_train_0_s, y_train_0, epochs=30)

predictions = model.predict(x_test_0_s)
predictions2 = np.argmax(predictions, axis = 1)
# Create a file to submit
submission = pd.DataFrame()
submission['ImageId'] = [i for i in range(1, len(predictions)+1)]
submission['Label'] = predictions2
submission.to_csv('../../Predictions/SNN.csv', index=False)
# %%
