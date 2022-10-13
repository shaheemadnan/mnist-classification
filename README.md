# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model


## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries
### STEP 2:
Build a Convolutional Neural Network Model
### STEP 3:
Create Model and then predict

## PROGRAM
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 0s 0us/step
11501568/11490434 [==============================] - 0s 0us/step

X_train.shape
(60000, 28, 28)

X_test.shape
(10000, 28, 28)

single_image= X_train[100]

single_image.shape
(28, 28)

plt.imshow(single_image,cmap='gray')


y_train.shape
(60000,)

X_train.min()
0

X_train.max()
255

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()
0.0

X_train_scaled.max()
1.0

y_train[0]
5

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)
numpy.ndarray

y_train_onehot.shape
(60000, 10)

single_image = X_train[500]

plt.imshow(single_image,cmap='gray')


y_train_onehot[500]
array([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], dtype=float32)

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 13, 13, 32)       0         
 )                                                               
                                                                 
 flatten (Flatten)           (None, 5408)              0         
                                                                 
 dense (Dense)               (None, 32)                173088    
                                                                 
 dense_1 (Dense)             (None, 10)                330       
                                                                 
=================================================================
Total params: 173,738
Trainable params: 173,738
Non-trainable params: 0
_________________________________________________________________

# Choose the appropriate parameters
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
              
model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))
Epoch 1/5
938/938 [==============================] - 25s 26ms/step - loss: 0.2395 - accuracy: 0.9294 - val_loss: 0.1105 - val_accuracy: 0.9658
Epoch 2/5
938/938 [==============================] - 25s 27ms/step - loss: 0.0822 - accuracy: 0.9755 - val_loss: 0.0664 - val_accuracy: 0.9772
Epoch 3/5
938/938 [==============================] - 24s 25ms/step - loss: 0.0576 - accuracy: 0.9824 - val_loss: 0.0579 - val_accuracy: 0.9816
Epoch 4/5
938/938 [==============================] - 24s 25ms/step - loss: 0.0445 - accuracy: 0.9862 - val_loss: 0.0506 - val_accuracy: 0.9836
Epoch 5/5
938/938 [==============================] - 24s 25ms/step - loss: 0.0356 - accuracy: 0.9891 - val_loss: 0.0574 - val_accuracy: 0.9824
<keras.callbacks.History at 0x7fde580dff50>

metrics = pd.DataFrame(model.history.history)

metrics.head()
loss	accuracy	val_loss	val_accuracy
0	0.239503	0.929367	0.110500	0.9658
1	0.082185	0.975467	0.066355	0.9772
2	0.057572	0.982383	0.057882	0.9816
3	0.044465	0.986250	0.050559	0.9836
4	0.035640	0.989117	0.057407	0.9824

metrics[['accuracy','val_accuracy']].plot()


metrics[['loss','val_loss']].plot()


x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
[[ 963    0    6    0    0    4    0    1    4    2]
 [   0 1131    3    0    0    0    1    0    0    0]
 [   0    2 1026    0    1    0    0    3    0    0]
 [   0    0    5  990    0    5    0    6    4    0]
 [   0    1    5    0  970    0    0    0    2    4]
 [   1    0    2    6    0  874    1    0    7    1]
 [   7    3    2    0    4    2  932    1    7    0]
 [   0    2   12    0    0    0    0 1013    1    0]
 [   3    0    8    1    0    0    0    4  954    4]
 [   1    5    0    1   10    3    0   12    6  971]]
 
print(classification_report(y_test,x_test_predictions))

              precision    recall  f1-score   support

           0       0.99      0.98      0.99       980
           1       0.99      1.00      0.99      1135
           2       0.96      0.99      0.98      1032
           3       0.99      0.98      0.99      1010
           4       0.98      0.99      0.99       982
           5       0.98      0.98      0.98       892
           6       1.00      0.97      0.99       958
           7       0.97      0.99      0.98      1028
           8       0.97      0.98      0.97       974
           9       0.99      0.96      0.98      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000

img = image.load_img('five.jpg')

type(img)

PIL.JpegImagePlugin.JpegImageFile

from tensorflow.keras.preprocessing import image
img = image.load_img('five.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)
     
print(x_single_prediction)
[5]

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
     
print(x_single_prediction)
[5]
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

loss	accuracy	val_loss	val_accuracy
0	0.239503	0.929367	0.110500	0.9658
1	0.082185	0.975467	0.066355	0.9772
2	0.057572	0.982383	0.057882	0.9816
3	0.044465	0.986250	0.050559	0.9836
4	0.035640	0.989117	0.057407	0.9824

![image](https://user-images.githubusercontent.com/114155368/195536426-0aa656fa-6a10-49ab-ade0-d17f4d8f6e10.png)
![image](https://user-images.githubusercontent.com/114155368/195536495-616fb06e-addf-4266-bd5b-75cbb0b38ef8.png)


### Classification Report
```
              precision    recall  f1-score   support

           0       0.99      0.98      0.99       980
           1       0.99      1.00      0.99      1135
           2       0.96      0.99      0.98      1032
           3       0.99      0.98      0.99      1010
           4       0.98      0.99      0.99       982
           5       0.98      0.98      0.98       892
           6       1.00      0.97      0.99       958
           7       0.97      0.99      0.98      1028
           8       0.97      0.98      0.97       974
           9       0.99      0.96      0.98      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000
```
### Confusion Matrix
```
[[ 963    0    6    0    0    4    0    1    4    2]
 [   0 1131    3    0    0    0    1    0    0    0]
 [   0    2 1026    0    1    0    0    3    0    0]
 [   0    0    5  990    0    5    0    6    4    0]
 [   0    1    5    0  970    0    0    0    2    4]
 [   1    0    2    6    0  874    1    0    7    1]
 [   7    3    2    0    4    2  932    1    7    0]
 [   0    2   12    0    0    0    0 1013    1    0]
 [   3    0    8    1    0    0    0    4  954    4]
 [   1    5    0    1   10    3    0   12    6  971]]
```

### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/114155368/195536672-e31addd5-8b53-4395-abee-368478b3c60d.png)


## RESULT
Thus a convolutional neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully
