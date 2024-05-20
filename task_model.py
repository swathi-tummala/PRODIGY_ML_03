# Importing the basic libraries
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = datagen.flow_from_directory(
        "./archive/training_set/training_set/",
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary"
      )

datagen1 = ImageDataGenerator(rescale=1./255)
test_set = datagen1.flow_from_directory(
        "./archive/test_set/test_set",
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary"
      )

cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32,padding="same",kernel_size=3, activation='relu', strides=2, input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32,padding='same',kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01),activation
             ='linear'))
cnn.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'])
r=cnn.fit(x = training_set, validation_data = test_set, epochs = 15)

cnn.save('./animal_classification')

# Cat prediction
test_image = image.load_img('archive/training_set/training_set/cats/cat.1028.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
if result[0]<0:
    print("The image classified is cat")
else:
    print("The image classified is dog")

# Dog prediction
test_image = image.load_img('archive/training_set/training_set/dogs/dog.1077.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
if result[0]<0:
    print("The image classified is cat")
else:
    print("The image classified is dog")
