from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
from basedata import train, validation

train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('basedata//train//',
                                          target_size=(200,200),
                                          batch_size = 32,
                                          class_mode = 'binary')

validation_dataset = train.flow_from_directory('basedata//validation//',
                                          target_size=(200,200),
                                          batch_size = 32,
                                          class_mode = 'binary')

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Flatten(),
                                    #
                                    #tf.keras.layers.Dense(33856,activation='relu'),
                                    tf.keras.layers.Dense(512, activation= 'relu'),
                                    #
                                    tf.keras.layers.Dense(1,activation='sigmoid')
                                   ]
                                   )

model.compile(loss='binary_crossentropy',
              optimizer = RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

# model_fit= model.fit(train_dataset,
#                      steps_per_epoch = 3,
#                      epochs= 30,
#                      validation_data= validation_dataset)

steps_per_epoch_train = len(train_dataset)//train_dataset.batch_size
steps_per_epoch_validation = len(validation_dataset)//validation_dataset.batch_size
model_fit = model.fit(train_dataset,
                     steps_per_epoch = steps_per_epoch_train,
                     epochs= 30,
                     validation_data= validation_dataset,
                     validation_steps = steps_per_epoch_validation)

# model_path = 'G://G drive//sem 6//project//SELF//Deepfake//'
model.save('cnn_model.h5')

# print("Model saved successfully as: ", model_path)