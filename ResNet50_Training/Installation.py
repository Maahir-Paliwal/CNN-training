import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python import keras
from keras import layers
from keras import applications
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import adam_v2


#getting flower dataset
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

print(data_dir)

#image of rose
roses = list(data_dir.glob('roses/*'))
print(roses[0])
PIL.Image.open(str(roses[0]))

#training data set
img_height,img_width=180,180
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#validation data set
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


#class names
class_names = train_ds.class_names
print(class_names)
     

#create own model
resnet_model = Sequential()

#get weights from pretrained model
pretrained_model = applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape= (180,180,3),
    pooling='avg',
    classes=5,
)

#make it so the weights can't be adjusted
for layer in pretrained_model.layers:
        layer.trainable=False


resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(units=512, activation='relu'))
resnet_model.add(Dense(units=5, activation='softmax'))

resnet_model.summary()

resnet_model.compile(optimizer=adam_v2.Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])


