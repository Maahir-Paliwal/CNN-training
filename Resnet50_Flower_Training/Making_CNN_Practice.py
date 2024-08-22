from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
from tensorflow.keras.models import Model 
from keras.models import Sequential 
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

ResNet = ResNet50(include_top=False, weights = 'imagenet', input_shape= [224,224,3])

for layer in ResNet.layers:
    layer.trainable = False
    model.add(layer)