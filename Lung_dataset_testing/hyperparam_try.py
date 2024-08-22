IMAGE_SIZE = [224,224]

import tensorflow as tf
import scikeras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

#Loaf Cifar dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#define pre-existing model
myResnet = ResNet50(input_shape=[32,32,3], weights = 'imagenet', include_top=False)

for layer in myResnet.layers:
    layer.trainable = False


#create model 
def create_model(learning_rate=0.001, dropout_rate=0.5):
    model = Sequential()
    model.add(myResnet)
        
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create a KerasClassifier wrapper for the model
model = KerasClassifier(model=create_model, epochs=10, batch_size=32, verbose=0)

param_grid = {
    'model__learning_rate': [0.001, 0.0001],
    'model__dropout_rate': [0.3, 0.5],
    'batch_size': [32, 64]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=1)
grid_result = grid.fit(x_train, y_train)

# Print the best parameters and best score
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Evaluate the best model on the test set
best_model = grid_result.best_estimator_.model_
test_loss, test_accuracy = best_model.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)