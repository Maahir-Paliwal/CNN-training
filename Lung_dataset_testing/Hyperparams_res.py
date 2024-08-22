IMAGE_SIZE = [224,224]

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model 
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.optimizers import Adam
import numpy as np
from glob import glob 
import matplotlib.pyplot as plt 
import pandas as pd


Train_folder = '/Users/maahirpaliwal/Downloads/lung_alrumeh_assem_train/output/train'
Validation_folder = '/Users/maahirpaliwal/Downloads/lung_alrumeh_assem_train/output/val'
Test_folder = '/Users/maahirpaliwal/Downloads/lung_alrumeh_assem_train/output/test'

myResnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights = 'imagenet', include_top=False)

myResnet.summary()


for layer in myResnet.layers:
    layer.trainable = False


classes = glob('/Users/maahirpaliwal/Downloads/lung_alrumeh_assem_train/output/train/*')
print(classes)

classesNum = len(classes)
print(classesNum)


#create model 
def create_model(learning_rate=0.001, dropout_rate=0.5):
    model = Sequential()
    model.add(myResnet)
        
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(classesNum, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


train_datagen = ImageDataGenerator(rescale = 1. /255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./ 255)

training_set = train_datagen.flow_from_directory(Train_folder, target_size=(224,224), batch_size= 32, class_mode='categorical')

val_set = val_datagen.flow_from_directory(Validation_folder, target_size=(224,224), batch_size= 32, class_mode='categorical')

STEP_SIZE_TRAIN = training_set.n // training_set.batch_size
STEP_SIZE_VALID = val_set.n // val_set.batch_size

model = create_model()

result = model.fit(training_set, 
                   validation_data=val_set, 
                   epochs=50, 
                   steps_per_epoch = STEP_SIZE_TRAIN, 
                   validation_steps = STEP_SIZE_VALID )


print(STEP_SIZE_TRAIN)
print(STEP_SIZE_VALID)


plt.plot(result.history['accuracy'], label='train_acc')
plt.plot(result.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

plt.plot(result.history['loss'], label='train_loss')
plt.plot(result.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

model.save('/Users/maahirpaliwal/Downloads/lung_alrumeh_assem_train/output/myResnetLung.h5')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(Test_folder, target_size=(224,224), batch_size= 1, class_mode=None, shuffle = False)

STEP_SIZE_TEST=test_set.n//test_set.batch_size
test_set.reset()
pred=model.predict(test_set,
steps=len(test_set),
verbose=1)


predicted_class_indices=np.argmax(pred,axis=1)

labels = (training_set.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

filenames=test_set.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)
