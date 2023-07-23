import numpy as np
from PIL import Image
import os
import cv2

image_directory='datasets/'

no_tumor_images=os.listdir(image_directory+ 'no/')
yes_tumor_images=os.listdir(image_directory+ 'yes/')
dataset=[]
label=[]

# print(no_tumor_images)

# path='no0.jpg'

# print(path.split('.')[1])


for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'no/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)

dataset=np.array(dataset)
label=np.array(label)

from sklearn.model_selection import train_test_split
#x = input features
#y = output feature
x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)

# Reshape = (n, image_width, image_height, n_channel)

# print(x_train.shape)
# print(y_train.shape)

# print(x_test.shape)
# print(y_test.shape)

#installing the normalization library from keras then assigning the training and testing data
from keras.utils import normalize
x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

# categorising the training and testing data
from keras.utils import to_categorical
y_train=to_categorical(y_train , num_classes=2)
y_test=to_categorical(y_test , num_classes=2)

# Building the model
# 64,64,3
# Required Libraries
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model=Sequential()

model.add(Conv2D(32, (3,3), input_shape=(64,64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])

model.fit(x_train, y_train, 
batch_size=16, 
verbose=1, epochs=10, 
validation_data=(x_test, y_test),
shuffle=False)


from sklearn.metrics import confusion_matrix

# Predict the labels for the test set
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1) # convert to class labels

# Calculate the confusion matrix
cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)

# Print the confusion matrix
print("Confusion Matrix:\n", cm)

from sklearn.metrics import classification_report
# Print the classification report
print(classification_report(np.argmax(y_test, axis=1), y_pred))


model.save('BrainTumor10EpochsCategorical.h5')
