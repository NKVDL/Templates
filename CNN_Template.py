
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 23:21:31 2018

@author: studyrelated
"""
##Convolutional Neural Network
"""
Separate N images of interest in folders test_set--test_category_i,
test_category_j...train_category_k and train_set--train_category_i,
train_category_j...train_category_k. This labels images for CNN to recognise
dog or cat ''.jpg format images. Ratio of train/test set is same as 
for ANN, if N > 10000 == 20/80 split, 2000 images in test_set, 8000 images in train_set. 
""" 
#Importing libraries
from tensorflow import keras
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout #apply to all layers if overfitting
import matplotlib.pyplot as plt

##Part 1 - Building the CNN
#Initialising the CNN
classifier = Sequential()

#1. Convolution
classifier.add(Conv2D(32, 3, 3,
                      input_shape=(64,64,3),
                      activation='relu'))
"""
To my classifier I add the function for my feature maps with rxc, rowxcolumn,
 heightxwidth, compressing my information to 3 by 3 rows and columns.
 My input is 64 by 64 pixels and has 3 layers Red, Green, Blue. My activation
function is as Rectifier Linear Unit to maximise the values into my rxc matrix:
    
    f(x)= x^{+} =max(0,x)},

where x is the input to a neuron.
"""
#2. Pooling feature maps
classifier.add(MaxPooling2D(pool_size=(2, 2)))

"""
Pooling layers into 2 and taking the maximum of the feature maps. Reducing size
without losing information.
"""
#-Adding layer and pooled feature maps
classifier.add(Conv2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(p=0.25))

#-Adding layer and pooled feature maps
classifier.add(Conv2D(64, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(p=0.25))

#-Adding layer and pooled feature maps
classifier.add(Conv2D(128, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(p=0.25))

#3.Flattening
classifier.add(Flatten())
"""
Flattening the pooled convolutional feature maps in a single vector with N
input features for the ANN. The sequential order of pooling and subsequent
flattening of all pooled feature maps in a single vector maintains the
spatial structure of the images. 
"""
#4.Create ANN for classification and full CNN connection
#create hidden layer
classifier.add(Dense(output_dim = 256, 
                     activation='relu'))
classifier.add(Dropout(p=0.1))
#create output layer
classifier.add(Dense(output_dim = 1,
                     activation='sigmoid'))

#Compile the CNN
classifier.compile(optimizer='adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
#Display CNN
classifier.summary()

## Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

#Keras documentation
#Augment images from the dataset
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
#Preprocessing the images
test_datagen = ImageDataGenerator(rescale=1./255)

#Creating 32 batches of hxw 64x64 pixels
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

#Creating 32 batches of hxw 64x64 pixels
test_set = test_datagen.flow_from_directory('dataset/training_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

#Create Tensorboard
tensorboard = TensorBoard(log_dir='dataset/logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=False)

#Running the CNN on the training set
classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=25,
                         validation_data=test_set, #test_set accuracy
                         validation_steps=2000,
                         callbacks=[tensorboard])


#Save classifier
classifier.save_weights('cat_dog_full_model_weights.h5')
classifier.save('cat_dog_classifier.h5')

"""
Tensorboard uses scalars, while Keras uses validation_data for evaluation.
Tensorboard; in terminal:
    tensorboard --logdir==training:model_dir --host=127.0.0.1
    tensorboard --host 0.0.0.0 --logdir=/dataset/logs

#Loss & accuracy
loss = classifier.history['loss']
val_loss = classifier.history['val_loss']
acc = classifier.history['acc']
val_acc = classifier.history['val_acc']

#Plot loss over epoch/time
tra = plt.plot(loss)
val = plt.plot(val_loss, 'r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend(["Training", "Validation"])
plt.show()
"""

## Part 3 - Run classifier on new data
#Importing library
import numpy as np
from keras.preprocessing import image
#Importing image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64)) # grayscale=True,optional
#Convert image to array of nxn dimensions
test_image = image.img_to_array(test_image)
#Image as new axis that will appear at the axis position in the expanded array shape.
test_image = np.expand_dims(test_image, axis = 0)
#Predict single image with classifier
result = classifier.predict(test_image)
#Dictionary containing mapping from class names to class indices
training_set.class_indices
#Return predicted class attrtibute
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

    
"""
Accuracy: 94.9%  
Loss: 0.13
Val_Accuracy: 98%, Loss
Val_Loss: 0.05
"""


