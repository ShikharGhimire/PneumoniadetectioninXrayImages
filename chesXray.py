#Convolutional neural network

#Data preprocessing is already done since we manually seperated images into multiple folder. 
#We have to feature scale it though

#Importing the libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

#Initialising the neural network
classifier = Sequential()

#Step 1-Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation = 'relu')) #Put 1 instead of 3 at the end id you are working with black and white image
#Step 2 -- Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Applying second layer of convolution
classifier.add(Convolution2D(32,3,3,activation = 'relu'))
#Applying second layer of Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))


#Step 3 -Flattening
classifier.add(Flatten()) #keras will understand what we are trying to flatten so no parameters needed

##Step 4 -Full connection
#First hidden layer
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.5))
#Creating another hidden layer
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.5))

#Creating another hidden layer
classifier.add(Dense(output_dim=128,activation = 'relu'))
classifier.add(Dropout(0.5))
#Last layer (Output node)
classifier.add(Dense(output_dim = 1, activation = "sigmoid"))

#Compiling the CNN
classifier.compile(optimizer='adam',loss = 'binary_crossentropy', metrics = ['accuracy'])  #adam is the stochastic gradient descent, metrics is to check the accuracy

#Fitting the CNN in the image
#Importing the class that allows us to use image datagenerator
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( #Flipping the test images in different ranges
        rescale = 1./255,  
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255) #changing the dataset

training_set = train_datagen.flow_from_directory( #Creating a training set
    'dataset/train', #Specifying the folder
    target_size = (64,64), #The size of our sample specified in the convolution stage is 64x64(see above)
    batch_size = 32,
    class_mode = 'binary') #Dependent variable. We have pneumonia and non pneumonia (0 and 1 so it's binary)

test_set  = test_datagen.flow_from_directory(
    'dataset/test',
    target_size = (64,64),
    batch_size = 32,
    class_mode = 'binary')

classifier.fit_generator( #this function uses training dataset to train the image and teste it on the test data set so we applied it in classifier
        training_set,
        samples_per_epoch = 5216, # number of images in the training set
        nb_epoch = 25,
        validation_data = test_set, #validation_data is where we want to evaluate our data which is the test set
        nb_val_samples = 624) #Number of images in test set

#Making a new single predicition
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/normal.jpeg',target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0]==0:
    prediction = 'There is NO pneumonia present'
else:
    prediction = 'There is pneumonia present in the X-Ray.Further examination is required'

