# Convulutional Neural Network (CNN) for image classification
# CNN is a type of deep learning model that is used for image classification tasks.
# CNN is a type of feed-forward artificial neural network that is designed to process structured data.
# CNN is inspired by the organization of the visual cortex in animals.
# CNN is composed of multiple layers of neurons that process visual information in a hierarchical and spatially invariant manner.
# CNN is used for tasks such as image recognition, object detection, and image segmentation.
# CNN is composed of three main types of layers: convolutional layers, pooling layers, and fully connected layers.
# CNN is trained using backpropagation and gradient descent to minimize the error between the predicted output and the actual output.

# Importing the libraries
import tensorflow as tf # Allows us to build and train the convolutional neural network
from keras.preprocessing.image import ImageDataGenerator # Allows us to preprocess the images

# Setting the path to the data folder
main_repo_folder = '/'.join(__file__.split('/')[:-1])
data_folder = f'{main_repo_folder}/data'

# Part 1 - Data Preprocessing
## Preprocessing the Training set
# We need to apply some transformations (image augmentation) to the training set to prevent overfitting and improve the performance of the model.
# We will use the ImageDataGenerator class to apply some transformations to the images in the training set.
# The transformations include rescaling, rotation, shear, zoom, and horizontal flipping.
train_datagen = ImageDataGenerator(rescale = 1./255, # Rescale the pixel values to the range [0, 1].
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, 
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set', # The path to the training set.
                                                 target_size = (64, 64), # Resize the images to 64x64 pixels.
                                                 batch_size = 32, # The number of images that will go through the network at once.
                                                 class_mode = 'binary') # The type of problem we are trying to solve. It can be binary or categorical.

## Preprocessing the Test set
# We need to rescale the pixel values of the test set to the range [0, 1] as we did for the training set.
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set', # The path to the test set.
                                            target_size = (64, 64), # Resize the images to 64x64 pixels.
                                            batch_size = 32, # The number of images that will go through the network at once.
                                            class_mode = 'binary') # The type of problem we are trying to solve. It can be binary or categorical.

# Part 2 - Building the CNN
## Initialising the CNN
cnn = tf.keras.models.Sequential() # Create an object of the Sequential class. This class allows us to build the convolutional neural network as a sequence of layers.

## Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3])) # Add a convolutional layer to the CNN.
# The filters parameter specifies the number of filters in the layer.
# The kernel_size parameter specifies the size of the filters.
# The activation parameter specifies the activation function to be used in the layer.
# The input_shape parameter specifies the shape of the input images. 3 stands for RGB images.
# The input_shape parameter is required only for the first layer of the CNN. 

## Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) # Add a pooling layer to the CNN.
# The pool_size parameter specifies the size of the pooling window. 2 means a 2x2 window.
# The strides parameter specifies the stride of the pooling window.

## Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')) # Add a convolutional layer to the CNN.
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2)) # Add a pooling layer to the CNN.

## Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten()) # Add a flattening layer to the CNN.

## Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu')) # Add a dense layer to the CNN.
# The units parameter specifies the number of neurons in the layer.
# The activation parameter specifies the activation function to be used in the layer.

## Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN
## Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # Compile the CNN.
# The optimizer parameter specifies the optimization algorithm to be used.
# The loss parameter specifies the loss function to be used.
# The metrics parameter specifies the metric to be used to evaluate the model.

## Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25) # Train the CNN on the training set and evaluate it on the test set.
# epochs: The number of times the entire training set is passed through the CNN.

# Part 4 - Making a single prediction
import numpy as np
from keras.utils import load_img, img_to_array # Allows us to load and convert images
test_image = load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64)) # Load the image and resize it to 64x64 pixels
test_image = img_to_array(test_image) # Convert the image to an array
test_image = np.expand_dims(test_image, axis=0) # Add an extra dimension to the image. The axis parameter specifies the position where the new axis is to be inserted.
result = cnn.predict(test_image) # Predict the class of the image
training_set.class_indices # Get the class indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)