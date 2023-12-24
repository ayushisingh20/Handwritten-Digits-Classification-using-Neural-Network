# Handwritten-Digits-Classification-using-Neural-Network
Handwritten Digits Classification using Neural Network


# Importing Libraries:

TensorFlow is like a toolbox for making computers learn things.
Keras is like a set of instructions inside TensorFlow that helps us build and train models easily.
Matplotlib is like a tool for drawing pictures.
NumPy helps with doing math on numbers.

# Loading the MNIST Dataset:

Imagine we have a big collection of pictures of handwritten numbers from 0 to 9.
We split this collection into two parts - one for training our computer and one for testing how well it learned.
x_train and x_test are the images, and y_train and y_test are the corresponding numbers each image represents.

# Data Preprocessing:

Imagine the pictures are made up of tiny dots, and each dot has a color from 0 to 255.
We want to make it easier for the computer to learn, so we change the colors to be between 0 and 1.
This is like saying, "Hey computer, when you see a dot, think of it as a number between 0 and 1."

# Visualization:

We want to see what the pictures look like, so we use Matplotlib to show us the first and second images.

# Data Inspection:

The labels (y_train) are like tags telling us which number each picture is.
We look at the first five tags.
The shape of x_train tells us how many pictures we have and how big each picture is.
We reshape the pictures into a simpler form, like stacking all the rows of pixels on top of each other.

# Final Data Shape:

x_train_flattened = x_train.reshape(len(x_train),28*28)
x_test_flattened = x_test.reshape(len(x_test),28*28)
We check how our reshaped test data looks.

# Displaying a Flattened Test Image:

We show the computer's simplified view of the first test picture after reshaping.


# Building a Neural Network (without hidden layer):

Created a sequential model with one Dense layer having 10 neurons (output neurons for digits 0-9) using the sigmoid activation function.
Compiled the model with the Adam optimizer, sparse categorical crossentropy loss, and accuracy as the metric.
Trained the model on the training data for 5 epochs.
Evaluated the model on the test data.
model = keras.Sequential([keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')]) #sequential means I have stack of layers in my neural network thus will accept every layer as one element.


 DENSE---->All neurons in one layer are connected with every other neuron in second layer
 Dense(10--->OUTPUTSHAPE(0-9 neurons), input_shape=(784,)----->784 1D NEURONS , activation='sigmoid')

## Model compilation
Model compilation is an activity performed after writing the statements in a model and before training starts. It checks for format errors, and defines the loss function, the optimizer or learning rate, and the metrics
optimizers allow you to train efficiently
loss = sparse_categorical_crossentropy --->output is categorical: 0-9
                                       --->sparse: output variable i.e y_train is an interger
accuracy--->we want an accurate model


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Predictions and Confusion Matrix (without hidden layer):

Predicted on the test data and visualized the prediction for the first sample.
Created a confusion matrix using TensorFlow's confusion_matrix function and visualized it using seaborn.

# Building a Neural Network (with hidden layer):

Created a sequential model with two Dense layers - one hidden layer with 100 neurons and ReLU activation, and one output layer with 10 neurons and sigmoid activation.
Compiled and trained the model similarly to the previous one.

# Predictions and Confusion Matrix (with hidden layer):

Predicted on the test data and created a new confusion matrix for the model with the hidden layer.
