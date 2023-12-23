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

We check how our reshaped test data looks.

# Displaying a Flattened Test Image:

We show the computer's simplified view of the first test picture after reshaping.
