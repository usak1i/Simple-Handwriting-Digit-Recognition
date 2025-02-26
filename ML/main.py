import os 
import cv2
import random
import numpy as np
import matplotlib.pyplot as plot
import tensorflow as tf
from tensorflow.keras import layers # type: ignore


mnist = tf.keras.datasets.mnist     # Load the mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()     # x contains the pixel data and y contains labels

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1) 

image_height = 28   
image_weight = 28
activate = 'relu'
epochs = 10          # Training iterations
batch_size = 32

# Define the model
model = tf.keras.models.Sequential([
   layers.Flatten(input_shape = (image_height, image_weight)),    # Input layer
   layers.Dense(128, activation = activate),                      # Hidden layers
   layers.Dense(128, activation = activate),
   layers.Dense(10, activation = 'softmax')                       # Output layer (0 ~ 9)
])

# Set up the model
model.compile(optimizer = 'adam', 
             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
             metrics = ['accuracy'])

# Print the model architecture
model.summary() 

# Train the codel
model.fit(x_train, y_train, epochs = epochs, batch_size = 32)

# Save the model so that we don't have to retrain the model
model.save('TestModel.keras')

# After Trained the model and save it, we can just load the model without retraining it
# model = tf.keras.models.load_model('TestModel.keras')

# Check the loss and accuarcy of the model
loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)

# Select digits from the Digits directory and let the model predict the digit
image_number = 1
while os.path.isfile(f"Digits/digit{image_number}.png"):
    image = cv2.imread(f"Digits/digit{image_number}.png", cv2.IMREAD_GRAYSCALE)
    image = np.invert(np.array([image]))
    prediction = model.predict(image)
    print(f"This digit is probably a {np.argmax(prediction)}")     # output the result
    plot.imshow(image[0], cmap = plot.cm.binary)
    plot.show()     # show the digit so that we can check if the model predict correctly
    image_number += 1


