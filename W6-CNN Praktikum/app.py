import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the CNN model structure
def create_cnn_model(input_shape=(224, 224, 3), num_classes=4):
    model = Sequential()

    # Convolution and Pooling layers
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Dropout after first convolutional layer

    # Second Convolution layer
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Dropout after second convolutional layer

    # Third Convolution layer
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # Dropout after third convolutional layer

    # Flatten the output of the convolution layers
    model.add(Flatten())

    # Fully connected (Dense) layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout after Dense layer
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # Multi-class classification

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Load the updated model
model = load_model("best_model.keras")

# Update class names
CLASS_NAMES = ["Cardboard", "Glass", "Plastic", "Vegetation"]

def predict_image(image):
    # Preprocess the image
    img = Image.fromarray(image).resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    confidences = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}

    return confidences

# Create Gradio interface
demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=4),
    title="Image Classification Demo",
    description="Upload an image and the model will classify it into one of the following categories: " + ", ".join(CLASS_NAMES),
    examples=[
        ["example1.jpg"],
        ["example2.jpg"]
    ]  # Optional: Add example images
)

demo.launch()