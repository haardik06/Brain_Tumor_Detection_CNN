# ==========================================================
# Brain Tumor Detection using CNN
# Predict.py
# ==========================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image

from tkinter import Tk
from tkinter.filedialog import askopenfilename

# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = "models/brain_tumor_cnn.keras"

IMAGE_SIZE = (224,224)

CLASS_NAMES = [

    "Glioma",

    "Meningioma",

    "No Tumor",

    "Pituitary"

]

# ==========================================================
# Load CNN Model
# ==========================================================

print("="*60)
print("Brain Tumor Detection")
print("="*60)

print("\nLoading Model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model Loaded Successfully")

# ==========================================================
# Open File Picker
# ==========================================================

Tk().withdraw()

IMAGE_PATH = askopenfilename(

    title="Select Brain MRI Image",

    filetypes=[

        ("Image Files","*.jpg *.jpeg *.png")

    ]

)

if IMAGE_PATH == "":

    print("\nNo Image Selected")

    exit()

print("\nSelected Image")

print(IMAGE_PATH)

# ==========================================================
# Read Image
# ==========================================================

img = image.load_img(

    IMAGE_PATH,

    target_size=IMAGE_SIZE

)

img_array = image.img_to_array(img)

img_array = img_array.astype("float32") / 255.0

img_array = np.expand_dims(

    img_array,

    axis=0

)

# ==========================================================
# Predict
# ==========================================================

print("\nPredicting...")

prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)

predicted_class = CLASS_NAMES[predicted_index]

confidence = prediction[0][predicted_index] * 100

print("\nPrediction Completed Successfully")

# ==========================================================
# Display Prediction Result
# ==========================================================

print("\n" + "=" * 60)
print("Prediction Result")
print("=" * 60)

print(f"\nPredicted Class : {predicted_class}")

print(f"Confidence      : {confidence:.2f}%")

print("\n" + "=" * 60)
print("Probability of Each Class")
print("=" * 60)

for i in range(len(CLASS_NAMES)):
    probability = prediction[0][i] * 100

    print(f"{CLASS_NAMES[i]:15} : {probability:.2f}%")

print("=" * 60)

# ==========================================================
# Find Whether Prediction is Confident
# ==========================================================

if confidence >= 90:

    print("\nModel Confidence : Excellent")

elif confidence >= 75:

    print("\nModel Confidence : Good")

elif confidence >= 60:

    print("\nModel Confidence : Moderate")

else:

    print("\nModel Confidence : Low")

# ==========================================================
# Display MRI Image
# ==========================================================

plt.figure(figsize=(8,8))

plt.imshow(img)

plt.axis("off")

plt.title(

    f"{predicted_class}\nConfidence : {confidence:.2f}%",

    fontsize=14,

    color="blue"

)

plt.show()

# ==========================================================
# End
# ==========================================================

print("\nPrediction Finished Successfully")

