import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1. Load the trained CNN model
MODEL_PATH = 'best_cnn_model.h5'
if not os.path.exists(MODEL_PATH):
    print("Error: 'best_cnn_model.h5' file not found!")
    exit()

model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# 2. Dynamic Path Search (ఇమేజ్ ఎక్కడున్నా వెతికి పట్టుకుంటుంది)
POSSIBLE_DIRS = [
    r'D:\pro1\extracted_data\archive\val\cat',
    r'D:\pro1\extracted_data\val\cat',
    r'D:\pro1\archive\val\cat',
    r'D:\pro1\val\cat'
]

TARGET_DIR = None
for directory in POSSIBLE_DIRS:
    if os.path.exists(directory):
        TARGET_DIR = directory
        break

if TARGET_DIR is None:
    print("Error: Could not find the 'val/cat' folder automatically.")
    print("Please check your extracted folder structure inside D:\\pro1\\")
    exit()

# ఫోల్డర్‌లో ఉన్న ఫైల్స్ లిస్ట్ తీసుకుంటుంది
all_files = os.listdir(TARGET_DIR)
image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print(f"Error: No images found inside the folder: {TARGET_DIR}")
    exit()

# మొదటి ఇమేజ్‌ను ఆటోమేటిక్‌గా సెలెక్ట్ చేస్తుంది
FILE_NAME = image_files[0]
IMAGE_PATH = os.path.join(TARGET_DIR, FILE_NAME)
print(f"\n[Path Found] Successfully located images at: {TARGET_DIR}")
print(f"[Selected Image]: {FILE_NAME}")

# Image preprocessing for CNN input
img = image.load_img(IMAGE_PATH, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# 3. Perform prediction
prediction = model.predict(img_array)

print("\n--- Prediction Result ---")
if prediction[0][0] > 0.5:
    confidence = prediction[0][0] * 100
    print(f"Result: DOG (Confidence: {confidence:.2f}%)")
else:
    confidence = (1 - prediction[0][0]) * 100
    print(f"Result: CAT (Confidence: {confidence:.2f}%)")
