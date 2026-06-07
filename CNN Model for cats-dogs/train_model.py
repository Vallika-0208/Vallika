import os
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ==========================================
# 1. DATA PATH CONFIGURATION
# ==========================================
EXTRACT_PATH = r'D:\pro1\extracted_data'
base_dir = EXTRACT_PATH
if 'archive' in os.listdir(EXTRACT_PATH):
    base_dir = os.path.join(EXTRACT_PATH, 'archive')

TRAIN_DIR = os.path.join(base_dir, 'train')
VAL_DIR = os.path.join(base_dir, 'val')

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# ==========================================
# 2. ADVANCED DATA AUGMENTATION
# ==========================================
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20, 
    shear_range=0.15, 
    zoom_range=0.2,
    horizontal_flip=True, 
    width_shift_range=0.1, 
    height_shift_range=0.1,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

print("Loading training images...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

print("Loading validation images...")
validation_generator = validation_datagen.flow_from_directory(
    VAL_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary'
)

# ==========================================
# 3. DEEPER CNN ARCHITECTURE FOR >90% ACCURACY
# ==========================================
model = Sequential([
    # Layer 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Layer 2
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Layer 3 (New Layer added for high accuracy)
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Fully Connected Network
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Using 'adam' optimizer for better performance
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks to adjust learning rate and prevent overfitting
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
earlystop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)

# ==========================================
# 4. START MODEL TRAINING (Increased Epochs)
# ==========================================
print("\n--- Starting Model Training for Higher Accuracy ---")
history = model.fit(
    train_generator, 
    epochs=25,  # Increased epochs to hit >90%
    validation_data=validation_generator,
    callbacks=[earlystop, learning_rate_reduction]
)
plt.figure(figsize=(12, 5))

# Accuracy Graph
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss Graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_validation_curves.png')
plt.show()

print("\nValidation curves saved as 'training_validation_curves.png'")

# ==========================================
# 6. CONFUSION MATRIX
# ==========================================

print("\nGenerating Confusion Matrix...")

validation_generator.reset()

predictions = model.predict(validation_generator)

y_pred = (predictions > 0.5).astype(int).reshape(-1)
y_true = validation_generator.classes

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Cat', 'Dog'],
    yticklabels=['Cat', 'Dog']
)

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.savefig('confusion_matrix.png')
plt.show()

print("\nConfusion Matrix saved as 'confusion_matrix.png'")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=['Cat', 'Dog']
))

# ==========================================
# 7. SAVE MODEL
# ==========================================

model.save('best_cnn_model.h5')
print("\nSuccess! High-accuracy model saved as 'best_cnn_model.h5'.")

