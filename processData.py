import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Image dimensions
width, height = 113, 113
num_labels = 3  # Number of emotion categories (happy, sad, angry)

# Path to the original folder
base_dir = 'F:\\github\\fer2013\\original_data_after_augmentation\\original'

# Folders for each emotion
categories = ['happy', 'angry', 'sad']

# Initialize lists to hold image data and labels
data = []
labels = []

# Load images from each folder
for category in categories:
    folder_path = os.path.join(base_dir, category)
    label = categories.index(category)  # Encode the labels (0, 1, 2)
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        # Load the image, resize it, and convert it to an array
        img = load_img(img_path, target_size=(width, height), color_mode='rgb')
        img_array = img_to_array(img)
        data.append(img_array)
        labels.append(label)

# Convert lists to numpy arrays
data = np.array(data, dtype='float32')
labels = np.array(labels)

# Normalize the image data (0-1 range)
data /= 255.0

# One-hot encode the labels
labels = to_categorical(labels, num_labels)

# Save the preprocessed data for future use
np.save('original_data_after_augmentation_fdataX.npy', data)
np.save('original_data_after_augmentation_flabels.npy', labels)

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")
