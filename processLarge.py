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
batch_size = 5000  # Define a batch size for processing
data_batches = []
label_batches = []

# Function to save each batch to disk
def save_batch(data_batch, label_batch, batch_index):
    np.save(f'original_data_batch_{batch_index}_X.npy', np.array(data_batch, dtype='float32'))
    np.save(f'original_data_batch_{batch_index}_Y.npy', np.array(label_batch, dtype='float32'))

# Load images from each folder in batches
batch_data = []
batch_labels = []
batch_index = 0

for category in categories:
    folder_path = os.path.join(base_dir, category)
    label = categories.index(category)  # Encode the labels (0, 1, 2)
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        # Load the image, resize it, and convert it to an array
        img = load_img(img_path, target_size=(width, height), color_mode='rgb')
        img_array = img_to_array(img)
        batch_data.append(img_array)
        batch_labels.append(label)
        
        # If batch size is reached, process the batch
        if len(batch_data) >= batch_size:
            # Normalize and one-hot encode the labels
            batch_data = np.array(batch_data, dtype='float32') / 255.0
            batch_labels = to_categorical(batch_labels, num_labels)
            save_batch(batch_data, batch_labels, batch_index)
            batch_index += 1
            batch_data = []  # Reset the batch data
            batch_labels = []  # Reset the batch labels

# Process any remaining images in the final batch
if len(batch_data) > 0:
    batch_data = np.array(batch_data, dtype='float32') / 255.0
    batch_labels = to_categorical(batch_labels, num_labels)
    save_batch(batch_data, batch_labels, batch_index)

print("All batches processed and saved.")
