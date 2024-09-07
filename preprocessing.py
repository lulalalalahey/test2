import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv('./fer2013.csv')

width, height = 48, 48

# Convert pixel values into arrays
datapoints = data['pixels'].tolist()

# Getting features for training
X = []
for xseq in datapoints:
    xx = [int(xp) for xp in xseq.split(' ')]
    xx = np.asarray(xx).reshape(width, height)
    X.append(xx.astype('float32'))

X = np.asarray(X)
X = np.expand_dims(X, -1)

# Filter the dataset to include only 'happy', 'sad', and 'angry' classes
desired_classes = [3, 4, 6]  # Adjust these based on actual class indices in your dataset
data_filtered = data[data['emotion'].isin(desired_classes)]

# Update X to only include the filtered data
X = X[data['emotion'].isin(desired_classes)]

# Getting labels for training
y = data_filtered['emotion']

# Re-map the labels to 0, 1, 2
class_mapping = {3: 0, 4: 1, 6: 2}
y = y.map(class_mapping)

# One-hot encode the labels
y = pd.get_dummies(y).values

# Storing them using numpy
np.save('fdataX', X)
np.save('flabels', y)

print("Preprocessing Done")
print("Number of Features: "+str(len(X[0])))
print("Number of Labels: "+ str(len(y[0])))
print("Number of examples in dataset:"+str(len(X)))
print("X,y stored in fdataX.npy and flabels.npy respectively")
