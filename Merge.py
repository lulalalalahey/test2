import numpy as np

all_data = []
all_labels = []

for batch_index in range(27):  # Replace total_batches with the number of batches
    batch_data = np.load(f'original_data_batch_{batch_index}_X.npy')
    batch_labels = np.load(f'original_data_batch_{batch_index}_Y.npy')
    all_data.append(batch_data)
    all_labels.append(batch_labels)

# Concatenate all batches into a single dataset
all_data = np.concatenate(all_data, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print(f"Final data shape: {all_data.shape}")
print(f"Final labels shape: {all_labels.shape}")
