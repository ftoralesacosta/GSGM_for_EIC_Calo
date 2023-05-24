import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from sklearn.metrics import roc_curve, auc
# import torch.nn.functional as F

nevts = 100_000


# Define a custom dataset class to load data from HDF5 files
class HDF5Dataset(Dataset):
    def __init__(self, file_path, dset_name, label):
        self.file_path = file_path
        self.data = None
        self.labels = None

        with h5py.File(file_path, 'r') as hf:
            self.data = hf[dset_name][:nevts]
            if dset_name == "hcal_cells":
                print(f"Shape of Input Cell Data = {np.shape(self.data)}")
                self.data = self.data[:, :, :-1]
                print(f"Shape of Input Cell Data = {np.shape(self.data)}")
            if dset_name == "cell_features":
                self.data[:, :, 0] = 10**self.data[:, :, 0]
            if dset_name == "cluster" or dset_name == "cluster_features":
                print(f"Shape of Input Cluster  Data = {np.shape(self.data)}")
                self.data = self.data[:, 2:]
                print(f"Shape of Cluster Data After Slice= {np.shape(self.data)}")
            self.labels = np.full(np.shape(self.data)[0], label)
            print(f"L:18 Labels in DataLoader = { self.labels }")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


# Define the classifier model
# class ClassifierModel(torch.nn.Module):
#     def __init__(self):
#         super(ClassifierModel, self).__init__()
#         self.fc1 = torch.nn.Linear(200 * 5, 128)
#         self.fc2 = torch.nn.Linear(128, 64)
#         self.fc3 = torch.nn.Linear(64, 2)

#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.fc1(x.float()))  # Convert input to float type
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


class ClassifierModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(ClassifierModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x.float()))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Set random seed for reproducibility
torch.manual_seed(42)

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Load the datasets
# geant4_dataset = HDF5Dataset('geant4_data.h5')
# diffusion_dataset = HDF5Dataset('diffusion_data.h5')

geant4_dataset = HDF5Dataset('improved_200cells_FPCD.hdf5', "hcal_cells", 1)
# diffusion_dataset = HDF5Dataset('improved_200cells_FPCD.hdf5', "hcal_cells", 0)
# diffusion_dataset = HDF5Dataset('improved_200cells_FPCD.hdf5', "hcal_cells", 1)
# g4_dset = HDF5Dataset('GSGM.h5', 'particle_features', 0)
diffusion_dataset = HDF5Dataset('GSGM_MaskedMean.h5', 'cell_features', 0)

# Combine datasets
combined_dataset = torch.utils.data.ConcatDataset([geant4_dataset, diffusion_dataset])

# Split the combined dataset into train and test
train_size = int(0.8 * len(combined_dataset))
test_size = len(combined_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(combined_dataset, [train_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model and optimizer
# Get the input size from the dataset
input_size = geant4_dataset[0][0].shape[0]*geant4_dataset[0][0].shape[1]
print("INPUT SIZE = ", input_size)
hidden_size1 = 128
hidden_size2 = 64
num_classes = 2

model = ClassifierModel(input_size, hidden_size1, hidden_size2, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print(f"Inputs = { inputs }")
        # print(f"Labels = { labels }")

        # Forward pass
        # for var in range(4):
        #     print("Input dtype = ", inputs[:,:,var].dtype)
        # print("Input Shape = ", inputs.shape)

        outputs = model(inputs)
        # outputs = labels
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
scores = []
true_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        scores.extend(probs.cpu().numpy())
        true_labels.extend(labels.numpy())

# Compute ROC curve and AUC
scores = np.array(scores)
true_labels = np.array(true_labels)
fpr, tpr, _ = roc_curve(true_labels, scores)
auc_score = auc(fpr, tpr)

# Plot ROC curve or perform further analysis as needed
# (code for plotting ROC curve and computing AUC is not included here)

# Print AUC score
print(f'AUC: {auc_score:.4f}')

