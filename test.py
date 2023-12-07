import torch 
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from torch.utils.data import random_split
from utils.meshing import Mesh3DStructured
from utils.dataset import TemporalDataset, HeteroDataset
from torch_geometric.data import Batch
import os
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import r2_score
from utils.models import GraphLSTM
import numpy as np

model_weights = 'model.pt'

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

export_root = './export/'

root = 'data.dat'
root = os.path.join(export_root, root)
print(f'Loading {os.path.join(root)}')
dataset_combined = torch.load(root, map_location=device)
print('Loaded')

train_set, test_set = random_split(dataset_combined, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

def collate(data_list):
    batch_list = []
    for seq in data_list:
        batch_list.extend(seq)

    batch = Batch.from_data_list(batch_list)
    batch.y_mat = torch.tensor([data[-1].y[0] for data in data_list], dtype=torch.float)
    batch.y_delam = torch.tensor([data[-1].y[1] for data in data_list], dtype=torch.float)
    batch.y = torch.tensor(list(zip(batch.y_mat, batch.y_delam)))
    return batch

train_loader = DataLoader(train_set, batch_size=64, shuffle=True, collate_fn=collate)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True, collate_fn=collate)

sequence = next(iter(dataset_combined))
sequence_length = len(sequence)
model = GraphLSTM(128, 16, 32, 2, sequence_length=sequence_length, data=sequence[0], device=device, num_layers_enc=4, num_layers_lstm=2).to(device)


outputs = {'matrix': [], 'delam': []}
labels = {'matrix': [], 'delam':[]}

model.load_state_dict(torch.load(os.path.join(export_root, model_weights), map_location=device))
model.eval()
for batch in test_loader:  # Iterate in batches over the training/test dataset.
    out = model(batch).squeeze(1)
    outputs['matrix'].extend(out[:,0].detach().cpu().numpy())
    labels['matrix'].extend(batch.y[:,0].cpu().numpy())
    outputs['delam'].extend(out[:,1].detach().cpu().numpy())
    labels['delam'].extend(batch.y[:,1].cpu().numpy())

outputs['matrix'] = [0 if m < 0 else 1 if m > 1 else m for m in outputs['matrix']]
outputs['delam'] = [0 if m < 0 else 1 if m > 1 else m for m in outputs['delam']]

print(f'R2 Score (Matrix): {r2_score(labels["matrix"], outputs["matrix"])}')
print(f'R2 Score (Delam): {r2_score(labels["delam"], outputs["delam"])}')


fig = plt.figure()
ax = fig.add_subplot()
plt.scatter(outputs["matrix"], labels['matrix'])
plt.scatter(outputs["delam"], labels['delam'])
plt.plot([0,1],[0,1], '--k')
ax.set_aspect('equal', adjustable='box')
plt.legend(['Matrix', 'Delamination'])
plt.xlabel('Predicted Damage $\hat{D}$')
plt.ylabel('Target Damage $D$')

plt.show()

error_mat = [np.sqrt((x-y)**2) for x,y in zip(outputs['matrix'],labels['matrix'])]
error_delam = [np.sqrt((x-y)**2) for x,y in zip(outputs['delam'],labels['delam'])]
plt.hist(error_mat, bins=15)
plt.hist(error_delam, bins=15)

error_mat_av = np.sum(error_mat)/len(error_mat)
error_delam_av = np.sum(error_delam)/len(error_delam)
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.legend(['Matrix', 'Delamination'])

plt.show()

print(f'RMS matrix error: {error_mat_av}')
print(f'RMS delam error: {error_delam_av}')