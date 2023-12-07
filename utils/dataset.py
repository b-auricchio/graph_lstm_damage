import torch
from torch_geometric.data import Data, HeteroData
import numpy as np
import os

class HeteroDataset: # node types: [solid, cohesive]  edge types: [in-plane, out_plane]
    def __init__(self, mesh, device, wrinkle_amp=0):
        self.mesh = mesh
        self.wrinkle_amp = wrinkle_amp
        self.device = device

    def get_data(self, i,j,k):
        data = HeteroData()
        assert k < self.mesh.dim[2]-1

        neighbours = [item for sublist in self.mesh.get_solid_neighbours(i,j,k,self.mesh.mat_strain)[0:-1] for item in sublist] # flatten list, ignore out-plane solid elements
        in_sources = list(np.arange(1,len(neighbours)+1))
        in_edge = torch.tensor([in_sources, [0 for i in range(len(in_sources))]])
        neighbours.insert(0, self.mesh.mat_strain[k][j,i])
        data['solid'].x = torch.tensor([[n, self.wrinkle_amp] for n in neighbours]).to(torch.float)

        coh_neighbour = self.mesh.get_item(i,j,k,self.mesh.delam_strain)
        data['cohesive'].x = torch.tensor([[coh_neighbour, self.wrinkle_amp]]).to(torch.float)
        out_edge = torch.tensor([[0],[0]])

        data.location = (i,j,k)
        data.y = torch.tensor([self.mesh.mat_damage[k][j,i], self.mesh.delam_damage[k][j,i]])
        data['solid','in_edge','solid'].edge_index = in_edge.long()
        data['cohesive','out_edge','solid'].edge_index = out_edge.long()

        return data.to(device=self.device)
    
    def get_dataset(self, transform=None):    
        dataset = []
        labels = []
        for i in range(self.mesh.dim[0]):
            for j in range(self.mesh.dim[1]):
                for k in range(self.mesh.dim[2]-1): #ignore final layer as no cohesive element present above 
                    data = self.get_data(i,j,k)
                    if not transform is None:
                        data = transform(data)
                    dataset.append(data)
                    labels.append(data.y)
        
        return dataset, labels

class TemporalDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.dataset = []
        self.labels = []

    def load_csv(self, folder:str, dim:tuple, Mesh, Data, angle, transform=None):
        print(f'Loading temporal dataset in {folder}: \n ----------------------')
        self.sequence_len = len(os.listdir(folder))
        dataset = np.ndarray((dim[0]*dim[1]*(dim[2]-1), self.sequence_len), dtype=np.object_)
        labels = np.ndarray((dim[0]*dim[1]*(dim[2]-1), self.sequence_len), dtype=np.object_)

        for i, file in enumerate(sorted(os.listdir(folder))):
            mesh = Mesh(os.path.join(folder, file), dim, char_length=1)
            data = Data(mesh, 'cpu', angle)
            mesh_dataset, mesh_labels = data.get_dataset(transform=transform)
            dataset[:,i] = mesh_dataset
            labels[:,i] = mesh_labels
            print(f'Loaded {file}')
        
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)