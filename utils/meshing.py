import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData

class Mesh2DStructured:
    def __init__(self, path, dim):
        self.path = path
        self.dim = dim
        # Reading data
        df = pd.read_csv(path)
        damage = df.iloc[:,1].to_numpy()
        strain = df.iloc[:,0].to_numpy()

        self.damage = np.reshape(damage, (dim,dim))
        self.strain = np.reshape(strain, (dim,dim))

    # Miscellaneous functions 
    def get_neighbours(self, x, y, array): #2d
        return [array[j, i] for i,j in ((x-1,y), (x+1,y), (x,y-1), (x,y+1)) if 0<=i<self.dim and 0<=j<self.dim and array[i][j] is not None]

    def make_data(self, i, j):
        neighbours = self.get_neighbours(i, j, self.strain)
        edge_index = torch.tensor([[i+1 for i in range(len(neighbours))],
                            [0 for i in range(len(neighbours))]], dtype=torch.long)
        neighbours.insert(0, self.strain[j,i])
        x = torch.tensor([[n] for n in neighbours], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=self.damage[j,i])

    def coord_to_index(self, i, j):
        return i + j*self.dim

    def index_to_coord(self, idx):
        return idx%self.dim, idx//self.dim
    
class Mesh3DStructured:
    def __init__(self, path:str, dim:tuple, char_length=None):
        self.path = path
        self.dim = dim

        # Initialise data
        df = pd.read_csv(path)
        df = df.sort_values(by='element')
        df_solid = df[df.type == 'C3D8'] # Get only elements of type C3D8
        df_coh = df[df.type == 'COH3D8'] # Get only elements of type COH3D8

        mat_damage = df_solid['sdv8'].to_numpy()
        mat_strain = df_solid['sdv9'].to_numpy()
        if char_length is not None:
            mat_strain = np.array([s * char_length for s in mat_strain])

        delam_damage = df_coh['sdv7'].to_numpy()
        sdv4 = df_coh['sdv4'].to_numpy()
        sdv8 = df_coh['sdv8'].to_numpy()
        delam_strain = np.array([np.sqrt(s4**2 + s8**2) for s4,s8 in zip(sdv4, sdv8)])

        mat_strain_temp = mat_strain.reshape(-1, dim[2])
        mat_damage_temp = mat_damage.reshape(-1, dim[2])
        delam_strain_temp = delam_strain.reshape(-1, dim[2]-1)
        delam_damage_temp = delam_damage.reshape(-1, dim[2]-1)

        self.mat_damage = []
        self.mat_strain = []
        for i in range(len(mat_strain_temp[0])):
            self.mat_strain.append(mat_strain_temp[:,i].reshape(dim[0],dim[1]))
        for i in range(len(mat_damage_temp[0])):
            self.mat_damage.append(mat_damage_temp[:,i].reshape(dim[0],dim[1]))


        self.delam_damage = []
        self.delam_strain = []
        for i in range(len(delam_strain_temp[0])):
            self.delam_strain.append(delam_strain_temp[:,i].reshape(dim[0],dim[1]))
        for i in range(len(delam_damage_temp[0])):
            self.delam_damage.append(delam_damage_temp[:,i].reshape(dim[0],dim[1]))

    # Misc functions
    def get_solid_neighbours(self, x, y, z, array): # return [x], [y], [z]
        return [[array[k][j, i] for i,j,k in ((x-1,y,z), (x+1,y,z)) if 0<=i<self.dim[0] and 0<=j<self.dim[1] and 0<=k<self.dim[2] and array[k][j, i] is not None],
                [array[k][j, i] for i,j,k in ((x,y-1,z), (x,y+1,z)) if 0<=i<self.dim[0] and 0<=j<self.dim[1] and 0<=k<self.dim[2] and array[k][j, i] is not None],
                [array[k][j, i] for i,j,k in ((x,y,z-1), (x,y,z+1)) if 0<=i<self.dim[0] and 0<=j<self.dim[1] and 0<=k<self.dim[2] and array[k][j, i] is not None]]
    

    def get_item(self, x, y, z, array):
        if 0<=x<self.dim[0] and 0<=y<self.dim[1] and 0<=z<self.dim[2] and array[z][y, x] is not None:
            return array[z][y, x]
        
    def get_coh_neighbours(self, x, y, z, array):
        return [array[k][j, i] for i,j,k in ((x,y,z-1), (x,y,z)) if 0<=i<self.dim[0] and 0<=j<self.dim[1] and 0<=k<self.dim[2] and array[k][j, i] is not None]
    
