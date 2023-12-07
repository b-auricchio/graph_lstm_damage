import torch 
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from torch.utils.data import random_split
from utils.meshing import Mesh3DStructured
from utils.dataset import TemporalDataset, HeteroDataset
from torch_geometric.data import Batch
import os 
import numpy as np
from utils.models import GraphLSTM
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import time

export_root = './export/'

parser = argparse.ArgumentParser(prog='Train GNN')

parser.add_argument('-g', dest='gen', help='Generate .dat file (otherwise load)', action='store_true')
parser.add_argument('-c', dest='copy',help='Give model directory to load before training starts (default None)', default=None, type=str)
parser.add_argument('-e', dest='epochs',default=100, type=int, help='Number of training epochs (default 100)')
parser.add_argument('-l', dest='lr', default=1e-4, help='Training learning rate (default 1e-4)')
parser.add_argument('--num_enc', default=4, type=int, help='Number of layers in GNN encoder (default 4)')
parser.add_argument('--num_lstm', default=2, type=int, help='Number of layers in LSTM (default 2)')


if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

print(f'Using {device} device')

args = parser.parse_args()

root = 'data.dat' # Dataset .dat file name stored in ./export/


root = os.path.join(export_root, root)
if args.gen:
    dataset_combined = None
    amps = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5]
    for i, folder in enumerate(sorted(os.listdir('./data/'))):
        print(f'\Amplitude: {amps[i]}')
        dataset = TemporalDataset()
        dataset.load_csv(os.path.join('./data/', folder), (10,10,24), Mesh3DStructured, HeteroDataset, angle=amps[i], transform=T.ToUndirected())
        if dataset_combined is None:
            dataset_combined = dataset
        else:
            dataset_combined = torch.utils.data.ConcatDataset([dataset_combined, dataset])

    torch.save(dataset_combined, root)
    print(f'Saved .dat file to {root}')
else:
    print(f'Loading {root}', end='... ')
    dataset_combined = torch.load(root, map_location=device)
    print('loaded')


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

model = GraphLSTM(128, 16, 32, 2, sequence_length=sequence_length, data=sequence[0], device=device, num_layers_enc=args.num_enc, num_layers_lstm=args.num_lstm).to(device)

if args.copy is not None:
    try:
        model.load_state_dict(torch.load(os.path.join(args.copy), map_location=device))
        print(f'Loaded model at {args.copy}')
    except:
        print(f'Error: {args.copy} is not a valid .pt file')

criterion = torch.nn.MSELoss()
lr = args.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=4e-5)

writer = SummaryWriter()

n_epochs = args.epochs

loss = None
model.train()
try:
    for i in range(n_epochs):
        for batch in tqdm(train_loader, leave=False, desc=f'Epoch {i+1}, Loss: {loss}'):
            batch.to(device)
            optimizer.zero_grad()
            out = model(batch).squeeze(1)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()

        writer.add_scalar('Loss/train', loss, i+1)

    torch.save(model.state_dict(), os.path.join(export_root, f'model{model.layers_enc}_{model.layers_lstm}_lr{lr}.pt'))
    print(f'model{model.layers_enc}_{model.layers_lstm}_lr{lr}.pt saved!')
except KeyboardInterrupt:
    torch.save(model.state_dict(), os.path.join(export_root, f'model{model.layers_enc}_{model.layers_lstm}_lr{lr}.pt'))
    print(f'model{model.layers_enc}_{model.layers_lstm}_lr{lr}.pt saved!')