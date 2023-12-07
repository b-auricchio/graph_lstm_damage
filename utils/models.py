import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import SAGEConv, BatchNorm, to_hetero, Linear
from torch import nn
import torch

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_features, num_layers):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), out_features)
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(self.conv1)
        for i in range(num_layers-2):
            self.convs.append(SAGEConv((-1, -1), hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))

        self.convs.append(self.conv3)

    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index).relu()
        for i in range(1, self.num_layers-1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i-1](x)
            x = F.relu(x)
        
        x = self.convs[-1](x,edge_index)
        
        return x

class LSTMRegressor(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, device, num_layers=1):
        super().__init__()
        self.hidden_features = hidden_features
        self.lstm = nn.LSTM(in_features, hidden_features, num_layers, batch_first=True)
        self.lin = nn.Linear(hidden_features, out_features)
        self.num_layers = num_layers
        self.device = device

    def forward(self, batch):
        batch_size = len(batch)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_features, requires_grad=True).to(self.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_features, requires_grad=True).to(self.device)
        _, (h_f, c_f) = self.lstm(batch, (h_0, c_0))

        out = self.lin(h_f[-1])
        return out

class GraphLSTM(nn.Module):
    def __init__(self, enc_hidden, enc_out, lstm_hidden, out_features, sequence_length, data, device, num_layers_enc=4, num_layers_lstm=2):
        super().__init__()
        self.encoder = GNNEncoder(enc_hidden, enc_out, num_layers=num_layers_enc)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='mean').to(device)
        
        with torch.no_grad():  # Initialize lazy modules.
            self.encoder.eval()
            _ = self.encoder(data.x_dict, data.edge_index_dict)

        self.layers_enc = num_layers_enc
        self.layers_lstm = num_layers_lstm

        self.enc_out = enc_out
        self.sequence_length = sequence_length
        self.lstm = LSTMRegressor(enc_out, lstm_hidden, out_features, device=device, num_layers=num_layers_lstm).to(device)

    def forward(self, batch):
        enc = self.encoder(batch.x_dict, batch.edge_index_dict)
        out = sum(global_mean_pool(enc[v], batch.batch_dict[v]) for v in batch.node_types)#.flatten(0)  # [batch_size, hidden_features]
        out = out.reshape(-1, self.sequence_length, self.enc_out) # x: [N, L, H]

        out = self.lstm(out)

        return out