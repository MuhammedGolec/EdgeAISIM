import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree



class MyGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, num_layers=3, hidden_channels=64):
        super(MyGNN, self).__init__(aggr='mean')
        self.lin1 = nn.Linear(in_channels, hidden_channels)                                   #Linear input layer
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_channels, hidden_channels) for _ in range(num_layers - 1)        #Linear hidden layers
        ])
        self.lin2 = nn.Linear(hidden_channels, out_channels)                                  #Linear ouptu layer

    def forward(self, x, edge_index):
        x = self.lin1(x)                                                                      #Pass thorugh input layer
        for hidden_layer in self.hidden_layers:                                               #Use relu actiavtion though hidden ayers
            x = hidden_layer(x).relu()
        x = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)                      #Use message propogation and aggregate information from 
        x = self.lin2(x)                                                                      #adjacent nodes, and pass though final output layer
        return x

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out