import pickle
import torch
import torch.nn as nn
import numpy as np

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

#Change file path
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

#Input (size = (1, sequence lenght, 4))
random_array = torch.tensor(np.random.rand(1,20, 4),dtype = torch.float32)
print(random_array.shape)
output = model(random_array)
print(output)