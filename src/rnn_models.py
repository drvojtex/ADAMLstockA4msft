import torch.nn as nn
import torch

from .customLSTM import CustomLSTM

# --- RNN Model Definition ---
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x, h=None):
        out, h = self.rnn(x, h)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1), h
    
# --- LSTM Model Definition ---
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x, h=None):
        out, h = self.lstm(x, h)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1), h
    
# --- Finall Model Definition ---
class FinallModel(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type = "LSTM", num_layers=2, dropout = 0.0, out_dim = 1):
        super().__init__()
        
        # Encode date one hot encoding
        date_encoded_size = 5
        self.date_linear = nn.Linear(in_features=22, out_features=date_encoded_size)

        rnn_input_size = input_size - 22 + date_encoded_size
        if(rnn_type == "LSTM"):
            self.rnn = nn.LSTM(rnn_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif(rnn_type == "GRU"):
            self.rnn = nn.GRU(rnn_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif(rnn_type == "RNN"):
            self.rnn = nn.RNN(rnn_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        elif(rnn_type == "customLSTM"):
            self.rnn = CustomLSTM(rnn_input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError(f"Unknown rnn type = {rnn_type}")
        
        # Out linear (predict return)
        self.out_linear = nn.Linear(hidden_size, out_dim)

    
    def forward(self, x, h=None):
        # split
        hist_returns = x[:, :, :-22] 
        date_one_hot = x[:, :, -22:]

        # encode date
        date_encoded = self.date_linear(date_one_hot)

        # concatenate
        x = torch.cat((hist_returns, date_encoded), dim=2)

        # RNN
        out, h = self.rnn(x, h)

        # predict return
        out = out[:, -1, :] 
        out = self.out_linear(out)

        return out, h
