import torch
from torch import nn
from torch.nn import functional as F







# Image LSTM
class ImageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(ImageLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attentionLayer = AttentionNet(self.input_size)

    def forward(self, x):
        out, _ = self.tlstm(x) # x:[batch*seq*feature], out:[batch*seq*feature]
        attention = self.attentionLayer(out) # attention:[batch*seq*1]
        h = torch.mul(attention, out) # h:[batch*seq*feature]
        sum_h = torch.sum(h, 1) # sum_h:[batch*feature]
        return sum_h

# Audio LSTM
class AudioLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(AudioLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attentionLayer = AttentionNet(self.input_size)

    def forward(self, x):
        out, _ = self.tlstm(x) # x:[batch*seq*feature], out:[batch*seq*feature]
        attention = self.attentionLayer(out) # attention:[batch*seq*1]
        h = torch.mul(attention, out) # h:[batch*seq*feature]
        sum_h = torch.sum(h, 1) # sum_h:[batch*feature]
        return sum_h

# Text LSTM
class TextLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(TextLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tlstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attentionLayer = AttentionNet(self.input_size)

    def forward(self, x):
        out, _ = self.tlstm(x) # x:[batch*seq*feature], out:[batch*seq*feature]
        attention = self.attentionLayer(out) # attention:[batch*seq*1]
        h = torch.mul(attention, out) # h:[batch*seq*feature]
        sum_h = torch.sum(h, 1) # sum_h:[batch*feature]
        return sum_h




# attention layer
class AttentionNet(nn.Module):
    def __init__(self, input_size):
        super(AttentionNet, self).__init__()
        self.linear = nn.Sequential(
                nn.Linear(input_size, input_size//2),
                nn.ReLU(),
                nn.Linear(input_size//2, input_size//4),
                nn.ReLU(),
                nn.Linear(input_size//4, 1)
            )

    def forward(self, x):
        out = self.linear(x)
        return out





