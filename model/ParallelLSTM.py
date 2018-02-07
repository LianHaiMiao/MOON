import torch
from torch import nn
from torch.nn import functional as F

class MOON(nn.Module):
    def __init__(self, img_in, img_h, au_in, au_h, text_in, text_h, common_size):
        super(MOON, self).__init__()
        self.ilstm = ImageLSTM(img_in, img_h)
        self.alstm = AudioLSTM(au_in, au_h)
        self.tlstm = TextLSTM(text_in, text_h)
        # map into the common space
        self.ilinear = nn.Linear(img_h, common_size)
        self.alinear = nn.Linear(au_h, common_size)
        self.tlinear = nn.Linear(text_h, common_size)

    def forward(self, image, audio, text):
        # lstm + attention
        h_i = self.ilstm(image) # batch*img_h
        h_a = self.alstm(audio) # batch*au_h
        h_t = self.tlstm(text) # batch*text_h
        # map into the common space
        x_i = self.ilinear(h_i) # batch*common_size
        x_a = self.alinear(h_a) # batch*common_size
        x_t = self.tlinear(h_t) # batch*common_size
        # return three kind of features
        return x_i, x_a, x_t

# Image LSTM
class ImageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(ImageLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attentionLayer = AttentionNet(self.hidden_size)

    def forward(self, x):
        out, _ = self.ilstm(x) # x:[batch*seq*feature], out:[batch*seq*feature]
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
        self.attentionLayer = AttentionNet(self.hidden_size)

    def forward(self, x):
        out, _ = self.alstm(x) # x:[batch*seq*feature], out:[batch*seq*feature]
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
        self.attentionLayer = AttentionNet(self.hidden_size)

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
