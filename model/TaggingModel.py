import torch
from torch import nn
from torch.nn import functional as F

class Tagging(nn.Module):
    def __init__(self, input_size):
        super(Tagging, self).__init__()
        self.predict = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, 1)
        )

    def forward(self, i_data, a_data, t_data, topic_data):
        # i_data: batch*common_size
        # we should concatenate the data
        x = torch.cat((i_data, a_data, t_data, topic_data), dim=1) # batch*(4common_size)
        out = self.predict(x)
        return out


