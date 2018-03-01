import torch
from torch import nn
from torch.nn import functional as F

class Tagging(nn.Module):
    def __init__(self, input_size, tag_num, embed_size):
        super(Tagging, self).__init__()
        self.predict = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.ReLU(),
            nn.Linear(input_size // 4, 1)
        )
        # embedding layer to catch the feature of tag
        self.tag_embed = nn.Embedding(tag_num, embed_size)

    def forward(self, i_data, a_data, t_data, tag_id):
        # i_data: batch*common_size
        # we should concatenate the data
        topic_data = self.tag_embed(tag_id)

        if len(topic_data.data.size()) == 2:
            # for train
            x = torch.cat((torch.mul(i_data, topic_data), torch.mul(a_data, topic_data), torch.mul(t_data, topic_data)), dim=1)
        else:
            # x = torch.cat((i_data, a_data, t_data, topic_data), dim=2)  # for evaluation
            # for evaluation
            x = torch.cat((torch.mul(i_data, topic_data), torch.mul(a_data, topic_data), torch.mul(t_data, topic_data)), dim=2)
        out = F.sigmoid(self.predict(x))
        return out