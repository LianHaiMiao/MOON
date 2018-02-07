import torch
from torch.autograd import Variable
import numpy as np
import math
import heapq

class Helper(object):
    """
        工具类： 用来提供各种工具函数
    """
    def __init__(self):
        self.timber = True

    def np2var(self, x, type=torch.FloatTensor):
        """
        :param x(numpy data), type (if we use the embedding layer, type should be torch.LongTensor)
        :return: y(Variable)
        """
        return Variable(torch.Tensor(x, type=type))


    def init_hidden(self, batch, hidden_size, num_layers=1):
        h_0 = Variable(torch.zeros(batch, num_layers, hidden_size))
        c_0 = Variable(torch.zeros(batch, num_layers, hidden_size))
        return h_0, c_0

    def to_var(self, x, use_gpu):
        x = Variable(x)
        if use_gpu:
            x = x.cuda()
        return x

    def eval_one_rating(self):