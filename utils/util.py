import torch
from torch.autograd import Variable
import numpy as np
import math

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

    def expend(self, data, hashtag_num):
        return data.view(data.data.size()[0], 1, data.data.size()[1]).expand(data.data.size()[0], hashtag_num, data.data.size()[1])


    def count_precision(self, pred_y, true_label, k=1):
        value, ranklist = torch.topk(pred_y, k)
        count = 0
        ranklist = ranklist.cpu().numpy()
        for j in ranklist:
            if j in true_label:
                count += 1
        p = float(count) / float(k)
        return p


    def count_recall(self, pred_y, true_label, k=1):
        value, ranklist = torch.topk(pred_y, k)
        count = 0
        ranklist = ranklist.cpu().numpy()
        for j in ranklist:
            if j in true_label:
                count += 1
        r = float(count) / float(len(true_label))
        return r

    def count_NDCG(self, pred_y, true_label, k=1):
        value, ranklist = torch.topk(pred_y, k)
        count = 1
        ranklist = ranklist.cpu().numpy()
        for j in ranklist:
            if j in true_label:
                return math.log(2) / math.log(count + 1)
            count += 1
        return 0


    # def get_rank_list(self, pred_y, k):
    #     temp = np.argsort(pred_y)
    #     ranklist = temp[-k:]
    #     return ranklist


