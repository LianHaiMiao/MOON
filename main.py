from config import Config
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils.util import Helper






if __name__ == '__main__':
    # 实例化 参数类
    config = Config()
    # 实例化 工具类
    helper = Helper()

    # 主程序入口
    lstm = nn.LSTM(config.text_in, config.text_h, batch_first=True)
    print("batch size is:", config.batch_size)
    print("input size is:", config.text_in)
    print("hidden size is:", config.text_h)
    print("seq_len is:", config.text_seq)


    print(lstm)
    # initial data
    h0, c0 = helper.init_hidden(config.batch_size, config.text_h)
    print(h0.size())
    print(c0.size())

    print("inputdata: \n")
    # input data
    data = Variable(torch.randn(config.batch_size, config.text_seq, config.text_in)) # batch*seq_len*input_size
    print(data.size())

    print("result: \n")
    output, hn = lstm(data, (h0, c0))

    print("output: \n")
    print(output.size())

    print("hn[0]: \n")
    print(hn[0].size())
    print("hn[1]: \n")
    print(hn[1].size())

    # b*s*f

    a = torch.FloatTensor([[[2, 2, 2, 2], [3, 3, 3, 3], [1, 1, 1, 1]], [[1, 2, 1, 2], [2, 3, 3, 2], [1, 2, 3, 3]]])

    print(a)

    b = torch.randn((2, 3, 1))

    print(b)

    c = torch.mul(b, a)

    print(c)

    sum_c = torch.sum(c, 1)

    print(sum_c)