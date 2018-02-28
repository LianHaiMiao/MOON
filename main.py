from config import Config
import h5py
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from utils.util import Helper
from model.ParallelLSTM import MOON
from model.TaggingModel import Tagging
from dataset import MircoDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import numpy as np

# presentation loss
def p_loss_fn(modal_i, modal_a, modal_t):
    loss = F.mse_loss(modal_i, modal_a) + F.mse_loss(modal_a, modal_t) + F.mse_loss(modal_i, modal_t)
    return loss


def train_presentation_model(config, helper, moon, mdataset):

    # define loss and optimizer
    # loss is p_loss_fn
    optimizer = optim.Adam(moon.parameters(), config.presentation_lr)

    moon.train()

    print("开始训练 presentation model")

    for epoch in range(config.presentation_epoch):
        total_loss = 0

        begin_time = time.time()

        for i, (i_data, a_data, t_data) in enumerate(mdataset.presentationTrain(config.batch_size)):
            i_data = helper.to_var(i_data, config.use_gpu) # batch*seq*feature
            a_data = helper.to_var(a_data, config.use_gpu) # batch*seq*feature
            t_data = helper.to_var(t_data, config.use_gpu) # batch*seq*feature

            optimizer.zero_grad()

            modal_i, modal_a, modal_t = moon(i_data, a_data, t_data) # batch*common_size

            loss = p_loss_fn(modal_i, modal_a, modal_t)

            total_loss += loss.data[0]

            loss.backward()
            optimizer.step()
        # end a epoch
        print('the loss of model Epoch[%d / %d]: is %.4f , time: %d s' % (epoch, config.presentation_epoch, total_loss, time.time() - begin_time))

        if epoch == 0:
            least_loss = total_loss
        # if the loss is the minimum, we save it

        if total_loss < least_loss:
            least_loss = total_loss
            # we save it in the bestmodel
            torch.save(moon.state_dict(), './bestmodel/moon_%d_EPOCHS_loss%.4f.pth' % (epoch, least_loss))

        # Decaying Learning Rate
        if (epoch + 1) % 5 == 0:
            config.presentation_lr /= 3
            optimizer = torch.optim.Adam(moon.parameters(), lr=config.presentation_lr)

    print("Done the train of presentation model")

    return True


def train_tagging_model(config, helper, moon, tag_model, mdataset):
    # define optimizer
    optimizer = optim.Adam(tag_model.parameters(), config.tagging_lr)

    tag_model.train()

    for epoch in range(config.tagging_epoch):

        total_loss = 0

        begin_time = time.time()

        for i, (i_data, a_data, t_data, p_topic, n_topic) in enumerate(mdataset.getTrainBatch(config.batch_size)):

            i_data = helper.to_var(i_data, config.use_gpu) # batch*seq*feature
            a_data = helper.to_var(a_data, config.use_gpu)
            t_data = helper.to_var(t_data, config.use_gpu)

            # get the common space feature
            i_data, a_data, t_data = moon(i_data, a_data, t_data)

            # p_topic = helper.to_var(p_topic, config.use_gpu)
            # n_topic = helper.to_var(n_topic, config.use_gpu)
            p_topic = helper.to_var(torch.LongTensor(p_topic), config.use_gpu)
            n_topic = helper.to_var(torch.LongTensor(n_topic), config.use_gpu)

            optimizer.zero_grad()

            pos_prediction = tag_model(i_data, a_data, t_data, p_topic)
            pos_taget = helper.to_var(torch.from_numpy(np.ones((pos_prediction.data.size())).astype(np.float32)), config.use_gpu)
            neg_prediction = tag_model(i_data, a_data, t_data, n_topic)
            neg_taget = helper.to_var(torch.from_numpy(np.zeros(neg_prediction.data.size()).astype(np.float32)), config.use_gpu)

            # pos-neg-loss
            # loss = torch.mean((pos_prediction - neg_prediction - 1) ** 2)
            # print(loss)
            # loss2 = torch.mean((pos_prediction - neg_prediction).add_(-1).pow(2))
            # print(loss2)

            # NCF cross entropy loss
            loss = F.binary_cross_entropy(pos_prediction, pos_taget) + F.binary_cross_entropy(neg_prediction, neg_taget)

            total_loss += loss.data[0]

            loss.backward()
            optimizer.step()

        # end a epoch
        print('the loss of model Epoch[%d / %d]: is %.4f , time: %d s' % (epoch, config.tagging_epoch, total_loss, time.time() - begin_time))

        if epoch == 0:
            least_loss = total_loss

        print(total_loss)

        # evaluate the precision and recall

        evaluation_model(config, helper, moon, tag_model, mdataset)

        # if the loss is the minimum, we save it
        if total_loss < least_loss:
            least_loss = total_loss
            # we save it in the bestmodel
            torch.save(tag_model.state_dict(), './tagmodel/tag_model_%d_EPOCHS_loss%.4f.pth' % (epoch, least_loss))

        # Decaying Learning Rate
        if (epoch + 1) % 5 == 0:
            config.tagging_lr /= 3
            optimizer = optim.Adam(tag_model.parameters(), config.tagging_lr)




    print("Done the train of tagging model")
    return True

def evaluation_model(config, helper, moon, tag_model, mdataset):
    begin_time = time.time()

    tag_model.eval()

    p_score = []
    r_score = []

    print("对模型进行评估咯～")
    # eval one by one
    for i, (i_data, a_data, t_data, all_tag, true_label) in enumerate(mdataset.getTestBatch(config.batch_size)):

        i_data = helper.to_var(i_data, config.use_gpu)
        a_data = helper.to_var(a_data, config.use_gpu)  # batch*seq*feature
        t_data = helper.to_var(t_data, config.use_gpu)

        # get the common space feature
        i_data, a_data, t_data = moon(i_data, a_data, t_data)  # batch*common_size


        i_data, a_data, t_data = helper.expend(i_data, config.hashtag_num), \
                                 helper.expend(a_data, config.hashtag_num), \
                                 helper.expend(t_data, config.hashtag_num)  # batch*hashtag_num*common_size

        # expend the all_tag to be batch*hashtag_num
        all_tag = torch.LongTensor(all_tag).unsqueeze(0)
        all_tag2 = all_tag.expand(i_data.data.size()[0], config.hashtag_num)

        all_tag_data = helper.to_var(all_tag2, config.use_gpu)  # batch*hashtag_num

        # loop the data

        prediction = tag_model(i_data, a_data, t_data, all_tag_data)

        prediction = torch.squeeze(prediction, 2)  # batch*hashtag_num

        # torch.topk
        temp_count = 0
        for true_l in true_label:
            p_score.append(helper.count_precision(prediction.data[temp_count], true_l, config.topk))
            r_score.append(helper.count_recall(prediction.data[temp_count], true_l, config.topk))
            temp_count += 1

    precision = np.mean(p_score)
    recall = np.mean(r_score)

    print('time consume %d s' % (time.time() - begin_time))

    print("precision is:", precision)

    print("recall is:", recall)


if __name__ == '__main__':

    # 实例化 参数类 / initial parameters class
    config = Config()

    # 实例化 工具类 / initial utils class
    helper = Helper()

    # initial presentation model
    moon = MOON(config.img_in, config.img_h, config.audio_in, config.audio_h, config
                .text_in, config.text_h, config.common_size)

    if config.use_gpu:
        moon = moon.cuda()

    # 数据初始化
    mdataset = MircoDataset()

    print("数据、模型初步构建完成")

    # train presentation model
    train_presentation_model(config, helper, moon, mdataset)

    # after train, we should choose the best one as the presentation model
    #
    #

    # initial tagging model
    tag_model = Tagging(config.tagging_size, config.hashtag_num, config.embed_size)

    if config.use_gpu:
        tag_model = tag_model.cuda()

    print("开始训练 hash tag model")
    # train tagging model
    train_tagging_model(config, helper, moon, tag_model, mdataset)

    # evaluation
    # 我们可以训练完之后直接进行测试，也可以每次训练的时候进行测试，都可以
    # evaluation_model(config, helper, moon, tag_model, mdataset)

