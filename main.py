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

    for epoch in range(config.presentation_epoch):
        total_loss = 0

        begin_time = time.time()

        for i, (i_data, a_data, t_data) in enumerate(mdataset.presentationTrain(config.batch_size)):
            i_data = helper.to_var(i_data, config.use_gpu) # batch*seq*feature
            a_data = helper.to_var(a_data, config.use_gpu) # batch*seq*feature
            t_data = helper.to_var(t_data, config.use_gpu) # batch*seq*feature

            optimizer.zero_grad()

            modal_i, modal_a, modal_t = moon(i_data, a_data, t_data)

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

            i_data = helper.to_var(i_data, config.use_gpu) # batch*common_size
            a_data = helper.to_var(a_data, config.use_gpu)
            t_data = helper.to_var(t_data, config.use_gpu)

            # get the common space feature
            i_data, a_data, t_data = moon(i_data, a_data, t_data)

            p_topic = helper.to_var(p_topic, config.use_gpu)
            n_topic = helper.to_var(n_topic, config.use_gpu)

            optimizer.zero_grad()

            pos_prediction = tag_model(i_data, a_data, t_data, p_topic)
            neg_prediction = tag_model(i_data, a_data, t_data, n_topic)

            loss = torch.mean((pos_prediction - neg_prediction - 1) ** 2)

            total_loss += loss.data[0]

            loss.backward()
            optimizer.step()

        # end a epoch
        print('the loss of model Epoch[%d / %d]: is %.4f , time: %d s' % (epoch, config.tagging_epoch, total_loss, time.time() - begin_time))
        if epoch == 0:
            least_loss = total_loss

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

    tag_model.eval()

    p_score = []
    r_score = []

    # eval one by one
    for i, (i_data, a_data, t_data, all_tat_feature, true_label) in enumerate(mdataset.getTestBatch()):

        i_data = helper.to_var(i_data, config.use_gpu)  # hash_tag_num*common_size
        a_data = helper.to_var(a_data, config.use_gpu)
        t_data = helper.to_var(t_data, config.use_gpu)

        # get the common space feature
        i_data, a_data, t_data = moon(i_data, a_data, t_data)

        all_tat_feature = helper.to_var(all_tat_feature, config.use_gpu)

        prediction = tag_model(i_data, a_data, t_data, all_tat_feature)

        prediction = torch.squeeze(prediction, 1)

        p_score.append(helper.count_precision(prediction.data.cpu().numpy(), true_label, config.topk))
        r_score.append(helper.count_recall(prediction.data.cpu().numpy(), true_label, config.topk))

    precision = np.mean(p_score)
    recall = np.mean(r_score)

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

    # train presentation model
    # train_presentation_model(config, helper, moon, mdataset)

    # initial tagging model
    tag_model = Tagging(config.tagging_size)

    if config.use_gpu:
        tag_model = tag_model.cuda()
    # train tagging model
    # train_tagging_model(config, helper, moon, tag_model, mdataset)

    # evaluation
    # 我们可以训练完之后直接进行测试，也可以每次训练的时候进行测试，都可以
    evaluation_model(config, helper, moon, tag_model, mdataset)

