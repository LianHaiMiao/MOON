from config import Config
import h5py
import torch
import numpy as np

class MircoDataset():
    """
        初始化数据的工作
    """
    def __init__(self):
        config = Config()
        self.lens = config.datalen
        self.test_path = config.test_path
        self.train_path = config.train_path  # 训练数据集读取的地方
        self.negative_num = config.negative_num  # 训练过程中的 negative num
        self.hash_tag_num = config.hashtag_num
        self.v2h_list = self.readDict(config.v2h)
        self.img_data = h5py.File(config.image_path, 'r')
        self.audio_data = h5py.File(config.audio_path, 'r')
        self.text_data = h5py.File(config.text_path, 'r')
        self.t_v_id_input, self.t_p_tag_input, self.t_n_tag_input = self.get_train_instances(self.train_path, self.negative_num)
        self.test_video_id_input, self.test_v2h_dict = self.get_test_instances(self.test_path)


    def presentationTrain(self, batch_size):
        """
        :param batch_size: int
        :return: img_feature, audio_feature, text_feature (Tensor)
        """
        sindex = 0
        eindex = batch_size
        presentation_data_list = [str(i) for i in range(self.lens)]
        while eindex < self.lens:
            batch = presentation_data_list[sindex:eindex]
            img_feature, audio_feature, text_feature = self.getBatchDataById(batch)
            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield img_feature, audio_feature, text_feature

        if eindex >= self.lens:
            batch = presentation_data_list[sindex:]
            img_feature, audio_feature, text_feature = self.getBatchDataById(batch)
            yield img_feature, audio_feature, text_feature


    # get train data
    def getTrainBatch(self, batch_size):
        """
        :param batch_size: int
        :return:  img_feature, audio_feature, text_feature (Tensor)
        """
        video_id_input, pos_tag_input, neg_tag_input = self.t_v_id_input, self.t_p_tag_input, self.t_n_tag_input
        sindex = 0
        eindex = batch_size
        while eindex < len(video_id_input):
            main_batch = video_id_input[sindex:eindex]
            pos_batch = pos_tag_input[sindex:eindex]
            neg_batch = neg_tag_input[sindex:eindex]
            # get the main feature
            img_feature, audio_feature, text_feature = self.getBatchDataById(main_batch)
            # get tag id
            pos_topic_feature = [int(j) for j in pos_batch]
            neg_topic_feature = [int(j) for j in neg_batch]
            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield img_feature, audio_feature, text_feature, pos_topic_feature, neg_topic_feature

        if eindex >= len(video_id_input):
            main_batch = video_id_input[sindex:]
            pos_batch = pos_tag_input[sindex:]
            neg_batch = neg_tag_input[sindex:]
            img_feature, audio_feature, text_feature = self.getBatchDataById(main_batch)
            pos_topic_feature = [int(j) for j in pos_batch]
            neg_topic_feature = [int(j) for j in neg_batch]
            yield img_feature, audio_feature, text_feature, pos_topic_feature, neg_topic_feature

    # get test data to evaluate
    def getTestBatch(self, batch_size):
        video_id_input, v2h_dict = self.test_video_id_input, self.test_v2h_dict
        sindex = 0
        eindex = batch_size # count by batch
        hash_tag_list = [int(i) for i in range(self.hash_tag_num)]  # get all hash tag id

        while eindex < len(video_id_input):
            main_batch = video_id_input[sindex:eindex]  # get the video id
            true_label = [v2h_dict[l] for l in main_batch]
            img_feature, audio_feature, text_feature = self.getBatchDataById(main_batch)
            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield img_feature, audio_feature, text_feature, hash_tag_list, true_label

        if eindex >= len(video_id_input):
            main_batch = video_id_input[sindex:]  # get the video id
            true_label = [v2h_dict[l] for l in main_batch]
            img_feature, audio_feature, text_feature = self.getBatchDataById(main_batch)
            yield img_feature, audio_feature, text_feature, hash_tag_list, true_label

    # construct test data
    def get_test_instances(self, test_path):
        """
        function: get the test data index, and make the form of (test_data, hash_Tag)
        :param test_path: String
        :param hash_tag_num: int
        :return:
        """
        video_id_input = []  # string list
        t_dict = {}
        with open(test_path, 'r') as fr:
            line = fr.readline()
            while line != None and line != "":
                arr = line.strip().split(":")
                video_id_input.append(arr[0])
                if arr[0] in t_dict:
                    t_dict[arr[0]].append(int(arr[1]))
                else:
                    t_dict[arr[0]] = [int(arr[1])]
                # read the next line
                line = fr.readline()
        video_id_set = list(set(video_id_input))
        return video_id_set, t_dict

    def get_train_instances(self, train_path, negative_num):
        """
        function: get the train data index
        :param train_path: String
        :param negtive_num: int
        :return:
        """
        video_id_input, pos_tag_input, neg_tag_input = [], [], []  # string list
        with open(train_path, 'r') as fr:
            line = fr.readline()
            while line != None and line != "":
                arr = line.strip().split(":")
                video_id = arr[0]
                for _ in range(negative_num):
                    pos_tag_input.append(arr[1])
                for _ in range(negative_num):
                    j = str(np.random.randint(self.hash_tag_num))
                    # 如果重复了就重新获取
                    while (video_id, j) in self.v2h_list:
                        j = str(np.random.randint(self.hash_tag_num))
                    # 得到训练数据
                    video_id_input.append(video_id)
                    neg_tag_input.append(j)
                # read the next line
                line = fr.readline()
        return video_id_input, pos_tag_input, neg_tag_input

    def readDict(self, path):
        v2h_list = []
        with open(path, 'r') as fr:
            line = fr.readline()
            while line != None and line != "":
                arr = line.strip().split(":")
                v = arr[0]
                h = arr[1].split(",")
                for i in h:
                    v2h_list.append((v, i))
                line = fr.readline()
        return v2h_list

    # get data feature
    def getBatchDataById(self, batch):
        """
        :param batch: String list
        :return: img_feature, audio_feature, text_feature (Tensor)
        """
        i_fe = []
        a_fe = []
        t_fe = []
        for key in batch:
            i_fe.append(torch.from_numpy(self.img_data[key].value))
            a_fe.append(torch.from_numpy(np.abs(self.audio_data[key].value)))
            t_fe.append(torch.from_numpy(self.text_data[key].value.astype(np.float32)))

        img_feature = torch.stack(i_fe, dim=0)
        audio_feature = torch.stack(a_fe, dim=0)
        text_feature = torch.stack(t_fe, dim=0)
        return img_feature, audio_feature, text_feature

    def __len__(self):
        return self.lens





