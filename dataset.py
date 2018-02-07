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
        self.train_path = config.train_path # 训练数据集读取的地方
        self.negative_num = config.negative_num # 训练过程中的 negative num
        self.hash_tag_num = config.hashtag_num
        self.v2h_list = self.readDict(config.v2h)
        self.img_data = h5py.File(config.image_path, 'r')
        self.audio_data = h5py.File(config.audio_path, 'r')
        self.text_data = h5py.File(config.text_path, 'r')
        self.topic_data = h5py.File(config.topic_path, 'r')

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
        :param batch_size:
        :return:  img_feature, audio_feature, text_feature (Tensor)
        """
        video_id_input, pos_tag_input, neg_tag_input = self.get_train_instances(self.train_path, self.negative_num)
        sindex = 0
        eindex = batch_size
        while eindex < len(video_id_input):
            main_batch = video_id_input[sindex:eindex]
            pos_batch = pos_tag_input[sindex:eindex]
            neg_batch = neg_tag_input[sindex:eindex]
            # get the main feature
            img_feature, audio_feature, text_feature = self.getBatchDataById(main_batch)
            # get topic feature
            pos_topic_feature = self.getBatchTopicDataById(pos_batch)
            neg_topic_feature = self.getBatchTopicDataById(neg_batch)
            temp = eindex
            eindex = eindex + batch_size
            sindex = temp
            yield img_feature, audio_feature, text_feature, pos_topic_feature, neg_topic_feature

        if eindex >= len(video_id_input):
            main_batch = video_id_input[sindex:]
            pos_batch = pos_tag_input[sindex:]
            neg_batch = neg_tag_input[sindex:]
            img_feature, audio_feature, text_feature = self.getBatchDataById(main_batch)
            pos_topic_feature = self.getBatchTopicDataById(pos_batch)
            neg_topic_feature = self.getBatchTopicDataById(neg_batch)
            yield img_feature, audio_feature, text_feature, pos_topic_feature, neg_topic_feature

    # get test data
    def getTestBatch(self):
        video_id_input, v2h_dict = self.get_test_instances(self.test_path)
        sindex = 0
        eindex = 1 # count one by one

        hash_tag_list = [str(i) for i in range(self.hash_tag_num)]
        all_tag_feature = self.getBatchTopicDataById(hash_tag_list) # get all hash tag topic feature

        while eindex < len(video_id_input):
            main_batch = video_id_input[sindex:eindex] * self.hash_tag_num
            true_label = v2h_dict[main_batch[0]]
            img_feature, audio_feature, text_feature = self.getBatchDataById(main_batch)
            temp = eindex
            eindex = eindex + 1
            sindex = temp
            print(eindex)
            yield img_feature, audio_feature, text_feature, all_tag_feature, true_label


    # construct test data
    def get_test_instances(self, test_path):
        """
        function: get the test data index, and make the form of (test_data, hash_Tag)
        :param test_path: String
        :param hash_tag_num: int
        :return:
        """
        video_id_input = [] # string list
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
        return video_id_input, t_dict

    def get_train_instances(self, train_path, negative_num):
        """
        function: get the train data index
        :param train_path: String
        :param negtive_num: int
        :return:
        """
        video_id_input, pos_tag_input, neg_tag_input = [], [], [] # string list
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

    def getBatchTopicDataById(self, batch):
        """
        :param batch: String list
        :return: topic_feature
        """
        fe = []
        for key in batch:
            fe.append(torch.from_numpy(self.topic_data[key].value))
        return torch.stack(fe, dim=0)

    def __len__(self):
        return self.lens



