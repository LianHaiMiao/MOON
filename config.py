class Config(object):
    def __init__(self):
        # model parameters
        self.img_h = 500
        self.audio_h = 300
        self.text_h = 80

        self.img_in = 4096
        self.audio_in = 512
        self.text_in = 300

        self.img_seq = 12
        self.audio_seq = 6
        self.text_seq = 10

        self.common_size = 150

        # data parameters
        self.text_path = './data/feature_data/v1_texts_feature.hdf5'
        self.image_path = './data/feature_data/v1_images_feature.hdf5'
        self.audio_path = './data/feature_data/v1_audio_feature.hdf5'

        # train and test data path for tagging model
        self.train_path = './data/v1_train_data.txt'
        self.test_path = './data/v1_test_data.txt'
        self.dev_path = './data/v1_dev_data.txt'
        self.v2h = './data/train_test_dataset.txt'

        self.datalen = 86901 # train video number
        self.hashtag_num = 10677 # hash tag number

        self.all_epoch = 10

        # the parameters we need to change is following:
        #
        # presentation model
        self.train_batch_size = 128

        self.test_batch_size = 64

        self.use_gpu = True

        self.presentation_lr = 0.0001

        self.presentation_epoch = 2

        # tagging model

        self.embed_size = self.common_size

        self.tagging_size = self.common_size * 3

        self.negative_num = 4

        self.topk = 10

        self.tagging_epoch = 5

        self.tagging_lr = 0.0005


