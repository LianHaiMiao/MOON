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

        # extra topic data
        self.topic_path = './data/topic_data.hdf5'
        # train and test data path for tagging model
        self.train_path = './data/v1_train_data.txt'
        self.test_path = './data/v1_test_data.txt'
        self.v2h = './data/train_test_dataset.txt'

        self.datalen = 27911 # video number
        self.hashtag_num = 503 #hash tag number

        # the parameters we need to change is following:
        #
        #
        # presentation model
        self.batch_size = 128

        self.use_gpu = True

        self.presentation_lr = 0.001

        self.presentation_epoch = 10

        # tagging model
        self.tagging_size = self.common_size * 4

        self.negative_num = 4

        self.topk = 10

        self.tagging_epoch = 30

        self.tagging_lr = 0.001


