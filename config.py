class Config(object):
    def __init__(self):
        # model parameters
        self.img_h = 300
        self.audio_h = 80
        self.text_h = 500

        self.img_in = 4096
        self.audio_in = 512
        self.text_in = 300

        self.img_seq = 12
        self.audio_seq = 6
        self.text_seq = 10


        # structure parameters
        self.batch_size = 128
        self.use_gpu = True
        self.lr = 0.1
        self.epoch_num = 30


        # hyper-parameters



