import os
import torch


class Config(object):
    def __init__(self):
        self.text_input = 300
        self.audio_input = 5
        self.video_input = 20
        self.video_output = 20  # 补
        self.device = torch.device('cuda')
        self.batch_size = 12
        self.align_length = 50
        self.LSTM_hidden = 128
        self.text_out_size = 64
        self.fusionLine = 64
        self.LSTM_dropout = 0.3
        self.DNNdrop = 0.5
        self.fusionDrop = 0.45
        self.squeezeLineOutput = 1
        self.DNN_V_hidden_size = 16
        self.DNN_A_hidden_size = 8
        self.MAX_EPOCH = 1000000
        self.manual_seed = 123
        self.manual_seed_all = 123
        self.patience = 8
        self.num_trials = 3
        self.train_size = 1284
        self.valid_size = 686

    def print_config(self):
        for key in self.__dict__:
            print(key, end=' = ')
            print(self.__dict__[key])


if __name__ == '__main__':
    config = Config()
    config.print_config()
