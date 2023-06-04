from abc import abstractmethod

import torch

import global_config
from config.network_config import ConfigHolder
from model import ffa_gan
from model import vanilla_cycle_gan as cycle_gan
from model import unet_gan
from model import usi3d_gan

class NetworkCreator():
    def __init__(self, gpu_device):
        self.gpu_device = gpu_device

    def initialize_img2img_network(self):
        config_holder = ConfigHolder.getInstance()
        network_config = config_holder.get_network_config()

        net_config = network_config["model_type"]
        input_nc = network_config["input_nc"]
        num_blocks = network_config["num_blocks"]
        dropout_rate = network_config["dropout_rate"]

        D_A = cycle_gan.Discriminator(input_nc=3).to(self.gpu_device)  # use CycleGAN's discriminator

        if (net_config == 1):
            G_A = cycle_gan.Generator(input_nc=input_nc, output_nc=3, n_residual_blocks=num_blocks, dropout_rate=dropout_rate, use_cbam=network_config["use_cbam"], norm=network_config["norm_mode"]).to(self.gpu_device)
        elif (net_config == 2):
            G_A = unet_gan.UnetGenerator(input_nc=input_nc, output_nc=3, num_downs=num_blocks).to(self.gpu_device)
        elif (net_config == 3):
            print("Using AdainGEN")
            params = {'dim': 64,  # number of filters in the bottommost layer
                      'mlp_dim': 256,  # number of filters in MLP
                      'style_dim': 8,  # length of style code
                      'n_layer': 3,  # number of layers in feature merger/splitor
                      'activ': 'relu',  # activation function [relu/lrelu/prelu/selu/tanh]
                      'n_downsample': 2,  # number of downsampling layers in content encoder
                      'n_res': network_config["num_blocks"],  # number of residual blocks in content encoder/decoder
                      'pad_type': 'reflect'}
            G_A = usi3d_gan.AdaINGen(input_dim=3, output_dim=3, params=params).to(self.gpu_device)
        else:
            G_A = ffa_gan.FFA(input_nc, num_blocks, dropout_rate=dropout_rate).to(self.gpu_device)

        return G_A, D_A

    def initialize_parsing_network(self, input_nc):
        G_P = unet_gan.UNetClassifier(num_channels=input_nc, num_classes=2).to(self.gpu_device)

        return G_P

class AbstractIIDTrainer():
    def __init__(self, gpu_device):
        self.gpu_device = gpu_device

    @abstractmethod
    def initialize_train_config(self):
        pass

    @abstractmethod
    def initialize_dict(self):
        # what to store in visdom?
        pass

    @abstractmethod
    def is_stop_condition_met(self):
        pass

    @abstractmethod
    def visdom_plot(self, iteration):
        pass

    @abstractmethod
    def visdom_infer(self, input_map):
        pass

    @abstractmethod
    def load_saved_state(self):
        pass

    @abstractmethod
    def save_states(self, epoch, iteration, is_temp:bool):
        pass
