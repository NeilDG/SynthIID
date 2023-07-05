from pathlib import Path

import kornia.metrics.psnr
import torchvision.utils

from config import network_config
from config.network_config import ConfigHolder
from trainers import abstract_iid_trainer
import global_config
import torch
import torch.cuda.amp as amp
import itertools
from model.modules import image_pool
from utils import plot_utils, tensor_utils
import torch.nn as nn
import numpy as np
from trainers import paired_trainer
from testers import test_metrics

class PairedTester():
    def __init__(self, gpu_device, network_name, iteration):
        self.gpu_device = gpu_device
        self.network_name = network_name
        self.iteration = iteration
        self.trainer = paired_trainer.PairedTrainer(self.gpu_device, self.network_name, self.iteration)
        self.trainer.load_best_state()
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')

        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()

        self.l1_results = []
        self.rmse_results = []
        self.ssim_results = []

    def measure_and_store(self, input_map, a_key, b_key):
        use_tanh = ConfigHolder.getInstance().get_network_attribute("use_tanh", False)
        target_like = self.trainer.test(input_map, a_key)
        target = input_map[b_key]

        if(use_tanh):
            target_like = tensor_utils.normalize_to_01(target_like)
            target = tensor_utils.normalize_to_01(target)

        l1_result = self.l1_loss(target_like, target).cpu()
        self.l1_results.append(l1_result)

        rmse_result = test_metrics.torch_rmse(target_like, target).cpu()
        self.rmse_results.append(rmse_result)

        ssim_result = test_metrics.ssim_measure(target_like, target).cpu()
        self.ssim_results.append(ssim_result)

    def visualize_results(self, input_map, a_key, b_key, dataset_title):
        version_name = network_config.ConfigHolder.getInstance().format_version_name(self.network_name, self.iteration)
        self.trainer.visdom_visualize(input_map, a_key, b_key, "Test - " + version_name + " " + dataset_title)

    def save_image_set(self, file_names, input_map, a_key, b_key, dataset_title):
        version_name = network_config.ConfigHolder.getInstance().format_version_name(self.network_name, self.iteration)
        SAVE_PATH = "./results/" + version_name + "/" + dataset_title + "/"
        try:
            path = Path(SAVE_PATH + "/input/")
            path.mkdir(parents=True)

            path = Path(SAVE_PATH + "/target/")
            path.mkdir(parents=True)

            path = Path(SAVE_PATH + "/target-like/")
            path.mkdir(parents=True)
        except OSError as error:
            pass
            # print(SAVE_PATH + " already exists. Skipping.", error)

        generated = self.trainer.test(input_map, a_key)
        input = input_map[a_key]
        ground_truth = input_map[b_key]
        for i in range(0, len(file_names)):
            img_save_file = SAVE_PATH + "/input/" + file_names[i] + ".png"
            torchvision.utils.save_image(input[i], img_save_file)

            img_save_file = SAVE_PATH + "/target/" + file_names[i] + ".png"
            torchvision.utils.save_image(ground_truth[i], img_save_file)

            img_save_file = SAVE_PATH + "/target-like/" + file_names[i] + ".png"
            torchvision.utils.save_image(generated[i], img_save_file)

        # print("Saved batch of images of size " + str(len(file_names)))


    def report_metrics(self, dataset_title):
        version_name = network_config.ConfigHolder.getInstance().format_version_name(self.network_name, self.iteration)

        l1_mean = np.round(np.float32(np.mean(self.l1_results)), 4) #ISSUE: ROUND to 4 sometimes cause inf
        self.l1_results.clear()

        rmse_mean = np.round(np.mean(self.rmse_results), 4)
        self.rmse_results.clear()

        ssim_mean = np.round(np.mean(self.ssim_results), 4)
        self.ssim_results.clear()

        last_epoch = global_config.last_epoch
        self.visdom_reporter.plot_text(dataset_title + " Results - " + version_name + " Last epoch: " + str(last_epoch) + "<br>"
                                       "Abs Rel: " + str(l1_mean) + "<br>"
                                       "RMSE: " + str(rmse_mean) + "<br>"
                                       "SSIM: " +str(ssim_mean) +"<br>")