import kornia.losses

from config.network_config import ConfigHolder
from trainers import abstract_iid_trainer, best_tracker
import global_config
import torch
import torch.cuda.amp as amp
import itertools
from model.modules import image_pool
from utils import plot_utils, tensor_utils
import torch.nn as nn
import numpy as np
from trainers import early_stopper
from losses import common_losses
from loaders import transform_operations

class PairedTrainer(abstract_iid_trainer.AbstractIIDTrainer):

    def __init__(self, gpu_device, network_name, iteration):
        super().__init__(gpu_device)
        self.network_name = network_name
        self.iteration = iteration
        self.initialize_train_config()

    def initialize_train_config(self):
        config_holder = ConfigHolder.getInstance()
        network_config = config_holder.get_network_config()
        self.common_losses = common_losses.LossRepository(self.gpu_device, self.iteration)

        self.D_A_pool = image_pool.ImagePool(50)
        self.D_B_pool = image_pool.ImagePool(50)

        self.fp16_scaler = amp.GradScaler()
        self.visdom_reporter = plot_utils.VisdomReporter.getInstance()

        self.load_size = global_config.load_size
        self.batch_size = global_config.batch_size

        self.initialize_dict()
        network_creator = abstract_iid_trainer.NetworkCreator(self.gpu_device)
        self.G_A2B, self.D_B = network_creator.initialize_img2img_network()

        patch_size = config_holder.get_network_attribute("patch_size", 64)

        self.optimizerG = torch.optim.Adam(itertools.chain(self.G_A2B.parameters()), lr=network_config["g_lr"], weight_decay=network_config["weight_decay"])
        self.optimizerD = torch.optim.Adam(itertools.chain(self.D_B.parameters()), lr=network_config["d_lr"], weight_decay=network_config["weight_decay"])
        self.schedulerG = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, patience=1000000 / self.batch_size, threshold=0.00005)
        self.schedulerD = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, patience=1000000 / self.batch_size, threshold=0.00005)

        self.NETWORK_VERSION = ConfigHolder.getInstance().format_version_name(self.network_name, self.iteration)
        self.NETWORK_CHECKPATH = 'checkpoint/' + self.NETWORK_VERSION + '.pth'
        self.load_saved_state()

        self.BEST_NETWORK_SAVE_PATH = "./checkpoint/best/"
        network_file_name = self.BEST_NETWORK_SAVE_PATH + self.NETWORK_VERSION + "_best" + ".pth"
        self.best_tracker = best_tracker.BestTracker(early_stopper.EarlyStopperMethod.L1_TYPE)
        self.best_tracker.load_best_state(network_file_name)

    def initialize_dict(self):
        # dictionary keys
        self.G_LOSS_KEY = "g_loss"
        self.G_ADV_LOSS_KEY = "g_adv"
        self.LIKENESS_LOSS_KEY = "likeness"
        self.SSIM_LOSS_KEY = "ssim_loss"

        self.D_OVERALL_LOSS_KEY = "d_loss"
        self.D_B_REAL_LOSS_KEY = "d_real_b"
        self.D_B_FAKE_LOSS_KEY = "d_fake_b"

        # what to store in visdom?
        self.losses_dict = {}
        self.losses_dict[self.G_LOSS_KEY] = []
        self.losses_dict[self.D_OVERALL_LOSS_KEY] = []
        self.losses_dict[self.LIKENESS_LOSS_KEY] = []
        self.losses_dict[self.G_ADV_LOSS_KEY] = []
        self.losses_dict[self.SSIM_LOSS_KEY] = []
        self.losses_dict[self.D_B_FAKE_LOSS_KEY] = []
        self.losses_dict[self.D_B_REAL_LOSS_KEY] = []

        self.caption_dict = {}
        self.caption_dict[self.G_LOSS_KEY] = "Shadow G loss per iteration"
        self.caption_dict[self.D_OVERALL_LOSS_KEY] = "D loss per iteration"
        self.caption_dict[self.LIKENESS_LOSS_KEY] = "L1 loss per iteration"
        self.caption_dict[self.G_ADV_LOSS_KEY] = "G adv loss per iteration"
        self.caption_dict[self.SSIM_LOSS_KEY] = "SSIM loss per iteration"
        self.caption_dict[self.D_B_FAKE_LOSS_KEY] = "D(B) fake loss per iteration"
        self.caption_dict[self.D_B_REAL_LOSS_KEY] = "D(B) real loss per iteration"

        # what to store in visdom?
        self.losses_dict_t = {}

        self.TRAIN_LOSS_KEY = "TRAIN_LOSS_KEY"
        self.losses_dict_t[self.TRAIN_LOSS_KEY] = []
        self.TEST_LOSS_KEY = "TEST_LOSS_KEY"
        self.losses_dict_t[self.TEST_LOSS_KEY] = []

        self.caption_dict_t = {}
        self.caption_dict_t[self.TRAIN_LOSS_KEY] = "Train L1 loss per iteration"
        self.caption_dict_t[self.TEST_LOSS_KEY] = "Test L1 loss per iteration"

    def train(self, epoch, iteration, input_map, a_key, b_key, a_key_test, b_key_test):
        input_tensor = input_map[a_key]
        target_tensor = input_map[b_key]

        accum_batch_size = self.load_size * iteration

        with amp.autocast():
            #discriminator update
            self.optimizerD.zero_grad()
            self.D_B.train()
            output = self.G_A2B(input_tensor)
            prediction = self.D_B(output)
            real_tensor = torch.ones_like(prediction)
            fake_tensor = torch.zeros_like(prediction)
            D_SM_real_loss = self.common_losses.compute_adversarial_loss(self.D_B(target_tensor), real_tensor)
            D_SM_fake_loss = self.common_losses.compute_adversarial_loss(self.D_B_pool.query(self.D_B(output.detach())), fake_tensor)

            errD = D_SM_real_loss + D_SM_fake_loss

            self.fp16_scaler.scale(errD).backward()
            if (accum_batch_size % self.batch_size == 0):
                self.fp16_scaler.step(self.optimizerD)
                self.schedulerD.step(errD)


            #generator update
            self.optimizerG.zero_grad()
            self.G_A2B.train()
            imga2b = self.G_A2B(input_tensor)
            IMG_likeness_loss = self.common_losses.compute_l1_loss(imga2b, target_tensor)
            IMG_ssim_loss = self.common_losses.compute_ssim_loss(imga2b, target_tensor)
            prediction = self.D_B(imga2b)
            real_tensor = torch.ones_like(prediction)
            IMG_adv_loss = self.common_losses.compute_adversarial_loss(prediction, real_tensor)

            errG = IMG_likeness_loss + IMG_ssim_loss + IMG_adv_loss

            self.fp16_scaler.scale(errG).backward()
            if (accum_batch_size % self.batch_size == 0):
                self.fp16_scaler.step(self.optimizerG)
                self.schedulerG.step(errG)
                self.fp16_scaler.update()

                # perform validation test
                imga2b_test = self.test(input_map, a_key_test)
                a2b_gt_test = input_map[b_key_test]

                # check and save best state
                self.try_save_best_state(imga2b_test, a2b_gt_test, epoch, iteration)

                # what to put to losses dict for visdom reporting?
                if (iteration > 50):
                    self.losses_dict[self.G_LOSS_KEY].append(errG.item())
                    self.losses_dict[self.D_OVERALL_LOSS_KEY].append(errD.item())
                    self.losses_dict[self.LIKENESS_LOSS_KEY].append(IMG_likeness_loss.item())
                    self.losses_dict[self.G_ADV_LOSS_KEY].append(IMG_adv_loss.item())
                    self.losses_dict[self.SSIM_LOSS_KEY].append(IMG_ssim_loss.item())
                    self.losses_dict[self.D_B_FAKE_LOSS_KEY].append(D_SM_fake_loss.item())
                    self.losses_dict[self.D_B_REAL_LOSS_KEY].append(D_SM_real_loss.item())

                self.losses_dict_t[self.TRAIN_LOSS_KEY].append(self.common_losses.compute_l1_loss(imga2b, target_tensor).item())
                self.losses_dict_t[self.TEST_LOSS_KEY].append(self.common_losses.compute_l1_loss(imga2b_test, a2b_gt_test).item())



    def test(self, input_map, a_key):
        with torch.no_grad():
            self.G_A2B.eval()
            img_a = input_map[a_key]
            a2b = self.G_A2B(img_a)

        return a2b

    def visdom_plot(self, iteration):
        self.visdom_reporter.plot_finegrain_loss("a2b_loss_a", iteration, self.losses_dict, self.caption_dict, self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_train_test_loss("train_test_loss", iteration, self.losses_dict_t, self.caption_dict_t, self.NETWORK_VERSION + str(self.iteration))

    def visdom_visualize(self, input_map, a_key, b_key, label="Train"):
        img_a = input_map[a_key]
        img_b = input_map[b_key]

        a2b = self.test(input_map, a_key)

        self.visdom_reporter.plot_image(img_a, str(label) + " A Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(a2b, str(label) + " B-Like Images - " + self.NETWORK_VERSION + str(self.iteration))
        self.visdom_reporter.plot_image(img_b, str(label) + " B Images - " + self.NETWORK_VERSION + str(self.iteration))

    def load_saved_state(self):
        try:
            checkpoint = torch.load(self.NETWORK_CHECKPATH, map_location=self.gpu_device)
        except:
            # check if a .checkpt is available, load it
            try:
                checkpt_name = 'checkpoint/' + self.NETWORK_VERSION + ".pth.checkpt"
                checkpoint = torch.load(checkpt_name, map_location=self.gpu_device)
                print("No available .pt/pth file found. Loading .checkpt file: ", checkpt_name)
            except:
                checkpoint = None
                print("No existing checkpoint file found. Creating new paired img2img network: ", self.NETWORK_CHECKPATH)

        if (checkpoint != None):
            global_config.last_epoch = checkpoint["epoch"]
            self.G_A2B.load_state_dict(checkpoint[global_config.GENERATOR_KEY])
            self.D_B.load_state_dict(checkpoint[global_config.DISCRIMINATOR_KEY])

            print("Loaded paired img2img network: ", self.NETWORK_CHECKPATH, "Epoch: ", checkpoint["epoch"])

    def save_states(self, epoch, iteration, is_temp:bool):
        save_dict = {'epoch': epoch, 'iteration': iteration}
        netGNS_state_dict = self.G_A2B.state_dict()
        netDNS_state_dict = self.D_B.state_dict()

        save_dict[global_config.GENERATOR_KEY] = netGNS_state_dict
        save_dict[global_config.DISCRIMINATOR_KEY] = netDNS_state_dict

        if (is_temp):
            torch.save(save_dict, self.NETWORK_CHECKPATH + ".checkpt")
            print("Saved checkpoint state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))
        else:
            torch.save(save_dict, self.NETWORK_CHECKPATH)
            print("Saved stable model state: %s Epoch: %d" % (len(save_dict), (epoch + 1)))

    def try_save_best_state(self, input, target, epoch, iteration):
        best_achieved = self.best_tracker.test(input, target)
        best_metric = self.best_tracker.get_best_metric()
        if(best_achieved):
            network_file_name = self.BEST_NETWORK_SAVE_PATH + self.NETWORK_VERSION + "_best" + ".pth"
            save_dict = {'epoch': epoch, 'iteration': iteration, 'best_metric' : best_metric}
            netGNS_state_dict = self.G_A2B.state_dict()
            netDNS_state_dict = self.D_B.state_dict()

            save_dict[global_config.GENERATOR_KEY] = netGNS_state_dict
            save_dict[global_config.DISCRIMINATOR_KEY] = netDNS_state_dict

            torch.save(save_dict, network_file_name)
            print("Saved best model state. Epoch: %d. Name: %s. Best metric: %f" % (epoch, network_file_name, best_metric))

    def load_best_state(self):
        network_file_name = self.BEST_NETWORK_SAVE_PATH + self.NETWORK_VERSION + "_best" + ".pth"
        try:
            checkpoint = torch.load(network_file_name, map_location=self.gpu_device)
            self.best_tracker = best_tracker.BestTracker(early_stopper.EarlyStopperMethod.L1_TYPE)
            self.best_tracker.load_best_state(network_file_name)
        except:
            checkpoint = None
            print("No best checkpoint found. ", network_file_name)

        if (checkpoint != None):
            self.G_A2B.load_state_dict(checkpoint[global_config.GENERATOR_KEY])
            self.D_B.load_state_dict(checkpoint[global_config.DISCRIMINATOR_KEY])

            print("Loaded best paired img2img network: ", self.NETWORK_CHECKPATH, "Epoch: ", checkpoint["epoch"],
                  " Best metric: ", self.best_tracker.get_best_metric())