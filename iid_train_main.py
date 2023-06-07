import itertools
import sys
from optparse import OptionParser
import random
from pathlib import Path

import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import yaml
from yaml import SafeLoader

from config import network_config
import global_config
from config.network_config import ConfigHolder
from loaders import dataset_loader
from trainers import paired_trainer
from utils import plot_utils
from tqdm import tqdm
from tqdm.auto import trange

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--plot_enabled', type=int, help="Min epochs", default=1)
parser.add_option('--network_version', type=str, default="VXX.XX")
parser.add_option('--iteration', type=int, default=1)
parser.add_option('--save_per_iter', type=int, default=500)

def update_config(opts):
    global_config.server_config = opts.server_config
    global_config.plot_enabled = opts.plot_enabled

    config_holder = ConfigHolder.getInstance()
    network_config = config_holder.get_network_config()

    ## COARE - 24GB/P40
    if (global_config.server_config == 0):
        global_config.num_workers = 6
        global_config.disable_progress_bar = True  # disable progress bar logging in COARE
        global_config.load_size = network_config["load_size"][0] - 4
        global_config.batch_size = network_config["batch_size"][0] - 4

        print("Using COARE configuration. Workers: ", global_config.num_workers)
        global_config.DATASET_PLACES_PATH = "/scratch3/neil.delgallego/Places Dataset/*.jpg"
        global_config.rgb_dir_ws = "/scratch3/neil.delgallego/SynthV3_Raw/{dataset_version}/rgb/*.*"
        global_config.rgb_dir_ns = "/scratch3/neil.delgallego/SynthV3_Raw/{dataset_version}/rgb_noshadows/*.*"
        global_config.albedo_dir = "/scratch3/neil.delgallego/SynthV3_Raw/{dataset_version}/albedo/*.*"
        global_config.depth_dir = "/scratch3/neil.delgallego/SynthV3_Raw/{dataset_version}/depth/*.*"
        global_config.shading_dir = "/scratch3/neil.delgallego/SynthV3_Raw/{dataset_version}/shading/*.*"

    # CCS JUPYTER
    elif (global_config.server_config == 1):
        global_config.num_workers = 20
        global_config.load_size = network_config["load_size"][1]
        global_config.batch_size = network_config["batch_size"][1]
        global_config.DATASET_PLACES_PATH = "/home/jupyter-neil.delgallego/Places Dataset/*.jpg"
        global_config.rgb_dir_ws = "/home/jupyter-neil.delgallego/SynthV3_Raw/{dataset_version}/rgb/*.*"
        global_config.rgb_dir_ns = "/home/jupyter-neil.delgallego/SynthV3_Raw/{dataset_version}/rgb_noshadows/*.*"
        global_config.albedo_dir = "/home/jupyter-neil.delgallego/SynthV3_Raw/{dataset_version}/albedo/*.*"
        global_config.depth_dir = "/home/jupyter-neil.delgallego/SynthV3_Raw/{dataset_version}/depth/*.*"
        global_config.shading_dir = "/home/jupyter-neil.delgallego/SynthV3_Raw/{dataset_version}/shading/*.*"

        print("Using CCS configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 2):
        global_config.num_workers = 6
        global_config.load_size = network_config["load_size"][2]
        global_config.batch_size = network_config["batch_size"][2]
        global_config.DATASET_PLACES_PATH = "X:/Datasets/Places Dataset/*.jpg"
        global_config.rgb_dir_ws = "X:/Datasets/SynthV3_Raw/{dataset_version}/rgb/*.*"
        global_config.rgb_dir_ns = "X:/Datasets/SynthV3_Raw/{dataset_version}/rgb_noshadows/*.*"
        global_config.albedo_dir = "X:/Datasets/SynthV3_Raw/{dataset_version}/albedo/*.*"
        global_config.depth_dir = "X:/Datasets/SynthV3_Raw/{dataset_version}/depth/*.*"
        global_config.shading_dir = "X:/Datasets/SynthV3_Raw/{dataset_version}/shading/*.*"
        global_config.cg_intrinsics_dir = "X:/Datasets/CGIntrinsics/rendered/"

        print("Using HOME RTX2080Ti configuration. Workers: ", global_config.num_workers)

    elif (global_config.server_config == 3):  # TITAN 3060
        global_config.num_workers = 4
        global_config.load_size = network_config["load_size"][2]
        global_config.batch_size = network_config["batch_size"][2]
        global_config.DATASET_PLACES_PATH = "/home/neildelgallego/Places Dataset/*.jpg"
        global_config.rgb_dir_ws = "/home/neildelgallego/SynthV3_Raw/{dataset_version}/rgb/*.*"
        global_config.rgb_dir_ns = "/home/neildelgallego/SynthV3_Raw/{dataset_version}/rgb_noshadows/*.*"
        global_config.albedo_dir = "/home/neildelgallego/SynthV3_Raw/{dataset_version}/albedo/*.*"
        global_config.depth_dir = "/home/neildelgallego/SynthV3_Raw/{dataset_version}/depth/*.*"
        global_config.shading_dir = "/home/neildelgallego/SynthV3_Raw/{dataset_version}/shading/*.*"

        print("Using TITAN configuration. Workers: ", global_config.num_workers)

    ## COARE - 40GB/A100
    elif (global_config.server_config == 4):
        global_config.num_workers = 12
        global_config.disable_progress_bar = True  # disable progress bar logging in COARE
        global_config.load_size = network_config["load_size"][1]
        global_config.batch_size = network_config["batch_size"][1]

        print("Using COARE configuration. Workers: ", global_config.num_workers)
        global_config.DATASET_PLACES_PATH = "/scratch3/neil.delgallego/Places Dataset/*.jpg"
        global_config.rgb_dir_ws = "/scratch3/neil.delgallego/SynthV3_Raw/{dataset_version}/rgb/*.*"
        global_config.rgb_dir_ns = "/scratch3/neil.delgallego/SynthV3_Raw/{dataset_version}/rgb_noshadows/*.*"
        global_config.albedo_dir = "/scratch3/neil.delgallego/SynthV3_Raw/{dataset_version}/albedo/*.*"
        global_config.depth_dir = "/scratch3/neil.delgallego/SynthV3_Raw/{dataset_version}/depth/*.*"
        global_config.shading_dir = "/scratch3/neil.delgallego/SynthV3_Raw/{dataset_version}/shading/*.*"
    else:
        global_config.num_workers = 12
        global_config.load_size = network_config["load_size"][0]
        global_config.batch_size = network_config["batch_size"][0]
        global_config.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
        global_config.rgb_dir_ws = "X:/SynthV3_Raw/{dataset_version}/rgb/*.*"
        global_config.rgb_dir_ns = "X:/SynthV3_Raw/{dataset_version}/rgb_noshadows/*.*"
        global_config.albedo_dir = "X:/SynthV3_Raw/{dataset_version}/albedo/*.*"
        global_config.depth_dir = "X:/SynthV3_Raw/{dataset_version}/depth/*.*"
        global_config.shading_dir = "X:/SynthV3_Raw/{dataset_version}/shading/*.*"
        global_config.normal_dir = "X:/SynthV3_Raw/{dataset_version}/normal/*.*"
        global_config.cg_intrinsics_dir = "X:/CGIntrinsics/rendered/"
        print("Using HOME RTX3090 configuration. Workers: ", global_config.num_workers)

def prepare_training():
    BEST_NETWORK_SAVE_PATH = "./checkpoint/best/"
    try:
        path = Path(BEST_NETWORK_SAVE_PATH)
        path.mkdir(parents=True)
    except OSError as error:
        print(BEST_NETWORK_SAVE_PATH + " already exists. Skipping.", error)

def train_albedo(device, opts):
    yaml_config = "./hyperparam_tables/albedo/{network_version}.yaml"
    yaml_config = yaml_config.format(network_version=opts.network_version)
    hyperparam_path = "./hyperparam_tables/common_iter.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader))

    update_config(opts)
    print(opts)
    print("=====================BEGIN - TRAIN ALBEDO============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (global_config.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    network_config = ConfigHolder.getInstance().get_network_config()
    global_config.albedo_network_version = opts.network_version
    global_config.a_iteration = opts.iteration
    global_config.img_to_load = ConfigHolder.getInstance().get_network_attribute("img_to_load", -1)
    global_config.test_size = 8

    tf = paired_trainer.PairedTrainer(device, global_config.albedo_network_version, global_config.a_iteration)

    iteration = 0
    start_epoch = global_config.last_epoch
    print("---------------------------------------------------------------------------")
    print("Started Training loop for mode: albedo", " Set start epoch: ", start_epoch)
    print("Network config: ", network_config)
    print("General config: ", global_config.albedo_network_version, global_config.a_iteration, global_config.img_to_load, global_config.load_size, global_config.batch_size, global_config.train_mode, global_config.last_epoch)
    print("---------------------------------------------------------------------------")

    dataset_version = network_config["dataset_version"]
    global_config.rgb_dir_ws = global_config.rgb_dir_ws.format(dataset_version=dataset_version)
    global_config.rgb_dir_ns = global_config.rgb_dir_ns.format(dataset_version=dataset_version)
    global_config.albedo_dir = global_config.albedo_dir.format(dataset_version=dataset_version)
    print("Dataset path WS: ", global_config.rgb_dir_ws)
    print("Dataset path NS: ", global_config.rgb_dir_ns)
    print("Dataset albedo: ", global_config.albedo_dir)

    gta_rgb_path = global_config.GTA_IID_PATH + "gta_trainfinal.webp/*/*.webp"
    gta_albedo_path = global_config.GTA_IID_PATH + "gta_trainalbedo.webp/*/*.webp"

    train_loader, dataset_count = dataset_loader.load_paired_train_dataset(global_config.rgb_dir_ns, global_config.albedo_dir)
    # test_loader, _ = dataset_loader.load_paired_test_dataset(gta_rgb_path, gta_albedo_path)
    test_loader, _ = dataset_loader.load_cgintrinsics_test_dataset()

    # compute total progress
    max_epochs = network_config["max_epochs"]
    needed_progress = int(max_epochs * (dataset_count / global_config.load_size))
    current_progress = int(start_epoch * (dataset_count / global_config.load_size))
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    for epoch in range(start_epoch, network_config["max_epochs"]):
        for i, (train_data, test_data) in enumerate(zip(train_loader, itertools.cycle(test_loader))):
            _, rgb_ns, albedo = train_data
            rgb_ns = rgb_ns.to(device)
            albedo = albedo.to(device)

            _, rgb_ns_test, albedo_test, _ = test_data
            rgb_ns_test = rgb_ns_test.to(device)
            albedo_test = albedo_test.to(device)

            input_map = {"rgb_train" : rgb_ns, "albedo_train" : albedo, "rgb_test" : rgb_ns_test, "albedo_test" : albedo_test}
            tf.train(epoch, iteration, input_map, "rgb_train", "albedo_train", "rgb_test", "albedo_test")

            if (iteration % opts.save_per_iter == 0):
                tf.save_states(epoch, iteration, True)

                if(global_config.plot_enabled == 1):
                    tf.visdom_plot(iteration)
                    tf.visdom_visualize(input_map, "rgb_train", "albedo_train", "Train")
                    tf.visdom_visualize(input_map, "rgb_test", "albedo_test", "Test")

            iteration = iteration + 1
            pbar.update(1)

        tf.save_states(epoch, iteration, True)

    pbar.close()

def train_normal(device, opts):
    yaml_config = "./hyperparam_tables/normal/{network_version}.yaml"
    yaml_config = yaml_config.format(network_version=opts.network_version)
    hyperparam_path = "./hyperparam_tables/common_iter.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader))

    update_config(opts)
    print(opts)
    print("=====================BEGIN - TRAIN NORMAL============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (global_config.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    network_config = ConfigHolder.getInstance().get_network_config()
    global_config.normal_network_version = opts.network_version
    global_config.n_iteration = opts.iteration
    global_config.test_size = 8

    tf = paired_trainer.PairedTrainer(device, global_config.normal_network_version, global_config.n_iteration)

    iteration = 0
    start_epoch = global_config.last_epoch
    print("---------------------------------------------------------------------------")
    print("Started Training loop for mode: normal", " Set start epoch: ", start_epoch)
    print("Network config: ", network_config)
    print("General config: ", global_config.normal_network_version, global_config.n_iteration, global_config.img_to_load, global_config.load_size, global_config.batch_size, global_config.train_mode, global_config.last_epoch)
    print("---------------------------------------------------------------------------")

    dataset_version = network_config["dataset_version"]
    global_config.rgb_dir_ws = global_config.rgb_dir_ws.format(dataset_version=dataset_version)
    global_config.rgb_dir_ns = global_config.rgb_dir_ns.format(dataset_version=dataset_version)
    global_config.normal_dir = global_config.normal_dir.format(dataset_version=dataset_version)
    print("Dataset path WS: ", global_config.rgb_dir_ws)
    print("Dataset path NS: ", global_config.rgb_dir_ns)
    print("Dataset normal: ", global_config.normal_dir)

    train_loader, dataset_count = dataset_loader.load_paired_train_dataset(global_config.rgb_dir_ns, global_config.normal_dir)
    test_loader, _ = dataset_loader.load_paired_test_dataset(global_config.rgb_dir_ns, global_config.normal_dir)

    # compute total progress
    max_epochs = network_config["max_epochs"]
    needed_progress = int(max_epochs * (dataset_count / global_config.load_size))
    current_progress = int(start_epoch * (dataset_count / global_config.load_size))
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    for epoch in range(start_epoch, network_config["max_epochs"]):
        for i, (train_data, test_data) in enumerate(zip(train_loader, itertools.cycle(test_loader))):
            _, rgb_ns, target = train_data
            rgb_ns = rgb_ns.to(device)
            target = target.to(device)

            _, rgb_ns_test, target_test = test_data
            rgb_ns_test = rgb_ns_test.to(device)
            target_test = target_test.to(device)

            input_map = {"rgb_train": rgb_ns, "target_train": target, "rgb_test": rgb_ns_test, "target_test": target_test}
            tf.train(epoch, iteration, input_map, "rgb_train", "target_train", "rgb_test", "target_test")

            if (iteration % opts.save_per_iter == 0):
                tf.save_states(epoch, iteration, True)

                if (global_config.plot_enabled == 1):
                    tf.visdom_plot(iteration)
                    tf.visdom_visualize(input_map, "rgb_train", "target_train", "Train")
                    tf.visdom_visualize(input_map, "rgb_test", "target_test", "Test")

            iteration = iteration + 1
            pbar.update(1)

        tf.save_states(epoch, iteration, True)

    pbar.close()

def main(argv):
    (opts, args) = parser.parse_args(argv)
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    plot_utils.VisdomReporter.initialize()
    prepare_training()

    if("albedo" in opts.network_version):
        train_albedo(device, opts)
    elif("normal" in opts.network_version):
        train_normal(device, opts)

if __name__ == "__main__":
    main(sys.argv)