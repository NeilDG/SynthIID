import itertools
import sys
from optparse import OptionParser
import random
import torch
import torch.nn.parallel
import torch.utils.data
import numpy as np
import yaml
from yaml import SafeLoader

import global_config
from config.network_config import ConfigHolder
from loaders import dataset_loader
from testers import paired_tester
from utils import plot_utils
from tqdm import tqdm

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)
parser.add_option('--cuda_device', type=str, help="CUDA Device?", default="cuda:0")
parser.add_option('--img_to_load', type=int, help="Image to load?", default=-1)
parser.add_option('--plot_enabled', type=int, help="Min epochs", default=1)
parser.add_option('--img_vis_enabled', type=int, default=1)
parser.add_option('--network_version', type=str, default="VXX.XX")
parser.add_option('--iteration', type=int, default=1)
parser.add_option('--load_best', type=int, default=0)
parser.add_option('--test_mode', type=str, default="albedo")

def update_config(opts):
    global_config.server_config = opts.server_config
    global_config.plot_enabled = opts.plot_enabled
    global_config.img_vis_enabled = opts.img_vis_enabled
    global_config.img_to_load = opts.img_to_load

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
        print("Using HOME RTX3090 configuration. Workers: ", global_config.num_workers)

def test_albedo(device, opts):
    yaml_config = "./hyperparam_tables/albedo/{network_version}.yaml"
    yaml_config = yaml_config.format(network_version=opts.network_version)
    hyperparam_path = "./hyperparam_tables/common_iter.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader))

    update_config(opts)
    print(opts)
    print("=====================BEGIN - TEST ALBEDO============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (global_config.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    network_config = ConfigHolder.getInstance().get_network_config()
    global_config.albedo_network_version = opts.network_version
    global_config.a_iteration = opts.iteration
    global_config.test_size = 64

    tester = paired_tester.PairedTester(device, global_config.albedo_network_version, global_config.a_iteration)

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

    test_loader_input, dataset_count = dataset_loader.load_paired_test_dataset(global_config.rgb_dir_ns, global_config.albedo_dir)
    # compute total progress
    needed_progress = int(dataset_count / global_config.test_size)
    current_progress = 0
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    print("============MEASURING ON TRAIN DATASET=================")
    for i, (_, rgb_ns, albedo) in enumerate(test_loader_input):
        rgb_ns = rgb_ns.to(device)
        albedo = albedo.to(device)

        input_map = {"rgb_test" : rgb_ns, "albedo_test" : albedo}
        tester.measure_and_store(input_map, "rgb_test", "albedo_test")

        if(i % 50 == 0 and global_config.img_vis_enabled == 1):
            tester.visualize_results(input_map, "rgb_test", "albedo_test", "Train")

        pbar.update(1)

    tester.report_metrics("Train")
    pbar.close()

    print("============MEASURING ON GTA-IID DATASET=================")
    test_loader_gta, dataset_count = dataset_loader.load_paired_test_dataset(gta_rgb_path, gta_albedo_path)

    # compute total progress
    needed_progress = int(dataset_count / global_config.test_size)
    current_progress = 0
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    for i, (_, rgb_ns, albedo) in enumerate(test_loader_gta):
        rgb_ns = rgb_ns.to(device)
        albedo = albedo.to(device)

        input_map = {"rgb_test": rgb_ns, "albedo_test": albedo}
        tester.measure_and_store(input_map, "rgb_test", "albedo_test")
        if (i % 50 == 0 and global_config.img_vis_enabled == 1):
            tester.visualize_results(input_map, "rgb_test", "albedo_test", "GTA-IID")

        pbar.update(1)

    tester.report_metrics("GTA-IID")
    pbar.close()

    print("============MEASURING ON CGINTRINSICS DATASET=================")
    test_loader, dataset_count = dataset_loader.load_cgintrinsics_test_dataset()

    # compute total progress
    needed_progress = int(dataset_count / global_config.test_size)
    current_progress = 0
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    for i, (_, rgb_ns, albedo, mask) in enumerate(test_loader):
        rgb_ns = rgb_ns.to(device)
        albedo = albedo.to(device)

        input_map = {"rgb_test": rgb_ns, "albedo_test": albedo}
        tester.measure_and_store(input_map, "rgb_test", "albedo_test")
        if (i % 50 == 0 and global_config.img_vis_enabled == 1):
            tester.visualize_results(input_map, "rgb_test", "albedo_test", "CGI")

        pbar.update(1)

    tester.report_metrics("CGI")
    pbar.close()

def test_normal(device, opts):
    yaml_config = "./hyperparam_tables/normal/{network_version}.yaml"
    yaml_config = yaml_config.format(network_version=opts.network_version)
    hyperparam_path = "./hyperparam_tables/common_iter.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader))

    update_config(opts)
    print(opts)
    print("=====================BEGIN - TEST NORMAL============================")
    print("Server config? %d Has GPU available? %d Count: %d" % (global_config.server_config, torch.cuda.is_available(), torch.cuda.device_count()))
    print("Torch CUDA version: %s" % torch.version.cuda)

    network_config = ConfigHolder.getInstance().get_network_config()
    global_config.normal_network_version = opts.network_version
    global_config.n_iteration = opts.iteration
    global_config.test_size = 64
    global_config.num_test_workers = 2

    tester = paired_tester.PairedTester(device, global_config.normal_network_version, global_config.n_iteration)

    start_epoch = global_config.last_epoch
    print("---------------------------------------------------------------------------")
    print("Started Training loop for mode: normal", " Set start epoch: ", start_epoch)
    print("Network config: ", network_config)
    print("General config: ", global_config.albedo_network_version, global_config.n_iteration, global_config.img_to_load, global_config.load_size, global_config.batch_size, global_config.train_mode, global_config.last_epoch)
    print("---------------------------------------------------------------------------")

    dataset_version = network_config["dataset_version"]
    global_config.rgb_dir_ws = global_config.rgb_dir_ws.format(dataset_version=dataset_version)
    global_config.rgb_dir_ns = global_config.rgb_dir_ns.format(dataset_version=dataset_version)
    global_config.normal_dir = global_config.normal_dir.format(dataset_version=dataset_version)
    print("Dataset path WS: ", global_config.rgb_dir_ws)
    print("Dataset path NS: ", global_config.rgb_dir_ns)
    print("Dataset normal: ", global_config.normal_dir)

    test_loader_input, dataset_count = dataset_loader.load_paired_test_dataset(global_config.rgb_dir_ns, global_config.normal_dir)
    # compute total progress
    needed_progress = int(dataset_count / global_config.test_size)
    current_progress = 0
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)

    print("============MEASURING ON TRAIN DATASET=================")
    for i, (_, rgb_ns, target) in enumerate(test_loader_input):
        rgb_ns = rgb_ns.to(device)
        target = target.to(device)

        input_map = {"rgb_test" : rgb_ns, "target_test" : target}
        tester.measure_and_store(input_map, "rgb_test", "target_test")

        if(i % 50 == 0 and global_config.img_vis_enabled == 1):
            tester.visualize_results(input_map, "rgb_test", "target_test", "Train-Normal")

        pbar.update(1)

    tester.report_metrics("Train-Normal")
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

    if(opts.test_mode == "albedo"):
        test_albedo(device, opts)
    elif(opts.test_mode == "normal"):
        test_normal(device, opts)

if __name__ == "__main__":
    main(sys.argv)