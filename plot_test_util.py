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

def update_config(opts):
    global_config.server_config = opts.server_config
    global_config.plot_enabled = opts.plot_enabled
    global_config.img_vis_enabled = opts.img_vis_enabled
    global_config.img_to_load = opts.img_to_load

    yaml_config = "./hyperparam_tables/albedo/rgb2albedo_v01.02_iid.yaml"
    hyperparam_path = "./hyperparam_tables/common_iter.yaml"
    with open(yaml_config) as f, open(hyperparam_path) as h:
        ConfigHolder.initialize(yaml.load(f, SafeLoader), yaml.load(h, SafeLoader))

    global_config.num_test_workers = 2
    global_config.DATASET_PLACES_PATH = "E:/Places Dataset/*.jpg"
    global_config.rgb_dir_ws = "X:/SynthV3_Raw/{dataset_version}/rgb/*.*"
    global_config.rgb_dir_ns = "X:/SynthV3_Raw/{dataset_version}/rgb_noshadows/*.*"
    global_config.albedo_dir = "X:/SynthV3_Raw/{dataset_version}/albedo/*.*"
    global_config.depth_dir = "X:/SynthV3_Raw/{dataset_version}/depth/*.*"
    global_config.shading_dir = "X:/SynthV3_Raw/{dataset_version}/shading/*.*"
    print("Using HOME RTX3090 configuration. Workers: ", global_config.num_test_workers)

def main(argv):
    (opts, args) = parser.parse_args(argv)
    device = torch.device(opts.cuda_device if (torch.cuda.is_available()) else "cpu")
    print("Device: %s" % device)

    update_config(opts)

    manualSeed = 0
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    np.random.seed(manualSeed)

    plot_utils.VisdomReporter.initialize()
    visdom_reporter = plot_utils.VisdomReporter.getInstance()

    path_a = "D:/Unused Datasets/Synth Hazy 3/hazy - styled/*.png"
    path_b = "D:/Unused Datasets/Synth Hazy 3/clean - styled/*.png"
    label = "DLSU-SYNSIDE"

    # path_a = global_config.GTA_IID_PATH + "gta_trainfinal.webp/*/*.webp"
    # path_b = global_config.GTA_IID_PATH + "gta_trainalbedo.webp/*/*.webp"
    # label = "GTA"

    path_a = "X:/SynthV3_Raw/v08_iid/rgb/*.*"
    path_b = "X:/SynthV3_Raw/v08_iid/rgb_noshadows/*.*"

    global_config.test_size = 128
    print(path_a, path_b)
    test_loader_input, dataset_count = dataset_loader.load_paired_test_dataset(path_a, path_b)

    # compute total progress
    needed_progress = int(dataset_count / global_config.test_size)
    current_progress = 0
    pbar = tqdm(total=needed_progress, disable=global_config.disable_progress_bar)
    pbar.update(current_progress)
    for i, (_, rgb_a, rgb_b) in enumerate(test_loader_input):
        rgb_a = rgb_a.to(device)
        rgb_b = rgb_b.to(device)

        visdom_reporter.plot_image(rgb_a, str(label) + " A Images", nrows=8, ncolumns=global_config.test_size)
        visdom_reporter.plot_image(rgb_b, str(label) + " B Images", nrows=8, ncolumns=global_config.test_size)

        # input()

if __name__ == "__main__":
    main(sys.argv)