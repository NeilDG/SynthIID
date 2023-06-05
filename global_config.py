# -*- coding: utf-8 -*-
import os

DATASET_PLACES_PATH = "X:/Places Dataset/"
CGINTRINSICS_PATH = "X:/CGIntrinsics/rendered/"
GTA_IID_PATH = "X:/GTA-IID/"

TEST_IMAGE_SIZE = (256, 256)

#========================================================================#
OPTIMIZER_KEY = "optimizer"
GENERATOR_KEY = "generator"
DISCRIMINATOR_KEY = "discriminator"

RELIGHTING_VERSION = "relighter_v1.00"
SHADING_VERSION = "rgb2shading_v7.00"
SHADOWMAP_VERSION = "rgb2shadowmap_v1.00"
ALBEDO_VERSION = "rgb2albedo_v1.00"
NORMAL_VERSION = "rgb2normal_v1.00"
DEPTH_VERSION = "rgb2depth_v1.00"

LAST_METRIC_KEY = "last_metric"

plot_enabled = 1
early_stop_threshold = 500
disable_progress_bar = False

server_config = -1
num_workers = -1

rgb_dir_ws = ""
rgb_dir_ns = ""
albedo_dir = ""
depth_dir = ""
shading_dir = ""
normal_dir = ""

depth_network_version = "VXX.XX"
d_iteration = -1
albedo_network_version = "VXX.XX"
a_iteration = -1
normal_network_version = "VXX.XX"
n_iteration = -1
shading_network_version = "VXX.XX"
sh_iteration = -1
shadowmap_network_version = "VXX.XX"
sm_iteration = -1

loaded_network_config = None

img_to_load = -1
load_size = -1
batch_size = -1
test_size = -1
train_mode = "all"


cuda_device = ""
save_images = 0
save_every_epoch = 5
epoch_to_load = 0

last_epoch = 0



