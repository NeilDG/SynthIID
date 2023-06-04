#Script to use for running heavy training.

import os

def train_albedo():
    # FOR TESTING
    os.system("python \"iid_train_main.py\" --server_config=5 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=50 --network_version=\"rgb2albedo_v01.test_v05_iid\" --iteration=1")

def train_shadow_removal():
    os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow\" "
              "--plot_enabled=0 --network_version=\"rgb2ns_v61.39_places\" --iteration=1")

    #FOR TESTING
    # os.system("python \"shadow_train_main.py\" --server_config=5 --img_to_load=-1 --train_mode=\"train_shadow\" "
    #           "--plot_enabled=1 --save_per_iter=50 --network_version=\"rgb2ns_v61.00_places\" --iteration=1")

def train_img2img():
    os.system("python \"train_img2img_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=50 --network_version=\"synth2istd_v01.00\" --iteration=1")

def main():
    train_albedo()

    # train_img2img()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()
