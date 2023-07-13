#Script to use for running heavy training.

import os

def train_albedo():
    # os.system("python \"iid_train_main.py\" --server_config=5 "
    #           "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2albedo_v01.02_iid\" --iteration=1")

    os.system("python \"iid_train_main.py\" --server_config=5 "
              "--plot_enabled=0 --save_per_iter=500 --network_version=\"rgb2albedo_v01.07_iid\" --iteration=1")

    # FOR TESTING
    # os.system("python \"iid_train_main.py\" --server_config=5 "
    #           "--plot_enabled=1 --save_per_iter=10 --network_version=\"rgb2albedo_v01.02_iid\" --iteration=1")

def test_albedo():
    os.system("python \"iid_test_main.py\" --server_config=5 --img_to_load=1000 "
              "--img_vis_enabled=0 --network_version=\"rgb2albedo_v01.02_iid\" --iteration=1")

    # os.system("python \"iid_test_main.py\" --server_config=5 --img_to_load=1000 "
    #           "--img_vis_enabled=0 --network_version=\"rgb2albedo_v01.03_iid\" --iteration=1")
    #
    # os.system("python \"iid_test_main.py\" --server_config=5 --img_to_load=1000 "
    #           "--img_vis_enabled=0 --network_version=\"rgb2albedo_v01.04_iid\" --iteration=1")
    #
    # os.system("python \"iid_test_main.py\" --server_config=5 --img_to_load=1000 "
    #           "--img_vis_enabled=0 --network_version=\"rgb2albedo_v01.05_iid\" --iteration=1")
    #
    # os.system("python \"iid_test_main.py\" --server_config=5 --img_to_load=1000 "
    #           "--img_vis_enabled=0 --network_version=\"rgb2albedo_v01.06_iid\" --iteration=1")
    #
    os.system("python \"iid_test_main.py\" --server_config=5 --img_to_load=1000 "
              "--img_vis_enabled=0 --network_version=\"rgb2albedo_v01.07_iid\" --iteration=1")

def plot_test():
    os.system("python \"plot_test_util.py\" --server_config=5 --img_to_load=-1 --img_vis_enabled=1")

def train_normal():
    # os.system("python \"iid_train_main.py\" --server_config=5 "
    #           "--plot_enabled=0 --save_per_iter=1000 --network_version=\"rgb2normal_v01.02_iid\" --iteration=1")

    os.system("python \"iid_train_main.py\" --server_config=5 --plot_enabled=0 --save_per_iter=1000 "
              "--network_version=\"rgb2normal_v01.07_iid\" --iteration=1")

    os.system("python \"iid_train_main.py\" --server_config=5 --plot_enabled=0 --save_per_iter=1000 "
              "--network_version=\"rgb2normal_v01.03_iid\" --iteration=1")

    # os.system("python \"iid_train_main.py\" --server_config=5 --plot_enabled=0 --save_per_iter=1000 "
    #           "--network_version=\"rgb2normal_v01.04_iid\" --iteration=1")
    # os.system("python \"iid_train_main.py\" --server_config=5 --plot_enabled=0 --save_per_iter=1000 "
    #           "--network_version=\"rgb2normal_v01.05_iid\" --iteration=1")

def test_normal():
    # os.system("python \"iid_test_main.py\" --server_config=5 --img_to_load=1000 --test_mode=\"normal\" "
    #           "--img_vis_enabled=0 --network_version=\"rgb2normal_v01.02_iid\" --iteration=1")
    #
    # os.system("python \"iid_test_main.py\" --server_config=5 --img_to_load=1000 --test_mode=\"normal\" "
    #           "--img_vis_enabled=0 --network_version=\"rgb2normal_v01.03_iid\" --iteration=1")
    #
    # os.system("python \"iid_test_main.py\" --server_config=5 --img_to_load=1000 --test_mode=\"normal\" "
    #           "--img_vis_enabled=0 --network_version=\"rgb2normal_v01.04_iid\" --iteration=1")

    # os.system("python \"iid_test_main.py\" --server_config=5 --img_to_load=1000 --test_mode=\"normal\" "
    #           "--img_vis_enabled=0 --network_version=\"rgb2normal_v01.05_iid\" --iteration=1")

    # os.system("python \"iid_test_main.py\" --server_config=5 --img_to_load=1000 --test_mode=\"normal\" "
    #           "--img_vis_enabled=0 --network_version=\"rgb2normal_v01.06_iid\" --iteration=1")

    os.system("python \"iid_test_main.py\" --server_config=5 --img_to_load=1000 --test_mode=\"normal\" "
              "--img_vis_enabled=0 --network_version=\"rgb2normal_v01.07_iid\" --iteration=1")

def train_img2img():
    os.system("python \"train_img2img_main.py\" --server_config=3 --img_to_load=-1 "
              "--plot_enabled=1 --save_per_iter=50 --network_version=\"synth2istd_v01.00\" --iteration=1")

def main():
    # train_albedo()
    # test_albedo()
    # plot_test()

    # train_normal()
    test_normal()
    # train_img2img()
    # os.system("shutdown /s /t 1")


if __name__ == "__main__":
    main()
