import os
import GPUtil
import multiprocessing
import time

def train_proper(gpu_device):
    os.system("python \"iid_train_main.py\" --server_config=1 --cuda_device=" +gpu_device+ " --plot_enabled=0 --save_per_iter=1000 "
              "--network_version=\"rgb2normal_v01.07_iid\" --iteration=1")

def main():
    EXECUTION_TIME_IN_HOURS = 48
    execution_seconds = 3600 * EXECUTION_TIME_IN_HOURS

    GPUtil.showUtilization()
    device_id = GPUtil.getFirstAvailable(maxMemory=0.1, maxLoad=0.1, attempts=2500, interval=30, verbose=True)
    gpu_device = "cuda:" + str(device_id[0])
    print("Available GPU device found: ", gpu_device)

    p = multiprocessing.Process(target=train_proper, name="train_proper", args=(gpu_device,))
    p.start()

    time.sleep(execution_seconds) #causes p to execute code for X seconds. 3600 = 1 hour

    #terminate
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print("\n Process " +p.name+ " has finished execution.")
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    p.terminate()
    p.join()


if __name__ == "__main__":
    main()