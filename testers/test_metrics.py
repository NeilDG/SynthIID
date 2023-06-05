import numpy as np
import math
import torch
import kornia
from utils import tensor_utils

def torch_rmse(tensor1, tensor2):
    return torch.sqrt(torch.nn.functional.mse_loss(tensor1, tensor2))

def torch_rmse_log(tensor1, tensor2):
    log_1 = torch.log(tensor1)
    log_2 = torch.log(tensor2)

    # print("Log result: ", torch.mean(log_depth1), torch.mean(log_depth2))
    return torch.sqrt(torch.nn.functional.mse_loss(log_1, log_2))

def ssim_measure(tensor1, tensor2):
    return 1.0 - kornia.losses.ssim_loss(tensor1, tensor2, 5)