import torch

from config import gpu_device_no

cuda_device_name = f'cuda:{gpu_device_no}' if torch.cuda.is_available() else 'cpu'
