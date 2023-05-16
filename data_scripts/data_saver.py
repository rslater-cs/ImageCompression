import os
from pathlib import Path
import shutil
from torch import cuda
from data_scripts.loggers.printing import Printer

NETWORK_NAME = 'SwinCompression'

# Checks if the save path of the model exists 
# Also checks if the model is a new run or a checkpoint
def make_path(data_dir):
    if(not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    else:
        if(not os.path.exists(f'{data_dir}/checkpoint.pt')):
            shutil.rmtree(data_dir)
            os.mkdir(data_dir)

    return Path(data_dir)

# Writes GPU statistics to log file
def save_gpu_stats(save_dir):
    writer = Printer(save_dir, name="gpu_info")
    writer.print("Has Cuda:")
    writer.print(str(cuda.is_available()))
    writer.print("Cuda Device Count:")
    writer.print(str(cuda.device_count()))
    writer.print("Current Device:")
    writer.print(str(cuda.current_device()))
    writer.print("Current Device ID:")
    writer.print(str(cuda.get_device_name(cuda.current_device())))
    writer.print("Device Names:")

    for i in range(cuda.device_count()):
        writer.print(str(cuda.get_device_name(i)))



