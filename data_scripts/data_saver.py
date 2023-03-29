import os
from pathlib import Path
import shutil
from torch import cuda
from loggers.printing import Printer

NETWORK_NAME = 'SwinCompression'

def make_path(data_dir):
    if(not os.path.exists(data_dir)):
        os.mkdir(data_dir)
    else:
        if(not os.path.exists(f'{data_dir}/checkpoint.pt')):
            shutil.rmtree(data_dir)
            os.mkdir(data_dir)

    return Path(data_dir)

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


# def save_model(model, path: Path, in_progress=False):
#     if(in_progress):
#         encoder_path = path / "encoder_progress.pt"
#         decoder_path = path / "decoder_progress.pt"
#     else:
#         encoder_path = path / "encoder.pt"
#         decoder_path = path / "decoder.pt"

#     torch.save(model.encoder.state_dict(), encoder_path)
#     torch.save(model.decoder.state_dict(), decoder_path)

#     return path



