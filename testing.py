from models.SwinCompression import Encoder
from data_loading import cifar_10
from torch.utils.data import DataLoader

encoder = Encoder(patch_size=[2,2], embed_dim=48, output_dim=8, depths=[2,2,2], num_heads=[2,2,2], window_size=[4,4])

dataset = cifar_10.CIFAR()
dataloader = DataLoader(dataset.trainset, batch_size=1)

for i in range(1):
    for inputs, _ in iter(dataloader):
        outputs = encoder(inputs)

        print(outputs.shape)
