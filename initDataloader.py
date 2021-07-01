import os
from torch.utils.data import DataLoader
from dataset import customCVSDataSet
import torchvision
import torch
from transforms import GroupScale,GroupCenterCrop,Stack,ToTorchFormatTensor,GroupNormalize

def get_dataloader( paras ):





    normalize = GroupNormalize( paras.model.input_mean, paras.model.input_std)
    transformer = torchvision.transforms.Compose([
                GroupScale(int(paras.model.scale_size)),
                GroupCenterCrop(paras.model.crop_size),
                Stack(roll=(paras.arch in ['BNInception', 'InceptionV3'])),
                ToTorchFormatTensor(div=(paras.arch not in ['BNInception', 'InceptionV3'])),
                normalize,
            ])

    dirImage     = '/home/withai/Pictures/LPDFrame'
    pathJson     = '/home/withai/dataset/LPDFrame.json'
    rule         = "phase"
    num_segments = 8
    traindataset = customCVSDataSet(dirImage,pathJson,'train', rule, transformer , num_segments, paras.dictCategories[rule])

    train_loader = DataLoader(
        dataset=traindataset,batch_size=paras.batch_size,shuffle=True,num_workers=paras.workers
    )

    valid_loader = DataLoader(
        dataset=traindataset,batch_size=paras.batch_size,shuffle=True,num_workers=paras.workers
    )

    test_loader = DataLoader(
        dataset=traindataset,batch_size=paras.batch_size,shuffle=True,num_workers=paras.workers
    )
    return paras




if __name__ == "__main__":

    dataloader = get_dataloader()