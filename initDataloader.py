from torch.utils.data import DataLoader
from model.TRN.dataset import customCVSDataSet
import torchvision
from model.TRN.transforms import GroupScale,GroupCenterCrop,Stack,ToTorchFormatTensor,GroupNormalize

def get_dataloader( sysparas ):

    if sysparas.modelname == "TRN":

        normalize   = GroupNormalize( sysparas.model.input_mean, sysparas.model.input_std)
        transformer = torchvision.transforms.Compose([
                    GroupScale(int(sysparas.model.scale_size)),
                    GroupCenterCrop(sysparas.model.crop_size),
                    Stack(roll=(sysparas.arch in ['BNInception', 'InceptionV3'])),
                    ToTorchFormatTensor(div=(sysparas.arch not in ['BNInception', 'InceptionV3'])),
                    normalize,
                ])

        dirImage     = sysparas.dataset_dir
        pathJson     = sysparas.labelfile
        rule         = "phase"
        num_segments = 8
        if not sysparas.interface:
            traindataset = customCVSDataSet(dirImage,pathJson,'train', rule, transformer , num_segments, sysparas.dictCategories)
            if traindataset.init == True:
                sysparas.dataloader_dict["train"] = DataLoader(
                    dataset=traindataset,batch_size=sysparas.batch_size,shuffle=True,num_workers=sysparas.workers
                )

            validataset = customCVSDataSet(dirImage, pathJson, 'valid', rule, transformer, num_segments, sysparas.dictCategories)
            if validataset.init == True:
                sysparas.dataloader_dict["valid"] = DataLoader(
                    dataset=validataset,batch_size=sysparas.batch_size,shuffle=True,num_workers=sysparas.workers
                )
        else:
            testataset = customCVSDataSet(dirImage, pathJson, 'test', rule, transformer, num_segments, sysparas.dictCategories)
            if testataset.init == True:
                sysparas.dataloader_dict["test"] = DataLoader(
                    dataset=testataset,batch_size=sysparas.batch_size,shuffle=True,num_workers=sysparas.workers
                )

    return sysparas




if __name__ == "__main__":

    dataloader = get_dataloader()