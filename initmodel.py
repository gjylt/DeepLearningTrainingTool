from model.TRN.models import TSN
import torch
import os
from tool import log


def load_checkpoint( model, checkpoint, optimizer = None ):
    if os.path.exists(checkpoint) :
        print("loading checkpoint...")
        model_dict      = model.state_dict()
        modelCheckpoint = torch.load(checkpoint)
        pretrained_dict = modelCheckpoint['state_dict']
        # 过滤操作
        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)
        # 打印出来，更新了多少的参数
        print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        model.load_state_dict(model_dict)
        print("loaded finished!")
        # 如果不需要更新优化器那么设置为false
        if optimizer != None:
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            print('loaded! optimizer')
        else:
            print('not loaded optimizer')
    else:
        print('No checkpoint is included')

    if optimizer != None:
        return model, optimizer
    else:
        return model

def get_model( paras ):
    modelname   = paras.modelname
    num_class   = paras.num_class
    modelresume = paras.modelresume

    gpunum      = torch.cuda.device_count()
    paras.log.log("gpunum:" + str(gpunum) )
    if gpunum > 0:
        paras.usegpu = True

    model = None
    #init model
    if modelname == "TRN":
        paras.log.log("init model, model name:" + modelname )

        num_segments    = 8
        modality        = "RGB"
        arch            = "BNInception"
        consensus_type  = 'TRNmultiscale'
        dropout         = 0.8
        img_feature_dim = 256
        no_partialbn    = False
        model = TSN(num_class, num_segments, modality,
                    base_model= arch,
                    consensus_type= consensus_type,
                    dropout= dropout,
                    img_feature_dim=img_feature_dim,
                    partial_bn=not no_partialbn)

    #load pretrain parameters
    if os.path.exists(modelresume):
        try:
            model = load_checkpoint(model, modelresume)

            paras.log.log( "load model:" + modelresume + " success!" )
        except:
            paras.log.log( "load model failed!!!" )
    else:
        paras.log.log("jump load model file")

    #use gpu
    if paras.usegpu:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count())).cuda()


    paras.model = model

    return model




from paras import paras
if __name__ == "__main__":

    myparas = paras()

    model   = get_model( myparas )

    print("end")