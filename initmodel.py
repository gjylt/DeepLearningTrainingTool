from   model.TRN.models import TSN
import torch
import os
from   torch.optim import SGD
import copy
from   initDataloader import get_dataloader

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


class trn_loss():
    def __init__(self, paras):

        self.metric = {
            "num":0
        }
        self.crossentropy =  torch.nn.CrossEntropyLoss()
        if paras.usegpu:
            self.crossentropy.cuda()


    def model_choice_metric(self, ):
        metric = copy.deepcopy( self.metric )
        metric["model_choice"] = metric["average_loss"]/metric["num"]
        return metric

    def calcloss(self, input, target):

        self.metric["num"] += input.shape(0)

        loss = self.crossentropy( input, target )
        self.metric["loss"] = loss
        if "average_loss" not in self.metric.keys():
            self.metric["average_loss"] = 0
        self.metric["average_loss"] += loss

        acc = torch.sum( input == target )
        self.metric["acc"] = acc
        if "average_acc" not in self.metric.keys():
            self.metric["average_acc"] = 0
        self.metric["average_acc"] += loss

        return self.metric



def get_model( sysparas ):
    modelname   = sysparas.modelname
    num_class   = sysparas.num_class
    modelresume = sysparas.modelresume

    gpunum      = torch.cuda.device_count()
    sysparas.log.log("gpunum:" + str(gpunum) )
    if gpunum > 0:
        sysparas.usegpu = True

    model = None
    #init model
    if modelname == "TRN":
        sysparas.log.log("init model, model name:" + modelname )

        num_segments    = 8
        modality        = "RGB"
        arch            = "BNInception"
        consensus_type  = 'TRNmultiscale'
        dropout         = 0.8
        img_feature_dim = 256
        no_partialbn    = False
        sysparas.model     = TSN(num_class, num_segments, modality,
                    base_model= arch,
                    consensus_type= consensus_type,
                    dropout= dropout,
                    img_feature_dim=img_feature_dim,
                    partial_bn=not no_partialbn)

        sysparas = get_dataloader(sysparas)

        # init optimizer
        if len(sysparas.learning_rate_policy.keys()) > 0:
            policies = sysparas.learning_rate_policy
        else:
            policies = sysparas.model.parameters()

        sysparas.optimizer = SGD(policies,
                                 sysparas.learning_rate,
                                 momentum=sysparas.momentum,
                                 weight_decay=sysparas.weight_decay
                                 )

        # init criterion
        sysparas.criterion = trn_loss(sysparas)


    #load pretrain parameters
    if os.path.exists(modelresume):
        try:
            sysparas.model = load_checkpoint(sysparas.model, modelresume)
            sysparas.log.log( "load model:" + modelresume + " success!" )
        except:
            sysparas.log.log( "load model failed!!!" )
    else:
        sysparas.log.log("jump load model file")

    #use gpu
    if sysparas.usegpu:
        sysparas.model = torch.nn.DataParallel( sysparas.model, device_ids=range(torch.cuda.device_count())).cuda()



    return sysparas




from paras import paras
if __name__ == "__main__":

    myparas = paras()

    model   = get_model( myparas )

    print("end")