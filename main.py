#coding=utf-8
from paras import paras
from initmodel import get_model
from initDataloader import get_dataloader



def process_epoch( paras ):

    loss = 0
    metric = {}

    return metric

def train():
    #init paras
    sysparas = paras()

    #init model
    get_model(sysparas)

    #init optimizer


    #init criterion

    #init dataloader


    #start train
    for iepoch in range(sysparas.epoch):

        sysparas.epoch_type = "train"
        train_metric = process_epoch( sysparas )
        if iepoch%sysparas.validstep == 0:
            sysparas.epoch_type = "valid"
            valid_metric = process_epoch(sysparas)










if __name__ == "__main__":



    print("end")