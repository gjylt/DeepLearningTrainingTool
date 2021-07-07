#coding=utf-8
from   paras import paras
from   initmodel import get_model
from   initDataloader import get_dataloader
import torch
import shutil

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def process_epoch( paras ):

    for i, (input, target) in enumerate( paras.dataloader_dict[paras.epoch_type]):
        if paras.usegpu:
            input = input.cuda()
            target= target.cuda()

        output = paras.model(input)
        metric = paras.criterion(output, target)

        if paras.epoch_type == "train":
            paras.optimizer.zero_grad()
            metric["loss"].backward()
            paras.optimizer.step()

    return paras.criterion.model_choice_metric()

def log_metric( iepoch, sysparas, metric):

    metrc_str = str(iepoch) + " " + sysparas.epoch_type + " : "
    for key in metric.keys():
        metrc_str += key + "=" + str(metric[key]) + "; "
    sysparas.log.log(metrc_str)


def save_checkpoint( sysparas, is_best = False):

    save_dict = {
        'epoch': sysparas.iepoch + 1,
        'state_dict': sysparas.model.state_dict(),
        'opt_dict': sysparas.optimizer.state_dict(),
    }
    savepath = '%s/%s_checkpoint.pth.tar' % (sysparas.savedir, sysparas.store_name)
    torch.save( save_dict, savepath )

    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth.tar' % (sysparas.savedir, sysparas.store_name),
                        '%s/%s_best.pth.tar' % (sysparas.savedir, sysparas.store_name))
        sysparas.log.log(f"best model is uqdate {'%s/%s_best.pth.tar' % (sysparas.savedir, sysparas.store_name)}")

def train():
    #init paras
    sysparas = paras()
    #init model
    sysparas = get_model(sysparas)

    #start train
    model_decide_metric = {}
    for iepoch in range(sysparas.epoch):
        sysparas.iepoch = iepoch

        # train
        sysparas.epoch_type = "train"
        train_metric = process_epoch( sysparas )
        #log
        log_metric(iepoch, sysparas, train_metric)

        #valid
        if iepoch%sysparas.validstep == 0:
            sysparas.epoch_type = "valid"
            valid_metric = process_epoch(sysparas)
            #log
            log_metric(iepoch, sysparas,  valid_metric)
            #save model
            if len(model_decide_metric.keys()) == 0 or model_decide_metric["model_choice"] < valid_metric["model_choice"]:
                model_decide_metric = valid_metric
                save_checkpoint(sysparas, is_best=False)

    sysparas.log.log("finish train")


if __name__ == "__main__":

    train()

    print("end")