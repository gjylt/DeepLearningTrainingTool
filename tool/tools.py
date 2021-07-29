import cv2
import torch
import numpy as  np
from   numpy import NaN
import copy
import os
import socket

def get_host_ip():
    """
    查询本机ip地址
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

def sig_class_IoU(  target, pred, nclass):
    pred = pred.cuda()
    target = target.cuda()

    # target = target.squeeze(0)
    pred_indx = pred == nclass
    target_indx = target == nclass

    # print( target.shape,pred.shape)

    intersection = pred_indx[target_indx].sum().float()
    union = (pred_indx.sum() + target_indx.sum() - intersection).float()

    if union == 0:
        iou = torch.tensor(np.array(NaN)).cuda()
    else:
        iou = intersection / union

    return intersection, union, iou

def sig_class_PR(  target, pred, nclass):
    pred = pred.cuda()
    target = target.cuda()
    # target = target.squeeze(0)

    pred_indx = pred == nclass
    target_indx = target == nclass
    pred_index_inverse = ~pred_indx
    target_indx_inverse = ~target_indx

    tp = pred_indx[target_indx].sum().float()
    fp = pred_indx[target_indx_inverse].sum().float()
    fn = pred_index_inverse[target_indx].sum().float()

    eps = 1e-7
    # if tp == 0:
    #     p = NaN
    #     r = NaN
    # else:
    p = tp / (tp + fp +eps)
    r = tp / (tp + fn +eps)

    return tp, fp, fn, p, r



def save_img( mask, pred, miou, filename, paras ):

    ROOT = ""
    if paras.train_paras["process_phase"] == "train":
        ROOT = paras.paras["train_dataset"]
    elif paras.train_paras["process_phase"] == "val":
        ROOT = paras.paras["val_dataset"]
    elif paras.train_paras["process_phase"] == "test":
        ROOT = paras.paras["test_dataset"]

    savepath         = paras.paras["test_result"]
    num_classes      = paras.data_paras["numclass"]
    bg_color         = paras.data_paras["color_list"]

    file_len = len(filename)
    pred = pred.cpu().data.numpy()
    mask = mask.cpu().data.numpy()


    for idx in range(file_len):

        name_t = filename[idx]
        pred_t = np.squeeze( pred[idx,:,:] )
        mask_t = np.squeeze( mask[idx,:,:] )

        # print(pred_t.shape,mask_t.shape )

        file_path = os.path.join(ROOT, name_t+".jpg")
        size_img  = paras.data_paras["input_shape"]
        im  = cv2.imread( file_path )
        img = cv2.resize(im, size_img)

        im1 = copy.deepcopy(img)
        im2 = copy.deepcopy(img)

        for nclass in range(1, num_classes ):
            tmpidx = mask_t  == nclass
            im1[ tmpidx ] = np.array( bg_color[nclass-1] )
            tmpidx = pred_t  == nclass
            im2[ tmpidx ] = np.array( bg_color[nclass-1] )

        img = np.concatenate([img,im1, im2], axis=1 )
        cv2.imwrite( os.path.join( savepath , name_t + '_' + str(miou)+'.jpg'), img)


def split_dir(pth):

    tmp = pth.split("/")
    idx = len(tmp)-1
    while tmp[idx] == "" and idx >= 0:
        idx -= 1

    if idx >= 0:
        return "/".join(tmp[0:idx]),tmp[idx]
    else:
        return pth,None