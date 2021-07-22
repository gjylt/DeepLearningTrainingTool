import numpy as np
import tqdm
import torch

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    ground truth label (np array),true positive为1,false positive为0
        conf:  Objectness value from 0-1 (np array).
        pred_cls: Predicted object classes (np array).
        target_cls: True object classes (np array).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((target_cls, pred_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()  # 累加和列表
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype(np.int8)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

import json
def confusematrix():

    pth = "/home/withai/Desktop/LCLabelFiles/LCPhase6Test_len24_version2_result.json"
    with open(pth,'r') as f:
        data = json.load(f)

    class_dic       = {}
    sequenceResult  = []
    sequenceLabel   = []
    conf            = []
    listJump = [
    ]

    for idkey in data.keys():
        bfind = False
        for jpstr in listJump:
            if idkey.__contains__(jpstr):
                bfind = True

        if bfind:
            print(idkey)
            continue

        ele = data[idkey]
        sequenceResult.append(ele[0])
        sequenceLabel.append(ele[1])
        conf.append(ele[2])
        if ele[1] in class_dic.keys():
            class_dic[ele[1]] +=1
        else:
            class_dic[ele[1]] = 1

    print(class_dic)

    classNum       = 7
    matrix = np.zeros((classNum, classNum))
    for i in range(len(sequenceLabel)):
      matrix[sequenceResult[i], sequenceLabel[i]] += 1

    print("confuse matrix")
    for i in range(classNum):
        strname = ""
        for j in range(classNum):
            strname = strname+ str(int(matrix[i,j]))+"\t"
        print(strname)


    print("confuse matrix prob")
    for i in range(classNum):
        strname = ""
        for j in range(classNum):
            strname = strname+ "{:.2f}".format( matrix[i,j]/class_dic[i] )  +"\t"
        print(strname)

    tp = np.array(torch.tensor(sequenceLabel) == torch.tensor(sequenceResult),np.int).flatten()

    tmpconf = conf

    p, r, ap, f1, labels = ap_per_class(np.array(tp), np.array(tmpconf), np.array(sequenceResult), np.array(sequenceLabel))

    strname = "p"+"\t"
    for i in p:
        strname += str(round(i,3))+"\t"
    print(strname)

    strname = "r"+"\t"
    for i in r:
        strname += str(round(i,3))+"\t"
    print(strname)

    strname = "ap"+"\t"
    for i in ap:
        strname += str(round(i,3))+"\t"
    print(strname)

    strname = "f1"+"\t"
    for i in f1:
        strname += str(round(i,3))+"\t"
    print(strname)


if __name__ =="__main__":
    confusematrix()


