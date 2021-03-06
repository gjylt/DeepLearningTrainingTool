import numpy as np
import tqdm


def ap_per_class( conf, pred_cls, target_cls):
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
    tp = [ int(t) for t in target_cls==pred_cls]
    tp = np.array(tp)
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    acc = tp.sum()/len(target_cls)
    print('acc:',acc)

    # Find unique classes
    unique_classes = np.unique(np.concatenate((target_cls, pred_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
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

    return p.tolist(), r.tolist(), ap.tolist(), f1.tolist(), unique_classes.astype(np.int8)


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
if __name__ == '__main__':

    pred_cls   = []#[1, 1, 1, 0, 0, 1, 1, 0, 0, 1]
    target_cls = []#[1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    conf       = []#[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]

    test_path = "/home/withai/Desktop/LCLabelFiles/LCPhase6Test_len24_version2_result.json"
    with open(test_path) as f:
        data = json.load(f)

    for key in data.keys():
        result = data[key]
        pred_cls.append(result[0])
        target_cls.append(result[1])
        conf.append(result[2])

    p, r, ap, f1, labels = ap_per_class( np.array(conf), np.array(pred_cls), np.array(target_cls))

    print(labels)
    print(p)
    print(r)
    print(ap)
    print(f1)
