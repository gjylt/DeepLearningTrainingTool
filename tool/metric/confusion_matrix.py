import os

import numpy as np
import tqdm
import torch

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
    i  = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    acc = tp.sum()/len(target_cls)
    print('acc:',acc)

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
def confusematrix(pth):

    # pth = "/home/withai/Desktop/LCLabelFiles/LCPhase_222_len24_2_annotator_test_result.json"

    # pth = "/Users/guan/Desktop/LCPhase_222_len24_2_annotator_test_6phase_limit_bg_num_result.json"

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

    p, r, ap, f1, labels = ap_per_class( np.array(tmpconf), np.array(sequenceResult), np.array(sequenceLabel))

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

from tool.LCDataProcess.LCJson2TRN import get_json_label
from tool.plot_figure import visualizationArray

def confuse_matrix( predlist, labelist):

    classNum   = 7
    matrix     = np.zeros((classNum, classNum))
    class_dic  = {}
    for i in range(len(predlist)):
        label = int(labelist[i])
        pred  = int(predlist[i])
        matrix[ label,  pred ] += 1
        if label not in class_dic.keys():
            class_dic[label]  = 1
        else:
            class_dic[label] += 1



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


def compute_postprocess():
    jsondir       = "/Users/guan/Desktop/trn_100_2_without10_6phase_2annotator_transform"
    visualize_dir = "/Users/guan/Desktop/100-2-visualize"
    jsonlist      = os.listdir(jsondir)
    extract_fps   = 1
    label_nane_dict = get_json_label(extract_fps)

    jsondir1      = "/Users/guan/Desktop/trn_100_2_without10_6phase_2annotator"
    # jsonlist1     = os.listdir(jsondir1)

    compare_dict = {}
    pred_total   = []
    label_total  = []
    pred_orig_total   = []
    for file in jsonlist:
        videoname   = file.split('.')[0]
        resident_id = videoname.split('-')[-1]
        for videoname1 in label_nane_dict.keys():
            if videoname1.__contains__(resident_id):
                jsonpth = os.path.join(jsondir, file)
                with open(jsonpth) as f:
                    predict = json.load(f)

                jsonpth1 = os.path.join(jsondir1,file)
                with open(jsonpth1) as f:
                    predict1 = json.load(f)

                keylis = [int(key) for key in predict1.keys()]
                keymax = np.max( np.array(keylis))+1

                pred_orig = np.zeros(keymax)
                for keyid in predict1.keys():
                    pred_orig[int(keyid)] = predict1[keyid]
                pred_orig = pred_orig.tolist()

                pred_list = predict['result']
                label_list= label_nane_dict[videoname1]

                len_pred      = len(pred_list)
                len_label     = len(label_list)
                len_pred_orig = len( pred_orig)

                max_len  = max(len_label,len_pred,len_pred_orig)
                pred_new = np.zeros(max_len)
                label_new= np.zeros(max_len)
                pred_orig_new = np.zeros(max_len)

                pred_list      = np.array( pred_list )
                pred_orig_list = np.array( pred_orig)
                label_list     = np.array( label_list )
                label_negative = label_list<0

                pred_new[0:len_pred]   = pred_list
                label_new[0:len_label] = label_list
                pred_orig_new[0:len_pred_orig] = pred_orig_list

                pred_new[label_negative]  = 0
                label_new[label_negative] = 0
                pred_orig_new[label_negative] = 0

                pred_total += pred_new.tolist()
                label_total+= label_new.tolist()
                pred_orig_total += pred_orig_new.tolist()

                compare_dict[videoname] = [ label_new, pred_orig_new, pred_new ]

                array_list = [ label_new.tolist(), pred_orig_new.tolist() ,pred_new.tolist() ]
                namelist  = [ "label", "pred_orig","pred"]
                if not os.path.exists(visualize_dir):
                    os.makedirs(visualize_dir)
                visualizationArray( array_list, videoname, visualize_dir, namelist)



    print("orig")
    p, r, ap, f1, label = ap_per_class( np.ones(len(pred_total)), np.array(pred_orig_total), np.array(label_total))

    print(label)
    print(p)
    print(r)
    print(ap)

    confuse_matrix( pred_orig_total, label_total)


    #
    print("after postprocess")
    p, r, ap, f1, label = ap_per_class( np.ones(len(pred_total)), np.array(pred_total), np.array(label_total))

    print(label)
    print(p)
    print(r)
    print(ap)

    confuse_matrix(pred_total, label_total)

    return label_nane_dict


import shutil
def extracted_test_result_sequnce( ):

    pth = "/home/withai/Desktop/LCLabelFiles/LCPhase_parkland_pure_test.json"
    with open(pth,'r') as f:
        data = json.load(f)

    savedir = "/home/withai/Desktop/LCLabelFiles/parkland_test_record"

    datadir = "/home/withai/Pictures/LCFrame/picture_for_parkland_test"
    error_dict = {}
    for path in data.keys():
        result = data[path]
        destpth = os.path.join(datadir,path+'.jpg')
        if not os.path.exists(destpth):
            continue
        label = result[1]
        pred  = result[0]
        # if label != pred:
        if label not in error_dict.keys():
            error_dict[label] ={}

        if pred not in error_dict[label].keys():
            error_dict[label][pred] = [path]
        else:
            error_dict[label][pred].append(path)

    tasklabel = error_dict.keys()

    for label in tasklabel:
        for pred in error_dict[label].keys():

            for errordata in error_dict[label][pred]:
                splistlist = errordata.split("_")
                endnum     = int( splistlist[-1] )
                start      = endnum - 23*8
                if start < 0:
                    continue
                startstr   = "{:0>5}".format(start)

                splistlist[-1] = startstr
                startline  = "_".join(splistlist)

                destpth = os.path.join(datadir, startline + '.jpg')

                if not os.path.exists(destpth):
                    print(destpth)
                    continue

                figdirname   = errordata.split("/")[-1]
                sub_save_dir = os.path.join(savedir,str(label), str(pred), figdirname)

                if not os.path.exists(sub_save_dir):
                    os.makedirs(sub_save_dir)

                for idx in range(start,endnum,8):

                    figlist     = figdirname.split("_")
                    startstr    = "{:0>5}".format(idx)
                    figlist[-1] = startstr
                    startline   = "_".join(figlist)
                    destpth     = os.path.join(sub_save_dir, startline + '.jpg')

                    startstr       = "{:0>5}".format(idx)
                    splistlist[-1] = startstr
                    startline      = "_".join(splistlist)
                    srcpth         = os.path.join(datadir, startline + '.jpg')

                    shutil.copy(srcpth,destpth)

    print(error_dict)


from tool.plot_figure import visualizationArray

def visualize_result():

    pth = "/Users/guan/Desktop/LCPhase_parkland_pure_test.json"
    with open(pth,'r') as f:
        data = json.load(f)

    arraylist_dict = {}
    for name in data.keys():
        videoname = name.split("/")[0]
        sequnce = data[name]
        id      = int( name.split("_")[-1])
        pred    = sequnce[0]
        label   = sequnce[1]
        if videoname not in arraylist_dict.keys():
            arraylist_dict[videoname] = {}
        arraylist_dict[videoname][id] = [label, pred]

    savedir = "/Users/guan/Desktop/visualize"
    namlist = ["pred","label"]

    for videoname in arraylist_dict.keys():

        listdata = arraylist_dict[videoname]

        predlist = []
        labelist = []
        idxlist  = []
        for i in sorted(listdata):
            pred  = listdata[i][1]
            label = listdata[i][0]
            predlist.append(pred-label)
            labelist.append(label)
            idxlist.append(i)

        arraylist = [predlist]
        visualizationArray( arraylist, videoname, savedir, namlist )

if __name__ =="__main__":

    # path = "/Users/guan/Desktop/videoname_phase_list_100-1.json"
    # with open(path) as f:
    #     data = json.load(f)

    # compute_postprocess()

    path = "/home/withai/Desktop/LCLabelFiles/LCPhase_222_len24_2_annotator_6phaseBgNoMoreThanTarget_result.json"
    confusematrix( path )

    # extracted_test_result_sequnce()

    # visualize_result()

    # confusematrix()

    # extracted_test_result_sequnce()


    print("")



