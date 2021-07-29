#coding=utf-8
import torch
from   torch.autograd import Variable as V
import cv2
import os
import numpy as np
from   numpy import NaN
from   model.dlinknet.dinknet import DinkNet34,DinkNet101,DinkNet50
import json
import torch.nn as nn
import shutil
from sklearn.metrics import confusion_matrix
from tool.tools import sig_class_IoU,sig_class_PR,save_img
# import xlwt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# instrument_classes_dict = {
#     "1": "atraumatic forceps",
#     "2": "cautery hook",
#     "3": "absorbable clip",
#     "4": "trocar",
#     "5": "gauze",
#     "6": "maryland dissecting forceps",
#     "7": "straight dissecting forceps",
#     "8": "dissecting forceps",
#     "9": "claw grasper",
#     "10": "metal clip",
#     "11": "clip applier",
#     "12": "scissor",
#     "13": "specimen bag",
#     "14": "aspirator"
# }
#
# instrument_color_dict = {
#     "1": [50,114,246],   #"atraumatic forceps"
#     "2": [238,114,109],  #cautery hook
#     "3": [18,0,255],     #"absorbable clip",
#     "4": [255,255,255],  #"trocar",
#     "5": [248,248,148],  #"gauze",
#     "6": [255,255,255],  #"maryland dissecting forceps",
#     "7": [255,255,255],  #"straight dissecting forceps",
#     "8": [13,44,246],    #"dissecting forceps",
#     "9": [113,1,255],    #"claw grasper",
#     "10": [255,255,255], #"metal clip",
#     "11": [233,114,109], #"clip applier",
#     "12": [255,255,255], #"scissor",
#     "13": [255,255,255], #"specimen bag",
#     "14": [255,255,255], #"aspirator"
# }

# color_dict =  {
#     "1": [238,114,109],  #"cautery hook",
#     "2": [255,255,255],  #"trocar",
#     "3": [13,44,246],    #"dissecting forceps",
#     "4": [18,0,255],     #"absorbable clip",
#     "5": [50,114,246],   #"atraumatic forceps",
#     "6": [233,114,109], #"clip applier",
#     "7": [255,255,255], #"metal clip",
#     "8": [255,255,255], #"grasper",
#     "9": [255,255,255], #"scissor"
#   }
#
# classes_dict =  {
#     "1": "cautery hook",
#     "2": "trocar",
#     "3": "dissecting forceps",
#     "4": "absorbable clip",
#     "5": "atraumatic forceps",
#     "6": "clip applier",
#     "7": "metal clip",
#     "8": "grasper",
#     "9": "scissor"
#   }

instrument_classes_dict = {
    "3": "cautery hook",
    "7": "maryland dissecting forceps",
    "11": "claw grasper",
    "13": "specimen bag",
    "12": "dissecting forceps",
    "15": "aspirator",
    "10": "scissor",
    "2": "atraumatic forceps",
    "1": "trocar",
    "5": "absorbable clip",
    "8": "gauze",
    "9": "straight dissecting forceps",
    "6": "metal clip",
    "4": "clip applier",
    "14": "puncture needle"
  }

instrument_color_dict = {
    "1": [50,114,246],   #"atraumatic forceps"
    "2": [238,114,109],  #cautery hook
    "3": [18,0,255],     #"absorbable clip",
    "4": [255,255,255],  #"trocar",
    "5": [248,248,148],  #"gauze",
    "6": [255,255,255],  #"maryland dissecting forceps",
    "7": [255,255,255],  #"straight dissecting forceps",
    "8": [13,44,246],    #"dissecting forceps",
    "9": [113,1,255],    #"claw grasper",
    "10":[255,255,255], #"metal clip",
    "11":[233,114,109], #"clip applier",
    "12":[255,255,255], #"scissor",
    "13":[255,255,255], #"specimen bag",
    "14":[255,255,255], #"aspirator"
    "15":[255,125,0]
}


anatomy_classes_dict ={
    "1": "gallbladder",
    "2": "liver",
    "3": "cystic bed",
    "4": "cystic artery",
    "5": "stomach and duodenum",
    "6": "cystic duct",
    "7": "common bile duct"
  }


anatomy_color_dict = {
    "1": [24,244,0], #"gallbladder",
    "2": [255,90,0], #"liver",
    "7": [6,8,212], #"common bile duct",
    "5": [255,255,255], #"stomach and duodenum",
    "6": [0,255,150], #"cystic duct",
    "4": [232,58,58], #"cystic artery",
    "3": [0,0,0], #"cystic bed"
}


cvs_anatomy_classes_dict ={
    "1": "gallbladder",
    "2": "cystic bed",
    "3": "cystic artery",
    "4": "tri angle"
  }


cvs_anatomy_color_dict = {
    "1": [24,244,0], #"gallbladder",
    "2": [255,90,0], #"liver",
    "3": [6,8,212], #"common bile duct",
    "4": [255,255,255], #"stomach and duodenum",
    # "5": [0,255,150], #"cystic duct",
    # "6": [232,58,58], #"cystic artery",
    # "7": [0,0,0], #"cystic bed"
}


classes_dict = cvs_anatomy_classes_dict
color_dict   = anatomy_color_dict


show_cls_list = [1,2,3,5,8,9,11]


def load(net,model_path):
    pretrained_dict = torch.load(model_path)
    """加载torchvision中的预训练模型和参数后通过state_dict()方法提取参数
       也可以直接从官方model_zoo下载：
       pretrained_dict = model_zoo.load_url(model_urls['resnet152'])"""
    print("begin load")
    model_dict = net.state_dict()

    # 将pretrained_dict里不属于model_dict的键剔除掉
    update_dict = {}
    for k, v in model_dict.items():

        if k in pretrained_dict:
            # print("load layer:",k)
            update_dict[k] = pretrained_dict[k]
        else:
            print(k, "not in pretrained model!")

    # 更新现有的model_dict
    model_dict.update(update_dict)

    # 加载我们真正需要的state_dict
    net.load_state_dict(model_dict)

    return net

import copy


def test_vedeio():


    num_classes = len(classes_dict.keys())+1


    net = DinkNet34(num_classes ).cuda()
    net = torch.nn.DataParallel( net, device_ids=range(torch.cuda.device_count()))
    size_img    = (512, 512)

    model_path  = "/home/withai/project/DinkNet34_anatomy_version2_valid.th"
    net         = load(net, model_path)

    videopth     = "/mnt/video/LC10000/CompleteVideo/hospital_id=14/surgery_id=612/video/20210111-LC-YR-MYH-CSR-71_HD1080.mp4"
    capture      = cv2.VideoCapture(videopth)
    frame_count  = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = float(capture.get(cv2.CAP_PROP_FPS))

    # 定义编码格式mpge-4
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # 定义视频文件输入对象
    ori_size = (frame_width,frame_height)
    # size = size_img

    save_video_name = "LC-CSR-4-anatomy-segment2.mp4"
    video_savepth   = os.path.join("./",save_video_name)
    outVideo        = cv2.VideoWriter(video_savepth, fourcc, fps, ori_size)

    ret = True

    instrument_statistics = {}

    capture.set(cv2.CAP_PROP_POS_FRAMES,int(4*60*fps))


    frame_id = 0
    video_detect_result = {}
    while(ret and frame_id < int(1*60*fps) ): #
        # if frame_id > 1000:
        #     break
        ret, frame = capture.read()
        if not ret:
            break

        ori_img = copy.deepcopy(frame)

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (frame_height != size_img[0]) or (frame_width != size_img[1]):
            frame = cv2.resize(frame, (size_img[0], size_img[1]) )


        frame = frame.transpose(2, 0, 1)
        frame = np.expand_dims(frame, axis=0)
        frame = V(torch.Tensor(np.array(frame, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        mask,_   = net.forward(frame)
        mask    = mask.squeeze().cpu().data.numpy()
        if num_classes == 1:
            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
            mask = torch.Tensor(mask)
        else:
            sp   = mask.shape
            mask = np.reshape(mask, (-1, num_classes ))
            mask = nn.Softmax(dim=1)(torch.Tensor(mask))
            mask = torch.argmax(mask, dim=1).view(sp[0], sp[1])

        mask = mask.numpy()
        mask = np.concatenate([mask[:,:,None],mask[:,:,None],mask[:,:,None]],axis=2 )
        mask = np.array(mask,np.uint8)
        mask = cv2.resize(mask,ori_size)
        mask = np.array(mask,np.uint8)
        # kernel = np.ones((3, 3), np.uint8)
        # mask = cv2.erode(mask, kernel)
        mask = mask[:,:,0].squeeze()

        video_detect_result[frame_id] = {}

        for class_id in range(1,num_classes):
            class_key   = str(class_id)
            indx = mask == class_id
            classname = classes_dict[class_key]

            gray = copy.deepcopy(mask)
            gray[indx] = 255
            gray[mask!= class_id] = 0
            ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            contours_tmp = []
            for idx in range(len(contours)):

                shp = contours[idx]
                lenth = len(shp)
                if lenth > 100:
                    contours_tmp.append(shp.tolist())

            video_detect_result[frame_id][class_id] = contours_tmp

            sum = np.sum(indx)
            if sum < 500:
                if classname not in instrument_statistics.keys():
                    instrument_statistics[classname] = {}
                    instrument_statistics[classname]["list"] = []
                instrument_statistics[classname]["state"] = False
                continue

            if classname not in instrument_statistics.keys():
                instrument_statistics[classname] = {}
                instrument_statistics[classname]["state"] = True
                instrument_statistics[classname]["list"]  = []
                instrument_statistics[classname]["list"].append( [frame_id/fps,frame_id/fps] )
            else:
                state = instrument_statistics[classname]["state"]
                if state == False:
                    instrument_statistics[classname]["state"] = True
                    if "list" not in instrument_statistics[classname].keys():
                        instrument_statistics[classname]["list"] = []
                    instrument_statistics[classname]["list"].append([frame_id/fps, frame_id/fps])
                else:
                    instrument_statistics[classname]["state"]       = True
                    instrument_statistics[classname]["list"][-1][1] = frame_id/fps

            tmp          = np.zeros_like(ori_img)
            tmp[indx, :] = np.array(color_dict[class_key])
            alpha = 0.4
            ori_img[indx, :] = np.array((1-alpha)*ori_img[indx, :] + alpha*tmp[indx, :], np.int)

        frame_id += 1

        outVideo.write(ori_img)

        print(frame_id,"/",frame_count)

    capture.release()
    outVideo.release()

    video_detect_result_pth = "./video_detect_result.json"
    with open(video_detect_result_pth, "w", encoding='utf-8') as f:
        json.dump(video_detect_result, f, indent=2, sort_keys=True, ensure_ascii=False)
    #



def averge_metric(mask,pred,phase_metric_dic,phase_id_name_dic):

    clsnum = len(phase_id_name_dic.keys())
    labels = np.arange(1,clsnum+1).tolist()
    total  = len(mask)

    for label in labels:
        labeltp  = [1,0]
        tmp_mask = copy.deepcopy(mask)
        tmp_pred = copy.deepcopy(pred)

        idx = tmp_mask==label
        tmp_mask[idx] = 1
        idx =(1-idx).astype(np.bool)
        tmp_mask[idx] = 0

        idx = tmp_pred==label
        tmp_pred[idx] = 1
        idx =(1-idx).astype(np.bool)
        tmp_pred[idx] = 0

        c3 = confusion_matrix(tmp_mask,tmp_pred,labels=labeltp)
        tp = c3[0,0]
        fn = c3[0,1]
        fp = c3[1,0]
        tn = c3[1,1]

        ac = np.nanmin( np.array([0.0,(tp+tn)/total]) )
        p  = np.nanmin( np.array([0.0, tp/(tp+fp)]) )
        r  = np.nanmin( np.array([0.0, tp/(tp+fn)]) )

        name = phase_id_name_dic[str(label)]
        if name in phase_metric_dic.keys():
            phase_metric_dic[name]["ac"].append(ac)
            phase_metric_dic[name]["p"].append(p)
            phase_metric_dic[name]["r"].append(r)
        else:
            phase_metric_dic[name] = {}
            phase_metric_dic[name]["ac"] = [ac]
            phase_metric_dic[name]["p"]  = [p]
            phase_metric_dic[name]["r"]  = [r]

    return phase_metric_dic



def test_picture():

    paras       = loadparas()

    num_classes = len( classes_dict.keys() )+1
    # num_classes = 4
    numclass    = num_classes

    paras.data_paras["numclass"] = num_classes
    net = DinkNet50(numclass ).cuda()
    net = torch.nn.DataParallel( net, device_ids=range(torch.cuda.device_count()))
    size_img    = (512, 512)

    bsave          = True
    excle_save_pth = "./statistic.xls"
    s3_statistic_result = "Instrument/cvs_statistic_dlink34_val.xls"
    save_dir    = "./result/"
    model_path  = "/home/withai/project/DinkNet34_anatomy_version2_valid.th"
    net         = load(net, model_path)

    dataset_dir = "/root/data/val/"
    dataset_dir = "/root/dlink_data/val/"
    imglis      = [pth for pth in os.listdir(dataset_dir) if pth.__contains__(".jpg") ]

    # 创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding='utf-8')

    # 创建一个worksheet
    # dataset_name = os.path.split(dataset_dir)
    worksheet    = workbook.add_sheet("instrument_merge")

    total_iou_list   = []
    total_p_list     = []
    total_r_list     = []

    class_iou = {}
    class_p   = {}
    class_r   = {}
    img_num   = 0
    for imgname in imglis:

        imgpth  = os.path.join(dataset_dir,imgname)
        frame   = cv2.imread(imgpth)
        pngpth  = imgpth.replace(".jpg",".png")
        if not os.path.exists(pngpth):
            continue
        mask    = cv2.imread(pngpth)

        # total_class = 16
        # tmpmask = copy.deepcopy(mask)
        # for id in range(1, total_class):
        #
        #     if id == 3:
        #         mask[tmpmask == id] = 1
        #     elif id == 5:
        #         mask[tmpmask == id] = 2
        #     elif id == 2:
        #         mask[tmpmask == id] = 3
        #     else:
        #         mask[tmpmask == id] = 0


        ori_img = copy.deepcopy(frame)

        shp     = ori_img.shape

        frame_height = shp[0]
        frame_width  = shp[1]

        ori_size = (frame_width,frame_height)

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (frame_height != size_img[0]) or (frame_width != size_img[1]):
            frame = cv2.resize(frame, (size_img[0], size_img[1]) )
            mask  = cv2.resize(mask, (size_img[0], size_img[1]) )


        ori_img = copy.deepcopy(frame)
        ori_img2= copy.deepcopy(frame)
        ori_img3= copy.deepcopy(frame)

        frame = frame.transpose(2, 0, 1)
        frame = np.expand_dims(frame, axis=0)
        frame = V(torch.Tensor(np.array(frame, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        pred,_   = net.forward(frame)
        pred   = pred.squeeze().cpu().data.numpy()
        if num_classes == 1:
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred = torch.Tensor(pred)
        else:
            sp = pred.shape
            pred = np.reshape(pred, (-1, num_classes ))
            pred = nn.Softmax(dim=1)(torch.Tensor(pred))
            pred = torch.argmax(pred, dim=1).view(sp[0], sp[1])

        img_iou_list = []
        img_p_list   = []
        img_r_list   = []
        mask = torch.from_numpy(mask[:, :, 0])

        for nclass in range(1, numclass):


            if bsave:
                class_key    = str(nclass)
                indx         = pred == nclass
                tmp          = np.zeros_like(ori_img)
                tmp[indx, :] = np.array(color_dict[class_key])
                alpha = 0.4
                ori_img[indx, :] = np.array((1-alpha)*ori_img[indx, :] + alpha*tmp[indx, :], np.int)

                class_key = str(nclass)
                indx      = mask == nclass
                tmp       = np.zeros_like(ori_img3)
                tmp[indx, :] = np.array(color_dict[class_key])
                alpha = 0.4
                ori_img3[indx, :] = np.array((1 - alpha) * ori_img3[indx, :] + alpha * tmp[indx, :], np.int)


            intersection, union, iou = sig_class_IoU(mask.squeeze(), pred.squeeze(), nclass)
            tp, fp, fn, p, r = sig_class_PR(mask.squeeze(), pred.squeeze(), nclass)

            img_iou_list.append(iou.cpu().numpy())
            img_p_list.append(p.cpu().numpy())
            img_r_list.append(r.cpu().numpy())

            nclass = str(nclass)
            if nclass in class_iou.keys():
                class_iou[nclass].append(iou.cpu().numpy())
                class_p[nclass].append(p.cpu().numpy())
                class_r[nclass].append(r.cpu().numpy())
            else:
                class_iou[nclass] = [iou.cpu().numpy()]
                class_p[nclass]   = [p.cpu().numpy()]
                class_r[nclass]   = [r.cpu().numpy()]

        img_num += 1

        miou = np.nanmean(img_iou_list)
        mp   = np.nanmean(img_p_list)
        mr   = np.nanmean(img_r_list)
        total_iou_list.append(miou)
        total_p_list.append(mp)
        total_r_list.append(mr)

        if bsave:
            save_path = os.path.join( save_dir, imgname )
            # save_path = save_path.replace(".jpg","_"+str(miou)+".jpg")
            total_img = np.concatenate([ori_img2, ori_img3, ori_img],axis=1)
            cv2.imwrite( save_path, total_img )

            # pred = pred[:,:,np.newaxis]
            # pred = np.concatenate([pred,pred,pred], axis=1)
            # save_path = save_path.replace(".jpg","_pred.png")
            # cv2.imwrite(save_path,pred)
            #
            # mask = mask[:,:,np.newaxis]
            # mask = np.concatenate([mask,mask,mask], axis=1)
            # save_path = save_path.replace("_pred.jpg","_mask.png")
            # cv2.imwrite(save_path,mask)

        print( img_num,  "/",  len(imglis) )

    class_average_iou = {}
    class_average_p   = {}
    class_average_r   = {}
    indx_name = paras.data_paras["indx_name"]

    worksheet.write(0, 0, "classs id")
    worksheet.write(0, 1, "P")
    worksheet.write(0, 2, "R")
    worksheet.write(0, 3, "mIoU")
    for nclass in range(1, numclass):

        nclass    = str(nclass)
        classname = classes_dict[nclass]
        # if nclass == "1":
        #     classname = classes_dict["3"]
        # elif nclass == "2":
        #     classname = classes_dict["5"]
        # elif nclass == "3":
        #     classname = classes_dict["2"]

        class_average_iou[ classname ] = np.nanmean(class_iou[nclass])
        class_average_p[ classname ]   = np.nanmean(class_p[nclass])
        class_average_r[ classname ]   = np.nanmean(class_r[nclass])

        worksheet.write( int(nclass), 0,  classname )
        worksheet.write( int(nclass), 1, float(class_average_p[ classname ]) )
        worksheet.write( int(nclass), 2, float(class_average_r[ classname ]) )
        worksheet.write( int(nclass), 3, float(class_average_iou[ classname ]) )


    if len(total_iou_list) > 0 and len(total_p_list) > 0 and len(total_r_list) > 0:
        average_iou = np.nanmean(total_iou_list)
        average_p   = np.nanmean(total_p_list)
        average_r   = np.nanmean(total_r_list)

        worksheet.write( numclass, 0, "average")
        worksheet.write( numclass, 1, float(average_p) )
        worksheet.write( numclass, 2, float(average_r) )
        worksheet.write( numclass, 3, float(average_iou) )


    workbook.save( excle_save_pth )

    data_reader = paras.awsreader["data"]
    try:
        data_reader.readwrite_file( excle_save_pth, s3_statistic_result, "w" )
    except:
        print("upload LC-CSR-4-anatomy.avi failed!")

def test_picture2():

    # paras       = loadparas()

    num_classes = len( anatomy_classes_dict.keys() )+1
    # num_classes = 4
    numclass    = num_classes

    # paras.data_paras["numclass"] = num_classes
    net = DinkNet34(numclass ).cuda()
    net = torch.nn.DataParallel( net, device_ids=range(torch.cuda.device_count()))
    size_img    = (512, 512)

    bsave          = True
    excle_save_pth = "./statistic.xls"
    save_dir    = "/home/withai/Desktop/extracted_lc_img_visualize_2"
    model_path  = "/home/withai/project/DinkNet34_anatomy_version2_valid.th"
    net         = load(net, model_path)

    dataset_dir = "/home/withai/Desktop/extracted_lc_img/"
    dirlist     = os.listdir(dataset_dir)
    imglis      = []
    for imgdir in dirlist:
        subdir  = os.path.join(dataset_dir,imgdir)
        filelist= os.listdir(subdir)
        for filename in filelist:
            imglis.append( os.path.join(subdir, filename) )

    # imglis      = [pth for pth in os.listdir(dataset_dir) if pth.__contains__(".jpg") ]

    have_label = False

    # if have_label:
    #     # 创建一个workbook 设置编码
    #     workbook = xlwt.Workbook(encoding='utf-8')
    #     # 创建一个worksheet
    #     # dataset_name = os.path.split(dataset_dir)
    #     worksheet    = workbook.add_sheet("instrument_merge")

    total_iou_list   = []
    total_p_list     = []
    total_r_list     = []

    class_iou = {}
    class_p   = {}
    class_r   = {}
    img_num   = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gallbladder_area = []
    for imgpth in imglis:

        subdir       = imgpth.split('/')[-2]
        sub_save_dir = os.path.join(save_dir,subdir)
        if not os.path.exists(sub_save_dir):
            os.makedirs(sub_save_dir)

        imgname      = imgpth.split('/')[-1]
        # imgpth  = os.path.join(dataset_dir,imgname)
        frame   = cv2.imread(imgpth)
        if have_label:
            pngpth  = imgpth.replace(".jpg",".png")
            if not os.path.exists(pngpth):
                continue
            mask    = cv2.imread(pngpth)

        # total_class = 16
        # tmpmask = copy.deepcopy(mask)
        # for id in range(1, total_class):
        #
        #     if id == 3:
        #         mask[tmpmask == id] = 1
        #     elif id == 5:
        #         mask[tmpmask == id] = 2
        #     elif id == 2:
        #         mask[tmpmask == id] = 3
        #     else:
        #         mask[tmpmask == id] = 0


        ori_img = copy.deepcopy(frame)

        shp     = ori_img.shape

        frame_height = shp[0]
        frame_width  = shp[1]

        ori_size = (frame_width,frame_height)

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (frame_height != size_img[0]) or (frame_width != size_img[1]):
            frame = cv2.resize(frame, (size_img[0], size_img[1]) )
            if have_label:
                mask  = cv2.resize(mask, (size_img[0], size_img[1]) )

        ori_img = copy.deepcopy(frame)
        ori_img2= copy.deepcopy(frame)
        ori_img3= copy.deepcopy(frame)

        frame = frame.transpose(2, 0, 1)
        frame = np.expand_dims(frame, axis=0)
        frame = V(torch.Tensor(np.array(frame, np.float32) / 255.0 * 3.2 - 1.6).cuda())

        pred,_   = net.forward(frame)
        pred   = pred.squeeze().cpu().data.numpy()
        if num_classes == 1:
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred = torch.Tensor(pred)
        else:
            sp = pred.shape
            pred = np.reshape(pred, (-1, num_classes ))
            pred = nn.Softmax(dim=1)(torch.Tensor(pred))
            pred = torch.argmax(pred, dim=1).view(sp[0], sp[1])

        img_iou_list = []
        img_p_list   = []
        img_r_list   = []

        if have_label:
            mask = torch.from_numpy(mask[:, :, 0])

        numclass = 2
        for nclass in range(1, numclass):

            if bsave:
                class_key    = str(nclass)
                indx         = pred == nclass

                area  =   torch.sum(indx).numpy()



                gallbladder_area.append([imgname,area,512*512, area*.10/512/512])

                tmp          = np.zeros_like(ori_img)
                tmp[indx, :] = np.array(color_dict[class_key])
                alpha = 0.4
                ori_img[indx, :] = np.array((1-alpha)*ori_img[indx, :] + alpha*tmp[indx, :], np.int)

                if have_label:
                    class_key = str(nclass)
                    indx      = mask == nclass
                    tmp       = np.zeros_like(ori_img3)
                    tmp[indx, :] = np.array(color_dict[class_key])
                    alpha = 0.4
                    ori_img3[indx, :] = np.array((1 - alpha) * ori_img3[indx, :] + alpha * tmp[indx, :], np.int)

            if have_label:
                intersection, union, iou = sig_class_IoU(mask.squeeze(), pred.squeeze(), nclass)
                tp, fp, fn, p, r = sig_class_PR(mask.squeeze(), pred.squeeze(), nclass)

                img_iou_list.append(iou.cpu().numpy())
                img_p_list.append(p.cpu().numpy())
                img_r_list.append(r.cpu().numpy())

                nclass = str(nclass)
                if nclass in class_iou.keys():
                    class_iou[nclass].append(iou.cpu().numpy())
                    class_p[nclass].append(p.cpu().numpy())
                    class_r[nclass].append(r.cpu().numpy())
                else:
                    class_iou[nclass] = [iou.cpu().numpy()]
                    class_p[nclass]   = [p.cpu().numpy()]
                    class_r[nclass]   = [r.cpu().numpy()]

        img_num += 1

        if have_label:
            miou = np.nanmean(img_iou_list)
            mp   = np.nanmean(img_p_list)
            mr   = np.nanmean(img_r_list)
            total_iou_list.append(miou)
            total_p_list.append(mp)
            total_r_list.append(mr)

        if bsave:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_path = os.path.join( save_dir, imgname )
            save_path = os.path.join( sub_save_dir,imgname)
            # save_path = save_path.replace(".jpg","_"+str(miou)+".jpg")

            if have_label:
                total_img = np.concatenate([ori_img2, ori_img3, ori_img],axis=1)
                cv2.imwrite( save_path, total_img )
            else:
                total_img = np.concatenate([ori_img2, ori_img],axis=1)
                cv2.imwrite( save_path, total_img )

            # pred = pred[:,:,np.newaxis]
            # pred = np.concatenate([pred,pred,pred], axis=1)
            # save_path = save_path.replace(".jpg","_pred.png")
            # cv2.imwrite(save_path,pred)
            #
            # mask = mask[:,:,np.newaxis]
            # mask = np.concatenate([mask,mask,mask], axis=1)
            # save_path = save_path.replace("_pred.jpg","_mask.png")
            # cv2.imwrite(save_path,mask)

        print( img_num,  "/",  len(imglis) )

    fp = open("save_aerea.txt","w")
    for subsequnce in gallbladder_area:
        linestr = subsequnce[0]+"\t"+ str(subsequnce[1]) + "\t"+ str(subsequnce[2]) + "\t"+  "%.3f"%( subsequnce[2] )
        fp.writelines( linestr)
    fp.close()


    if have_label:
        class_average_iou = {}
        class_average_p   = {}
        class_average_r   = {}
        # indx_name = paras.data_paras["indx_name"]

        # worksheet.write(0, 0, "classs id")
        # worksheet.write(0, 1, "P")
        # worksheet.write(0, 2, "R")
        # worksheet.write(0, 3, "mIoU")
        for nclass in range(1, numclass):

            nclass    = str(nclass)
            classname = classes_dict[nclass]
            # if nclass == "1":
            #     classname = classes_dict["3"]
            # elif nclass == "2":
            #     classname = classes_dict["5"]
            # elif nclass == "3":
            #     classname = classes_dict["2"]

            class_average_iou[ classname ] = np.nanmean(class_iou[nclass])
            class_average_p[ classname ]   = np.nanmean(class_p[nclass])
            class_average_r[ classname ]   = np.nanmean(class_r[nclass])

            # worksheet.write( int(nclass), 0,  classname )
            # worksheet.write( int(nclass), 1, float(class_average_p[ classname ]) )
            # worksheet.write( int(nclass), 2, float(class_average_r[ classname ]) )
            # worksheet.write( int(nclass), 3, float(class_average_iou[ classname ]) )


        if len(total_iou_list) > 0 and len(total_p_list) > 0 and len(total_r_list) > 0:
            average_iou = np.nanmean(total_iou_list)
            average_p   = np.nanmean(total_p_list)
            average_r   = np.nanmean(total_r_list)

            # worksheet.write( numclass, 0, "average")
            # worksheet.write( numclass, 1, float(average_p) )
            # worksheet.write( numclass, 2, float(average_r) )
            # worksheet.write( numclass, 3, float(average_iou) )

        # workbook.save( excle_save_pth )

def draw_label( save_path, img_pth, json_path, name2idx ,img_size):

    info  = {}
    data  = json.load(open(json_path))
    info["h"]   = data['imageHeight']
    info["w"]   = data['imageWidth']
    info["obj"] = []
    shapes      = data['shapes']

    img    = cv2.imread(img_pth)
    mask   = np.zeros_like(img)

    for shpid in range(len(shapes)):
        shp       = data['shapes'][shpid]

        labelname = shp['label']
        labelname = labelname.lower()
        data['shapes'][shpid]['label'] = labelname
        shapetype = shp['shape_type']
        points    = shp['points']

        if labelname == '0' or labelname == "" or labelname == "''":
            continue

        if shapetype == "polygon":

            if labelname not in name2idx.keys():
                continue
            else:
                classid = name2idx[labelname]

            points = np.array([points], np.int)
            mask   = cv2.drawContours(mask, points, 0, (classid, classid, classid), -1)

    save_path = save_path.replace('.jpg', '.png')
    mask      = cv2.resize(mask, img_size)
    cv2.imwrite(save_path,mask)

    return mask

def file_name(file_dir,tag):
    dir_list = []
    for root, dirs, files in os.walk(file_dir):
        #print('root_dir:', root)  # 当前目录路径
        #print('sub_dirs:', dirs)  # 当前路径下所有子目录
        #print('files:', files)    # 当前路径下所有非目录子文件
        filenames = [ name for name in files if name.__contains__(tag) ]
        if len(filenames)>0:
            dir_list.append([root,filenames])

    return dir_list

def TransposeJson2PNG():

    dataRoot  = "/root/service-code/data/ori/total/"
    tag       = ".jpg"
    dir_lists = file_name(dataRoot, tag)

    name2idx  = {
    "absorbable clip": 3,
    "aspirator": 14,
    "atraumatic forceps": 1,
    "cautery hook": 2,
    "claw grasper": 9,
    "clip applier": 11,
    "dissecting forceps": 8,
    "gauze": 5,
    "maryland dissecting forceps": 6,
    "metal clip": 10,
    "scissor": 12,
    "specimen bag": 13,
    "straight dissecting forceps": 7,
    "trocar": 4
   }

    img_size  = (512,512)
    dir_idx   = 0
    for dir_list in dir_lists:
        dir       = dir_list[0]
        file_list = dir_list[1]


        jsondir = dir
        imgdir  = dir
        savedir = dir

        for file in file_list:
            filename  = file.split('.')[0]
            json_path = os.path.join(jsondir, filename + '.json')
            img_pth   = os.path.join(imgdir,  filename + '.jpg')
            save_path = os.path.join(savedir, filename + '.png')

            if not os.path.exists(json_path):
                continue

            if not os.path.exists(img_pth):
                continue

            # if os.path.exists(save_path):
            #     continue

            draw_label(save_path, img_pth, json_path, name2idx, img_size)
            dir_idx = dir_idx + 1
            print(dir_idx,"/",len(file_list))

    return dir_lists

if __name__ == '__main__':
    # test_vedeio()

    # TransposeJson2PNG()

    test_picture2()

    print("finish")
