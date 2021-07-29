import os
import time

from   ctypes import *
import cv2
import imageio
import numpy as np
import torch
import torch.nn as nn
# from ffmpy import FFmpeg as ff
from   torchvision import models
import json
import tqdm
# from . import phase_post_process
# from phase_post_process import phase_post_process
import pytz
import datetime
import sys

PATH = "/root/EFS/20201209-LC-HX-12.mp4"


#SAVE = "/root/Algorithm/testPic/test"
class ResNet34(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.

        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock.
        """

    def __init__(self, num_classes):
        super(ResNet34, self).__init__()

        self.resnet34 = models.resnet34(pretrained=False)
        fc_in_channel = self.resnet34.fc.in_features
        self.resnet34.fc = nn.Linear(fc_in_channel, num_classes)

    def forward(self, x):
        x = self.resnet34(x)

        return x


class PhaseRecongnition():
    def __init__(self, modelpath):
        num_classes = 7
        model = ResNet34(num_classes)

        # check if there was a previously saved checkpoint
        if os.path.exists(modelpath):
            pretrained_dict = torch.load(
                modelpath, map_location=lambda storage, loc: storage)["state_dict"]
            model_dict = model.state_dict()
            tmpdict = {}
            for k, v in pretrained_dict.items():
                if k.startswith("module."):
                    k = k.replace("module.", "")
                if k in model_dict:
                    tmpdict[k] = v
                else:
                    print(k)

            model.load_state_dict(tmpdict)

        model = model.cuda()
        # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.eval()
        self.model = model

    def inference(self, imglist):
        newlist = []

        for img in imglist:
            # img = img[:, :, ::-1]
            img = cv2.resize(img, (512, 512))
            img = (img - 128.0) / 128.0
            newlist.append(img[np.newaxis, :, :, :])

        imgs = np.concatenate(newlist)
        imgs = torch.Tensor(imgs).cuda()
        inputs = imgs.permute(0, 3, 1, 2).float()

        outputs = self.model(inputs)
        outputs = outputs.cpu()
        softmax = nn.Softmax(dim=1)
        outputs = softmax(outputs)
        confs, preds = torch.max(outputs, 1)
        preds = preds.numpy().tolist()
        confs = confs.data.numpy().tolist()

        return preds, confs


def index_list(targetnum, list):
    number = list.count(targetnum)
    i = number
    index = []
    while i > 0:
        for x in range(len(list)):
            if list[x] == targetnum:
                index.append(x + 1)
                i = i - 1
    return index


# 获取阶段数据并格式化
def get_time():
    local_time = datetime.datetime.now(pytz.timezone(
        'Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S")
    return local_time


def get_phase_data(path, gap):
    data = {
        "video": {},
        "phase": {
            "config": {
                "model_verison": "",
                "frame_gap": "",
                "label_name": [],
            },
            "overview": [],
            "detail": [],
        }
    }

    phase_total = 7

    label_list = (
        (1, "抓取胆囊"),
        (2, "建立气腹"),
        (3, "分离粘连"),
        (4, "游离胆囊三角"),
        (5, "分离胆囊床"),
        (6, "清理术野"),
    )
    for label in label_list:
        data["phase"]["config"]["label_name"].append(
            {"id": label[0], "content": label[1]})
    if not os.path.exists(path):
        raise FileNotFoundError
    file_name = os.path.basename(path)
    start_time = get_time()
    # with open("/root/EFS/logs/log", "a+") as f:
    #     f.write("filename:" + file_name + "开始分析模型" + " time:" + str(start_time) + "\n")

    reader = imageio.get_reader(path)
    width, height = reader.get_meta_data()["size"]
    fps = reader.get_meta_data()["fps"]
    duration = reader.get_meta_data()["duration"]

    data["video"].update({"file_name": file_name})
    data["video"].update({"width": width, "height": height})
    data["video"].update({"fps": fps})
    data["video"].update({"duration": duration})
    modelpath = "model_resnet34_float.pth"
    if not os.path.exists(modelpath):
        raise Exception
    model = PhaseRecongnition(modelpath)
    predlist = []
    conflist = []

    compute_start = time.time()

    reader = imageio.get_reader(path)
    fps = reader.get_meta_data()["fps"]
    frame_gap = fps / gap
    data["phase"]["config"]["frame_gap"] = frame_gap
    duration = reader.get_meta_data()["duration"]
    total_frame_num = duration * fps
    imgList = []
    timeList = []
    for num, image in enumerate(tqdm.tqdm(reader, desc=file_name[:file_name.rfind('.')], total=total_frame_num)):
        if (num // frame_gap) > ((num - 1) // frame_gap):
            time_stamp = int(num / fps)
            timeList.append(time_stamp)
            imgList.append(image)
            if len(imgList) == 1:
                pred, conf = model.inference(imgList)
                predlist += pred
                conflist += conf
                imgList.clear()

    model_end_time = get_time()
    # with open("/root/EFS/logs/log", "a+") as f:
    #     f.write("filename:" + file_name + "模型分析结束" + " time:" + str(model_end_time) + "\n")

    # 做后处理
    phase_post_process_C = CDLL('./phasePostProcessV0.2.so')
    arr = (c_int * len(predlist))(*predlist)
    phase_post_process_C.runThis(arr, len(predlist))

    post_process_end_time = get_time()
    # with open("/root/EFS/logs/log", "a+") as f:
    #     f.write("filename:" + file_name + "后处理结束结束" + " time:" + str(post_process_end_time) + "\n")

    # with open("/root/EFS/logs/log", "a+") as f:
    #     f.write("filename:" + file_name + "输出JSON结束" + " time:" + str(local_time) + "\n")
    #     f.write("filename:" + file_name + "\n模型输出结果" + str(predlist) + "\n" + "后处理结果" + str(PostProcessPredList) + "\n")
    #     f.write("-------------------------------\n")
    PostProcessPredList = list(arr)
    PostProcessPredList = [-1] + PostProcessPredList + [-1]
    detailMsg = [(x, i) for i, x in enumerate(PostProcessPredList[1:]) if
                 PostProcessPredList[i + 1] != PostProcessPredList[i]]
    for num in range(len(detailMsg)):
        if detailMsg[num][0] != -1:
            data['phase']['detail'].append({
                "label": detailMsg[num][0],
                "start_time": detailMsg[num][1] + 1,
                "end_time": detailMsg[num + 1][1]
            })
    for cls in label_list:
        clsID = cls[0]
        overview = sum([x['end_time'] - x['start_time'] for x in data['phase']['detail'] if x['label'] == clsID])
        data['phase']['overview'].append({
            "label": clsID,
            "total": overview
        })
    json_file = f"fileBag/{file_name[:file_name.rfind('.')]}_phase_{gap}.json"
    log_file = "fileBag/log.txt"
    # with open(json_file, 'w', encoding='utf-8') as f:
    #     json.dumps(f, data, indent=2, ensure_ascii=False)
    #     f.close()
    data = json.dumps(data, indent=2, ensure_ascii=False)
    with open(json_file, 'w', encoding='utf-8') as f:
        f.write(data)
        f.close()
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{file_name[:file_name.rfind('.')]}\n")
        f.write(f"识别:{predlist}\n")
        f.write(f"后处理:{PostProcessPredList[1:-1]}\n")
        f.close()
    return data


if __name__ == "__main__":
    import cgf
    # pathVideo = '/home/withai/Desktop/LC-MY-A01621288.mp4'
    # get_phase_data(pathVideo, 1)

    dir0 = '/home/withai/Videos/20210301_merge'
    for operationName in os.listdir(dir0):
        pathVideo = os.path.join(dir0, operationName)
        if pathVideo.split('/')[-1].split('.')[0] in cgf.jump:
            continue
        if pathVideo.split('.')[-1] in ['mp4', 'avi', 'mpg']:
            try:
                get_phase_data(pathVideo, 1)
            except:
                print(f'{operationName} reture Error!!!')
        else:
            print(pathVideo)
