# -*- coding: utf-8 -*-

import json
import os
import cv2
import numpy as np
import tqdm
import base64


def str2Image(str):
    """
    ascII编码string数据转成图像数据
    :params str:类型(string)ascii编码的图像数据
    :return 类型（np.array-uint8）图像
    """
    image_str = str.encode('ascii')
    image_byte = base64.b64decode(image_str)

    img = cv2.imdecode(np.asarray(bytearray(image_byte), dtype='uint8'), cv2.IMREAD_COLOR)  # for jpg
    return img


def image2Str(img):
    """
    图像数据转成ascii编码string数据
    :params img:类型（np.array-uint8）需要转换的图像
    :return 类型(string)ascii编码的图像数据
    """

    img_encode = cv2.imencode('.jpg', img)[1]

    data_encode = np.array(img_encode)
    str_encode = data_encode.tostring()

    imgByteArray = base64.b64encode(str_encode)
    imgStr = imgByteArray.decode("ascii")
    return imgStr


def drawTrackmap(sequence_msg, labels, start_time, have_time_track=True):
    """
    生成识别结果轨道图
    :params sequence_msg:类型（list-int）结果数据，背景为0，其他的类别转换成从1、2、3...类别id
    :params labels:类型（list-int）类型ID列表，sequenceMsg中出现的ID都在在这出现，但只出现一次
    :params start_time:类型（int），序列第一个数据是第多少秒的结果
    :params have_time_track:类型（bool）是否在轨道下面显示时间轨道
    :return 类型(np.array-uint8)BGR类别轨道图
    """
    one_track_wide = 16
    len_max = len(sequence_msg)

    heatmap = np.ones((one_track_wide * len(labels), len_max, 3), dtype=np.uint8) * 255
    for i, label in enumerate(sequence_msg):
        if label == -1:
            continue
        label_index = label
        heatmap[label_index * one_track_wide:(label_index + 1) * one_track_wide, i] = (0, 216, 0)
    heatmap = heatmap[::-1]

    track_time = np.ones((one_track_wide, len_max, 3), dtype=np.uint8) * 255
    for i in range(track_time.shape[1]):
        if (i + start_time) % 60 == 0:
            track_time[-1 * int(one_track_wide * 0.25):, i] = (0, 0, 128)
        if (i + start_time) % 600 == 0:
            track_time[-1 * int(one_track_wide * 0.75):, i] = (0, 0, 128)
            cv2.putText(track_time, str((i + start_time) // 60), (i + 2, track_time.shape[0] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 128))
    if have_time_track:
        heatmap = np.concatenate((heatmap, track_time), 0)
    else:
        heatmap = np.concatenate((heatmap, np.zeros((8, heatmap.shape[1], 3), dtype=np.uint8)), 0)

    num_interval = len(labels) + 1
    for i in range(num_interval):
        heatmap[i * one_track_wide - 1:i * one_track_wide + 2] = (128, 128, 128)
    return heatmap


def drawConfigLine(listcfg_track_msg, tf=None):
    """
    生成识别结果置信度折线图
    tf如果不为None则在画点的时候会用红绿区分正确和错误识别，正确为绿点，错误为红点，折线均为黄色
    有三条线分别表示30%、50%、70%置信度
    :params listcfg_track_msg:类型（list-float）置信度数据，取值范围0~1，也允许-1表示没有结果，跳过
    :params tf:类型（list-int），长度必须与listcfgTrackMsg一致且能对应，该识别结果正确与否，正确为1，错误为0，没有结果为-1
    :return 类型(np.array-uint8)BGR置信度轨道图
    """
    colors = ((0, 0, 218), (0, 218, 0), (0, 218, 218), (255, 255, 255))
    if tf is None:
        tf = [1, ] * len(listcfg_track_msg)
    background = np.ones((100, len(listcfg_track_msg), 3), dtype=np.uint8) * 255
    for i in [3, 5, 7]:
        cv2.line(background, (0, i * 10), (len(listcfg_track_msg) - 1, i * 10), (64, 64, 64))
    last_set = (-1, -1)
    for x, cfg_value in enumerate(listcfg_track_msg):
        if cfg_value == -1:
            last_set = (-1, -1)
            continue
        if sum(last_set) == -2:
            now_set = (x, 100 - int(cfg_value * 100))
            cv2.circle(background, now_set, 1, colors[tf[x]], -1)
            last_set = now_set
        else:
            now_set = (x, 100 - int(cfg_value * 100))
            cv2.circle(background, now_set, 1, colors[tf[x]], -1)
            cv2.line(background, last_set, now_set, colors[2])
            last_set = now_set

    background = np.concatenate((background, np.zeros_like(background, dtype=np.uint8)[:4]), 0)
    return background


def normHeatmap2RGBHeatmap(norm_heatmap):
    """
    使用归一化矩阵生成热力图
    :params normHeatmap:类型（np.array-float）单通道归一化矩阵，形状（H，W），取值范围0~1
    :return 类型（np.array-uint8）轨道热图，通道顺序B G R
    """

    # 大于1的取1，小于0的取0
    norm_heatmap[norm_heatmap > 1] = 1
    norm_heatmap[norm_heatmap < 0] = 0

    norm_heatmap = np.expand_dims(norm_heatmap, 2).repeat(3, 2)
    min_values = np.min(norm_heatmap)
    max_values = np.max(norm_heatmap)
    norm_heatmap = (norm_heatmap - min_values) / (max_values - min_values)

    norm_heatmap[..., 2] = norm_heatmap[..., 2] * 510 - 255
    norm_heatmap[..., 1] = (-4 * norm_heatmap[..., 1] * norm_heatmap[..., 1] + 4 * norm_heatmap[..., 1]) * 255
    norm_heatmap[..., 0] = norm_heatmap[..., 0] * (-510) + 255
    norm_heatmap[norm_heatmap < 0] = 0
    norm_heatmap[norm_heatmap > 255] = 255
    return norm_heatmap


def colorHistogram(image, formula_mode="AVG"):
    """
    获取单帧图像BGR各通道色彩值直方图/值
    根据formula_mode决定返回图和各通道计算结果值
    :param image:类型（np.array-uint8）计算用图，BGR通道顺序
    :param formula_mode:类型（string）返回图还是BGR通道平均像素值或其他图。后期根据需要添加计算。如果为None则只返回色彩直方图
    :return 类型（np.array-uint8）BGR直方图[和各通道结果值]
    """
    histogram = np.zeros((100, 256, 3), dtype=np.uint8)
    channel_sorted = ["B", "G", "R"]
    reture_value = {"B": 0,
                    "G": 0,
                    "R": 0}

    for channel_id in range(3):
        color_values_statistic = np.unique(image[..., channel_id], return_counts=True)
        channel_count_max = np.max(color_values_statistic[1])
        for i in range(color_values_statistic[0].shape[0]):
            color_values = color_values_statistic[0][i]
            color_values_percent = round(color_values_statistic[1][i] / channel_count_max * 100)
            histogram[100 - color_values_percent:, color_values, channel_id] = 218
        if formula_mode == "AVG":
            reture_value[channel_sorted[channel_id]] = np.average(image[..., channel_id]).astype(np.float16)

    # image[:100, image.shape[1] - 256:] = histogram

    if formula_mode is None:
        return image
    else:
        return image, reture_value


if __name__ == '__main__':
    hp = np.expand_dims(np.array(list(range(256))) / 255., 0).repeat(16, 0)
    hpOut = normHeatmap2RGBHeatmap(hp).astype(np.uint8)
    # hpOut = cv2.cvtColor(hpOut,cv2.COLOR_RGB2BGR)
    cv2.imshow('a', hpOut)
    cv2.waitKey()

    img = cv2.imread(r"C:\Users\98759\Desktop\wxp\Screenshot from 2021-05-31 11-20-31.png")
    img = cv2.resize(img, (512, 512))
    bb, _ = colorHistogram(img)
    cv2.imshow('b', bb)
    cv2.waitKey()

    cc = drawConfigLine([x / 10 for x in range(11)])
    cv2.imshow('c', cc)
    cv2.waitKey()

    dd = drawTrackmap(list(range(11)), list(range(11)), 0, True)
    cv2.imshow('d', dd)
    cv2.waitKey()

    # jsonPath = r"D:\work\Data\CVSSegmentation\temp\LC-HX-0021011325_0775_01.json"
    # with open(jsonPath, encoding="utf-8") as f:
    #     labelMsg = json.load(f)
    #     f.close()
    # imgStr = labelMsg["imageData"]
    # img = str2Image(imgStr)
    # imgStr1 = image2Str(img)
    # img1 = str2Image(imgStr1)
    # cv2.imshow('a', img1)
    # cv2.waitKey()
    # print(imgStr == imgStr1)
    # stop = 0