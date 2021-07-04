from xml.dom.minidom import parse

def readXML(xmlpth, choise=False):
    domTree = parse(xmlpth)
    rootNode = domTree.documentElement
    # print(rootNode.nodeName)

    # 所有顾客
    tracks = rootNode.getElementsByTagName("track")
    videoName = rootNode.getElementsByTagName("video")[0].getAttribute("src")
    # print("****所有顾客信息****")

    track_dict = {}
    native_dict = {}
    for track in tracks:
        if track.hasAttribute("name"):
            track_name = track.getAttribute("name")

            track_tpye = track.getAttribute("type")

            track_dict[track_name] = {}
            native_dict[track_name] = {}
            # print("ID:", track_name )
            # name 元素

            if track_tpye == "primary":
                index = ""
                start = 0
                end = 0
                els = track.getElementsByTagName("el")
                for el in els:
                    if el.hasAttribute("index") and el.hasAttribute("start") and el.hasAttribute("end"):
                        index = el.getAttribute("index")
                        start = float(el.getAttribute("start"))
                        end = float(el.getAttribute("end"))
                    else:
                        continue

                    native_dict[track_name][index] = {}
                    native_dict[track_name][index]["range"] = [start, end]

                    attributes = el.getElementsByTagName("attribute")
                    nodelen = attributes.length
                    if nodelen > 0:
                        attribute_name = attributes[0].childNodes[0].data

                        native_dict[track_name][index]["name"] = attribute_name

                        if attribute_name in track_dict[track_name].keys():
                            track_dict[track_name][attribute_name].append([index, start, end])
                        else:
                            track_dict[track_name][attribute_name] = []
                            track_dict[track_name][attribute_name].append([index, start, end])

            if track_tpye == "subdivision":

                index = ""
                start = 0
                end = 0

                el_grps = track.getElementsByTagName("el-group")

                # ----------------------------------------------
                el_groups = track.getElementsByTagName("el-group")
                for el_group in el_groups:
                    ref = el_group.getAttribute("ref")
                    els = el_group.getElementsByTagName("el")
                    for el in els:
                        ref_index = el.getAttribute("index")
                        ref_start = el.getAttribute("start")
                        if ref_start is '':
                            ref_start = native_dict['Phase.main procedures'][ref]['range'][0]
                        attributes = el.getElementsByTagName("attribute")
                        attribute_name = attributes[0].childNodes[0].data
                        native_dict[track_name][ref_index] = {}
                        native_dict[track_name][ref_index]['range'] = [ref_start]
                        native_dict[track_name][ref_index]['name'] = attribute_name

                # for el_grp in el_grps:
                #     ref_idx = el_grp.getAttribute("ref")
                #     ref_track_name = track.getAttribute("ref")
                #
                #     ref_start = native_dict[ref_track_name][ref_idx]["range"][0]
                #     ref_end = native_dict[ref_track_name][ref_idx]["range"][1]
                #
                #     # el = el_grp.getElementsByTagName("el")
                #     # ref_start = el_grp.getAttribute("start")
                #     # ref_start = el_grp.getAttribute("ref")
                #     # ref_start = el.getAttribute("start")
                #
                #     els = el_grp.getElementsByTagName("el")
                #     el_idx = 0
                #     for el in els:
                #         if el.hasAttribute("index"):
                #             index = int(el.getAttribute("index"))
                #         else:
                #             continue
                #
                #         if el_idx > 0:
                #             start = float(el.getAttribute("start"))
                #             native_dict[track_name][str(index - 1)]["range"][1] = start
                #         else:
                #             start = ref_start
                #
                #         el_idx += 1
                #
                #         index = str(index)
                #         native_dict[track_name][index] = {}
                #         native_dict[track_name][index]["range"] = [start, ref_end]
                #
                #         attributes = el.getElementsByTagName("attribute")
                #         nodelen = attributes.length
                #         if nodelen > 0:
                #             attribute_name = attributes[0].childNodes[0].data
                #             native_dict[track_name][index]["name"] = attribute_name
                #
                # for index in native_dict[track_name].keys():
                #     name = native_dict[track_name][index]["name"]
                #     value_range = native_dict[track_name][index]["range"]
                #
                #     if name in track_dict[track_name].keys():
                #         track_dict[track_name][name].append([index, value_range[0], value_range[1]])
                #     else:
                #         track_dict[track_name][name] = []
                #         track_dict[track_name][name].append([index, value_range[0], value_range[1]])
    track_dict['videoName'] = videoName
    native_dict['videoName'] = videoName
    if choise:
        return track_dict
    else:
        return native_dict
    # return track_dict
    return native_dict


import os
import cv2
import copy
from collections import Counter
import numpy as np
# import xlwt
import json


def statistic_label(dir_anvil, dir_save):
    dirpth = dir_anvil  # 待处理数据文件夹

    # 获取 .anvil格式文件
    pthlis = os.listdir(dirpth)
    lablelist = [pth for pth in pthlis if pth.__contains__('.anvil') and not pth.startswith('.')]

    save_labeldir = dir_save

    # 对所有文件进行遍历
    for label in lablelist:

        name = label.split('.')[0]  # 获取文件名

        pth = os.path.join(dirpth, label)  # 文件绝对路径

        if name == "LC-JQB-N3":
            print("tmp")

        try:
            track_dict = readXML(pth)  # 读取文件中信息
        except:
            print(pth)

        statistic_dict = {}
        # 数据在 statistic_dict 中汇总
        for action_name in track_dict.keys():

            if action_name not in statistic_dict.keys():
                statistic_dict[action_name] = {}

            for type_name in track_dict[action_name].keys():

                valuelist = track_dict[action_name][type_name]
                t_frame = 0

                if type_name not in statistic_dict[action_name].keys():
                    # statistic_dict[action_name][type_name] = 0
                    statistic_dict[action_name][type_name] = []

                for valuetmp in valuelist:
                    value = copy.deepcopy(valuetmp)

                    # statistic_dict[action_name][type_name] += value[2] - value[1]
                    statistic_dict[action_name][type_name].append([value[1], value[2]])

                    # statistic_dict[action_name][type_name].append( [ value[1],value[2], (value[2] - value[1]),name ])
        # print(statistic_dict)
        f = open(os.path.join(save_labeldir, label.replace('.anvil', '.json')), "w", encoding='utf-8')
        json.dump(statistic_dict, f, indent=2, ensure_ascii=False)
        '''
        # 创建一个workbook 设置编码
        workbook = xlwt.Workbook(encoding='utf-8')
        # 创建一个worksheet
        worksheet = workbook.add_sheet(name)

        # 打印统计结果
        row_idx = 0
        name_col = 0
        value_col = 1
        for action_name in statistic_dict.keys():

            # 参数对应 行, 列, 值
            worksheet.write(row_idx, name_col, action_name)
            row_idx += 1

            for phase_name in statistic_dict[action_name].keys():
                phas_list = statistic_dict[action_name][phase_name]
                worksheet.write(row_idx, name_col, phase_name)
                worksheet.write(row_idx, value_col, phas_list)
                row_idx += 1
                # print('\t',phase_name,len(phas_list))   # 子 阶段 数量

        save_excel = os.path.join(save_labeldir, name + ".xls")
        workbook.save(save_excel)
        '''


phase_class_dict = {
    "Extract the gallbladder": 1,
    "Establish access": 2,
    "Adhesion lysis": 3,
    "Mobilize the Calot's triangle": 4,
    "Dissect gallbladder from liver bed": 5,
    "Clear the operative region": 6,
}


def compare_anvil_statistic():
    dir1 = "/home/wyx/proj/yolo_pose/YOLOv3/anvil"
    dir2 = "/home/wyx/proj/yolo_pose/YOLOv3/anvil"

    video_dir = "/home/wyx/proj/yolo_pose/YOLOv3/video"

    list1 = os.listdir(dir1)
    list1 = [pth for pth in list1 if pth.endswith(".anvil")]

    list2 = os.listdir(dir2)
    list2 = [pth for pth in list2 if pth.endswith(".anvil")]

    common_list = []
    for pth in list1:
        if pth in list2:
            common_list.append(pth)

    total_list1 = []
    total_list2 = []
    for pth in common_list:
        label1 = os.path.join(dir1, pth)
        label2 = os.path.join(dir2, pth)

        video_path = os.path.join(video_dir, pth.replace(".anvil", ".mp4"))
        if not os.path.exists(video_path):
            video_path.replace(".mp4", ".MP4")
            if not os.path.exists(video_path):
                continue

        video_capture = cv2.VideoCapture(video_path)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(video_capture.get(cv2.CAP_PROP_FPS))

        if frame_count == 0 or frame_height == 0 or frame_width == 0 or fps == 0:
            continue

        track_dict1 = readXML(label1)
        track_dict2 = readXML(label2)

        keyname = "Phase.main procedures"

        video_array1 = np.zeros(frame_count, np.int)
        video_array2 = np.zeros(frame_count, np.int)

        for subkey in track_dict1[keyname].keys():

            class_id = phase_class_dict[subkey]

            if subkey in track_dict1[keyname].keys():
                action_list1 = track_dict1[keyname][subkey]
                for act in action_list1:
                    start = int(act[1] * fps)
                    end = int(act[2] * fps)
                    video_array1[start:end] = class_id

            if subkey in track_dict2[keyname].keys():
                action_list2 = track_dict2[keyname][subkey]
                for act in action_list2:
                    start = int(act[1] * fps)
                    end = int(act[2] * fps)
                    video_array2[start:end] = class_id

        equal_idx = video_array1 == video_array2

        total_list1.append(copy.deepcopy(video_array1))
        total_list2.append(copy.deepcopy(video_array2))

        print(pth, np.sum(equal_idx) / frame_count)

    total_list1 = np.concatenate(total_list1, axis=0)
    total_list2 = np.concatenate(total_list2, axis=0)

    for key in phase_class_dict.keys():
        classid = phase_class_dict[key]
        indx1 = total_list1 == classid
        indx2 = total_list2 == classid

        idnx1_2 = indx2[indx1]
        idnx2_1 = indx1[indx2]

        sum1_2 = np.sum(idnx1_2)
        sum2_1 = np.sum(idnx2_1)

        sum1 = np.sum(indx1)
        sum2 = np.sum(indx2)

        if sum1 == 0 or sum2 == 0:
            continue

        print(classid, key, sum1_2 / sum1, sum2_1 / sum2)

        # print("")


# import xlwt


def class_dic(dir_anvil):
    dirpth = dir_anvil  # 待处理数据文件夹

    # 获取 .anvil格式文件
    pthlis = os.listdir(dirpth)
    lablelist = [pth for pth in pthlis if pth.__contains__('.anvil') and not pth.startswith('.')]

    workbook = xlwt.Workbook()
    # 对所有文件进行遍历
    for label in lablelist:

        name = label.split('.')[0]  # 获取文件名

        pth = os.path.join(dirpth, label)  # 文件绝对路径

        if name == "LC-JQB-N3":
            print("tmp")

        track_dict = readXML(pth)  # 读取文件中信息
        print(track_dict)
        x = 0
        y = 0

        worksheet = workbook.add_sheet(name)

        for key in track_dict.keys():
            if key in track_dict.keys():
                if len(key.split('.')) == 2:
                    worksheet.write(y, x, key.split('.')[0])
                    x += 1
                    worksheet.write(y, x, key.split('.')[1])
                    x += 1
                else:
                    worksheet.write(y, x, key)
                    x += 1
                for points in track_dict[key].values():
                    if points == 'name':
                        continue
                    point = points['range']
                    worksheet.write(y, x, point[0])
                    worksheet.write(y + 1, x, point[1])
                    x += 1
            else:
                worksheet.write(y, x, 0)
                worksheet.write(y + 1, x, 0)
            y += 2
            x = 0
    workbook.save(f'all.xls')


if __name__ == "__main__":
    import os

    dir_anvil = r'K:\wangyx\instrument_time\labeled_phase'
    list_name_all = os.listdir(dir_anvil)
    list_name_anvil = [x for x in list_name_all if ".anvil" in x]
    for name in list_name_anvil:
        path_need = os.path.join(dir_anvil, name)
        track_dict = readXML(path_need)
        print(f"'{name}',")
