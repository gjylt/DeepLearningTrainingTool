import os
import xlrd
from sys import platform
import json
import tqdm
import numpy as np
from tool.readXML import readXML


def traversalDir(dir1, returnX='path'):
    """
    params:
        returnX: choise [path,name]
    """
    if platform == "win32":
        separatorSymbol = "\\"
    elif platform == "linux":
        separatorSymbol = "/"
    else:
        assert SystemError("分隔符号检测不是windows也不是linux")
    out = []
    list_name = os.listdir(dir1)
    for name in list_name:
        dp = os.path.join(dir1, name)
        if os.path.isfile(dp):
            # list_x = dp.replace(dir_local + separatorSymbol, "").split(separatorSymbol)
            # out.append(os.path.join(*list_x))
            if returnX == 'path':
                out.append(dp)
            elif returnX == 'name':
                out.append(name)
            else:
                assert ValueError("returnX choise in [path, name]")
        elif os.path.isdir(dp):
            out.extend(traversalDir(dp, returnX=returnX))
        else:
            print(f"不知道这是个啥{dp}")
    return out


def anvil2list(anvilPath, listPhase,fps=1):
    dictAnvilMsg = readXML(anvilPath)
    listMsg = [0] * 9999
    maxTime = 0
    for phaseMsg in dictAnvilMsg['Phase.main procedures'].values():
        label = phaseMsg['name']
        if label not in listPhase:
            # print(f"jump {label}")
            continue
        start, end = phaseMsg['range']
        start, end = round(start*fps), round(end*fps)
        if end > maxTime:
            maxTime = end
        listMsg[start:end + 1] = [listPhase.index(label)] * (end - start + 1)
    for phaseMsg in dictAnvilMsg['key action'].values():
        label = phaseMsg['name']
        if label not in listPhase:
            # print(f"jump {label}")
            continue
        start, end = phaseMsg['range']
        start, end = round(start*fps), round(end*fps)
        if end > maxTime:
            maxTime = end
        listMsg[start:end + 1] = [listPhase.index(label)] * (end - start + 1)
    return listMsg[:maxTime + 1]


def mk0502():
    listAllPhase = ["bg",
                    "Establish access",
                    "Adhesion lysis",
                    "Mobilize the Calot\'s triangle",
                    "Dissect gallbladder from liver bed",
                    "Extract the gallbladder",
                    "Clear the operative region",
                    "clip the cystic artery",
                    "clip the cystic duct",
                    "cut the cystic artery",
                    "cut the cystic duct"]
    listMainPhase = ["bg",
                     "Establish access",
                     "Adhesion lysis",
                     "Mobilize the Calot\'s triangle",
                     "Dissect gallbladder from liver bed",
                     "Extract the gallbladder",
                     "Clear the operative region"]
    listKeyAction = ["bg",
                     "clip the cystic artery",
                     "clip the cystic duct",
                     "cut the cystic artery",
                     "cut the cystic duct"]
    listPhase    = listAllPhase
    anvilFolder  = "/home/withai/Desktop/LCLabelFiles/LCPhase/100-1"
    excelPath    = "/home/withai/Desktop/LCLabelFiles/视频对应名称_路径.xls"
    videoTabPath = "/home/withai/Desktop/LCLabelFiles/Result_1_2.xls"
    savePath     = "/home/withai/Desktop/LCLabelFiles/LCPhase{}V1_action_len40_8.json".format(len(listPhase)-1)
    savePath_statistic = "/home/withai/Desktop/LCLabelFiles/LCPhase{}V1_action_len40_8_statistic.json".format(len(listPhase) - 1)
    sequence_len = 40
    extract_fps  = 8

    listAnvilPath = [x for x in traversalDir(anvilFolder) if '.anvil' in x]
    listAnvilPath = sorted(listAnvilPath, key=lambda x: x.split('\\')[-1])

    workbook = xlrd.open_workbook(excelPath)
    sheet = workbook.sheet_by_index(0)
    colName0 = sheet.col_values(1)
    colName1 = sheet.col_values(3)
    colIsComplete = sheet.col_values(4)
    colName1 = [x.split('/')[-1].split('.')[0] for x in colName1]

    workbook = xlrd.open_workbook(videoTabPath)
    sheet = workbook.sheet_by_index(0)
    colTags = sheet.col_values(3)[1:]
    colS3Key = sheet.col_values(4)[1:]
    colS3KeyName = [x.split('/')[-1].split('.')[0] for x in colS3Key]

    dictS = {}
    for anvilIndex in range(len(listAnvilPath)):
        operationName = listAnvilPath[anvilIndex].split('/')[-1].split('_')[0].split('.')[0]
        imgPrefix = colName1[colName0.index(operationName)]
        if imgPrefix == '20210122-LC-YB-0000803345':
            continue
        hs = imgPrefix.split('-')[2]
        tags = colTags[colS3KeyName.index(imgPrefix)]
        if type(tags) == str and tags != '':
            tags = tags.split(',')
        elif type(tags) == float:
            tags = [int(tags), ]
        else:
            tags = []
        diff = [x for x in [1, 3, 7, 9] if x in tags]
        if hs not in dictS.keys():
            dictS[hs] = {'df': [],
                         'ez': []}
        if imgPrefix in dictS[hs]['df'] or imgPrefix in dictS[hs]['ez']:
            continue
        if len(diff) != 0:
            dictS[hs]['df'].append(imgPrefix)
        else:
            dictS[hs]['ez'].append(imgPrefix)

    dictTVT = {'train': [], 'valid': []}
    for hs in dictS:
        for de in dictS[hs]:
            cc = 0
            if len(dictS[hs][de]) == 0:
                continue
            for imgPrefix in dictS[hs][de]:
                if cc % 9 == 1:
                    tvt = 'valid'
                else:
                    tvt = 'train'
                if type(colIsComplete[colName1.index(imgPrefix)]) == 'float':
                    tvt = 'train'
                cc += 1
                dictTVT[tvt].append(imgPrefix)

    dictOut = {'phase': {}}
    dictOut_statistic = {'phase': {}}

    # video_label_dict = {}

    for anvilIndex in tqdm.tqdm(range(len(listAnvilPath))):
        operationName = listAnvilPath[anvilIndex].split('/')[-1].split('_')[0].split('.')[0]
        imgPrefix = colName1[colName0.index(operationName)]
        if imgPrefix == '20210122-LC-YB-0000803345':
            continue
        isCompare = False
        if anvilIndex == len(listAnvilPath) - 1:
            pass
        elif anvilIndex != 0 and listAnvilPath[anvilIndex - 1] == listAnvilPath[anvilIndex]:
            continue
        else:
            if listAnvilPath[anvilIndex].split('\\')[-1] == listAnvilPath[anvilIndex + 1].split('\\')[-1]:
                isCompare = True
        # if listAnvilPath[anvilIndex].split('\\')[-1] in ["LC-CHZH-629577_phase_1.anvil",
        #                                                  "LC-MY-602473791_phase_1.anvil",
        #                                                  "LC-MY-A01723005_phase_1.anvil",
        #                                                  "LC-MY-a00677619_phase_1.anvil",
        #                                                  "LC-ZG-1021514_phase_1.anvil",
        #                                                  "LC-ZG-1023819_phase_1.anvil"]:
        #     continue

        if isCompare:
            anvilPath0 = listAnvilPath[anvilIndex]
            listMsg0 = anvil2list(anvilPath0, listPhase,extract_fps)
            anvilPath1 = listAnvilPath[anvilIndex + 1]
            listMsg1 = anvil2list(anvilPath1, listPhase,extract_fps)
            if len(listMsg0) < len(listMsg1):
                listMsg0 = listMsg0 + [0] * (len(listMsg1) - len(listMsg0))
            elif len(listMsg0) > len(listMsg1):
                listMsg1 = listMsg1 + [0] * (len(listMsg0) - len(listMsg1))
            listMsg = [-1] * len(listMsg0)
            for i in range(len(listMsg0)):
                if listMsg0[i] == listMsg1[i]:
                    listMsg[i] = listMsg0[i]
        else:
            anvilPath = listAnvilPath[anvilIndex]
            listMsg = anvil2list(anvilPath, listPhase,extract_fps)

        tvt = [x for x in dictTVT.keys() if imgPrefix in dictTVT[x]][0]
        if tvt not in dictOut['phase']:
            dictOut['phase'][tvt] = {}
            dictOut_statistic['phase'][tvt] = {}


        # video_label_dict[imgPrefix] = copy.deepcopy(listMsg)
        # continue

        for i in range(len(listMsg)):
            if i + sequence_len > len(listMsg) - 1:
                break
            label = listMsg[i + sequence_len-1]
            FindDiffirend = False
            for idx in np.arange(i, i + sequence_len):
                curlabel = listMsg[idx]
                if curlabel != label:
                    FindDiffirend = True
                    break
            if FindDiffirend:
                continue
            if label == -1:
                continue
            if label not in dictOut['phase'][tvt]:
                dictOut['phase'][tvt][label] = []
                dictOut_statistic['phase'][tvt][label] = []
            #sequence = [ ["{}/{}_{:0>5}.jpg".format(imgPrefix, imgPrefix, j + 1),listMsg[j]] for j in range(i, i + sequence_len)]
            sequence = [ "{}/{}_{:0>5}.jpg".format(imgPrefix, imgPrefix, j + 1) for j in range(i, i + sequence_len)]
            sequence2 = [ ["{}/{}_{:0>5}.jpg".format(imgPrefix, imgPrefix, j + 1), listMsg[j]] for j in range(i, i + sequence_len)]
            if sequence not in dictOut['phase'][tvt][label]:
                dictOut['phase'][tvt][label].append(sequence)
                dictOut_statistic['phase'][tvt][label].append(sequence2)

    # savePath_video_label = "/home/withai/Desktop/LCLabelFiles/Lc200TrainVal_fps8_video_label.json"
    # with open(savePath_video_label,'w') as f:
    #     json.dump(video_label_dict,f)


    with open(savePath, 'w', encoding='utf-8') as f:
        json.dump(dictOut, f, ensure_ascii=False, indent=2)
        f.close()
    with open(savePath_statistic, 'w', encoding='utf-8') as f:
        json.dump(dictOut_statistic, f, ensure_ascii=False, indent=2)
        f.close()
    stop = 0


if __name__ == '__main__':
    mk0502()
