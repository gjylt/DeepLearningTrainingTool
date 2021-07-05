import os
import json
import numpy as np
import tqdm

from tool.readXML import readXML


def dict2list(dictAnvilMsg, listPhase,fps = 1):
    listMsg = [0] * 9999
    maxTime = 0
    for phaseMsg in dictAnvilMsg['Phase.main procedures'].values():
        label = phaseMsg['name']
        if label not in listPhase:
            continue
        start, end = phaseMsg['range']
        start, end = round(start*fps), round(end*fps)
        if end > maxTime:
            maxTime = end
        listMsg[start:end + 1] = [listPhase.index(label)] * (end - start + 1)
    for phaseMsg in dictAnvilMsg['key action'].values():
        label = phaseMsg['name']
        if label not in listPhase:
            continue
        start, end = phaseMsg['range']
        start, end = round(start*fps), round(end*fps)
        if end > maxTime:
            maxTime = end
        listMsg[start:end + 1] = [listPhase.index(label)] * (end - start + 1)
    return listMsg[:maxTime]



listAnvil = ["bg",
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
listJson = ["bg",
            2,
            3,
            4,
            5,
            1,
            6,
            2,
            1,
            4,
            3
            ]
listUse = ["bg",
           "Establish access",
           "Adhesion lysis",
           "Mobilize the Calot\'s triangle",
           "Dissect gallbladder from liver bed",
           "Extract the gallbladder",
           "Clear the operative region",
           "clip the cystic artery",
           "clip the cystic duct",
           "cut the cystic artery",
           "cut the cystic duct"
           ]


def generate_label():
    dictAnvil = {}
    dictJson  = {}


    savePath     = "./LCPhase{}TestV1_action_len40_8.json".format(len(listUse) - 1)
    savePath_statistic= ".s/LCPhase{}TestV1_action_len40_8_statistic.json".format(len(listUse) - 1)
    extract_fps  = 8
    sequence_len = 40



    cc=0
    dictOut = {"phase": {}}
    dictOut_statistic = {"phase": {}}
    tvt = "test"



    label_nane_dict = get_json_label()



    for videoname in tqdm.tqdm(label_nane_dict.keys()):

        listMsg = label_nane_dict[videoname].tolist()

        if tvt not in dictOut['phase'].keys():
            dictOut['phase'][tvt] = {}
            dictOut_statistic['phase'][tvt] = {}
        # if imgPrefix == "20201117-LC-HX-0016893143_ORIGIN":
        #     stop = 0
        # else:
        #     continue
        for i in range(len(listMsg) - sequence_len - 1):
            label = listMsg[i + sequence_len-1]
            FindDiffirend = False
            for idx in np.arange(i,i+sequence_len):
                curlabel = listMsg[idx]
                if curlabel != label:
                    FindDiffirend = True
                    break
            if FindDiffirend:
                continue

            if label == -1:
                cc+=1
                continue
            if label not in dictOut['phase'][tvt]:
                dictOut['phase'][tvt][label] = []
                dictOut_statistic['phase'][tvt][label] = []
            sequence = [ "{}/{}_{:0>5}.jpg".format(videoname, videoname, j + 1) for j in range(i, i + sequence_len)]
            sequence2 = [["{}/{}_{:0>5}.jpg".format(videoname, videoname, j + 1), listMsg[j]] for j in range(i, i + sequence_len)]
            if sequence not in dictOut['phase'][tvt][label]:
                dictOut['phase'][tvt][label].append(sequence)
                dictOut_statistic['phase'][tvt][label].append(sequence2)

    # savePath_video_label = "/home/withai/Desktop/LCLabelFiles/Lc200Test_fps8_video_label.json"
    # with open(savePath_video_label,'w') as f:
    #     json.dump(video_label_dict,f)

    print(cc)
    print('write in')
    with open(savePath, 'w', encoding='utf-8') as f:
        json.dump(dictOut, f, ensure_ascii=False, indent=2)
        f.close()
    with open(savePath_statistic, 'w', encoding='utf-8') as f:
        json.dump(dictOut_statistic, f, ensure_ascii=False, indent=2)
        f.close()


from tool.readwrite import read_xls_rows
import math


lable_dict = {'建立气腹':1, '分离粘连':2, '游离胆囊三角':3, '分离胆囊床':4, '清理术区':6, '取出胆囊':5}

def get_json_label():
    pth1 = "/Users/guan/Desktop/100-3/35"
    pth2 = "/Users/guan/Desktop/100-3/36"
    filelist1 = os.listdir(pth1)
    filelist2 = os.listdir(pth2)


    pth = "/Users/guan/Desktop/videopth_info.json"
    with open(pth) as f:
        data = json.load(f)

    videolist = read_xls_rows("/Users/guan/Desktop/100-3.xls")
    read_dict = {}
    for videoname in videolist:
        videoname = videoname[0]
        for videoname1 in data.keys():

            if videoname1.__contains__(videoname) or videoname.__contains__(videoname1):
                read_dict[videoname] = data[videoname1]
                break

    file_dict = {}
    for filename in filelist1:
        videoname = filename.split(".")[0]
        if videoname in read_dict.keys():
            read_dict[videoname]["lable"] = [os.path.join(pth1, filename)]
        # file_dict[filename] = [os.path.join(pth1, filename)]

    for filename in filelist2:
        videoname = filename.split(".")[0]
        if videoname  in read_dict.keys():
            if "lable" not in read_dict[videoname].keys():
                read_dict[videoname]["lable"] = [os.path.join(pth2, filename)]
            else:
                read_dict[videoname]["lable"].append( os.path.join(pth2, filename) )

    fps = 8
    label_name_dict = {}
    lablenames = []
    for filename in read_dict.keys():
        filelist = read_dict[filename]["lable"]
        duration = math.ceil( read_dict[filename]["duration"] )*8
        labelarray_list = []
        for filepth in filelist:
            labelarray = np.zeros(duration, np.int)
            with open(filepth) as f:
                labelist = json.load(f)

            for label in labelist:
                labelname = label["label"]
                if labelname not in lablenames:
                    lablenames.append(labelname)

                labelid = lable_dict[labelname]
                start = min(math.floor(label["start"]*8),duration)
                end   = min(math.floor(label["end"]*8), duration )
                labelarray[start:end] = labelid

            labelarray_list.append(labelarray)

        if len( labelarray_list) == 2:
            idxs = np.where( labelarray_list[0] != labelarray_list[1])
            labelarray_list[0][idxs] = -1
        label_name_dict[filename] = labelarray_list[0]
    # print(lablenames)

    return label_name_dict



if __name__ == "__main__":

    # get_json_label()

    generate_label()

    print("end")