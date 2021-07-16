import os
import json
import numpy as np
import tqdm
from  tool.readwrite import anvil2list
from  tool.readwrite import traversalDir
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

    pth = "./LCPhase_version1_len8_1.json"
    with open(pth) as f:
        LCPhase6V1 = json.load(f)

    version1_train = []
    version1_valid = []
    for phase in LCPhase6V1["phase"].keys():

        for label in LCPhase6V1["phase"][phase].keys():

            for sequnce in LCPhase6V1["phase"][phase][label]:
                videoname = sequnce[-1].split("/")[0]
                if phase == "train" and videoname not in version1_train:
                    version1_train.append(videoname)
                if phase == "valid" and videoname not in version1_valid:
                    version1_valid.append(videoname)

    savePath           = "./LCPhase_version1_len8_1.json"
    savePath_statistic = "./LCPhase_version1_len8_1_statistic.json"
    extract_fps        = 8
    sequence_len       = 8
    cc                 = 0
    dictOut            = {"phase": {}}
    dictOut_statistic  = {"phase": {}}
    tvt                = "test"
    # label_nane_dict    = get_json_label()
    # pth = "/Users/guan/Desktop/100-1.txt"
    # with open(pth) as f:
    #     version1 = f.readlines()
    # version1_train = [ data.replace("\'","").replace("[","").replace("]","").replace("\n","") for data in version1[1].split(',') ]
    # version1_valid = [ data.replace("\'","").replace("[","").replace("]","").replace("\n","") for data in version1[3].split(',') ]


    pth = "/Users/guan/Desktop/Lc200_fps8_video_label.json"
    with open(pth) as f:
        label_nane_dict = json.load(f)

    train_dict = {
        "train":{},
        "valid":{}
    }
    for videoname in label_nane_dict.keys():
        if videoname in version1_train:
            train_dict["train"][videoname] = label_nane_dict[videoname]
        if videoname in version1_valid:
            train_dict["valid"][videoname] = label_nane_dict[videoname]

    ori_fps   = 8
    label_fps = 1
    videoidx = 0
    for phase in train_dict.keys():

        if phase not in dictOut['phase'].keys():
            dictOut['phase'][phase] = {}
            dictOut_statistic['phase'][phase] = {}

        for videoname in train_dict[phase].keys():

            print( videoidx,"/", len( train_dict["valid"].keys()) + len( train_dict["train"].keys()))
            videoidx += 1

            listMsg = train_dict[phase][videoname]
            for i in range(len(listMsg) - sequence_len*ori_fps - 1):
                label = listMsg[i + sequence_len*ori_fps-1]
                FindDiffirend = False
                for idx in np.arange(i,i+sequence_len*ori_fps, ori_fps ):
                    curlabel = listMsg[idx]
                    if curlabel != label:
                        FindDiffirend = True
                        break
                if FindDiffirend:
                    continue

                if label == -1:
                    cc+=1
                    continue
                if label not in dictOut['phase'][phase]:
                    dictOut['phase'][phase][label] = []
                    dictOut_statistic['phase'][phase][label] = []
                sequence  = [ "{}/{}_{:0>5}.jpg".format(videoname, videoname, int( (j + 1)*1.0/ori_fps) ) for j in range(i, i + sequence_len*ori_fps,ori_fps)]
                sequence2 = [["{}/{}_{:0>5}.jpg".format(videoname, videoname, int( (j + 1)*1.0/ori_fps) ), listMsg[j]] for j in range(i, i + sequence_len*ori_fps,ori_fps)]
                if sequence not in dictOut['phase'][phase][label]:
                    dictOut['phase'][phase][label].append(sequence)
                    dictOut_statistic['phase'][phase][label].append(sequence2)

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
    pth1 = "/Users/guan/Desktop/10video_2_analyze/58"
    pth2 = "/Users/guan/Desktop/10video_2_analyze/59"
    extract_fps = 1
    filelist1 = os.listdir(pth1)
    filelist2 = os.listdir(pth2)

    pth = "/Users/guan/Desktop/videopth_info.json"
    with open(pth) as f:
        data = json.load(f)

    # videolist     = read_xls_rows("/Users/guan/Desktop/100-3.xls")
    videolist     = []
    filenamelist1 = os.listdir(pth1)
    filenamelist2 = os.listdir(pth2)
    for filename in filenamelist1:
        if filename in filenamelist2:
            videolist.append(filename)

    read_dict = {}
    for videoname in videolist:
        videoname = videoname.split('.')[0]
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


    label_name_dict = {}
    lablenames      = []
    prelabel = 0
    for filename in read_dict.keys():
        filelist = read_dict[filename]["lable"]
        duration = math.ceil( read_dict[filename]["duration"] )*extract_fps
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
                start   = min(math.floor(label["start"]*extract_fps),duration)
                end     = min(math.floor(label["end"]*extract_fps), duration )
                if labelid > 6:
                    labelid = prelabel
                labelarray[start:end] = labelid
                prelabel = labelid

            labelarray_list.append(labelarray)

        merge = False
        if len( labelarray_list) == 2:
            if merge == True:
                idxs = np.where( labelarray_list[0] != labelarray_list[1])
                labelarray_list[0][idxs] = -1
                label_name_dict[filename] = labelarray_list[0].tolist()
            else:
                label_name_dict[filename] = [labelarray_list[0].tolist(), labelarray_list[1].tolist()]



    # print(lablenames)

    return label_name_dict


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

def get_anvil_label( namelist ):
    pthdir = r"/Users/guan/Desktop/LCPhase/100-1"

    listAnvilPath = [x for x in traversalDir(pthdir) if '.anvil' in x]
    listAnvilPath = sorted(listAnvilPath, key=lambda x: x.split('\\')[-1])

    anvilist = {}
    anvil_pth_dict = {}
    for videoname in namelist:
        surgid = videoname.split('-')[-1]
        for pth in listAnvilPath:
            videoname2 = pth.split('/')[-1].split('.')[0]
            surgid2    = videoname2.split('-')[-1].split('_')[0]
            # if videoname2.__contains__(videoname) or videoname.__contains__(videoname2):
            # if surgid.__contains__(surgid2) or surgid2.__contains__(surgid):
            if surgid == surgid2:

                if videoname not in anvil_pth_dict.keys():
                    labels = anvil2list(pth, listAllPhase)
                    anvil_pth_dict[videoname] = [labels]

                    anvilist[videoname] = [ pth]
                else:
                    labels = anvil2list(pth, listAllPhase)
                    anvil_pth_dict[videoname].append(labels)

                    anvilist[videoname].append(pth)


    return anvil_pth_dict


from tool.math_tool   import align_list
from tool.plot_figure import visualizationArray

def visualize_lable_different():
    jsonlabel    = get_json_label()
    anvil_label  = get_anvil_label( jsonlabel.keys())
    compare_dict = {}
    savedir = "/Users/guan/Desktop/10video_2_analyze/picture/"
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    namelist = ['czx','wsd','gjy','wyx']
    for videoname in anvil_label.keys():
        if videoname in jsonlabel.keys():
            arraylist = [jsonlabel[videoname][0],jsonlabel[videoname][1],anvil_label[videoname][0],anvil_label[videoname][1]]
            compare_dict[videoname] =  align_list( arraylist )
            visualizationArray( compare_dict[videoname][0], videoname, savedir, namelist)

    savepth = "/Users/guan/Desktop/10video_2_analyze/compare_dict.json"
    with open(savepth,"w") as f:
        json.dump(compare_dict, f)

if __name__ == "__main__":

    generate_label()



    print("end")