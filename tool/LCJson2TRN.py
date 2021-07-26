#coding=utf-8
import os
import json
import numpy as np
import tqdm
from  tool.readwrite import anvil2list
from  tool.readwrite import traversalDir
from tool.readXML import readXML


def dict2list(dictAnvilMsg, listPhase, fps = 1):
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

    version1_train_val_split_path = "/home/withai/Desktop/LCLabelFiles/videoname_phase_list_100-1.json"
    with open(version1_train_val_split_path) as f:
        train_val = json.load(f)
    version1_train = train_val['train']
    version1_valid = train_val['valid']

    version2_test_dir = "/home/withai/Pictures/LCFrame/100-2"
    version2_test     = os.listdir(version2_test_dir)
    version2_test_new = []
    for video2 in version2_test:
        resident_id2 = video2.split('_')[0].split('-')[-1]
        find = False
        for video1 in version1_train+version1_valid:
            resident_id1 = video1.split('_')[0].split('-')[-1]
            if resident_id1==resident_id2:
                find = True
        if not find:
            version2_test_new.append(video2)


    savePath           = "/home/withai/Desktop/LCLabelFiles/LCPhase_version2_len24_2_annotator.json"
    savePath_statistic = "/home/withai/Desktop/LCLabelFiles/LCPhase_version2_len24_2_annotator_statistic.json"
    extract_fps        = 8
    sequence_len       = 24
    cc                 = 0
    dictOut            = {"phase": {}}
    dictOut_statistic  = {"phase": {}}
    label_nane_dict    = get_json_label( extract_fps )
    train_dict = {
        "train":{},
        "valid":{},
        "test":{}
    }

    for videoname in label_nane_dict.keys():
        resident_id = videoname.split('-')[-1]
        find = False

        # for videoname1 in version1_train:
        #     if videoname1.__contains__(resident_id):
        #         train_dict["train"][videoname1] = label_nane_dict[videoname]
        #         find = True
        #         break
        #
        # for videoname1 in version1_valid:
        #     if videoname1.__contains__(resident_id):
        #         train_dict["valid"][videoname1] = label_nane_dict[videoname]
        #         find = True
        #         break

        for videoname1 in version2_test_new:
            if videoname1.__contains__(resident_id):
                train_dict["test"][videoname1] = label_nane_dict[videoname]
                find = True
                break

        if not find:
            print(videoname,'not found')


    videoidx  = 0
    label_fps = 1  #extract_fps should be exact divided by label_fps
    for phase in train_dict.keys():

        if phase not in dictOut['phase'].keys():
            dictOut['phase'][phase] = {}
            dictOut_statistic['phase'][phase] = {}

        for videoname in train_dict[phase].keys():

            print( videoidx,"/", len( train_dict["valid"].keys()) + len( train_dict["train"].keys()) + len( train_dict["test"].keys()))
            videoidx += 1

            listMsg = train_dict[phase][videoname]
            i_start = 0
            i_end   = max(int( len(listMsg)*label_fps/extract_fps )-sequence_len,0)
            i_step  = 1
            transform_ration = extract_fps/label_fps
            max_extract_len  = len(listMsg)

            for i in range( i_start, i_end, i_step):

                indx  = min(  int((i+sequence_len-1)*transform_ration), max_extract_len)
                label = listMsg[ indx ]
                if label == -1:
                    cc+=1
                    continue

                FindDiffirend = False
                sequence2 = []
                sequence  = []
                for idx in np.arange( i, i+sequence_len, 1 ):

                    indx     = min( int((idx) * transform_ration) , max_extract_len)
                    curlabel = listMsg[indx]

                    if curlabel != label:
                        FindDiffirend = True
                        break

                    subpath = "{}/{}_{:0>5}.jpg".format(videoname, videoname, indx )

                    sequence.append(subpath)
                    sequence2.append( [subpath, curlabel] )

                if FindDiffirend:
                    continue

                if label not in dictOut['phase'][phase]:
                    dictOut['phase'][phase][label] = []
                    dictOut_statistic['phase'][phase][label] = []

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


lable_dict = {'建立气腹':1,
              '分离粘连':2,
              '游离胆囊三角':3,
              '分离胆囊床':4,
              '清理术区':6,
              '抓取胆囊':5,
              '取出胆囊':5,
              "清理术野":6}


video_path ={
"LC-CD-6689898":"/mnt/video/LC10000/CompleteVideo/hospital_id=9/surgery_id=476/video/20210107-LC-CD-6689898_ORIGIN.mp4",
"LC-DY-778970":"/mnt/video/LC10000/CompleteVideo/hospital_id=11/surgery_id=394/video/20201116-LC-DY-778970_ORIGIN.mp4",
"LC-DY-574665":"/mnt/video/LC10000/CompleteVideo/hospital_id=11/surgery_id=365/video/20201103-LC-DY-574665_ORIGIN.mp4",
"LC-DY-793367":"/mnt/video/LC10000/CompleteVideo/hospital_id=11/surgery_id=721/video/20210115-LC-DY-793367_ORIGIN.mp4",
"LC-DY-792089":"/mnt/video/LC10000/CompleteVideo/hospital_id=11/surgery_id=436/video/20210105-LC-DY-792089_ORIGIN.mp4",
"LC-CHB-468797":"/mnt/video/LC10000/CompleteVideo/hospital_id=12/surgery_id=338/video/20201223-LC-CHB-468797_ORIGIN.mp4",
"LC-CD-6498741":"/mnt/video/LC10000/CompleteVideo/hospital_id=9/surgery_id=433/video/20210105-LC-CD-6498741_ORIGIN.mp4",
"LC-XN-IP00000320162":"/mnt/video/LC10000/CompleteVideo/hospital_id=2/surgery_id=268/video/20201208-LC-XN-IP00000320162_ORIGIN.mp4",
"LC-XN-IP00000320761":"/mnt/video/LC10000/CompleteVideo/hospital_id=2/surgery_id=269/video/20201208-LC-XN-IP00000320761_ORIGIN.mp4"
}

import imageio

def get_json_label(extract_fps):
    pth1 = "/Users/guan/Desktop/100-2/100-2/CZX/"
    pth2 = "/Users/guan/Desktop/100-2/100-2/WSD/"
    # extract_fps = 8
    filelist1   = os.listdir(pth1)
    filelist2   = os.listdir(pth2)

    pth = "/Users/guan/Desktop/videopth_info_.json"
    with open(pth) as f:
        data = json.load(f)

    # for videoname in video_path.keys():
    #
    #     try:
    #         pth = video_path[videoname]
    #         vid = imageio.get_reader(pth, 'ffmpeg')
    #         metaData = vid.get_meta_data()
    #         fps = metaData['fps']
    #         duration = metaData['duration']
    #         data[videoname] = {}
    #         data[videoname]['path'] = pth
    #         data[videoname]['fps'] = fps
    #         data[videoname]['duration'] = duration
    #     except:
    #         print(videoname, "faild read video info")
    #
    # pth = "/home/withai/Desktop/videopth_info_.json"
    # with open(pth,'w') as f:
    #     json.dump(data,f)

    #find out the videos both in two directory
    videolist     = []
    filenamelist1 = os.listdir(pth1)
    filenamelist2 = os.listdir(pth2)
    for filename in filenamelist1:
        if filename in filenamelist2:
            videolist.append(filename)

    #remaind the video name exist video infomation
    read_dict = {}
    for videoname in videolist:
        videoname = videoname.split('.')[0]
        surgid    = videoname.split('-')[-1]
        find = False
        for videoname1 in data.keys():
            if videoname1.__contains__(surgid) or videoname.__contains__(videoname1):
                read_dict[videoname] = data[videoname1]
                find = True
                break
        if not find:
            print(videoname,'not find')

    #
    for filename in filelist1:
        videoname = filename.split(".")[0]
        if videoname in read_dict.keys():
            read_dict[videoname]["lable"] = [os.path.join(pth1, filename)]

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

                #
                if labelid > 6:
                    labelid = prelabel

                labelarray[start:end] = labelid
                prelabel = labelid

            labelarray_list.append(labelarray)

        merge_label = True
        if len( labelarray_list) == 2:
            if merge_label == True:
                idxs = np.where( labelarray_list[0] != labelarray_list[1])
                labelarray_list[0][idxs] = -1
                label_name_dict[filename] = labelarray_list[0].tolist()
            else:
                label_name_dict[filename] = [labelarray_list[0].tolist(), labelarray_list[1].tolist()]


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

    # get_json_file()


    print("end")