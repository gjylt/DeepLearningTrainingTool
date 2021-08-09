#coding=utf-8
import os
import json
import numpy as np
import tqdm
from  tool.readwrite import anvil2list
from  tool.readwrite import traversalDir
from  tool.readXML   import readXML

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




def get_videolist_of_dataset( test_data ,phase):

    test_video_list = []
    video_data_dict = {}

    for labelid in test_data["phase"][phase].keys():

        for sequnce in test_data["phase"][phase][labelid]:

            videoname =  os.path.split(sequnce[0])[0]

            if videoname not in test_video_list:
                test_video_list.append(videoname)

            if videoname not in video_data_dict.keys():
                video_data_dict[videoname]={}
            if int(labelid) not in video_data_dict[videoname].keys():
                video_data_dict[videoname][ int(labelid) ] = []

            video_data_dict[videoname][ int(labelid) ].append(sequnce)



    return test_video_list,video_data_dict

def get_find_not_find_video(xlsdata_train, total_list_old):
    find_dict = {}
    not_find_dict = []
    for videonew in xlsdata_train:
        video_id = videonew[0].split('-')[-1]
        find = False
        for videold in total_list_old:
            if videold.__contains__(video_id):
                find = True
                find_dict[videonew[0]] = videold
                break

        if not find:
            not_find_dict.append(videonew[0])

    return find_dict,not_find_dict


def generate_train_sequnce(label_name_dict , train_list, valid_list, test_list ):


    savePath           = "/home/withai/Desktop/LCLabelFiles/LCPhase_version2_len8_fps8_2_annotator.json"
    savePath_statistic = "/home/withai/Desktop/LCLabelFiles/LCPhase_version2_len8_fps8_2_annotator_statistic.json"
    extract_fps        = 8
    label_fps          = 1  # extract_fps should be exact divided by label_fps
    sequence_len       = 24
    cc                 = 0
    dictOut            = {"phase": {}}
    dictOut_statistic  = {"phase": {}}
    new_train_dict = {
        "train":{},
        "valid":{},
        "test":{}
    }

    #split labels to train,vaild,test
    for videoname in label_name_dict.keys():
        resident_id = videoname.split('-')[-1]
        find = False

        for videoname1 in train_list:
            resident_id = videoname1.split("_")[0].split("-")[-1]
            if videoname.__contains__(resident_id):
                new_train_dict["train"][videoname1] = label_name_dict[videoname]
                find = True
                break
        #
        for videoname1 in valid_list:
            resident_id = videoname1.split("_")[0].split("-")[-1]
            if videoname.__contains__(resident_id):
                new_train_dict["valid"][videoname1] = label_name_dict[videoname]
                find = True
                break

        for videoname1 in test_list:
            resident_id = videoname1.split("_")[0].split("-")[-1]
            if videoname.__contains__(resident_id):
                new_train_dict["test"][videoname1] = label_name_dict[videoname]
                find = True
                break

        if not find:
            print(videoname,'not found')



    #generate label sequnce
    videoidx  = 0

    cc        = 0
    for phase in new_train_dict.keys():

        if phase not in dictOut['phase'].keys():
            dictOut['phase'][phase] = {}
            dictOut_statistic['phase'][phase] = {}

        for videoname in new_train_dict[phase].keys():

            print( videoidx,"/", len( new_train_dict["valid"].keys()) + len( new_train_dict["train"].keys()) + len( new_train_dict["test"].keys()))
            videoidx += 1

            listMsg = new_train_dict[phase][videoname]
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


    return dictOut,dictOut_statistic



def merge_data( dictOut, append_dict ,phase):

    for videoname in append_dict.keys():
        for labelid in append_dict[videoname].keys():

            if labelid not in dictOut["phase"][ phase].keys():
                dictOut["phase"][ phase][labelid] = []
            # print( len(dictOut["phase"]["test"][labelid]), len(test_dict[videoname][labelid]) )
            dictOut["phase"][phase][labelid].extend( append_dict[videoname][labelid] )
            # print( len(dictOut["phase"]["test"][labelid]) )

    return  dictOut

def get_video_list():

    path    = "/home/withai/Desktop/videolist_8_3_wsd.xlsx"
    xlsdata_train = read_xls_rows(path)

    path    = "/home/withai/Desktop/videolist_8_3_wsd.xlsx"
    xlsdata_valid = read_xls_rows(path,1)

    path    = "/home/withai/Desktop/videolist_8_3_wsd.xlsx"
    xlsdata_test = read_xls_rows(path,2)

    #load old train val data sequnce
    old_test_json = "/home/withai/Desktop/LCLabelFiles/LCPhase_version1_len24_2_annotator_checked.json"
    with open( old_test_json) as f:
        train_val_data = json.load(f)

    #load old test data sequnce
    old_test_json = "/home/withai/Desktop/LCLabelFiles/LCPhase_version2_len24_2_annotator_checked.json"
    with open( old_test_json) as f:
        test_data = json.load(f)

    #get old test video list
    test_video_list,  test_dict  = get_videolist_of_dataset( test_data,      "test" )
    train_video_list, train_dict = get_videolist_of_dataset( train_val_data, "train")
    valid_video_list, valid_dict = get_videolist_of_dataset( train_val_data, "valid")

    # train_set = set(train_video_list)
    # valid_set = set(valid_video_list)
    # test_set  = set(test_video_list)
    # train_val  = train_set&valid_set
    # train_test = train_set&test_set
    # val_test   = valid_set&test_set

    total_list_old = train_video_list + valid_video_list + test_video_list

    find_dict_train,not_find_dict_train  = get_find_not_find_video(xlsdata_train, total_list_old)
    find_dict_valid, not_find_dict_valid = get_find_not_find_video(xlsdata_valid, total_list_old)
    find_dict_test, not_find_dict_test   = get_find_not_find_video(xlsdata_test, total_list_old)

    videoinfopth = "/home/withai/Desktop/LCLabelFiles/videopth_info.json"
    path1 = "/home/withai/Desktop/LCLabelFiles/phase_specialevents/phases/35"
    phase_list1 = os.listdir(path1)

    path2 = "/home/withai/Desktop/LCLabelFiles/phase_specialevents/phases/36"
    phase_list2 = os.listdir(path2)
    extract_fps = 8
    video_list = not_find_dict_test + not_find_dict_valid + not_find_dict_train
    label_name_dict = get_json_label(extract_fps, path1, path2, videoinfopth, video_list)

    dictOut,dictOut_statistic = generate_train_sequnce(label_name_dict, not_find_dict_train, not_find_dict_valid, not_find_dict_test)

    dictOut = merge_data(dictOut, test_dict, "test"  )
    dictOut = merge_data(dictOut, train_dict, "train")
    dictOut = merge_data(dictOut, valid_dict, "valid")

    savePath = "/home/withai/Desktop/LCLabelFiles/LCPhase_222_len24_2_annotator_checked.json"
    with open(savePath, 'w', encoding='utf-8') as f:
        json.dump(dictOut, f, ensure_ascii=False, indent=2)
        f.close()




def generate_label():

    version1_train_val_split_path = "/home/withai/Desktop/LCLabelFiles/videoname_phase_list_100-1.json"
    with open(version1_train_val_split_path) as f:
        train_val = json.load(f)
    version1_train = train_val['train']
    version1_valid = train_val['valid']

    version2_test_dir = "/home/withai/Pictures/LCFrame/100-1-2-8fps"
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

    savePath           = "/home/withai/Desktop/LCLabelFiles/LCPhase_version2_len8_fps8_2_annotator.json"
    savePath_statistic = "/home/withai/Desktop/LCLabelFiles/LCPhase_version2_len8_fps8_2_annotator_statistic.json"
    extract_fps        = 8
    sequence_len       = 8
    cc                 = 0
    dictOut            = {"phase": {}}
    dictOut_statistic  = {"phase": {}}
    train_dict = {
        "train":{},
        "valid":{},
        "test":{}
    }

    #get label
    label_nane_dict    = get_json_label( extract_fps )


    #split labels to train,vaild,test
    for videoname in label_nane_dict.keys():
        resident_id = videoname.split('-')[-1]
        find = False

        # for videoname1 in version1_train:
        #     resident_id = videoname1.split("_")[0].split("-")[-1]
        #     if videoname.__contains__(resident_id):
        #         train_dict["train"][videoname1] = label_nane_dict[videoname]
        #         find = True
        #         break
        #
        # for videoname1 in version1_valid:
        #     resident_id = videoname1.split("_")[0].split("-")[-1]
        #     if videoname.__contains__(resident_id):
        #         train_dict["valid"][videoname1] = label_nane_dict[videoname]
        #         find = True
        #         break

        for videoname1 in version2_test_new:
            resident_id = videoname1.split("_")[0].split("-")[-1]
            if videoname.__contains__(resident_id):
                train_dict["test"][videoname1] = label_nane_dict[videoname]
                find = True
                break

        if not find:
            print(videoname,'not found')

    #generate label sequnce
    videoidx  = 0
    label_fps = 8  #extract_fps should be exact divided by label_fps
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

    #save label as json file
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


lable_dict = {'建立气腹':1,
              '分离粘连':2,
              '游离胆囊三角':3,
              '分离胆囊床':4,
              '清理术区':6,
              '抓取胆囊':5,
              '取出胆囊':5,
              "清理术野":6}



def get_video_info():

    dir = ""
    traversalDir()

    video_info = {}
    for videoname in video_path.keys():

        try:
            pth = video_path[videoname]
            vid = imageio.get_reader(pth, 'ffmpeg')
            metaData = vid.get_meta_data()
            fps = metaData['fps']
            duration = metaData['duration']
            video_info[videoname] = {}
            video_info[videoname]['path'] = pthF
            video_info[videoname]['fps']  = fps
            video_info[videoname]['duration'] = duration
        except:
            print(videoname, "faild read video info")

    videoinfopth = "/home/withai/Desktop/LCLabelFiles/videopth_info_.json"
    with open(videoinfopth, 'w') as f:
        json.dump(video_info, f)


def get_json_label(extract_fps, pth1,pth2, videoinfopth, videolist):
    # pth1 = "/home/withai/Desktop/phase-100-1/100-1/100-1/CZX"
    # pth2 = "/home/withai/Desktop/phase-100-1/100-1/100-1/WSD"
    # extract_fps = 8
    filelist1   = os.listdir(pth1)
    filelist2   = os.listdir(pth2)

    #get video information
    # videoinfopth = "/mnt/FileExchange/withai/project/LC阶段识别/videopth_info_update_2021_8_4.json"
    video_info   = {}
    if os.path.exists(videoinfopth):
        with open(videoinfopth) as f:
            video_info = json.load(f)
    else:

        for videoname in video_path.keys():

            try:
                pth = video_path[videoname]
                vid = imageio.get_reader(pth, 'ffmpeg')
                metaData = vid.get_meta_data()
                fps      = metaData['fps']
                duration = metaData['duration']
                video_info[videoname] = {}
                video_info[videoname]['path'] = pth
                video_info[videoname]['fps']  = fps
                video_info[videoname]['duration'] = duration
            except:
                print(videoname, "faild read video info")

        with open(videoinfopth,'w') as f:
            json.dump(video_info,f)

    #find out the videos both in two directory
    # videolist     = []
    # filenamelist1 = os.listdir(pth1)
    # filenamelist2 = os.listdir(pth2)
    # for filename in filenamelist1:
    #     if filename in filenamelist2:
    #         videolist.append(filename)

    #remaind the video name exist video infomation
    read_dict = {}
    for videoname in videolist:
        videoname = videoname.split('.')[0]
        surgid    = videoname.split('-')[-1]
        find = False
        for videoname1 in video_info.keys():
            if videoname1.__contains__(surgid) or videoname.__contains__(videoname1):
                read_dict[videoname] = video_info[videoname1]
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
                # labelid = label['id']
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


def get_append_video():

    train_val_txt = "/home/withai/Desktop/train_val.txt"
    with open(train_val_txt) as f:
        video_list = f.readlines()


    path    = "/home/withai/Desktop/videolist_8_3_wsd.xlsx"
    xlsdata_train = read_xls_rows(path)

    path    = "/home/withai/Desktop/videolist_8_3_wsd.xlsx"
    xlsdata_valid = read_xls_rows(path,1)

    path    = "/home/withai/Desktop/videolist_8_3_wsd.xlsx"
    xlsdata_test = read_xls_rows(path,2)

    video_list_train_val = xlsdata_valid + xlsdata_train

    not_find_list = []
    find_list     = []
    for video in video_list_train_val:
        resident_id = video[0].split("-")[-1]
        find = False
        for videoexist in video_list:
            if videoexist.__contains__(resident_id) :
                find_list.append(video[0])
                find = True
                break
        if not find:
            not_find_list.append(video[0])


    video_dir1 = "/home/withai/Pictures/LCFrame/100-1-2-8fps"
    video_dir_list1 = os.listdir(video_dir1)

    video_dir2      = "/home/withai/Pictures/LCFrame/append_video-8fps"
    video_dir_list2 = os.listdir(video_dir2)

    video_not_find_dict = {}
    for video in not_find_list:
        resident_id = video.split("-")[-1]
        for video1 in video_dir_list1:
            if video1.__contains__(resident_id):
                if video not in video_not_find_dict.keys():
                    video_not_find_dict[video] = [ os.path.join(video_dir1,video1)]
                else:
                    video_not_find_dict[video].append( os.path.join(video_dir1, video1))


    for video in not_find_list:
        resident_id = video.split("-")[-1]
        for video1 in video_dir_list2:
            if video1.__contains__(resident_id):
                if video not in video_not_find_dict.keys():
                    video_not_find_dict[video] = [ os.path.join(video_dir2,video1)]
                else:
                    video_not_find_dict[video].append( os.path.join(video_dir2, video1))

    # for videoname in video_not_find_dict.keys():

    save_dir = "/home/withai/Pictures/LCFrame/video_append"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for video in video_not_find_dict.keys():

        srcpth = video_not_find_dict[video][0]
        videoname = os.path.split(srcpth)[-1]
        despth = os.path.join( save_dir,videoname)
        if os.path.exists(despth):
            continue

        cmd = "cp -r "+srcpth+" "+despth
        os.system( cmd )




    print("end")




def re_create():
    path = "/home/withai/Desktop/LCLabelFiles/LCPhase_222_len24_2_annotator_checked.json"
    with open(path) as f:
        ori_data = json.load(f)

    save_path = "/home/withai/Desktop/LCLabelFiles/LCPhase_222_len24_2_annotator_test.json"
    phse_list = ["test"]
    new_dict = {
        "phase":{

        }
    }
    for phase in ori_data["phase"].keys():
        if phase in phse_list:
            new_dict["phase"][phase] = ori_data["phase"][phase]


    with open(save_path,"w") as f:
        json.dump(new_dict,f)


def check_video_exist():

    path1     = "/home/withai/Pictures/LCFrame/100-1-2-8fps"
    videolist1 = os.listdir(path1)

    path2     = "/home/withai/Pictures/LCFrame/append_video-8fps"
    videolist2 = os.listdir(path2)

    path3     = "/home/withai/Pictures/LCFrame/100-3-8fps"
    videolist3 = os.listdir(path3)

    jsonpath  = "/home/withai/Desktop/LCLabelFiles/LCPhase_222_len24_2_annotator_test.json"
    with open(jsonpath) as f:
        data = json.load(f)


    path    = "/home/withai/Desktop/videolist_8_3_wsd.xlsx"
    xlsdata_test = [pth[0] for pth in  read_xls_rows(path,2) ]

    test_video_list = []
    new_dict = {
        "phase":{
            "test":{

            }
        }
    }
    for labelid in data["phase"]["test"].keys():
        for sequnce in data["phase"]["test"][labelid]:
            videoname = sequnce[0].split("/")[0]
            find = False
            for videoname1 in xlsdata_test:
                resident_id = videoname1.split("-")[-1]
                if videoname.__contains__(resident_id):
                    find = True
                    if labelid not in new_dict["phase"]["test"].keys():
                        new_dict["phase"]["test"][labelid] = [sequnce]
                    else:
                        new_dict["phase"]["test"][labelid].append(sequnce)
                    if videoname not in test_video_list:
                        test_video_list.append(videoname)

                    break

    # jsonpath  = "/home/withai/Desktop/LCLabelFiles/LCPhase_222_len24_2_annotator_test.json"
    # with open(jsonpath, "w") as f:
    #     json.dump( new_dict,f)

    not_find_list1 = []
    find_dict1 = {}
    for video in test_video_list:
        find = False
        resident_id = video.split("-")[-1]
        for video1 in videolist1:
            if video1.__contains__(resident_id):
                find = True
                find_dict1[video] = video1
                break
            # print(video1)
        if not find:
            not_find_list1.append(video)


    not_find_list2 = []
    find_dict2 = {}
    for video in not_find_list1:
        find = False
        resident_id = video.split("-")[-1]
        for video1 in videolist2:
            if video1.__contains__(resident_id):
                find = True
                find_dict2[video] = video1
                break
            # print(video1)
        if not find:
            not_find_list2.append(video)

    not_find_list3 = []
    find_dict3 = {}
    for video in not_find_list2:
        find = False
        resident_id = video.split("-")[-1]
        for video1 in videolist3:
            if video1.__contains__(resident_id):
                find = True
                find_dict2[video] = video1
                break
            # print(video1)
        if not find:
            not_find_list3.append(video)


    for video in find_dict1.keys():

        videoname = find_dict1[video]
        srcpth    = os.path.join( path1, videoname)
        despth    = os.path.join( path2,videoname)
        if os.path.exists(despth):
            continue

        cmd = "cp -r "+srcpth+" "+despth
        os.system( cmd )

    print(not_find_list3)


if __name__ == "__main__":
    # get_video_list()

    # get_append_video()

    check_video_exist()

    # re_create()
    # generate_label()

    print("end")