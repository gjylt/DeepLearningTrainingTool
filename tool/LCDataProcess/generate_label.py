import os
import numpy as np
import json
import math


lable_dict = {'建立气腹':1,
              '分离粘连':2,
              '游离胆囊三角':3,
              '分离胆囊床':4,
              '清理术区':6,
              '抓取胆囊':5,
              '取出胆囊':5,
              "清理术野":6}

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