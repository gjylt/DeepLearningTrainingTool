import imageio
import tqdm
import os
import cv2
from tool.parse_config import get_config
import shutil
from tool.LCDataProcess.LCJson2TRN import get_json_label
from tool.readwrite import read_xls_rows
from tool.LCDataProcess.datalist_config import train_list_g,valid_list_g,test_list_g

platform = "linux"
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

def extract_frame(videoS,savedir, path_info_dict):
    for path in videoS:

        endpoint = path_info_dict[path]

        reader = imageio.get_reader(path)
        width, height = reader.get_meta_data()["size"]
        fps = reader.get_meta_data()["fps"]
        duration = reader.get_meta_data()["duration"]
        total_frame_num = duration * fps

        saveImageFolder = os.path.join(savedir, "{}".format(path.split('.')[0].split('/')[-1]) )
        os.makedirs(saveImageFolder, exist_ok=True)

        exist_num = len(os.listdir(saveImageFolder))
        if exist_num >= endpoint:
            continue

        print("extract video", path.split('/')[-1],duration,fps,total_frame_num)
        for num, image in enumerate(tqdm.tqdm(reader, desc=path.split('.')[0].split('/')[-1], total=total_frame_num)):


            if (num // (fps/8)) > ((num - 1) // (fps/8)):
                if int(num // (fps / 8)) + 1 > endpoint:
                    break

                imagePath = "{}/{}_{:0>5}.jpg".format(saveImageFolder, saveImageFolder.split('/')[-1], int(num // (fps/8)) + 1)
                if os.path.exists(imagePath):
                    continue
                if width == 1920:
                    image = cv2.resize(image, ( 1024, int(height*1024/width) ))
                imageio.imwrite(imagePath, image)

import json

from tool.readwrite import read_xls_rows


def get_video_info( savedir ):


    dir1        = os.path.join( get_config('nas_video_lc10000_path'),'CompleteVideo' )
    pth_list    = traversalDir(dir1, returnX='path')

    pth_list    = [pth for pth in pth_list if pth.__contains__("ORIGIN") and (pth.endswith("mp4") or pth.endswith("MP4")) ]
    video_dict  = {}

    oldpath = "/home/withai/Desktop/LCLabelFiles/videopth_info_.json"
    with open(oldpath) as f:
        video_dict = json.load(f)

    jump_num = 0
    read_fail_num = 0
    for pth in pth_list:
        appendx = pth.split('.')[-1]
        if appendx == "mp4" or appendx == "MP4":
            videoname = pth.split('/')[-1].split('.')[0]
            if videoname in video_dict.keys():
                jump_num += 1
                continue
                # print(videoname,pth,video_dict[videoname])
            else:
                try:
                    vid = imageio.get_reader(pth, 'ffmpeg')
                    metaData = vid.get_meta_data()
                    fps      = metaData['fps']
                    duration = metaData['duration']
                    video_dict[videoname] = {}
                    video_dict[videoname]['path'] = pth
                    video_dict[videoname]['fps']  = fps
                    video_dict[videoname]['duration'] = duration
                except:
                    read_fail_num += 1
                    print(videoname,"faild read video info")

    print("already exist video:", jump_num)
    print("read fail video:", read_fail_num)

    savepth = os.path.join( savedir, "videopth_info.json" )
    with open(savepth,'w') as f:
        json.dump(video_dict,f)

    return video_dict


video_path ={
"LC-CD-6689898":"/mnt/video/LC10000/CompleteVideo/hospital_id=9/surgery_id=476/video/20210107-LC-CD-6689898_ORIGIN.mp4",
"LC-DY-778970":"/mnt/video/LC10000/CompleteVideo/hospital_id=11/surgery_id=394/video/20201116-LC-DY-778970_ORIGIN.mp4",
"LC-DY-574665":"/mnt/video/LC10000/CompleteVideo/hospital_id=11/surgery_id=365/video/20201103-LC-DY-574665_ORIGIN.mp4",
"LC-DY-793367":"/mnt/video/LC10000/CompleteVideo/hospital_id=11/surgery_id=721/video/20210115-LC-DY-793367_ORIGIN.mp4",
"LC-DY-792089":"/mnt/video/LC10000/CompleteVideo/hospital_id=11/surgery_id=436/video/20210105-LC-DY-792089_ORIGIN.mp4",
"LC-CHB-468797":"/mnt/video/LC10000/CompleteVideo/hospital_id=12/surgery_id=338/video/20201223-LC-CHB-468797_ORIGIN.mp4",
"LC-CD-6498741":"/mnt/video/LC10000/CompleteVideo/hospital_id=9/surgery_id=433/video/20210105-LC-CD-6498741_ORIGIN.mp4",
"LC-XN-IP00000320162":"/mnt/video/LC10000/CompleteVideo/hospital_id=2/surgery_id=268/video/20201208-LC-XN-IP00000320162_ORIGIN.mp4"
}



def extract_frame_from_videos():

    for videoname in video_path.keys():
        path = os.path.join( "/mnt/video/LC10000/CompleteVideo/", video_path[videoname] )
        vid  = imageio.get_reader(path, 'ffmpeg')
        metaData = vid.get_meta_data()
        print(videoname,metaData)

    savedir = get_config('savedir')

    get_video_info( savedir )

    nas_video_path = get_config("nas_video_path")
    pth_list    = traversalDir(nas_video_path, returnX='path')
    video_dict  = {}
    appendlist  = []
    for pth in pth_list:
        if not pth.__contains__('ORIGIN'):
            continue
        appendx = pth.split('.')[-1]
        if appendx == "mp4" or appendx == "MP4":
            videoname = pth.split('/')[-1].split('.')[0]
            if videoname in video_dict.keys():
                print(videoname,pth,video_dict[videoname])
            else:
                video_dict[videoname] = pth

    # savepth = "/home/withai/Desktop/videopth.json"
    # with open(savepth,'w') as f:
    #     json.dump(video_dict,f)
    videolist_xls = get_config("videolist_xls")

    videos     = read_xls_rows(videolist_xls)
    videopaths = []
    for video in videos:
        videoname = video[0]
        if videoname in video_dict.keys():
            videopaths.append( video_dict[videoname] )

    savedir = get_config("savedir")
    extract_frame( videopaths, savedir)

def extracted_parkland_picture():


    extrcat_save_dir = "/home/withai/Desktop/extracted_img_4_parkland_100_1"

    extrcat_fps = 1
    pth1 = "/Users/guan/Desktop/phase_specialevents_parkland/phase_specialevents_parkland/phases/35"
    pth2 = "/Users/guan/Desktop/phase_specialevents_parkland/phase_specialevents_parkland/phases/36"
    videoinfopth = "/Users/guan/Desktop/document/medical/项目/LC/8月LC实验/8月LC/videopth_info.json"
    videolist = [pth.split(".")[0] for pth in os.listdir(pth1)]
    label_name_dict = get_json_label(extrcat_fps, pth1,pth2, videoinfopth, videolist)
    picture_dir = "/home/withai/Pictures/LCFrame/100-1-2-8fps"
    videonamelist = os.listdir(picture_dir)

    parkland_path = "/home/withai/Desktop/LCLabelFiles/Parklands.xlsx"
    xlsrows       = read_xls_rows(parkland_path)
    parkland_sequnce = {}
    # rows = int((len(xlsrows)-1)/2)
    rows = len(xlsrows)
    diffirent_sequnce = []
    for rowid in range(1,rows):
        # wsd_data = xlsrows[rowid*2+1]
        # czx_data = xlsrows[rowid*2+2]
        row_data = xlsrows[rowid]
        if row_data[1] == row_data[2]:
            parkland_sequnce[row_data[0]] = row_data[2]
        else:
            diffirent_sequnce.append( row_data[0])
            print(row_data[0],row_data[1],row_data[2])


    new_label_dict1 = {}
    for videoname in videonamelist:
        resident_id = videoname.split("_")[0].split("-")[-1]
        for videoname1 in label_name_dict.keys():
            if videoname1.__contains__(resident_id):
                new_label_dict1[videoname] = label_name_dict[videoname1]
            # print("")

    new_label_dict = {}
    for videoname in parkland_sequnce.keys():
        resident_id = videoname.split("-")[-1]
        for videoname1 in new_label_dict1.keys():
            if videoname1.__contains__(resident_id):
                new_label_dict[videoname1] = new_label_dict1[videoname1]



    if not os.path.exists(extrcat_save_dir):
        os.makedirs(extrcat_save_dir)

    # extract picture
    videoid = 0
    dont_have_AL_MCT = []
    for videoname in new_label_dict.keys():
        videoid += 1
        print(videoid,"/",len(new_label_dict.keys()))

        labelist = new_label_dict[videoname]
        if 2 in labelist:
            AL_indx = labelist.index(2)
        else:
            AL_indx = len(labelist)

        if 3 in labelist:
            MCT_indx = labelist.index(3)
        else:
            MCT_indx = len(labelist)

        if MCT_indx == AL_indx:
            dont_have_AL_MCT.append(videoname)
            continue

        # MCT_indx = labelist.index(3)
        extract_indx = min(AL_indx, MCT_indx)
        start =  max(0, extract_indx-10)
        end = extract_indx + 10

        saveImageFolder = os.path.join(extrcat_save_dir, videoname)
        if not os.path.exists(saveImageFolder):
            os.makedirs(saveImageFolder)
        srcImageFolder = os.path.join(picture_dir, videoname)
        for imgid in range(start, end):

            srcimgpth = "{}/{}_{:0>5}.jpg".format(srcImageFolder, srcImageFolder.split('/')[-1], imgid * 8)
            saveimgpth = "{}/{}_{:0>5}.jpg".format(saveImageFolder, saveImageFolder.split('/')[-1], imgid * 8)
            if os.path.exists(srcimgpth):
                shutil.copy(srcimgpth, saveimgpth)

    for videoname in dont_have_AL_MCT:
        print( videoname )
    # print("end")


def get_append_video_list():

    path    = "/home/withai/Desktop/videolist_8_3_wsd.xlsx"
    xlsdata1 = read_xls_rows(path)

    path    = "/home/withai/Desktop/videolist_8_3_wsd.xlsx"
    xlsdata2 = read_xls_rows(path,1)

    path    = "/home/withai/Desktop/videolist_8_3_wsd.xlsx"
    xlsdata3 = read_xls_rows(path,2)

    exist_data = "/home/withai/Pictures/LCFrame/100-1-2-8fps"
    exist_videos1 = os.listdir(exist_data)

    exist_data = "/home/withai/Pictures/LCFrame/append_video-8fps"
    exist_videos2 = os.listdir(exist_data)

    exist_videos = exist_videos1 + exist_videos2

    xlsdata = xlsdata1 + xlsdata2 + xlsdata3
    not_find_list = []
    find_videos = {}
    for rowdata in xlsdata:
        resident_id = rowdata[0].split("-")[-1]
        find = False
        if rowdata[0] in find_videos.keys():
            print( rowdata[0],"?")
        for video1 in exist_videos:
            if video1.__contains__(resident_id):
                find = True
                find_videos[rowdata[0]] = video1
                break

        if not find:
            not_find_list.append(rowdata[0])

    nas_video_path = get_config("nas_video_path")
    pth_list    = traversalDir(nas_video_path, returnX='path')
    info_data   = {}
    appendlist  = []
    for pth in pth_list:
        if not pth.__contains__('ORIGIN'):
            continue
        appendx = pth.split('.')[-1]
        if appendx == "mp4" or appendx == "MP4":
            videoname = pth.split('/')[-1].split('.')[0]
            if videoname in info_data.keys():
                print(videoname,pth,info_data[videoname])
            else:
                info_data[videoname] = pth

    video_list_4_extracted = []
    for video1 in not_find_list:
        resident_id = video1.split("-")[-1]
        find = False
        for video2 in info_data.keys():
            if video2.__contains__(resident_id):
                video_list_4_extracted.append( info_data[video2] )
                find = True
                break

        if not find:
            print(video1)

    return video_list_4_extracted

def extracted_append_video():

    savedir    = get_config("savedir")
    # videopaths = get_append_video_list()

    videopaths = [
        "/mnt/video/LC10000/CompleteVideo/hospital_id=10/surgery_id=1408/video/20210420-LC-PZHH-0000109665_ORIGIN.mp4"
    ]

    extract_frame( videopaths, savedir)

def statistic_parkland():

    path    = "/Users/guan/Desktop/document/medical/项目/LC/8月LC实验/8月LC/videolist_8_3_wsd.xlsx"
    xlsdata1 = [pth[0] for pth in read_xls_rows(path) ]

    path    = "/Users/guan/Desktop/document/medical/项目/LC/8月LC实验/8月LC/videolist_8_3_wsd.xlsx"
    xlsdata2 = [pth[0] for pth in read_xls_rows(path,1) ]

    path    = "/Users/guan/Desktop/document/medical/项目/LC/8月LC实验/8月LC/videolist_8_3_wsd.xlsx"
    xlsdata3 = [pth[0] for pth in read_xls_rows(path,2) ]

    path          = "/Users/guan/Desktop/document/medical/项目/LC/parkland分级/Parklands_222.xlsx"
    parkland_data = read_xls_rows(path)

    video_label   = {
        "equal":{

        },
        "different":
        {

        },
        "label":{

        },
        "phase":{

        }
    }


    for rowid in range(1, len(parkland_data)):
        rowdata   = parkland_data[rowid]
        videoname = rowdata[0]
        wsd       = int(rowdata[1])
        czx       = int(rowdata[2])
        phase     = ""
        for video1 in xlsdata1:
            if video1.__contains__(videoname):
                phase = "train"
                break

        for video1 in xlsdata2:
            if video1.__contains__(videoname):
                phase = "valid"
                break

        for video1 in xlsdata3:
            if video1.__contains__(videoname):
                phase = "test"
                break
        if phase == "":
            continue

        if wsd == czx:
            if phase not in video_label["phase"].keys():
                video_label["phase"][phase] = {}

            if wsd not in video_label["phase"][phase].keys():
                video_label["phase"][phase][wsd] = [videoname]
            else:
                video_label["phase"][phase][wsd].append(videoname)


            # video_label["phase"][phase][videoname] = wsd
            video_label["equal"][videoname]        = wsd
            if wsd not in video_label["label"].keys():
                video_label["label"][wsd] = [videoname]
            else:
                video_label["label"][wsd].append(videoname)
        else:
            video_label["different"][videoname] = [wsd,czx]

    for phase in video_label["phase"].keys():
        print(phase)
        for labelid in video_label["phase"][phase].keys():

            print(labelid,  len( video_label["phase"][phase][labelid] ) )
            print( video_label["phase"][phase][labelid] )


def statistic_parkland1():
    save_json_path = "/Users/guan/Desktop/parkland_train_val_test.json"
    used_video_json_path = "/Users/guan/Desktop/used_video.json"
    with open( save_json_path) as f:
        data = json.load(f)

    video_list ={}
    for phase in data["phase"].keys():
        if phase not in video_list.keys():
            video_list[phase] = {}

        for labelid in data["phase"][phase].keys():
            if labelid not in video_list[phase].keys():
                video_list[phase][labelid] = []

            for sequnce in data["phase"][phase][labelid]:
                videoname = sequnce[0].split("/")[0]
                if videoname not in video_list[phase][labelid]:
                    video_list[phase][labelid].append( videoname )


    for phase in video_list.keys():

        print(phase)
        for labelid in video_list[phase].keys():
            if labelid == '0':
                continue

            print( labelid, len( video_list[phase][labelid] ) )
            print( video_list[phase][labelid]  )

    print("end")


from tool.LCDataProcess.LCJson2TRN import find_exist_not_exist

def extract_video_frame_according_label():
    videopaths    = []
    savedir       = "/home/withai/Pictures/LCFrame/picture_for_parkland"
    train_val_txt = "/home/withai/Desktop/LCLabelFiles/train_val_new.txt"
    with open(train_val_txt) as f:
        train_val_videos = f.readlines()

    target_list = train_list_g + valid_list_g

    #find video list not in train and valid
    not_find_list = []
    for videoname in target_list:
        resident_id = videoname.split("-")[-1]
        find = False
        for videoname1 in train_val_videos:
            if videoname1.__contains__(resident_id):
                find = True
                break
        if not find:
            not_find_list.append(videoname)

    videopath1 = "/home/withai/Pictures/LCFrame/100-1-2-8fps"
    videolist1 = os.listdir(videopath1)
    not_find_list1 = []
    find_in_1  = []
    for videoname in not_find_list:
        resident_id = videoname.split("-")[-1]
        find = False
        for videoname1 in videolist1:
            if videoname1.__contains__(resident_id):
                find = True
                find_in_1.append(videoname1)
                break
        if not find:
            not_find_list1.append(videoname)

    videopath2     = "/home/withai/Pictures/LCFrame/append_video-8fps"
    videolist2     = os.listdir(videopath2)
    not_find_list2 = []
    find_in_2      = []
    for videoname in not_find_list1:
        resident_id = videoname.split("-")[-1]
        find = False
        for videoname1 in videolist2:
            if videoname1.__contains__(resident_id):
                find = True
                find_in_2.append( videoname1 )
                break
        if not find:
            not_find_list2.append(videoname)




    #######################################################
    train_val_list = test_list_g

    path1      = "/home/withai/Pictures/LCFrame/100-1-2-8fps"
    find_dict1, not_find_list1= find_exist_not_exist(path1, train_val_list)

    path2      = "/home/withai/Pictures/LCFrame/append_video-8fps"
    find_dict2, not_find_list4 = find_exist_not_exist(path2, not_find_list1)

    path3      = "/home/withai/Pictures/LCFrame/100-3-8fps"
    find_dict3, not_find_list3 = find_exist_not_exist(path3, not_find_list4)

    append_new = "/home/withai/Pictures/LCFrame/new_append"
    find_dict4, not_find_list2 = find_exist_not_exist(append_new, not_find_list3)



    #######################################################


    videoinfo = "/home/withai/Desktop/LCLabelFiles/videopth_info.json"
    with open(videoinfo) as f:
        video_info = json.load(f)

    for videoname in video_info.keys():
        if not videoname.__contains__("ORIGIN"):
            continue
        for notfind in not_find_list2:
            resident_id = notfind.split("-")[-1]
            if videoname.__contains__(resident_id):
                path = video_info[videoname]['path']
                videopaths.append(path)



    extrcat_fps     = 1
    pth1            = "/home/withai/Desktop/LCLabelFiles/phase_specialevents/phases/35"
    pth2            = "/home/withai/Desktop/LCLabelFiles/phase_specialevents/phases/36"
    videoinfopth    = "/home/withai/Desktop/LCLabelFiles/videopth_info.json"
    train_list      = train_list_g
    valid_list      = valid_list_g
    test_list       = test_list_g
    videolist       = train_list + valid_list + test_list
    label_name_dict = get_json_label(extrcat_fps, pth1, pth2, videoinfopth, videolist)

    path_info_dict = {}
    not_find_list_ = []
    for path in videopaths:

        find = False
        for videoname in label_name_dict.keys():
            resident_id = videoname.split("-")[-1]
            if path.__contains__(resident_id):
                find = True
                labelist = label_name_dict[videoname]

                al_idx = len(labelist)
                mct_idx= len(labelist)
                if 2 in labelist:
                    al_idx = labelist.index(2)
                if 3 in labelist:
                    mct_idx= labelist.index(3)

                target_idx = min(al_idx,mct_idx)
                path_info_dict[path] = (target_idx+10)*8

                break

        if not find:
            not_find_list_.append(path)

    extract_frame( videopaths, savedir, path_info_dict )

if __name__ == "__main__":

    # extract_video_frame_according_label()

    # statistic_parkland()
    # genearate_parkland_train_label()
    # extracted_parkland_picture()

    extracted_append_video()
    # savedir = "/home/withai/Desktop/"
    # get_video_info(savedir)

    print("end")