import imageio
import tqdm
import os
import cv2
from tool.parse_config import get_config

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

def extract_frame(videoS,savedir):
    for path in videoS:

        reader = imageio.get_reader(path)
        width, height = reader.get_meta_data()["size"]
        fps = reader.get_meta_data()["fps"]
        duration = reader.get_meta_data()["duration"]
        total_frame_num = duration * fps

        saveImageFolder = os.path.join(savedir, "{}".format(path.split('.')[0].split('/')[-1]) )
        os.makedirs(saveImageFolder, exist_ok=True)

        print("extract video", path.split('/')[-1],duration,fps,total_frame_num)
        for num, image in enumerate(tqdm.tqdm(reader, desc=path.split('.')[0].split('/')[-1], total=total_frame_num)):
            if (num // (fps/8)) > ((num - 1) // (fps/8)):
                imagePath = "{}/{}_{:0>5}.jpg".format(saveImageFolder, saveImageFolder.split('/')[-1], int(num // (fps/8)) + 1)
                if width == 1920:
                    image = cv2.resize(image, ( 1024, int(height*1024/width) ))
                imageio.imwrite(imagePath, image)

import json

from tool.readwrite import read_xls_rows


def get_video_info( savedir ):


    dir1        = os.path.join( get_config('nas_video_lc10000_path'),'CompleteVideo' )
    pth_list    = traversalDir(dir1, returnX='path')
    video_dict  = {}

    for pth in pth_list:
        appendx = pth.split('.')[-1]
        if appendx == "mp4" or appendx == "MP4":
            videoname = pth.split('/')[-1].split('.')[0]
            if videoname in video_dict.keys():
                print(videoname,pth,video_dict[videoname])
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
                    print(videoname,"faild read video info")

    savepth = os.path.join( savedir, "videopth_info.json" )
    with open(savepth,'w') as f:
        json.dump(video_dict,f)


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



# def extract_frame():
#
#     for videoname in video_path.keys():
#         path = os.path.join( "/mnt/video/LC10000/CompleteVideo/", video_path[videoname] )
#         vid = imageio.get_reader(path, 'ffmpeg')
#         metaData = vid.get_meta_data()
#         print(videoname,metaData)
#
#     savedir = get_config('savedir')
#
#     get_video_info( savedir )
#
#     nas_video_path = get_config("nas_video_path")
#     pth_list    = traversalDir(nas_video_path, returnX='path')
#     video_dict  = {}
#     appendlist  = []
#     for pth in pth_list:
#         if not pth.__contains__('ORIGIN'):
#             continue
#         appendx = pth.split('.')[-1]
#         if appendx == "mp4" or appendx == "MP4":
#             videoname = pth.split('/')[-1].split('.')[0]
#             if videoname in video_dict.keys():
#                 print(videoname,pth,video_dict[videoname])
#             else:
#                 video_dict[videoname] = pth
#
#     # savepth = "/home/withai/Desktop/videopth.json"
#     # with open(savepth,'w') as f:
#     #     json.dump(video_dict,f)
#     videolist_xls = get_config("videolist_xls")
#
#     videos     = read_xls_rows(videolist_xls)
#     videopaths = []
#     for video in videos:
#         videoname = video[0]
#         if videoname in video_dict.keys():
#             videopaths.append( video_dict[videoname] )
#
#     savedir = get_config("savedir")
#     extract_frame( videopaths, savedir)



import shutil

from tool.LCDataProcess.LCJson2TRN import get_json_label
from tool.readwrite import read_xls_rows

def extracted_parkland_picture():

    extrcat_fps = 1
    label_name_dict = get_json_label(extrcat_fps)
    picture_dir = "/home/withai/Pictures/LCFrame/100-1-2-8fps"
    videonamelist = os.listdir(picture_dir)

    parkland_path = "/home/withai/Desktop/Parklands_3.xlsx"
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


    extrcat_save_dir = "/home/withai/Desktop/extracted_img_4_parkland_100_1"
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

if __name__ == "__main__":


    # extracted_parkland_picture()

    extracted_append_video()

    print("end")