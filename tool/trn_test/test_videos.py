from tool.trn_test import demo
import os
import json
from tool.parse_config import get_config
from tool.readwrite import read_xls_rows, traversalDir

def get_video_list():
    datasetdir = "/mnt/video/LC10000/CompleteVideo/"
    bachid = 0
    json_pth = "/home/withai/Desktop/trn_test/LC_7times_dataset.json"
    savedir = "/home/withai/Desktop/trn_test"
    with open(json_pth) as f:
        jsondata = json.load(f)

    infolist = jsondata[str(bachid)]
    videolist= []
    notexist = []
    for info in infolist:
        videoname = info[1]
        surgeid   = int(info[0])
        hospitalid= int(info[4])
        hospitalstr = "hospital_id="+str(hospitalid)
        surgestr    = "surgery_id="+str(surgeid)
        subdir      = os.path.join(datasetdir,hospitalstr,surgestr,"video")
        if not os.path.exists(subdir):
            notexist.append(subdir)
            print(subdir)
            continue

        filelist = os.listdir(subdir)
        origname = ""
        for filename in filelist:
            if filename.__contains__("ORIGIN"):
                origname = filename
                break
        if origname== "":
            break
        filepath = os.path.join(subdir,origname)
        if os.path.exists(filepath):
            videolist.append(filepath)

    videofind_result_dict = {}
    videofind_result_dict["exit"] = videolist
    videofind_result_dict["disappear"] = notexist

    savepth = os.path.join(savedir, str(bachid)+".json" )
    with open(savepth,"w") as f:
        json.dump( videofind_result_dict,f)

    return videolist


def get_100_2_without10video_list():
    datasetdir = "/mnt/video/LC10000/CompleteVideo/"
    bachid     = 0
    json_pth   = "/home/withai/Desktop/100-2-without-10video.json"
    savedir    = "/home/withai/Desktop/trn_test"
    with open(json_pth) as f:
        jsondata = json.load(f)

    # infolist = jsondata[str(bachid)]
    videolist= []
    notexist = []
    idx = 0
    for key in jsondata.keys():
        idx += 1
        # if idx <= 10:
        #     continue
        info = jsondata[key]
        videoname = info[1]
        surgeid   = int(info[0])
        hospitalid= int(info[4])
        hospitalstr = "hospital_id="+str(hospitalid)
        surgestr    = "surgery_id="+str(surgeid)
        subdir      = os.path.join(datasetdir,hospitalstr,surgestr,"video")
        if not os.path.exists(subdir):
            notexist.append(subdir)
            print(subdir)
            continue

        filelist = os.listdir(subdir)
        origname = ""
        for filename in filelist:
            if filename.__contains__("ORIGIN"):
                origname = filename
                break
        if origname== "":
            break
        filepath = os.path.join(subdir,origname)
        if os.path.exists(filepath):
            videolist.append(filepath)

    videofind_result_dict = {}
    videofind_result_dict["exit"] = videolist
    videofind_result_dict["disappear"] = notexist

    # savepth = os.path.join(savedir, str(bachid)+".json" )
    # with open(savepth,"w") as f:
    #     json.dump( videofind_result_dict,f)

    return videolist


video_path ={
"20210107-LC-CD-6689898_ORIGIN":"/mnt/video/LC10000/CompleteVideo/hospital_id=9/surgery_id=476/video/20210107-LC-CD-6689898_ORIGIN.mp4",
"20201116-LC-DY-778970_ORIGIN":"/mnt/video/LC10000/CompleteVideo/hospital_id=11/surgery_id=394/video/20201116-LC-DY-778970_ORIGIN.mp4",
"20201103-LC-DY-574665_ORIGIN":"/mnt/video/LC10000/CompleteVideo/hospital_id=11/surgery_id=365/video/20201103-LC-DY-574665_ORIGIN.mp4",
"20210115-LC-DY-793367_ORIGIN":"/mnt/video/LC10000/CompleteVideo/hospital_id=11/surgery_id=721/video/20210115-LC-DY-793367_ORIGIN.mp4",
"20210105-LC-DY-792089_ORIGIN":"/mnt/video/LC10000/CompleteVideo/hospital_id=11/surgery_id=436/video/20210105-LC-DY-792089_ORIGIN.mp4",
"20201223-LC-CHB-468797_ORIGIN":"/mnt/video/LC10000/CompleteVideo/hospital_id=12/surgery_id=338/video/20201223-LC-CHB-468797_ORIGIN.mp4",
"20210105-LC-CD-6498741_ORIGIN":"/mnt/video/LC10000/CompleteVideo/hospital_id=9/surgery_id=433/video/20210105-LC-CD-6498741_ORIGIN.mp4",
"20201208-LC-XN-IP00000320162_ORIGIN":"/mnt/video/LC10000/CompleteVideo/hospital_id=2/surgery_id=268/video/20201208-LC-XN-IP00000320162_ORIGIN.mp4"
}

def get_100_1_train():
    datasetdir = "/mnt/video/LC10000/CompleteVideo/"
    path       = "/home/withai/Desktop/LCLabelFiles/videoname_phase_list_100-1.json"
    with open(path) as f:
        data        = json.load(f)
    video_info_path = "/home/withai/Desktop/LCLabelFiles/videopth_info_.json"
    with open(video_info_path) as f:
        videoinfos  = json.load(f)

    train_val_list  = data["train"] + data["valid"]
    video_path_dict = {}
    for videoname1 in train_val_list:
        resident_id = videoname1.split("_")[0].split("-")[-1]
        for videoname in videoinfos.keys():
            info = videoinfos[videoname]
            if videoname.__contains__(resident_id) and videoname.__contains__("ORIGIN"):
                if videoname1 not in video_path_dict.keys() :
                    video_path_dict[videoname1] = [[videoname, info]]
                else:
                    video_path_dict[videoname1].append( [videoname,info] )

    exist_list = list(video_path_dict.keys())
    videopath_list = []
    for videoname in train_val_list:
        if videoname not in exist_list:
            if videoname in list(video_path.keys()):
                videopath_list.append( video_path[videoname])
            else:
                print(videoname)
        else:
            info = video_path_dict[videoname][0][1]['path']
            videopath_list.append( info  )

    # print("")
    return  videopath_list





def get_22_testvideo_list():

    path      = "/mnt/FileExchange/withai/project/LC阶段识别/8月LC/videolist_8_3_wsd.xlsx"
    xlsdata3  = [pth[0] for pth in read_xls_rows(path,2) ]

    videoinfo = "/home/withai/Desktop/LCLabelFiles/videopth_info.json"
    with open(videoinfo) as f:
        data = json.load(f)

    findlist = []
    not_find = []
    for videoname in xlsdata3:
        resident_id = videoname.split("-")[-1].replace(" ","")
        find = False
        for videoname1 in data.keys():
            if videoname1.__contains__(resident_id):
                findlist.append([videoname, data[videoname1]["path"]])
                find = True
                break
        if not find:
            not_find.append(videoname)

    dir1        = os.path.join( get_config('nas_video_lc10000_path'),'CompleteVideo' )
    pth_list    = traversalDir( dir1, returnX='path')
    pth_list  = [pth for pth in pth_list if pth.__contains__("ORIGIN") and pth.endswith(".mp4") or pth.endswith(".MP4") ]
    for videoname in not_find:
        resident_id = videoname.split("-")[-1]

        for path in pth_list:
            if path.__contains__(resident_id) :
                findlist.append([videoname,path])
                break




    print(not_find)
    return_list = []
    for listdata in findlist:
        return_list.append( listdata[1])


    return return_list





def process_videos():
    # videoS = [
    #           "/mnt/video/LC10000/CompleteVideo/hospital_id=1/surgery_id=783/video/20210203-LC-HX-0033834527_HD1080.mp4",
    #          ]
    videoS         = get_22_testvideo_list()
    checkPointPath = f"/home/withai/wangyx/TRN/action_len8_balance_pg_bg.csv_LC10_best.pth"
    checkPointPath = "/home/withai/Desktop/LCLabelFiles/TRN_something_RGB_BNInception_TRNmultiscale_segment8_6phaseBgNoMoreThanTarget_best.pth"
    saveDir        = "/home/withai/Desktop/LCLabelFiles/22video6phase_result"
    os.makedirs(saveDir, exist_ok=True)
    seqlen = 24
    subfps = 1
    for videoPath in videoS:
        demo.go(videoPath, checkPointPath, saveDir, seqlen, subfps)


if __name__ == "__main__":
    # get_22_testvideo_list()
    # get_video_list()
    process_videos()
    print("end")