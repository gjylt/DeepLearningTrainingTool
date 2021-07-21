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


def get_video_info():

    savedir     = get_config('savedir')
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
                    fps = metaData['fps']
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

if __name__ == "__main__":

    for videoname in video_path.keys():
        path = os.path.join( "/mnt/video/LC10000/CompleteVideo/", video_path[videoname] )
        vid = imageio.get_reader(path, 'ffmpeg')
        metaData = vid.get_meta_data()
        print(videoname,metaData)

    get_video_info()

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

    print("end")