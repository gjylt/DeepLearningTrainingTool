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

if __name__ == "__main__":

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