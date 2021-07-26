import json
import matplotlib.pyplot as plt
import os
import numpy as np

def visualizationArray( arraylist, videoname , savedir ="",namlist = [] ):

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    plt.figure()
    pic_index = len(arraylist)*100+10
    idx = 1

    for array in arraylist:

        x = np.arange(0, len(array))
        ax = plt.subplot(pic_index+idx)
        plt.axhline(y=1, ls="--")
        plt.axhline(y=3, ls="--")
        plt.axhline(y=5, ls="--")
        if idx == 1:
            plt.title(videoname)
        if len(namlist) > 0:
            titlename = namlist[idx-1]
        else:
            titlename = ""
        ax.text(int(len(array) / 2) - 10, 3,  titlename)
        ax.plot(x, np.array(array), color=colors[idx])
        # plt.ylabel( tool_name_list1[idx])
        plt.yticks([0, 2, 4,6])
        idx += 1

    if savedir !="" and os.path.exists( savedir ):
        savepth = os.path.join(savedir, videoname+".jpg")
        plt.savefig(savepth)


import imageio
import cv2
import math
from tqdm import tqdm

def visualize_video( videoinfo, labelist ):


    datadir   = "/mnt/video/LC10000"

    videoPath = videoinfo['path'].split('LC10000')[-1]
    videoPath1=  datadir + videoPath
    reader    = imageio.get_reader(videoPath1)
    videoMeta = reader.get_meta_data()
    fps       = videoMeta["fps"]
    duration  = videoMeta["duration"]

    if fps != videoinfo['fps'] or duration != videoinfo['duration']:
        return None

    lable_len = int(len(labelist[0])*fps)
    videoframe= int(duration * fps)
    frame_len = min( lable_len, videoframe)

    savedir   = "/home/withai/Desktop/10video_visualize"
    videoSavePath = os.path.join( savedir, videoPath.split('/')[-1] )
    writer    = imageio.get_writer(videoSavePath, fps=fps,macro_block_size=1)
    for idx in tqdm( range( frame_len ) ):
        frame = reader.get_data(idx)

        lableidx = math.floor(idx/fps)
        czx   = labelist[0][lableidx]
        wsd   = labelist[1][lableidx]
        gjy   = labelist[2][lableidx]
        wyx   = labelist[3][lableidx]

        cv2.putText(frame, "czx=" + str(czx), (5,50 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frame, "wsd=" + str(wsd), (5, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frame, "gjy=" + str(gjy), (5, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frame, "wyx=" + str(wyx), (5, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


        # different = False
        # labels = [czx,wsd,gjy,wyx]
        # for idx in labels:
        #     for idy in labels:
        #         if idx != idy:
        #             different = True
        #             break
        # if different:
        #     frame[:,:,0] = frame[:,:,0]*1.3
        #     tmp = frame[:,:,0].squeeze()
        #     idxs = np.where(tmp>255)
        #     frame[:,:,idxs] = 255

        # cv2.imshow("test",frame)
        # cv2.waitKey(3)

        writer.append_data(frame)
    writer.close()





if __name__ == "__main__":
    pth = "/mnt/FileExchange/withai/project/LC阶段识别/10video_annotation_analyze/compare_dict.json"
    with open(pth) as f:
        data = json.load(f)

    pth = "/home/withai/Desktop/LCLabelFiles/lc200_videopth.json"
    with open(pth) as f:
        videoinfo = json.load(f)

    video_info_10 = {}
    for videoname in data.keys():
        for videoname1 in videoinfo.keys():
            if videoname.__contains__(videoname1) or videoname1.__contains__(videoname):
                video_info_10[videoname] = videoinfo[videoname1][0]

    indx = 0
    for videoname in video_info_10.keys():
        print("process video ",videoname, indx , "/",len(video_info_10.keys()))
        visualize_video( video_info_10[videoname], data[videoname][0] )
        indx += 1
    print("end")