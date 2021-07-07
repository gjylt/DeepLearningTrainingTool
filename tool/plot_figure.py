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