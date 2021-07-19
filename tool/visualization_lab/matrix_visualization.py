import json
import os

import matplotlib.pyplot as plt
import numpy
import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D


def plot_surface(X,Y,Z):

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.show()

def plot_line(x,y):

    ax = plt.figure()

    plt.plot(x,y)

    plt.show()


def visualization_image():

    pth  = "/Users/guan/Desktop/document/medical/dataset/instrument/test/LC-CSR-7VID014-319.jpg"
    img  = cv2.imread( pth )
    sp   = img.shape

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    x    = np.linspace(1,sp[0],sp[0])
    y    = np.linspace(1,sp[1],sp[1])
    X,Y  = np.meshgrid(y,x)
    #Z   = gray
    # plot_surface( X, Y, Z )

    # z   = gray[rows, :]
    # z_b = img[rows, :, 0]
    # z_g = img[rows, :, 1]
    # z_r = img[rows, :, 2]
    # b   = img[:, :,  0]
    # g   = img[:, :,  1]
    # r   = img[:, :,  2]

    fig = plt.figure()
    row = 5
    col = 1

    idx  = row*100+10*col + 1
    ax2  = fig.add_subplot(idx)
    rows = 98
    z_b  = img[rows, :, 0]
    z_g  = img[rows, :, 1]
    z_r  = img[rows, :, 2]
    ax2.plot(y, z_b)
    ax2.plot(y,z_g)
    ax2.plot(y,z_r)

    # idx = row*100+10*col + 3
    # ax3 = fig.add_subplot(idx)
    # ax3.plot(y, z_g)

    # idx = row*100+10*col + 4
    # ax4 = fig.add_subplot(idx)
    # ax4.plot(y, z_r)
    #
    # idx = row*100+10*col + 5
    # ax5 = fig.add_subplot(idx)
    # ax5.plot(y, z)

    # idx = row*100+10*col+1
    # ax1 = fig.add_subplot(122)
    # # ax1.imshow(gray,cmap='gray')
    # ax1.imshow(img)
    plt.show()
    # plot_line(x,z)

from labelme import utils
import  imageio

def visualization_instrument():
    dirpth  = '/Users/guan/Desktop/document/medical/dataset/instrument/LC-CSR-7VID014'
    dirlist = [ pth for pth in  os.listdir(dirpth) if pth.endswith('.json') ]

    for jsonname in dirlist:
        filepth = os.path.join( dirpth, jsonname)
        with open(filepth,encoding='utf-8') as f:
            data = json.load(f)

        img = utils.img_b64_to_arr(data['imageData'])
        shpaes = data['shapes']
        for shape in shpaes:
            shape_type = shape['shape_type']
            points     = shape['points']
            if shape_type == "polygon":

                points  = np.array([points],np.int)
                classid = 255
                mask    = np.zeros_like(img)
                cv2.drawContours(mask, points, -1, (classid, classid, classid), -1)
                tmp     = mask[:,:,0].squeeze()
                idxs    = np.where( tmp == 255)
                target  = img[idxs[0], idxs[1],:]
                cv2.imshow('mask', mask)
                cv2.waitKey()


        # plt.figure()
        # plt.imshow(img)
        # plt.show()
        # cv2.imshow('test',img)
        # cv2.waitKey()


if __name__ == "__main__":

    visualization_instrument()

    print("end")


