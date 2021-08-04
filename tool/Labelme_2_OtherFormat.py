from tool.image_tools import str2Image
from tool.readwrite import traversalDir
import json
import os
import cv2

# anatomy_name_id_dict = {
#  'liver': 33872,
#  'gallbladder': 21003,
#  'stomach and duodenum': 6372,
#  'common bile duct': 6914,
#  'cystic artery': 6326,
#  'cystic duct': 10966,
#  'cystic bed': 8472
#
#  }

anatomy_name_id_dict = {
 'liver': 1,
 'gallbladder': 2,
 'stomach and duodenum': 3,
 'common bile duct': 4,
 'cystic artery': 5,
 'cystic duct': 6,
 'cystic bed': 7

 }



def Lableme2Darknet():

    src_dir  = "/mnt/FileExchange/withai/dataset/Annatomy/"
    save_dir = "/home/withai/Pictures/Anotomy4Darknet"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    imglist = traversalDir(src_dir)

    name_dict = {}
    for jsonpth in imglist:
        with open(jsonpth) as f:
            data = json.load(f)
        if "imageData" not in data.keys():
            continue
        imgdata = data["imageData"]
        image_h = data["imageHeight"]
        image_w = data["imageWidth"]
        shapes  = data["shapes"]

        filename = jsonpth.split("/")[-1]

        find_object = False
        recort_list = []
        for shape in shapes:
            shape_type = shape["shape_type"]
            lablename  = shape["label"]

            if lablename != "gallbladder":
                continue

            if lablename not in name_dict.keys():
                name_dict[lablename]  = 1
            else:
                name_dict[lablename] += 1


            if shape_type == "rectangle":
                p1 = shape['points'][0]
                p2 = shape['points'][1]
                x_l = min( p1[0],p2[0] )
                x_r = max( p1[0],p2[0] )
                y_u = min( p1[1],p2[1] )
                y_d = max( p1[1],p2[1] )
                w = x_r - x_l
                h = y_d - y_u
                x_c = ( x_l+x_r)/2
                y_c = ( y_u + y_d)/2

                recort_list.append( [ 0, x_c/image_w, y_c/image_h, w/image_w, h/image_h ]  )
                find_object = True

        if find_object:
            imgpth = os.path.join(save_dir, filename.replace(".json",".jpg"))
            txtpth = os.path.join(save_dir, filename.replace(".json", ".txt"))
            img    = str2Image(imgdata)
            with open(txtpth,'w') as f:
                for record in recort_list:
                    linstr = "%d\t%f\t%f\t%f\t%f"%(record[0],record[1],record[2],record[3],record[4])
                    f.writelines(linstr)

            cv2.imwrite(imgpth, img)

    print(name_dict)
    print("end")

import shutil
import random

def splitdataset():
    srcdir = "/home/withai/Pictures/Anotomy4Darknet"
    savedir = "/home/withai/Pictures/Anotomy/"
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    files = [ pth for pth in os.listdir(srcdir) if pth.endswith("jpg")]

    totalnum = len(files)
    validnum = int(totalnum*0.1)
    tsetnum  = int(totalnum * 0.1)

    valid_list    = random.sample( files, validnum)
    remaind_list  = list(set(files).difference(set(valid_list)))

    test_list = random.sample(remaind_list, tsetnum)
    train_list = list(set(remaind_list).difference(set(test_list)))

    savetxt = "/home/withai/Pictures/Anotomy/train.txt"
    with open(savetxt,"w") as f:
        for file in train_list:
            f.writelines( os.path.join("data/obj/", file)+"\n")

    savetxt = "/home/withai/Pictures/Anotomy/valid.txt"
    with open(savetxt,"w") as f:
        for file in valid_list:
            f.writelines( os.path.join("data/obj/", file)+"\n")

    savetxt = "/home/withai/Pictures/Anotomy/test.txt"
    with open(savetxt,"w") as f:
        for file in test_list:
            f.writelines( os.path.join("data/obj/", file)+"\n")


    print("end")




if __name__ == "__main__":

    # Lableme2Darknet()

    splitdataset()