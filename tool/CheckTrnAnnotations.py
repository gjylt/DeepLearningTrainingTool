import os
import json
import copy

def CheckFileExist():
    old_pth = "/home/withai/Desktop/LCLabelFiles/LCPhase_version1_len8_2_annotator.json"
    imgdir = r"/home/withai/Pictures/LCFrame/100-1-2-8fps"
    save_pth = r"/home/withai/Desktop/LCLabelFiles/LCPhase_version1_len8_2_annotator_checked.json"
    with open(old_pth) as f:
        old_dict = json.load(f)

    new_dict = {}
    new_dict['phase'] = {}

    dir_list = os.listdir(imgdir)

    error_dir_list  = []
    error_file_list = []
    num = 0
    for phase in old_dict['phase'].keys():

        if phase not in new_dict['phase'].keys():
            new_dict['phase'][phase] = {}

            for labelid in old_dict['phase'][phase].keys():

                if labelid not in new_dict['phase'][phase].keys():
                    new_dict['phase'][phase][labelid] = []

                for sequnce in old_dict['phase'][phase][labelid]:

                    videoname = sequnce[0].replace('\\','/').split('/')[0]
                    if videoname not in dir_list :
                        if videoname not in error_dir_list:
                            error_dir_list.append(videoname)
                            print(videoname,'dir not exist')
                        continue

                    not_exist = False
                    new_sequnce = []
                    for imgpthid in range(len(sequnce)):
                        imgpth = sequnce[imgpthid]
                        imgpth = imgpth.replace('\\','/')
                        new_sequnce.append(imgpth)
                        filepth = os.path.join(imgdir, imgpth)
                        if not os.path.exists(filepth):
                            not_exist = True
                            error_file_list.append(filepth)
                            print(filepth,"file not found")
                            break

                    if not not_exist:
                        #print(new_sequnce)
                        new_dict['phase'][phase][labelid].append(new_sequnce)
                    else:
                        num += 1
    print( "not exist sequnce=",num )
    with open(save_pth, 'w') as f:
        json.dump(new_dict, f)


def StatisticData():

    dir_pth = r"G:\DataSet\LCPhase\20-8-test"
    dirlist = os.listdir(dir_pth)

    old_pth = r"G:\DataSet\LCPhase\LCPhaseTest200V1_action_len8_fps8_new2.json"
    with open(old_pth) as f:
        old_dict = json.load(f)

    new_dict = {}
    new_dict['phase'] = {}

    for phase in old_dict['phase'].keys():

        if phase not in new_dict['phase'].keys():
            new_dict['phase'][phase] = []

            for labelid in old_dict['phase'][phase].keys():

                for sequnce in old_dict['phase'][phase][labelid]:

                    videoname = sequnce[0].split('/')[0]
                    if videoname not in new_dict['phase'][phase]:
                        new_dict['phase'][phase].append( videoname )


    for phase in new_dict['phase'].keys():

        for videoname in new_dict['phase'][phase]:
            if videoname not in dirlist:
                print(phase,videoname,"not exist")

        # print(phase, ':', len( new_dict['phase'][phase] ))

def modify_error():

    old_pth = r"G:\DataSet\LCPhase\LCPhaseTest200V1_action_len8_fps8.json"
    save_pth = r"G:\DataSet\LCPhase\LCPhaseTest200V1_action_len8_fps8_new2.json"
    with open(old_pth) as f:
        old_dict = json.load(f)


    dir_pth = r"G:\DataSet\LCPhase\20-8-test"
    dirlist = os.listdir(dir_pth)

    new_dict = {}
    new_dict['phase'] = {}

    video_list = []
    for phase in old_dict['phase'].keys():

        if phase not in new_dict['phase'].keys():
            new_dict['phase'][phase] = {}

            for labelid in old_dict['phase'][phase].keys():

                if labelid not in new_dict['phase'][phase].keys():

                    new_dict['phase'][phase][labelid] = []

                tmp = []
                for imgpth in old_dict['phase'][phase][labelid]:
                    videoname = imgpth.split('/')[0]
                    if videoname not in video_list:
                        video_list.append(videoname)


                    tmp.append(imgpth)


                    if len(tmp) == 8:

                        new_dict['phase'][phase][labelid].append( copy.deepcopy(tmp))
                        tmp = []

    with open(save_pth,'w') as f:
        json.dump( new_dict, f)



if __name__ == "__main__":

    CheckFileExist()

    # StatisticData()
    # modify_error()
    print("end")
