

def align_list( arrylist, align_type = "left" ):

    maxlen = 0
    for idx in range(len(arrylist)):
        len1 = len(arrylist[idx])
        maxlen = max(len1,maxlen)

    new_arraylist = []
    for idx in range( len( arrylist)):
        lenth = len(arrylist[idx])
        if lenth < maxlen:
            if align_type =="left":
                arrylist[idx] = arrylist[idx] + [0]*(maxlen-lenth)
            elif align_type == "right":
                arrylist[idx] = [0] * (maxlen - lenth) + arrylist[idx]
        new_arraylist.append(arrylist)

    return new_arraylist


import random

def split_list( ori_list, ration ):

    assert 0<=ration<=1,"ration range is [0,1]"

    split_num  = int( ration*len(ori_list) )
    split_list = random.sample(ori_list,  split_num)

    # 将已经选定的测试集数据从数据集中删除
    for data in split_list:
        ori_list.remove(data)

    return ori_list,split_list