

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