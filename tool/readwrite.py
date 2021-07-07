import xlrd
import json
from   tool.readXML import readXML
import os

def read_xls_rows(pth):
    workbook = xlrd.open_workbook(pth)
    sheet1_object = workbook.sheet_by_index(0)
    nrows    = sheet1_object.nrows

    row_list = []
    for idx in range(nrows):
        row_list.append( sheet1_object.row_values(rowx=idx) )

    return row_list

def anvil2list(anvilPath, listPhase,fps=1):
    dictAnvilMsg = readXML(anvilPath)
    listMsg = [0] * 9999
    maxTime = 0
    for phaseMsg in dictAnvilMsg['Phase.main procedures'].values():
        label = phaseMsg['name']
        if label not in listPhase:
            # print(f"jump {label}")
            continue
        start, end = phaseMsg['range']
        start, end = round(start*fps), round(end*fps)
        if end > maxTime:
            maxTime = end
        listMsg[start:end + 1] = [listPhase.index(label)] * (end - start + 1)
    # for phaseMsg in dictAnvilMsg['key action'].values():
    #     label = phaseMsg['name']
    #     if label not in listPhase:
    #         # print(f"jump {label}")
    #         continue
    #     start, end = phaseMsg['range']
    #     start, end = round(start*fps), round(end*fps)
    #     if end > maxTime:
    #         maxTime = end
    #     listMsg[start:end + 1] = [listPhase.index(label)] * (end - start + 1)
    return listMsg[:maxTime + 1]


def traversalDir(dir1, returnX='path', platform = "linux"):
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

if __name__ == "__main__":



    print("end")