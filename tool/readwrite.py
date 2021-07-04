import xlrd

def read_xls_rows(pth):
    workbook = xlrd.open_workbook(pth)
    sheet1_object = workbook.sheet_by_index(0)
    nrows = sheet1_object.nrows

    row_list = []
    for idx in range(nrows):
        row_list.append( sheet1_object.row_values(rowx=idx))

    return row_list


import json
if __name__ == "__main__":

    pth = "/home/withai/Desktop/LCLabelFiles/videopth_info.json"
    with open(pth) as f:
        data = json.load(f)

    videolist = read_xls_rows("/home/withai/Desktop/100-3.xls")
    read_dict = {}
    for videoname in videolist:
        videoname = videoname[0]
        for videoname1 in data.keys():

            if videoname1.__contains__(videoname) or videoname.__contains__(videoname1):
                read_dict[videoname] = data[videoname1]
                break

    print("end")