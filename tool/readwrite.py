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



    print("end")