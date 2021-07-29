import os
import json
import numpy as np
import tqdm

from tool.readXML import readXML


def dict2list(dictAnvilMsg, listPhase,fps = 1):
    listMsg = [0] * 9999
    maxTime = 0
    for phaseMsg in dictAnvilMsg['Phase.main procedures'].values():
        label = phaseMsg['name']
        if label not in listPhase:
            continue
        start, end = phaseMsg['range']
        start, end = round(start*fps), round(end*fps)
        if end > maxTime:
            maxTime = end
        listMsg[start:end + 1] = [listPhase.index(label)] * (end - start + 1)
    for phaseMsg in dictAnvilMsg['key action'].values():
        label = phaseMsg['name']
        if label not in listPhase:
            continue
        start, end = phaseMsg['range']
        start, end = round(start*fps), round(end*fps)
        if end > maxTime:
            maxTime = end
        listMsg[start:end + 1] = [listPhase.index(label)] * (end - start + 1)
    return listMsg[:maxTime]


listImgPrefix = ['20201021-LC-HX-0000345663_ORIGIN',
                 '20201028-LC-HX-0004140719_ORIGIN',
                 '20201028-LC-HX-0018412128_ORIGIN',
                 '20201029-LC-HX-0033423772_ORIGIN',
                 '20201106-LC-HX-0033451505_ORIGIN',
                 '20201111-LC-HX-0018939470_ORIGIN',
                 '20201111-LC-HX-0019647768_ORIGIN',
                 '20201112-LC-HX-0001117671_ORIGIN',
                 '20201116-LC-HX-0013576581_ORIGIN',
                 '20201117-LC-HX-0016893143_ORIGIN',
                 '20201126-LC-HX-0021104485_ORIGIN',
                 '20201204-LC-CHZH-628202_ORIGIN',
                 '20201207-LC-MY-601588526_ORIGIN',
                 '20201207-LC-MY-A00087221_ORIGIN',
                 '20201207-LC-MY-A00773514_ORIGIN',
                 '20201207-LC-MY-A01128029_ORIGIN',
                 '20201207-LC-MY-A01402158_ORIGIN',
                 '20201210-LC-HX-0011369974_ORIGIN',
                 '20201210-LC-MY-a01247183_ORIGIN',
                 '20201211-LC-ZG-1024951_ORIGIN',
                 '20201215-LC-CHZH-629285_ORIGIN',
                 '20201216-LC-CHZH-629398_ORIGIN',
                 '20201216-LC-HX-0033595725_ORIGIN',
                 '20201217-LC-HX-0032489847_ORIGIN',
                 '20201217-LC-ZG-1026928_ORIGIN',
                 '20201218-LC-CHZH-629485_ORIGIN',
                 '20201218-LC-ZG-1026910_ORIGIN',
                 '20201221-LC-HX-0033709241_ORIGIN',
                 '20201223-LC-HX-0013701278_ORIGIN',
                 '20201223-LC-HX-0033525871_ORIGIN',
                 '20201224-LC-MY-600698364_ORIGIN',
                 '20201225-LC-CHZH-630236_ORIGIN',
                 '20201225-LC-LSH-20128806_ORIGIN',
                 '20201228-LC-CHZH-629973_ORIGIN',
                 '20201228-LC-LSH-20139001_ORIGIN',
                 '20201228-LC-ZG-1027226_ORIGIN',
                 '20201230-LC-CHZH-630381_ORIGIN',
                 '20201230-LC-HX-0033301729_ORIGIN',
                 '20210105-LC-CHZH-631191_ORIGIN',
                 '20210106-LC-LSH-20131306_ORIGIN',
                 '20210108-LC-CHZH-631634_ORIGIN',
                 '20210108-LC-MY-601383906_ORIGIN',
                 '20210108-LC-MY-a01399087_ORIGIN',
                 '20210108-LC-ZG-1031890_ORIGIN',
                 '20210111-LC-LSH-20255350_ORIGIN',
                 '20210111-LC-MY-603761847_ORIGIN',
                 '20210111-LC-ZG-1032500_ORIGIN',
                 '20210113-LC-CHZH-632131_ORIGIN',
                 '20210113-LC-HX-0016710143_ORIGIN',
                 '20210113-LC-HX-0033697907_ORIGIN',
                 '20210114-LC-CHZH-632150_ORIGIN',
                 '20210114-LC-MY-600057774_ORIGIN',
                 '20210114-LC-MY-600381214_ORIGIN',
                 '20210114-LC-MY-a01823365_ORIGIN',
                 '20210114-LC-MY-A01950806_ORIGIN',
                 '20210115-LC-LSH-20213356_ORIGIN',
                 '20210115-LC-ZG-1033678_ORIGIN',
                 '20210118-LC-CHZH-632574_ORIGIN',
                 '20210118-LC-LSH-20256829_ORIGIN',
                 '20210119-LC-LSH-20257074_ORIGIN',
                 '20210125-LC-CHZH-633108_ORIGIN',
                 '20210125-LC-ZG-1035924_ORIGIN',
                 '20210127-LC-LSH-20258681_ORIGIN',
                 '20210127-LC-ZG-1036806_ORIGIN',
                 '20210128-LC-LSH-20258863_ORIGIN',
                 '20210128-LC-MY-601501744_ORIGIN',
                 '20210128-LC-MY-a01955674_ORIGIN',
                 '20210129-LC-CHZH-633576_ORIGIN',
                 '20210203-LC-CHZH-633602_ORIGIN',
                 '20210203-LC-CHZH-645321_ORIGIN',
                 '20210204-LC-MY-a01846681_ORIGIN',
                 '20210204-LC-MY-a01968054_ORIGIN',
                 '20210204-LC-ZG-1038499_ORIGIN',
                 '20210208-LC-ZG-1040005_ORIGIN',
                 '20210218-LC-ZG-1041373_ORIGIN',
                 '20210218-LC-ZG-1041403_ORIGIN',
                 '20210219-LC-ZG-1041640_ORIGIN',
                 '20210219-LC-ZG-1041764_ORIGIN',
                 '20210219-LC-ZG-1041913_ORIGIN',
                 '20210219-LC-ZG-1042633_ORIGIN',
                 '20210220-LC-LSH-20261634_ORIGIN',
                 '20210223-LC-LSH-20260570_ORIGIN',
                 '20210223-LC-LSH-20262409_ORIGIN',
                 '20210224-LC-LSH-20166541_ORIGIN',
                 '20210303-LC-LSH-20271128_ORIGIN',
                 '20210305-LC-CHZH-636336_ORIGIN',
                 '20210311-LC-LSH-20263425_ORIGIN',
                 '20210311-LC-MY-a00827719_ORIGIN',
                 '20210312-LC-ZG-1048257_ORIGIN',
                 '20210315-LC-LSH-20390026_ORIGIN',
                 '20210315-LC-LSH-20478891_ORIGIN',
                 '20210315-LC-MY-a00651637_ORIGIN',
                 '20210316-LC-LSH-20430982_ORIGIN',
                 '20210316-LC-LSH-20470028_ORIGIN',
                 '20210316-LC-LSH-20482211_ORIGIN',
                 '20210323-LC-CHZH-637335_ORIGIN',
                 '20210324-LC-ZG-1050810_ORIGIN',
                 '20210325-LC-CHZH-638144_ORIGIN',
                 '20210325-LC-CHZH-638315_ORIGIN',
                 '20210325-LC-ZG-1051667_ORIGIN']
listAnvil = ["bg",
             "Establish access",
             "Adhesion lysis",
             "Mobilize the Calot\'s triangle",
             "Dissect gallbladder from liver bed",
             "Extract the gallbladder",
             "Clear the operative region",
             "clip the cystic artery",
             "clip the cystic duct",
             "cut the cystic artery",
             "cut the cystic duct"]
listJson = ["bg",
            2,
            3,
            4,
            5,
            1,
            6,
            2,
            1,
            4,
            3
            ]
listUse = ["bg",
           "Establish access",
           "Adhesion lysis",
           "Mobilize the Calot\'s triangle",
           "Dissect gallbladder from liver bed",
           "Extract the gallbladder",
           "Clear the operative region",
           "clip the cystic artery",
           "clip the cystic duct",
           "cut the cystic artery",
           "cut the cystic duct"
           ]

dictAnvil = {}
dictJson  = {}

anvilFolder  = "/home/withai/Desktop/LCLabelFiles/LCPhase/100-2/anvil"
jsonFolder   = "/home/withai/Desktop/LCLabelFiles/LCPhase/100-2/json"
savePath     = "/home/withai/Desktop/LCLabelFiles/LCPhase{}TestV1_action_len40_8.json".format(len(listUse) - 1)
savePath_statistic= "/home/withai/Desktop/LCLabelFiles/LCPhase{}TestV1_action_len40_8_statistic.json".format(len(listUse) - 1)
extract_fps  = 8
sequence_len = 40

for anvilName in os.listdir(anvilFolder):
    op = anvilName.split(".")[0]
    if op not in dictAnvil.keys():
        dictAnvil[op] = []
    else:
        print("重复")
    anvilPath = os.path.join(anvilFolder, anvilName)
    dictMsg = readXML(anvilPath)
    listMsg = dict2list(dictMsg, listUse, extract_fps)
    dictAnvil[op].extend(listMsg)

for jsonName in os.listdir(os.path.join(jsonFolder, "phase")):
    dictMsg = {}
    for msgName in ["phase", "key action"]:
        jsonPath = os.path.join(jsonFolder, msgName, jsonName)
        with open(jsonPath, encoding="utf-8") as f:
            listTemp = json.load(f)
            f.close()
        if msgName == "phase":
            title = "Phase.main procedures"
        else:
            title = "key action"
        dictMsg[title] = {}
        for i, msg in enumerate(listTemp):
            labelID = msg["id"]
            listIndex = [j for j, x in enumerate(listJson) if x == labelID]
            if msgName == "phase":
                label = listAnvil[listIndex[0]]
            else:
                label = listAnvil[listIndex[-1]]
            start = msg["start"]
            end = msg["end"]
            dictMsg[title][str(i)] = {"range": [start, end], "name": label}

    op = jsonName.split('.')[0]
    if op not in dictJson.keys():
        dictJson[op] = []
    else:
        print("重复")
    listMsg = dict2list(dictMsg, listUse, extract_fps)
    dictJson[op].extend(listMsg)

cc=0
dictOut = {"phase": {}}
dictOut_statistic = {"phase": {}}
tvt = "test"
listOp = sorted(set(list(dictAnvil.keys()) + list(dictJson.keys())))

# video_label_dict = {}

for op in tqdm.tqdm(listOp):
    if op in dictAnvil.keys() and op in dictJson.keys():
        listMsg0 = dictAnvil[op]
        listMsg1 = dictJson[op]
        aa = np.array(listMsg0)
        # bb = np.array(listMsg1)
        listMsg = [-1] * max(len(listMsg0), len(listMsg1))
        for i in range(min(len(listMsg0), len(listMsg1))):
            if listMsg0[i] == listMsg1[i]:
                listMsg[i] = listMsg0[i]
        for i in range(min(len(listMsg0), len(listMsg1)), max(len(listMsg0), len(listMsg1))):
            if i < len(listMsg0):
                if listMsg0[i] == 0:
                    listMsg[i] = 0
            elif i < len(listMsg1):
                if listMsg1[i] == 0:
                    listMsg[i] = 0
            else:
                print("索引有问题")
    elif op in dictAnvil.keys():
        listMsg = dictAnvil[op]
    elif op in dictJson.keys():
        listMsg = dictJson[op]
    else:
        print("手术名记录有问题")
        continue

    # video_label_dict[op] = listMsg
    # continue

    imgPrefix = [x for x in listImgPrefix if op in x][0]
    if tvt not in dictOut['phase'].keys():
        dictOut['phase'][tvt] = {}
        dictOut_statistic['phase'][tvt] = {}

    for i in range(len(listMsg) - sequence_len - 1):
        label = listMsg[i + sequence_len-1]
        FindDiffirend = False
        for idx in np.arange(i,i+sequence_len):
            curlabel = listMsg[idx]
            if curlabel != label:
                FindDiffirend = True
                break
        if FindDiffirend:
            continue

        if label == -1:
            cc+=1
            continue
        if label not in dictOut['phase'][tvt]:
            dictOut['phase'][tvt][label] = []
            dictOut_statistic['phase'][tvt][label] = []
        sequence  = [ "{}/{}_{:0>5}.jpg".format(imgPrefix, imgPrefix, j + 1) for j in range(i, i + sequence_len)]
        sequence2 = [["{}/{}_{:0>5}.jpg".format(imgPrefix, imgPrefix, j + 1), listMsg[j]] for j in range(i, i + sequence_len)]
        if sequence not in dictOut['phase'][tvt][label]:
            dictOut['phase'][tvt][label].append(sequence)
            dictOut_statistic['phase'][tvt][label].append(sequence2)


# savePath_video_label = "/home/withai/Desktop/LCLabelFiles/Lc200Test_fps8_video_label.json"
# with open(savePath_video_label,'w') as f:
#     json.dump(video_label_dict,f)

print(cc)
print('write in')
with open(savePath, 'w', encoding='utf-8') as f:
    json.dump(dictOut, f, ensure_ascii=False, indent=2)
    f.close()
with open(savePath_statistic, 'w', encoding='utf-8') as f:
    json.dump(dictOut_statistic, f, ensure_ascii=False, indent=2)
    f.close()
