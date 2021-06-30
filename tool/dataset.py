import torch.utils.data as data
import json
import os
import torch
import numpy as np
import random
from PIL import Image

class customCVSDataSet(data.Dataset):
    def __init__(self, dirImage, pathJson, dataSetName, rule, transform, numSegments, categories):
        self.dirImage = dirImage
        self.transform = transform
        self.numSegments = numSegments
        self.dataSetName = dataSetName
        self.categories = categories
        self.rule       = rule
        with open(pathJson, encoding='utf-8') as f:
            dictMsg = json.load(f)
            f.close()

        video_list = []
        new_dict = {}
        for phase in dictMsg['phase'].keys():

            for labelid in dictMsg['phase'][phase].keys():

                for sequnceid in range( len( dictMsg['phase'][phase][labelid])):

                    sequnce = dictMsg['phase'][phase][labelid][sequnceid]
                    for imgpthid in range(len(sequnce)):
                        sequnce[imgpthid] = sequnce[imgpthid].replace('\\','/')
                    dictMsg['phase'][phase][labelid][sequnceid] = sequnce

                    videoname = sequnce[0].split('/')[0]
                    if videoname not in video_list:
                        video_list.append(videoname)


        labels = categories #sorted(list(dictMsg[rule][dataSetName].keys()))
        self.listSequences = []
        self.listLabels = []
        self.listdir    = os.listdir(dirImage)

        pg_totalnum = 0
        for label in labels:
            pg_totalnum += len(dictMsg[rule][dataSetName][label])
            for sequnce in dictMsg[rule][dataSetName][label]:
                videoname = sequnce[0].split('/')[0]
                # print(videoname)
                if videoname not in self.listdir:
                    continue
                self.listSequences.append( sequnce )
                self.listLabels.append(label)

        self.label_map_dic={
            '0':0,
            '7': 1,
            '8': 2,
            '9': 3,
            '10': 4,
        }

        self.balance_bg_pg_fortest( dictMsg, pg_totalnum)


    def balance_bg_pg_fortest(self,dictMsg, pg_totalnum):
        appendlabel = np.arange(0, 7).tolist()
        for label in appendlabel:
            label    = str(label)
            need_num = len(dictMsg[self.rule][self.dataSetName][label])
            for sequnce in dictMsg[self.rule][self.dataSetName][label]:
                videoname = sequnce[0].split('/')[0]
                # print(videoname)
                if videoname not in self.listdir:
                    continue
                self.listSequences.append( sequnce )
                self.listLabels.append('0')

    def balance_bg_pg_augmentpg(self,dictMsg, pg_totalnum):
        appendlabel     = np.arange(0,7).tolist()
        appenddata_dict = {}
        bg_totalnum     = 0
        for label in appendlabel:
            label    = str(label)
            labellen = len(dictMsg[self.rule][self.dataSetName][label])
            appenddata_dict[label] = labellen
            bg_totalnum += labellen

        for label in appendlabel:
            label    = str(label)
            need_num = len(dictMsg[self.rule][self.dataSetName][label])
            self.listSequences.extend(dictMsg[self.rule][self.dataSetName][label])
            self.listLabels.extend(['0', ] * need_num)

        appendlabel = np.arange(7, 11).tolist()
        for label in appendlabel:
            label = str(label)
            multi_times  = bg_totalnum*1.0/pg_totalnum
            multi_times1 = int(np.floor(multi_times))
            tmp = dictMsg[self.rule][self.dataSetName][label] * multi_times1
            self.listSequences.extend( tmp )
            self.listLabels.extend([label, ] * len(tmp))

            # print("")

    def balance_bg_pg(self,dictMsg, pg_totalnum):
        appendlabel     = np.arange(0,7).tolist()
        appenddata_dict = {}
        bg_totalnum        = 0
        for label in appendlabel:
            label    = str(label)
            labellen = len(dictMsg[self.rule][self.dataSetName][label])
            appenddata_dict[label] = labellen
            bg_totalnum += labellen

        for label in appendlabel:
            label    = str(label)
            need_num = int(pg_totalnum*1.0*appenddata_dict[label]/bg_totalnum)
            if need_num > appenddata_dict[label]:
                need_num = appenddata_dict[label]
                self.listSequences.extend(dictMsg[self.rule][self.dataSetName][label])
                self.listLabels.extend(['0', ] * need_num )
            else:
                choied_data = random.sample(dictMsg[self.rule][self.dataSetName][label], need_num)
                self.listSequences.extend( choied_data )
                self.listLabels.extend(['0', ] * need_num)


    def randomShift(self, sequenceLen=8):
        interval = int(sequenceLen // self.numSegments)
        listInd = list(range(interval // 2, sequenceLen, interval))
        if self.dataSetName == 'train':
            useInd = [random.choice(list(range(-1 * (interval // 2), interval // 2 + 1))) + x for x in listInd]
        else:
            useInd = listInd
        return useInd

    def __len__(self):
        return len(self.listSequences)

    def __getitem__(self, index):
        sequence = self.listSequences[index]
        label = int(self.listLabels[index])
        choiseInd = self.randomShift()
        lastImage = sequence[-1]
        sequence = [sequence[x] for x in choiseInd]
        listImg = []
        for imgName in sequence:
            img = Image.open(os.path.join(self.dirImage, imgName)).convert('RGB')
            listImg.append(img)
        processData = self.transform(listImg)

        if self.dataSetName == 'test':
            return processData, torch.tensor( self.label_map_dic[str(label)] ), lastImage
        else:
            return processData, torch.tensor( self.label_map_dic[str(label)] ),lastImage