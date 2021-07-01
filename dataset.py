import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import json
import random
import torch
import time


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
                return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
        elif self.modality == 'Flow':
            try:
                idx_skip = 1 + (idx - 1) * 5
                flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip))).convert(
                    'RGB')
            except Exception:
                print('error loading flow file:',
                      os.path.join(self.root_path, directory, self.image_tmpl.format(idx_skip)))
                flow = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')
            # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
            flow_x, flow_y, _ = flow.split()
            x_img = flow_x.convert('L')
            y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        # check the frame number is large >3:
        # usualy it is [video_id, num_frames, class_idx]
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [VideoRecord(item) for item in tmp]
        print('video number:%d' % (len(self.video_list)))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        while not os.path.exists(os.path.join(self.root_path, record.path, self.image_tmpl.format(1))):
            print(os.path.join(self.root_path, record.path, self.image_tmpl.format(1)))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)



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


if __name__ == '__main__':
    dirImage = r''
    pathJson = r"W:\outputs\sequence\V0.1\TRN.json"
    dataSetName = 'train'
    rule = 'cvsi'
    dataSet = customCVSDataSet(dirImage, pathJson, dataSetName, rule, '1', 8)
    dataSet.__getitem__(1)
    print(dataSet.listSequences[:10])
    print(dataSet.listLabels)
