import os
import torch.nn.parallel
import torch.optim
import imageio
# from utils.TRN import GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor, normalizeFunc
from model.TRN.transforms import *
import numpy as np
import tqdm
from   PIL import Image
import json
from   ctypes import *


def post_process(native_list):
    arr = (c_int * len(native_list))(*native_list)
    phase_post_process_C = CDLL(
        '/home/withai/Desktop/phasePostProcess/V0.2/post_phase_c.so')
    phase_post_process_C.runThis(arr, len(native_list))
    result_list = list(arr)

    return result_list

class Calculate:
    def __init__(self, modelName, transformer, checkPointPath, saveDir, seqlen, subfps):
        self.modelName = modelName
        self.transformer = transformer
        self.modelInit(checkPointPath)
        self.saveDir = saveDir
        if not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)
        self.seqlen = seqlen
        self.subfps = subfps

    def modelInit(self, checkPointPath):

        if not os.path.isfile(checkPointPath):
            raise FileNotFoundError("model path Error!")
        print(checkPointPath)

        self.model = torch.load(checkPointPath)
        self.model.cuda()
        self.model.eval()
        # self.input_mean = self.model.input_mean
        # self.input_std = self.model.input_std
        print("loaded!")

    def modelCalculate(self, input):
        output = self.model(input)
        return output

    def dealModelOutput(self, output):
        return torch.argmax(output).item()

    def Main(self, videopath, fpsUse=1):
        fpsUse = self.subfps
        reader = imageio.get_reader(videopath)
        videoMsg = reader.get_meta_data()
        videoFps = videoMsg['fps']
        videoDuration = videoMsg['duration']
        videoTime = round(videoFps * videoDuration)

        postResult = {}
        frameGroup = []
        for i, frame in enumerate(tqdm.tqdm(reader, desc=videopath.split('/')[-1].split('_')[0], total=videoTime)):
            secondLastFrame = int((i - 1) / (videoFps / fpsUse))
            secondNowFrame = int(i / (videoFps / fpsUse))
            if secondLastFrame < secondNowFrame:
                if len(frameGroup) < self.seqlen:
                    pass
                elif len(frameGroup) == self.seqlen:
                    frameGroup = frameGroup[1:]
                else:
                    raise ValueError('!!!')
                frameGroup.append(frame)
                if len(frameGroup) == self.seqlen:
                    inputFrames = self.dealFrame(frameGroup, self.transformer)
                    output = self.modelCalculate(inputFrames)
                    postResult[secondNowFrame] = self.dealModelOutput(output)

        operationName = videopath.split('/')[-1].split('_')[0]
        with open(f'{self.saveDir}/{operationName}.json', 'w', encoding='utf-8') as f:
            json.dump(postResult, f, ensure_ascii=False, indent=2)
            f.close()

    def dealFrame(self, frameGroup, transformer):
        if self.modelName == 'TRN':
            if self.seqlen == 24:
                frameGroup = [Image.fromarray(x).convert('RGB') for i, x in enumerate(frameGroup) if
                              i in [1, 4, 7, 10, 13, 16, 19, 22]]
            else:
                frameGroup = [Image.fromarray(x).convert('RGB') for i, x in enumerate(frameGroup) ]
            inputFrames = transformer(frameGroup)
            inputFrames = torch.unsqueeze(inputFrames, 0)
        else:
            raise NameError("modelName Error!")
        return inputFrames.cuda()


def go(videopath, checkPointPath, saveDir, seqlen, subfps):
    modelName = 'TRN'
    videopath = videopath
    checkPointPath = checkPointPath
    saveDir = saveDir
    normalize = GroupNormalize([104, 117, 128], [1])
    transformer = torchvision.transforms.Compose([
        GroupScale(int(scaleSize)),
        GroupCenterCrop(cropSize),
        Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
        ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
        normalize,
    ])

    # modelName, transformer, checkPointPath, saveDir, seqlen, subfps

    tool = Calculate(modelName, transformer, checkPointPath, saveDir, seqlen, subfps)
    tool.Main(videopath)



laelTransedict = {
    0:0,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 1,
    6: 6,
    -1:0
}

laelTransedict_reverse = {
    0:0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    1: 5,
    6: 6,
    -1:0
}

def TransformTest2Web():

    jsodir  = "/home/withai/Desktop/trn_100_2_without10_6phase_2annotator"
    savedir = "/home/withai/Desktop/trn_100_2_without10_6phase_2annotator_transform"
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    jsonlist = os.listdir(jsodir)
    jsonlist = [filename for filename in jsonlist if filename.endswith('.json') ]

    for filename in jsonlist:
        jsonpth =  os.path.join( jsodir, filename)  #"/home/withai/Desktop/trn_test/20210203-LC-HX-0033834527.json"
        with open(jsonpth) as f:
            test_result = json.load(f)

        keys = [ int(key) for key in list( test_result.keys()) ]
        keys = sorted(keys)
        videolen = keys[-1]+1

        predlist = np.zeros(videolen,np.int)
        for idx in test_result.keys():
            if test_result[idx] < 0:
                predlist[ int(idx)] = 0 #.append(0)
            else:
                predlist[ int(idx)] = laelTransedict[ test_result[idx] ]
                # predlist.append( laelTransedict[ test_result[idx] ] )
        # predlist = [ test_result[idx] for idx in  test_result.keys()  ]

        predlist = predlist.tolist()
        arr = (c_int * len(predlist))(*predlist)
        phase_post_process_C = CDLL('/home/withai/Desktop/phasePostProcess/V0.2/post_phase_c.so')
        phase_post_process_C.runThis(arr, len(predlist))
        result_list = list(arr)

        new_result  = [ laelTransedict_reverse[label] for label in result_list ]
        #json_dict  = getJSON( result_list, label_list, data)
        result_dict = { 'result':new_result }
        savepth     = os.path.join( savedir, filename)
        with open(savepth,'w') as f:
            json.dump( result_dict, f)


if __name__ == '__main__':
    TransformTest2Web()



