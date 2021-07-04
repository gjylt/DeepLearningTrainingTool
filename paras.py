from tool.log import loger
from tool.parse_config import get_config

class paras():


    def __init__(self):
        self.modelresume = "/home/withai/Desktop/LCLabelFiles/action_len24_fps8_pg_bg.csv_LC10_best.pth.tar"
        self.dataset_dir  = '/home/withai/Pictures/LCFrame/LCTrainVal_189'
        self.labelfile   = "/home/withai/Desktop/LCLabelFiles/LCPhaseTrainValid200V1_action_len8_fps8_new_check2.json"
        self.logfilepth  = "./log.txt"
        self.modelname   = "TRN"
        self.interface   = False
        self.dictCategories = ['0', '1', '2', '3', '4']
        self.num_class   = len( self.dictCategories )
        self.arch        = "BNInception"
        self.usegpu      = False
        self.dataloader_dict  = {
            "train":None,
            "valid":None,
            "test":None
        }
        self.optimizer   = None
        self.model       = None
        self.criterion   = None

        self.workers     = 8
        self.batch_size  = 2
        self.epoch       = 2
        self.validstep   = 2
        self.epoch_type  = "train" # train or valid or test
        self.iepoch      = 0
        self.learning_rate_policy = {}
        self.learning_rate = 1e-3
        self.momentum    =  0.9
        self.weight_decay= 5e-4

        if self.logfilepth != "":
            self.log = loger( filepth = self.logfilepth )
        else:
            self.log = loger()

        self.savedir    = ""
        self.store_name = ""
        print("init paras")

