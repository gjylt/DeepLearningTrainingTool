from tool.log import loger

class paras():


    def __init__(self):
        self.modelresume = "/home/withai/Desktop/trn_test/LC6Best_20210507.tar"
        self.logfilepth  = ""
        self.modelname   = "TRN"
        self.num_class   = 4
        self.arch        = "BNInception"
        self.usegpu      = False
        self.dataloader_dict  = {}
        self.optimizer   = None
        self.model       = None
        self.criterion_dict = {}
        self.dictCategories ={
            'phase': ['0', '1', '2', '3', '4']
        }
        self.workers     = 8
        self.batch_size  = 8
        self.epoch       = 32
        self.validstep   = 5
        self.epoch_type  = "train" # train or valid or test

        if self.logfilepth != "":
            self.log = loger( filepth = self.logfilepth )
        else:
            self.log = loger()
        print("init paras")

