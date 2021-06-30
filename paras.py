from tool.log import loger

class paras():


    def __init__(self):
        self.modelresume = "/home/withai/Desktop/trn_test/LC6Best_20210507.tar"
        self.logfilepth  = ""
        self.modelname   = "TRN"
        self.num_class   = 4
        self.usegpu      = False

        if self.logfilepth != "":
            self.log = loger( filepth= self.logfilepth )
        else:
            self.log = loger()
        print("init paras")

