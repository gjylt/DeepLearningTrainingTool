import os



import datetime,time



class loger():
    def __init__(self, filepth="./log.txt", save=True):
        self.filepth = filepth
        self.save    = save
        if save:
            with open(filepth,"w") as f:
                print("init log")

    def log( self, logstr ):

        # 获取当前时间
        now_time = datetime.datetime.now()
        # 格式化时间字符串
        str_time = now_time.strftime("%Y-%m-%d %X")

        logstr = str_time +" -> "+logstr
        print(logstr)
        if self.save:
            with open( self.filepth,"a+") as f:
                f.write(logstr+"\n")


if __name__ == "__main__":

    pth = "/home/withai/Desktop/log.txt"
    mylog = loger(filepth=pth)
    mylog.log("1")
    mylog.log("2")
    mylog.log("2")

    # # 获取当前时间
    # now_time = datetime.datetime.now()
    # # 格式化时间字符串
    # str_time = now_time.strftime("%Y-%m-%d %X")
    # tup_time = time.strptime(str_time, "%Y-%m-%d %X")
    # time_sec = time.mktime(tup_time)
    # # 转换成时间戳 进行计算
    # time_sec += 1
    # tup_time2 = time.localtime(time_sec)
    # str_time2 = time.strftime("%Y-%m-%d %X", tup_time2)
    # print(str_time)
    # print(str_time2)

    print("end")