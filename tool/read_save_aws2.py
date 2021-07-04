from boto3 import Session
import os


class AwsData():
    def __init__(self,access_key,secret_key,region, bucketname):
        self.access_key = access_key
        self.secret_key = secret_key
        self.region     = region
        self.s3         = self.init_seesion()
        self.bucketname = bucketname

    def init_seesion(self):
        session = Session(
            aws_access_key_id       =  self.access_key,
            aws_secret_access_key   =  self.secret_key,
            region_name             =  self.region
        )
        s3  = session.resource('s3')

        return s3


    def readwrite_file(self, localpth, s3pth, mode="w" ):

        obj = self.s3.Object( self.bucketname, s3pth )

        if mode == "r":
            obj.download_file(localpth)
        elif mode == "w":
            obj.upload_file(localpth)
        elif mode == "d":
            obj.delete()
        else:
            print(mode,"is not a correct para,mode should be r or w!")

    def list_file(self, s3dir, tag="" ):
        bucket = self.s3.Bucket(self.bucketname)
        # object_summary_iterator = bucket.objects.all()
        object_summary_iterator = bucket.objects.filter(
            # Delimiter='string',
            #     EncodingType='url',
            #     Marker='string',
            #     MaxKeys=123,
            Prefix=s3dir,
            # RequestPayer='requester'
        )

        filelist = []
        for ite in object_summary_iterator:

            if tag != "" :
                if ite.key.__contains__(tag):
                    filelist.append(ite)
            else:
                filelist.append(ite)

        return filelist


    def copy_file(self,old_path,new_path):

        self.s3.Object(self.bucketname, new_path).copy_from(CopySource=self.bucketname + "/" + old_path)
        self.s3.Object(self.bucketname, old_path).delete()


    def chech_file_exist(self, s3pth):

        localpth = "./chech_file_exist.txt"
        mode     = "r"
        try:
            self.readwrite_file( localpth, s3pth, mode)
        except:
            return False

        try:
            os.remove(localpth)
        except:
            return False

        return True





    def mkdir(self, s3dir):
        localpth    = "./tmp.txt"
        mode        = "w"
        fp          = open("./tmp.txt","w")
        fp.close()
        s3path      = os.path.join(s3dir,"tmp.txt")
        self.readwrite_file( localpth, s3path, mode)

    def LoadDirsFromS3(self,s3dir,localdir,overwrite = False):

        s3dir = os.path.join(s3dir,'')

        tag   = ""
        filelist = self.list_file(s3dir, tag)

        if not os.path.exists(localdir):
            os.makedirs(localdir)

        idx = 0
        for fi in filelist:

            s3path = fi.key
            s3tmpth = s3path.replace(s3dir,"")
            # if s3tmpth == "/":
            #     s3tmpth = ''
            localpth = os.path.join(localdir,s3tmpth)

            localdirtmp = os.path.split(localpth)
            if not os.path.exists(localdirtmp[0]):
                os.makedirs(localdirtmp[0])

            if os.path.exists(localpth) and not overwrite:
                continue

            if localdirtmp[1] != '':
                self.readwrite_file(localpth, s3path, mode="r")

            idx = idx + 1
            print(idx,"/",len(filelist) )

    def dirlist(self, path, tags=[], TfRecusive=False):
        filelist = os.listdir(path)
        result = []
        for filename in filelist:

            if filename.startswith('.'):
                continue

            if filename.startswith('__pycache__'):
                continue

            filepath = os.path.join(path, filename)
            if os.path.isdir(filepath):

                if TfRecusive == False:
                    continue

                tmp    = self.dirlist(filepath, tags, TfRecusive)
                result = result + tmp
            else:
                if len(tags) > 0:
                    for tag in tags:
                        if os.path.splitext(filename)[1] == tag:
                            result.append( os.path.join(path,filename) )
                else:
                    result.append( os.path.join(path,filename) )

        return result





import shutil

from tool.parse_config import get_config

if __name__ == "__main__":



    access_key  = get_config("access_key")
    secret_key  = get_config("secret_key")
    region      = get_config("region")
    bucketname  = get_config("bucketname")
    s3dir       = get_config("s3dir")
    rootdir     = get_config("rootdir")

    awsdata     = AwsData(access_key,secret_key,region, bucketname)

    dirlist     = [ "merge"]


    localdir    = ["train_val_8_2",
                   "train_val_8_1"]
    dirindx = 0
    for local in localdir:
        dirindx += 1
        destdir = os.path.join(rootdir,local)
        dirlist = awsdata.dirlist(destdir,TfRecusive= True)

        # print("dir:",dirindx)
        numidx = 0
        for pth in dirlist:
            numidx += 1
            print( dirindx,numidx, "/",len(dirlist) )
            subpth = pth.split(rootdir)[-1]
            localpth = pth
            s3pth    = os.path.join(s3dir,subpth)
            try:
                awsdata.readwrite_file(localpth,s3pth,'w')
            except:
                print(localpth,s3pth)


    # if not os.path.exists(localdir):
    #     os.makedirs(localdir)
    #
    # try:
    #     filist = awsdata.LoadDirsFromS3(s3dir, localdir)
    # except:
    #     print("list failed!")

    # for dirpth in dirlist:
    #
    #     s3dir = os.path.join( "腹腔镜胆囊/CSR", dirpth )
    #     localdir1 = os.path.join( localdir, dirpth)
    #
    #     #打印指定目录视频文件列表
    #     # filelsit = awsdata.list_file(s3dir)
    #     # for fi in filelsit:
    #     #     print(fi.key)
    #
    #     # localdir   = os.path.join(savedir, s3dir.split("/")[-1] )
    #     if not os.path.exists(localdir1):
    #         os.makedirs(localdir1)
    #
    #     try:
    #         filist = awsdata.LoadDirsFromS3( s3dir,localdir1 )
    #     except:
    #         print("list failed!")

    print("end")
