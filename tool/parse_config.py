import os


def get_config( findname, config_path = ""):

    if config_path == "":
        curdir = os.getcwd()
        print( curdir )

        if curdir.__contains__("DeepLearningProject"):
            parentdir = curdir.split("DeepLearningProject")[0]
        else:
            print("can't find config.txt")
            return None

        parentdir = os.path.join( parentdir, "DeepLearningProject")
        os.chdir(parentdir)

        config_path = "./config.txt"

        print(os.getcwd())


    if not os.path.exists( config_path ):
        print("can't find config.txt")
        return None

    with open( config_path ) as f:
        configdatas = f.readlines()

    for linedata in configdatas:
        splits = linedata.split("=")
        name   = splits[0].replace(" ","").replace("\n","").replace("\'","").replace("\"","")
        value  = splits[1].replace(" ","").replace("\n","").replace("\'","").replace("\"","")

        if name == findname:
            return value
