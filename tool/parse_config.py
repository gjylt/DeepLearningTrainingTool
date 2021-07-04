import os


def get_config( findname):

    curdir = os.getcwd()
    print( curdir )

    if curdir.__contains__("DeepLearningProject"):
        parentdir = curdir.split("DeepLearningProject")[0]
    else:
        print("can't find config.txt")
        return None

    parentdir = os.path.join( parentdir, "DeepLearningProject")
    os.chdir(parentdir)

    print(os.getcwd())
    if not os.path.exists("./config.txt"):
        print("can't find config.txt")

    with open("./config.txt") as f:
        configdatas = f.readlines()

    for linedata in configdatas:
        splits = linedata.split("=")
        name = splits[0].replace(" ","").replace("\n","").replace("\'","").replace("\"","")
        value= splits[1].replace(" ","").replace("\n","").replace("\'","").replace("\"","")

        if name == findname:
            return value
