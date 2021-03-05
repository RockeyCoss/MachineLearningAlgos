from xml.dom.minidom import parse
import xml.dom.minidom

def loadMainConfig(name:str):
    try:
        domTree=parse("..\config\mainConfig.xml")
        config=domTree.documentElement
        result=config.getElementsByTagName(name)[0].childNodes[0].data
    except:
        result=None
    finally:
        return result

def loadConfigWithName(configName:str,name:str):
    try:
        domTree = parse(f"..\config\{configName}.xml")
        config = domTree.documentElement
        result=config.getElementsByTagName(name)[0].childNodes[0].data
    except:
        result=None
    finally:
        return result

def loadMultipleConfigWithName(configName:str,name:str):
    try:
        domTree = parse(f"..\config\{configName}.xml")
        config = domTree.documentElement
        #resultList = config.getElementsByTagName(name)[0].childNodes[0].data
        resultList=config.getElementsByTagName(name)
        result=[]
        for element in resultList:
            result.append(element.childNodes[0].data)
    except:
        result=None
    finally:
        return result

if __name__ == '__main__':
    print(loadMainConfig("modelName"))