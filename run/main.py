from run import train,test
from utilities import loadMainConfig
if __name__ == '__main__':
    print(f"Model Name : {loadMainConfig('modelName')}")
    print("Start Training")
    train()
    print("Training Completed")
    print("Start Evaluating")
    test()
    print("Evaluating Completed")