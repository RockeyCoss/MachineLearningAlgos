from run import train,test
import time
from utilities import loadMainConfig
if __name__ == '__main__':
        start=time.time()
        print(f"Model Name : {loadMainConfig('modelName')}")
        print("Start Training")
        train()
        print("Training Completed")
        print("Start Evaluating")
        test()
        print("Evaluating Completed")
        print(f"time spent: {time.time()-start}")
