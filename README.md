# Implementation of Some ML Algorithms 
Implementation of the algorithms in the book *Statistical Learning Methods*  

《统计学习方法》python实现

Use the MNIST dataset to validate the algorithms.

>  This repository is still in the process of updating.

----

### Currently Implemented:

* [Perceptron](https://github.com/RockeyCoss/machineLearningImplementation/blob/main/models/Perceptron.py)

* [Naive Bayesian](https://github.com/RockeyCoss/machineLearningImplementation/blob/main/models/NaiveBayesian.py)

* [KNN](https://github.com/RockeyCoss/machineLearningImplementation/blob/main/models/KNN.py)

  > kd-tree is used to implement KNN

* [Logistic](https://github.com/RockeyCoss/machineLearningImplementation/blob/main/models/Logistic.py)

  > Use either SGD or quasi-Newton to optimize the target function.
  >
  > When using quasi-Newton, some numerical problems may be encountered. This bug has not been fixed yet. （懒得修了，以后再说吧md）

* [Maximum Entropy](https://github.com/RockeyCoss/machineLearningImplementation/blob/main/models/MaximumEntropy.py)

  > Use quasi-Newton to optimize the target function.
  >
  > The derivative of the likelihood function is different from that written in *Statistical Learning Methods* . For MNIST dataset, the derivative is:
  >
  > <img src="E:\学习笔记\machine_learning\machineLearningFrame\README.assets\derivative-1615695073673.png" alt="derivative" style="zoom:40%;" />

  

