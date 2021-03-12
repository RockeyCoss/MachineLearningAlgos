import numpy as np

a=np.array([[1,2],[3,4],[5,6]])
print(a)
b=np.array([3,3,3])
c=np.insert(a,a.shape[1],b,axis=1)
print(b)
print(c)