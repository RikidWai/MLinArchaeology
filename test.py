import numpy as np

mylist = []
a = np.zeros(shape=(2, 2, 3))
b = np.vstack(a)/255
print(a.shape)
print(type(a))
print(b.shape)
print(type(b))
for i in range(5):
    mylist.append(b.copy())
    print(mylist)
    print(len(mylist))
