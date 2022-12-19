import numpy as np

pth = '/home/kun/'
f = open(pth + 'k.txt', 'w')
k = np.float64(np.concatenate((np.arange(1e-5,3501e-5,1e-5),np.arange(3.5e-2,0.19999,1e-4),np.arange(0.21,2.21,1e-2),np.arange(2.2,10.000001,0.2),np.arange(10.0,2990.1,20.0))))
for kk in k:
    print(kk, file=f)

f.close()