import scipy.misc
import os

print(dir(scipy.misc))

imgpath = '.\\testdata\\012.jpg'
#img = scipy.misc.imread(imgpath)
import matplotlib.image as mp

img = mp.imread(imgpath)

print(type(img))

#print(img)

cwd=os.getcwd()
print(cwd)

import scipy.io

scipy.io.savemat('./result/tmp_result.mat', result)