

from keras.datasets import cifar10

#load datasets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.reshape(-1, 3072)
x_test = x_test.reshape(-1, 3072)



x_train = x_train[1:50000, :]
y_train = y_train[1:50000, :]


import math 
import numpy as np
def cosine_similarity(v1,v2): 
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)" 
    sumxx, sumxy, sumyy = 0, 0, 0 
    for i in range(len(v1)): 
        x = v1[i] 
        y = v2[i] 
        
        x = np.float32(x)
        y = np.float32(y)
        
        sumxx += x*x 
        sumyy += y*y 
        sumxy += x*y 
#        
#        sumxx = np.float32(sumxx)
#        sumyy = np.float32(sumyy)
#        sumxy = np.float32(sumxy)
    
    return sumxy/math.sqrt(sumxx*sumyy)

def knnClassifier(x_train, y_train, sample_test, k ):
#''' This function is used to classify sample_test using KNN algorithm and cifar data set.
#Cosine similarity is used for distance.
#
#        Args:
#        x_train: is a matrix which consists cifar dataset
#        y_train: includes class information from 0 to 9
#        sample_test: is vector form of test 
#        k: is the nearest neighbor size
#
#        Returns:
#        The return value is class name (0-9)
#'''   
    dx,dy = x_train.shape
    distance = np.zeros((dx,1))
    class_count =  np.zeros((10,1))
    
    #cosine similarity between sample_test and x_train is calculated and dasitance array holds values
    for i in range (0,dx):
        distance[i,0] = cosine_similarity(sample_test,x_train[i,:])
   
    #y_train vector is added to last column of distance vector and then distance is sorted from large to small 
    distance=np.hstack((distance,y_train))
    sortedDis = distance[distance[:,0].argsort()[::-1][:dx]]

    #first k biggest distance values is counted to detect which class is more    
    for i in range(k):
        x=int(sortedDis[i,1])
        class_count[x,0] = class_count[x,0] + 1
    
    #counted class number is sorted to find class_name. max_index_col holds class name
    max_index_col = np.argmax(class_count, axis=0)
  
    #return class name
    return max_index_col

#test code
sample_test = x_test[1,:]
k=3

similar_class_name = knnClassifier(x_train, y_train, sample_test, k )
print('similar class name:', similar_class_name)

