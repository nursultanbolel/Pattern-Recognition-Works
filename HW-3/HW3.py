

from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


#convert data to vector
x_train = x_train.reshape(-1, 3072)
x_test = x_test.reshape(-1, 3072)

# In[41]:



import numpy as np

#This function returns bayes models
#which are covariance matrix (S) and mean vector (u) of each class
def getBayesModels(x_train, y_train ):

    models = {} 
    # models keeps u:mean and S: Covariance Matrix
    # models[i,0] = u
    # models[i,1] = S
    for i in range(10):
	    #class labels are stored in idx1
        idx1 = np.array(np.where(y_train==i)).T
        n = idx1.shape[0]
        x_data = np.zeros((n,3072), np.int32)
		#taking 5000 samples from each class
        x_data[0:5000] = x_train[idx1[:,0]] 
        #calculating mean vector of x_data 
        models[i,0] = np.mean(x_data,axis=0)
        #calculating covarience matrix  of x_data
        models[i,1] = np.cov(x_data.T)
        
    return models

#By using the covariance matrix (S) and mean vector (u), you are gone to copute
#the Mahalanbois distance for given sample test (x)
def getBayesProbabilities(x, models ):
#''' This function is used to calculate distance between x and cifar dataset rows using mahalanobies distance formula. 
#
#        Args:
#        x: is a test vector
#        models: includes 10 mean vectors and covariance matrices about classes in cifar dataset
#
#        Returns:
#        The return value is a vector holds distance between x and cifar data set
#'''      
    p=0
    distances = np.zeros(10, np.float32)
    
    for i in range(10):
        #u is mean vector and S is covarience matrix of class i
        u =  models[i,0]
        S = models[i,1]
        
        #mahalanobies distance=(x-u)^T.S^-1.(x-u)
        #step1 = (x-u)
        step1 = np.subtract(x,u).T
        #step2 = (x-u)^T.S^-1
        step2 = np.dot(step1,np.linalg.inv(S))
        #p holds distance value
        #p = (x-u)^T.S^-1.(x-u)
        p = np.dot(step2,step1)
        
        #distance vector holds all distance values between x and rows of cifar data set
        distances[i] = p
    #return distance vector     
    return distances

sample_test = x_test[0,:];

models = getBayesModels(x_train, y_train )
distances = getBayesProbabilities(sample_test, models )
predicted_class = np.argmin(distances)
print('distances between 10-classess and sample_test:', distances)
print('predicted_class:', predicted_class)
