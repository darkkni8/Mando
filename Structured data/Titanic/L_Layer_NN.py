#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


def init_params(layer_dims):
    parameters={}
    L=len(layer_dims)
    for i in range(1,L):
        parameters['W'+str(i)]=np.random.randn(layer_dims[i],layer_dims[1-1])*0.01
        parameters['b'+str(i)]=np.zeros((layer_dims[i],1))
        #assert(parameters['W' + str(i)].shape == (layer_dims[i], layer_dims[i-1]))
        #assert(parameters['b' + str(i)].shape == (layer_dims[i], 1))
    return parameters


# In[3]:


def liner_forward(A,W,b):
    Z=np.dot(W,A)+b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache


# In[4]:


def sigmoid(Z):
    A= 1/(1+np.exp(-Z))
    cache= Z
    return A, cache


# In[5]:


def relu(Z):
    A= Z*(Z>0)
    cache= Z
    return A, cache


# In[6]:


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache= linear_forward(A_prev, W, b)
    if activation== "sigmoid":
        A, activation_cache= sigmoid(Z)
    elif activation== "relu":
        A, activation_cache= relu(Z)
    assert(A.shape== (W.shape[0], A_prev.shape[1]))
    cache= (linear_cache, activation_cache)
    return A, cache


# In[7]:


def L_model_forward(X, parameters):
    caches=[]
    A=X
    L=len(parameters)//2
    for i in range(1,L):
        A_prev= A
        A, cache= linear_activation_forward(A_prev, parameters('W'+str(i)), parameters('b'+str(i)), activation="relu")
        caches.append(cache)
    AL,cache= linear_activation_forward(A, parameters('W'+str(L)), parameters('b'+str(L)), activation="sigmoid")
    caches.append(cache)
    assert(AL.shape==(1,X.shape[1]))
    return AL, caches


# In[8]:


def compute_cost(AL, Y):
    m=Y.shape[1]
    cost = -sum(sum(np.multiply(Y,np.log(AL))+np.multiply(1-Y,np.log(1-AL))))/m
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost


# In[9]:


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ,A_prev.T)/m  
    db = np.sum(dZ, axis=1, keepdims=True )/m
    dA_prev = np.dot(W.T,dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db


# In[10]:


def sigmoid_derivative(Z):
    f, g= sigmoid(Z)
    return f*(1-f)


# In[11]:


def relu_derivative(Z):
    Z[Z<=0] = 0
    Z[Z>0] = 1
    return Z


# In[12]:


def sigmoid_backward(dA, activation_cache):
    Z= activation_cache
    return dA*sigmoid_derivative(Z)


# In[13]:


def relu_backward(dA, activation_cache):
    Z= activation_cache
    return dA*relu_derivative(Z)


# In[14]:


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###
        
    elif activation == "sigmoid":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        ### END CODE HERE ###
    
    return dA_prev, dW, db


# In[15]:


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 
    
    dAL =  -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))   
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation= "sigmoid")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation= "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


# In[16]:


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 
    for l in range(L):
        parameters["W" + str(l+1)] =  parameters["W" + str(l+1)] -learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] -learning_rate*grads["db" + str(l+1)]
    return parameters


# In[ ]:





# In[ ]:





# In[17]:


#import pandas as pd
#import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


# In[18]:


data= 'titanic_train.csv'
test_data="titanic_test.csv"


# In[19]:


X= pd.read_csv(data, index_col='PassengerId')
X_test=pd.read_csv(test_data, index_col="PassengerId")
Y= X.Survived
X=X.drop(['Survived'], axis=1)
print(X.head())


# In[20]:


# Get names of columns with missing values
cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
print("cols with missing: ",cols_with_missing)
object_cols = [col for col in X.columns if X[col].dtype == "object"]
print("object cols: ",object_cols)
object_nunique = list(map(lambda col: X[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))
print("no. of unique entries: ", d)
low_cardinality_cols = [col for col in object_cols if X[col].nunique() < 10]
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))
print("low_cardinality_cols: ",low_cardinality_cols)
print("high_cardinality_cols: ", high_cardinality_cols)
numerical_cols= list(set(X.columns)-set(object_cols))
print("numerical_cols", numerical_cols)


# In[21]:


X= X.drop(high_cardinality_cols, axis=1)
X_test= X_test.drop(high_cardinality_cols, axis=1)


# In[22]:


numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[ ('imputer', SimpleImputer(strategy='most_frequent')),
                                           ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols),
                                                 ('cat', categorical_transformer, low_cardinality_cols)])


# In[23]:


X_transformed= preprocessor.fit_transform(X)
X_transformed= pd.DataFrame(StandardScaler().fit_transform(X_transformed))
X_transformed.index= X.index
print(X)
print(X_transformed)


# In[24]:


X_test_trans= preprocessor.fit_transform(X_test)
X_test_trans= pd.DataFrame(StandardScaler().fit_transform(X_test_trans))
X_test_trans.index= X_test.index
print(X_test)
print(X_test_trans)


# In[25]:


X_train, X_val, Y_train, Y_val= train_test_split(X_transformed, Y, test_size=0.3, random_state= 1)
X_train=X_train.T
X_val= X_val.T
train_indices=Y_train.index
val_indices=Y_val.index
Y_train= np.array([Y_train])
Y_val= np.array([Y_val])
X_test= X_test.T


# In[26]:


### CONSTANTS ###
layers_dims = [10, 6, 4, 1] 


# In[27]:


def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    np.random.seed(1)
    costs = []                      
    parameters = init_params(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters


# In[28]:


parameters = L_layer_model(X_train, Y_train, layers_dims, num_iterations = 2500, print_cost = True)


# In[ ]:





# In[ ]:




