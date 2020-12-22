# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:16:54 2020

@author: DELL

"""

import numpy as np
from sklearn.datasets import load_iris
from scipy.spatial import distance
import pandas as pd

"""def eulicen_distance(dim,djm):
    return np.linalg.norm(dim - djm, axis=1)"""

def is_unique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return (a[0] == a).all()

def merge_dataset(D,i,j):
    cluster = np.vstack((D[i], D[j])) #Merging to one array
    D = np.vstack(cluster) # adding to dataset
    return D
    
# Dataset will be stored in a numpy array

def OPE_HCA(dataset,param1):
    # param1 is theeta
    # Eucliden distance based merging
    target = np.arange(0,150,1)
    i_values = []
    j_values= []
    for i in range(len(dataset)):
        if( ( i in i_values) == False):
            j=0
            while(j<len(dataset)):
                if(j != i ):
                    if( ( j in j_values) == False and ( i in i_values) == False):
                        distance_ =distance.euclidean(dataset[i,],dataset[j,])
                        if(distance_<= param1):
                            target[j] = target[i]
                            i_values = np.append(i_values,i)
                            j_values = np.append(j_values, j)
                            
                j=j+1
    return dataset,target   
    
def OPE(D,minMd):
    occurence_probability_pi = []
    target = D['label'].to_numpy()
    
    # number of clusters 
    sum_ = []
    label_unique = np.unique(target)
    
    #################### ESTIMATION ###########################
    
    
    # Finding sum of the same cluster
    for i in range(len(label_unique)):
        c = (D.loc[D.label == target[i],["column1", "column2","column3", "column4"]].values)
        sum_ = np.append(sum_,np.sum(c))   
    
    occurence_probability_pi = sum_/np.sum(sum_)
    
    # creating a dataframe with labels and occurence probability
    df_label_probability = pd.DataFrame({'label':label_unique, 'occurence_probability':occurence_probability_pi})
    
    # Lets sort above dataframe using occurence probability in decending
    df_label_probability = df_label_probability.sort_values(by='occurence_probability',ascending=False)  
    
    ################################ SAMPLING ################################
    # Recurssion condition 
    condition_1 = is_unique(df_label_probability['occurence_probability'])
    condition_2 = is_unique(df_label_probability['label'])
    if (condition_1  or condition_2  ):
        return D
    else:
        for i in range(len(df_label_probability)):
            j = 0
            while(j< len(df_label_probability)):                
                if (df_label_probability.occurence_probability[i]> df_label_probability.occurence_probability[j] ):
                    c_i = (D.loc[D.label == df_label_probability.label[i],["column1", "column2","column3", "column4"]].values)
                    c_j = (D.loc[D.label == df_label_probability.label[j],["column1", "column2","column3", "column4"]].values)
                    mean_value = np.mean(c_i) - np.mean(c_j)
                    
                    if (mean_value < 0):
                        mean_value = mean_value *-1
                       
                    if(mean_value <= minMd ):
                        # then change the cluster of the dataset
                        D['label'] = D['label'].replace(df_label_probability.label[j],df_label_probability.label[i])
                        return(OPE(D,minMd))  
                j = j+1
        
    
def main():
    # import iris data 
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html
    iris = load_iris()
    df = iris.data
    D,t= OPE_HCA(df,0.6)
    print("View 1 ")
    print(t)
    # converting to a datafrmae
    df = pd.DataFrame(data=D,columns=["column1", "column2","column3", "column4"])
    # adding label column
    df['label'] = t
    D = OPE(df,2.2)
    print("Clustered dataset")
    print("************")
    print(D)
    print("Final View:")
    print(D['label'].unique())
  
if __name__ == "__main__":
    main()
                
    
            
                     
    
        
