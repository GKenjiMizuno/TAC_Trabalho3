# Training data
import numpy as np
import pandas as pd
import sys
import pickle
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("./Datasets/revised_kddcup_dataset.csv",index_col=0)



def train_udp(df, classifier=0):
    """
    Only two best classifiers have been employed on these datasets
    """
    udp_df = df[df.loc[:,"protocol_type"] == "udp"]
    
    service_values = np.unique(udp_df.loc[:,"service"])
    mid = (len(service_values)+1)/2
    for i in range(len(service_values)):
        udp_df = udp_df.replace(service_values[i], (i-mid)/10)
    
    udp_features = ["dst_bytes","service","src_bytes","dst_host_srv_count","count"]
    udp_target = "result"
    
    X = udp_df.loc[:,udp_features]
    y = udp_df.loc[:,udp_target]
    classes = np.unique(y)
    for i in range(len(classes)):
        if i == 2:
            udp_df = udp_df.replace(classes[i], 0)
        else:
            udp_df = udp_df.replace(classes[i], 1)
    
    #turning the service attribute to categorical values
    #icmp_df=icmp_df.replace("eco_i",-0.1)
    #icmp_df=icmp_df.replace("ecr_i",0.0)
    #icmp_df=icmp_df.replace("tim_i",0.1)
    #icmp_df=icmp_df.replace("urp_i",0.2)
    
    y = udp_df.loc[:,udp_target] #updating the y variables
    print("Data preprocessing done.")
    
    #choose KNN if classifier == 0 else choose Decision Tree
    if str(classifier) == "0":
        k = 3
        model = KNeighborsClassifier(n_neighbors=k)
    elif str(classifier) == "1":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
    else:
        print("Wrong model chosen! Placing default model 0 to model training!")
        k = 3
        model = KNeighborsClassifier(n_neighbors=k)
    
    
    #fitting our model
    model.fit(X,y)
    print("The model has been fit.")
    
    print("Save the fitted model?(y/n):")
    choice = input().lower()
    if choice == "y":
        pickle.dump(model, open("./modelo/udp_data.sav", 'wb'))

train_udp(df)