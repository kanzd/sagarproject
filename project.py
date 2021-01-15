#=======================================DATA LOADING=====================================================
#LOAD ALL THE PYTHON LIBRARIES
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb
import tkinter as tk
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import subprocess

pip_install = "pip install "
subprocess.call(pip_install+"numpy")
subprocess.call(pip_install+"pandas")
subprocess.call(pip_install+"matplotlib")
subprocess.call(pip_install+"seaborn")
subprocess.call(pip_install+"tkinter")
subprocess.call(pip_install+"sklearn")


try:
    Wholesale=pd.read_csv("wholesale customer data.csv")
    print("==============================ORIGINAL DATASET================================")
    
    print("===========================NEW TRANSFORMED DATASET============================")
    Wholesale_df=pd.DataFrame(Wholesale)
    data=pd.DataFrame(Wholesale)
    
    Wholesale_df.drop(["Region","Fresh","Frozen","Delicassen"],axis=1,inplace=True)
    data.drop(["Channel","Region"],axis=1,inplace=True)
   
    print(data)
    
    
except:
    print("No Dataset loaded here. Please Try again with another dataset!")
try:
        #===================================DATA EXPLORING=======================================================
        #========DATA DESCRIPTION==================
        #FRESH: Annual spending on Fresh products.
        #MILK: Annual spending on Milk products.
        #GROCERY: Annual spending on Grocery products.
        #FROZEN: Annual spending on Frozen products.
        #DETERGENTS_PAPER: Annual spending on Detergent and Paper Products.
        #DELICATESSEN: Annual spending on Delicaressen Products.

        #===Display description of dataset.
        desc=data.describe()
        print("the data description is below:")
        print(desc)
        #Now we gotta understand the customer's behaviour while buying products, and to get better understanding of the buyers.
        #====TAKING A SAMPLE OD DATA FOR BETTER UNDERSTANDING====
        data1=data.loc[[100,200,300],:]
        print(data1)
        #Column names
        a1=data.columns
        print("The data's columns are:",a1)
        #We have first and third quartile.Using that,we can filter out starkly different products.
        print("=====LOW Fresh Products,HIGH Grocery Products====")
        fresh_q1 = 3127.750000 #25th percentile for the fresh products.
        print(data.loc[data.Fresh < fresh_q1, :].head())
        #Here we see, 43: "LOW" Fresh products but, "HIGH" Grocery products.
        print("=====LOW Frozen Products,HIGH Fresh Products=====")
        frozen_q1 = 742.250000#25th percentile for the frozen products.
        print(data.loc[data.Frozen < frozen_q1, :].head())
        #Here we see, 12: "LOW" Frozen products but, "HIGH" fresh products.
        print("=====HIGH Frozen Products,LOW Detergent products==")
        frozen_q3 = 3554.250000#75th percentile for the frozen products.
        print(data.loc[data.Frozen > frozen_q3, :].head(7))
        #Here we see,39: "HIGH" frozen products, but "LOW" detergent products.
        indices = [43, 12, 39]#Now,choose any three sample.
        #Create a DataFrame of the chosen samples
        # reset_index(drop = True) resets the index from 0, 1 and 2 instead of 100, 200 and 300 
        samples = pd.DataFrame(data.loc[indices], columns = data.columns).reset_index(drop = True)
        print("Chosen samples of the given dataset are:")
        print(samples)

        #====================================DATA VISUALISATION============================================
        #Now, comes the use of seaborn, a very strong library for data visualisation
        #==first we gotta find the mean==
        mean_data=data.describe().loc['mean',:]
        #==append mean to sample's data==
        sample_data=samples.append(mean_data)
        print(sample_data)
        sample_data.index=indices+[1]
        #==Now, we will plot the points==
        sample_data.plot(kind='bar')
        plt.show()
        # First, calculate the percentile ranks of the whole dataset.
        percentiles = data.rank(pct=True)
        print("************************************************************")
        print(percentiles)
        print("************************************************************")
        # Then, round it up, and multiply by 100
        percentiles = 100*percentiles.round(decimals=3)
        print("************************************************************")
        print(percentiles.head(50))
        print("************************************************************")
        # Select the indices you chose from the percentiles dataframe
        percentiles = percentiles.iloc[indices]
        print("************************************************************")
        print(percentiles)
        print("************************************************************")
        # Now, create the heat map using the seaborn library
        sb.heatmap(percentiles, vmin=20, vmax=99, annot=True)
        plt.show()
        #=======TO PREDICT THE BEHAVIOUR OF THESE THREE SAMPLES=======
        def info():
            top=tk.Toplevel()
            top.geometry("700x500")
            top.title("Some Predicted Information!")
            foo=open("textproject_predc.txt","r")
            text1=foo.read()
            label1=tk.Label(top,text=text1,bg="White",fg="Black",height="18",width="90")
            label1.place(x="40",y="100")
            button2=tk.Button(top,text="GO BACK!",bg="Brown",fg="White",height="1",width="10",command=top.destroy)
            button2.place(x="290",y="400")
        root=tk.Tk()
        root.geometry("800x400")
        root.title("INFORMATION PREDICTED REGARDING THE SAMPLES")
        root.configure(background="Grey")
        button=tk.Button(root,text="FIND OUT WHAT ESTABILISHMENT THESE THREE SAMPLES CAN REPRESENT",bg="Black",fg="White",width="60",height="4",bd="10",command=info)
        button.place(x="150",y="120")
        button1=tk.Button(root,text="EXIT",command=root.destroy,bd="10",bg="Black",fg="White")
        button1.place(x="360",y="300")
        root.mainloop()
        Wholesale_df.drop([],axis=1,inplace=True)
        df2=Wholesale_df
        X=Wholesale_df.drop(["Channel"],axis=1)
        Y=df2["Channel"]
        Logistic_Accuracy_Mean=0.0
        X1_train,X1_test,Y1_train,Y1_test=train_test_split(X,Y,test_size=0.2)
        lr=LogisticRegression()
        lr.fit(X1_train,Y1_train)
        pre=lr.predict(X1_test)
        print(pre)
        correct=0
        lis=Y1_test.to_list()
        print(type(Y1_test))
        for i in range(len(pre)):
            if(pre[i]==lis[i]):
                correct=correct+1
        print("Accuracy : ",correct/len(pre))

        for i in range(101):
            X1_train,X1_test,Y1_train,Y1_test=train_test_split(X,Y,test_size=0.01)
            lr=LogisticRegression()
            lr.fit(X1_train,Y1_train)
            pre=lr.predict(X1_test)
            Logistic_Accuracy_Mean=Logistic_Accuracy_Mean+accuracy_score(Y1_test,pre)
        print("Mean of Accuracy: ",Logistic_Accuracy_Mean/100)

        #================================CUSTOMER SEGMENTATION===============================================
        customer=pd.read_csv("Customer Density.csv")
        customer_df=pd.DataFrame(customer)
        customer_df=customer_df.reindex(np.random.permutation(customer_df.index))
        customer_df.reset_index(inplace=True,drop=True)
        print(customer_df[0:10])
        #print(customer_df.head())#top 5 columns
        #print(len(customer_df)) # of row

        #descriptive statistics of the dataset
        print("the descriptive statistics of the datasets")
        print(customer_df.describe())
        #Visualizing using jointplot
        join=sb.jointplot(x='Customer Density',y='Financial Viability',data=customer_df,kind='hex')
        plt.show()

        #Identifying no. of clusters using Elbow method
        wcss = []
        for i in range(1,11):
            km=KMeans(n_clusters=i)
            km.fit(customer_df)
            wcss.append(km.inertia_)
        plt.plot(range(1,11),wcss)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('wcss')
        plt.show()
        ##Fitting kmeans to the dataset - k=3
        km4=KMeans(n_clusters=3)
        y_means = km4.fit_predict(customer_df)
        #Visualizing the clusters
        plt.scatter(customer_df.iloc[y_means==0,0],customer_df.iloc[y_means==0,1],s=20, c='purple',label='Cluster1')
        plt.scatter(customer_df.iloc[y_means==1,0],customer_df.iloc[y_means==1,1],s=20, c='blue',label='Cluster2')
        plt.scatter(customer_df.iloc[y_means==2,0],customer_df.iloc[y_means==2,1],s=50, c='green',label='Cluster3')
        plt.scatter(km4.cluster_centers_[:,0], km4.cluster_centers_[:,1],s=200,marker='s', c='red', alpha=0.7, label='Centroids')
        plt.title('Customer segments of the dataset:')
        plt.xlabel('Financial Viability')
        plt.ylabel('Customer Density')
        plt.legend()
        plt.show()

        def theory30():
            top00=tk.Toplevel()
            top00.geometry("700x500")
            top00.title("Some Predicted Information!")
            foo2=open("textproject_predc5.txt","r")
            text3=foo2.read()
            labelp=tk.Label(top00,text=text3,bg="White",fg="Black",height="18",width="90")
            labelp.place(x="40",y="80")
            button29=tk.Button(top00,text="GO BACK!",bg="Brown",fg="White",height="1",width="10",command=top00.destroy)
            button29.place(x="290",y="400")
        root2=tk.Tk()
        root2.geometry("500x200")
        root2.title("INFORMATION!")
        root2.configure(background="Grey")
        button100=tk.Button(root2,text="What do we understand by the clusters?",bg="Black",fg="White",command=theory30,bd="10")
        button100.place(x="100",y="20")
        button20=tk.Button(root2,text="EXIT",command=root2.destroy,bd="10",bg="Black",fg="White")
        button20.place(x="180",y="100")
        root2.mainloop()
except Exception as e:
    print(e)

