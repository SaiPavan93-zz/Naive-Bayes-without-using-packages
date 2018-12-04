from __future__ import division
import numpy as np
import math
import sys
import pandas as pd


def convert(matrs):
    mean=[]
    for i in range(len(matrs)):
        a=(matrs[i].split(','))
        if(i==0):
            mean_virginica=np.array(a,dtype=float)
        elif(i==1):
            mean_Setosa=np.array(a,dtype=float)
        else:
            mean_versicolor=np.array(a,dtype=float)
    return  (mean_virginica,mean_Setosa,mean_versicolor)

def convertcov(mat):
    cov=[]
    for i in range(len(mat)):
        a=mat[i].split(',')
        cov.append(np.array(a,dtype=float))
    #print(np.matrix(cov))
    return(np.matrix(cov))

def convertprior(mat):
    for i in range(len(mat)):
        a=(mat[i].split(':'))
        if(i==0):
            prior_virginica=float(a[1])
        elif(i==1):
            prior_versicolor=float(a[1])
        else:
            prior_setosa=float(a[1])
    return (prior_virginica,prior_versicolor,prior_setosa)

def calmul(x):
    k=1
    for each in x:
        k = k * each
    return k

def estimateLikelyhood(covariance,mean,point,prior):
    a=math.pow(math.sqrt(2*np.pi),4)
    #b=math.sqrt(np.exp(np.log(np.diag(covariance)).sum()))
    b1=math.sqrt(calmul(np.diag(covariance)))
    ktt=np.matrix(point-mean)
    c=((ktt).dot(np.linalg.inv(covariance)).dot(ktt.transpose()))/2
    c1=0
    #m=point-mean
    #print(m.dot(np.linalg.inv(covariance)).dot(m.transpose()))
    for (i,j,k) in zip(point,mean,np.diag(covariance)):
        #print(i,j,k)
        c1 = c1 + math.pow((i - j), 2) / k
        #print(c1)
    #c=np.matmul(covariance,(np.linalg.inv(covariance)))
    #print(c1/2,c)
    d=(1/(a*b1))*math.exp(-(c1/2))
    #print(b,(b1),np.diag(covariance))
    return(d*prior)

def likelyhood(priors,virginica,setosa,versicolor,means,names):
    mean_virginica,mean_setosa,mean_versicolor=convert(means)
    cov_virginica=convertcov(virginica)
    cov_setosa=convertcov(setosa)
    cov_versicolor=convertcov(versicolor)
    prior_virginica,prior_versicolor,prior_setosa=convertprior(priors)
    prob=[]
    arry=[]
    for i in range(len(names)):
        p=[]
        arr=np.array(names[i], dtype=float)
        post_virginica=estimateLikelyhood(cov_virginica, mean_virginica, arr, prior_virginica)
        post_setosa=estimateLikelyhood(cov_setosa,mean_setosa,arr,prior_setosa)
        post_versicolor=estimateLikelyhood(cov_versicolor,mean_versicolor,arr,prior_versicolor)
        p.append([post_virginica,post_setosa,post_versicolor])
        prob.append(p)
        arry.append(arr)
    #print(len(prob))
    classlabel=[]
    for each in prob:
        classlabel.append(each[0].index(max(each[0])))
    return  (classlabel,arry)
    #arr=np.array(names[0],dtype=float)

def confusionmat(labels,lab, classnames):
    classnames=list(classnames)
    mat=np.zeros((len(classnames),len(classnames)))
    for (each,i) in zip(labels,lab):
        if (each==i):
            mat[classnames.index(each)][classnames.index(i)]+=1
        else:
            mat[classnames.index(each)][classnames.index(i)] += 1
    print(mat.T)

def main():
    f = open(sys.argv[1], "r")
    ff=open(sys.argv[2],"r")
    file = f.read()
    file1=ff.read()
    f2=file1.split("\n")
    try:
        names = []
        labels=[]
        for each in f2:
            classvar = each.split(',')
            names.append(classvar[0:4])
            labels.append(classvar[4])
    except IndexError:
        names = []
        labels=[]
        for each in f2[0:(len(f2) - 1)]:
            classvar = each.split(',')
            names.append(classvar[0:4])
            labels.append(classvar[4])
    #print(names)
    f1=file.rstrip(",").split("\n")
    means=f1[0:3]
    virginica=(f1[3:7])
    setosa=f1[7:11]
    versicolor=f1[11:15]
    priors=(f1[15:18])
    #sys.stdout=open("pavan.txt","w")
    classlabel,arry=likelyhood(priors, virginica, setosa, versicolor, means, names)
    #print(len(classlabel))
    lab=[]
    for each in classlabel:
        if(each== 2):
            lab.append('Iris-versicolor')
        elif(each==0):
            lab.append('Iris-virginica')
        elif(each==1):
            lab.append('Iris-setosa')
    #print(lab,len(lab),classlabel)
    for i,j in zip(arry,lab):
        print(i,j)
    actual=pd.Series(labels,name='Actual')
    predicted=pd.Series(lab,name='Predicted')
    #print(len(actual),len(predicted))
    #print(confusion_matrix(lab, classlabel))
    df_confusion = pd.crosstab(predicted,actual)
    print(df_confusion)
    confusionmat(labels, lab, set(lab))

if __name__ == "__main__":
    main()


