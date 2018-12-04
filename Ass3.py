from __future__ import division
import numpy as np
import sys

def meancov(f1):
    virginica=[]
    versicolor=[]
    setosa=[]
    try:
        for each in f1:
            classvar=each.split(',')
            if(classvar[4]=="Iris-virginica"):
                virginica.append(classvar[0:4])
            elif(classvar[4]=="Iris-setosa"):
                setosa.append(classvar[0:4])
            elif(classvar[4]=="Iris-versicolor"):
                versicolor.append((classvar[0:4]))
    except IndexError:
        for each in f1[0:(len(f1)-1)]:
            classvar=each.split(',')
            if (classvar[4] == "Iris-virginica"):
                virginica.append(classvar[0:4])
            elif (classvar[4] == "Iris-setosa"):
                setosa.append(classvar[0:4])
            elif (classvar[4] == "Iris-versicolor"):
                versicolor.append((classvar[0:4]))
    virginica=np.array(virginica,dtype='float')
    versicolor=np.array(versicolor,dtype='float')
    setosa=np.array(setosa,dtype='float')
    virginica=np.matrix(virginica)
    versicolor=np.matrix(versicolor)
    setosa=np.matrix(setosa)
    mean_virginica=virginica.mean(0)
    mean_setosa=setosa.mean(0)
    mean_versicolor=versicolor.mean(0)
    covar_virginica=np.cov(virginica,rowvar=False,bias=True)
    covar_setosa=np.cov(setosa,rowvar=False,bias=True)
    covar_versicolor=np.cov(versicolor,rowvar=False,bias=True)
    return(np.around(covar_virginica,2),np.around(covar_setosa,2),np.around(covar_versicolor,2),np.around(mean_virginica,2),
           np.around(mean_setosa,2),np.around(mean_versicolor,2))

def findPriors(f1):

    d={}
    priors={}
    try:
        names = []
        for each in f1:
            classvar=each.split(',')
            names.append(classvar[4])
        for i in names:
            d[i]=d.get(i,0)+1
    except IndexError:
        names=[]
        for each in f1[0:(len(f1)-1)]:
            classvar=each.split(',')
            names.append(classvar[4])
        for i in names:
            d[i]=d.get(i,0)+1
    #print(d.keys())
    for (values,keys) in zip(d.values(),d.keys()):
        priors[keys]=np.around((values/len(names)),2)
    #sys.stdout = open("assign3-iusaichoud.txt", "w")
    covar_virginica,covar_setosa,covar_versicolor,mean_virginica,mean_setosa,mean_versicolor=meancov(f1)
    return(priors,covar_virginica,covar_setosa,covar_versicolor,mean_virginica,mean_setosa,mean_versicolor)

def printing(matrices):
    for a,b,c,d in matrices:
        print(a,",",b,",",c,",",d)

        #print("\n")

def main():
    f = open(sys.argv[1], "r")
    file = f.read()
    f1=file.split("\n")
    priors,covar_virginica,covar_setosa,covar_versicolor,mean_virginica,mean_setosa,mean_versicolor=findPriors(f1)
    sys.stdout=open("testfile.txt","w")
    printing(mean_virginica)
    printing(mean_setosa)
    printing(mean_versicolor)
    printing(covar_virginica)
    printing(covar_setosa)
    printing(covar_versicolor)
    for each in priors.keys():
        print(str(each),":",priors[each])


if __name__ == "__main__":
    main()