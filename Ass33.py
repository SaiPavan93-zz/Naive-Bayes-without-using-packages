from __future__ import division
import numpy as np
import sys
import math
import pandas as pd

def meancov(f1):
    virginica=[]
    versicolor=[]
    setosa=[]

    for each in f1:
        classvar=each.split(',')
        if(classvar[4]=="Iris-virginica"):
            virginica.append(classvar[0:4])
        elif(classvar[4]=="Iris-setosa"):
            setosa.append(classvar[0:4])
        else:
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

def priorsfinding(traindata):
    d={}
    priors = {}
    try:
        names = []
        for each in traindata:
            classvar = each.split(',')
            names.append(classvar[4])
        for i in names:
            d[i] = d.get(i, 0) + 1
    except IndexError:
        names = []
        for each in traindata:
            classvar = each.split(',')
            names.append(classvar[4])
        for i in names:
            d[i] = d.get(i, 0) + 1
    for (values, keys) in zip(d.values(), d.keys()):
        priors[keys] = np.around((values / len(names)), 2)
    #print(priors)
    # sys.stdout = open("assign3-iusaichoud.txt", "w")
    covar_virginica, covar_setosa, covar_versicolor, mean_virginica, mean_setosa, mean_versicolor = meancov(traindata)
    return (priors, covar_virginica, covar_setosa, covar_versicolor, mean_virginica, mean_setosa, mean_versicolor)

def findPriors(f1):

    kfolds = 3
    instances = (len(f1) - 1) / kfolds
    # print(instances)
    f1 = f1[:-1]
    train_1fold = f1[0:int(instances)]
    train_2fold = f1[int(instances):int(instances * 2)]
    train_3fold = f1[int(instances * 2):]
    # print(len(train_1fold),len(train_2fold),len(test_3fold))
    traindata = train_1fold + train_2fold
    testdata=train_3fold
    traindata1=train_2fold+train_3fold
    testdata1=train_1fold
    traindata2=train_1fold+train_3fold
    testdata2=train_2fold
    train=[traindata,traindata1,traindata2]
    test=[testdata,testdata1,testdata2]
    answ=[]
    for each,i in zip(train,test):
        priors, covar_virginica, covar_setosa, covar_versicolor, mean_virginica, mean_setosa, mean_versicolor=priorsfinding(each)
        answ.append((priors, covar_virginica, covar_setosa, covar_versicolor, mean_virginica, mean_setosa,
                     mean_versicolor,i))
    return answ

def estimateLikelyhood(covariance,mean,point,prior):
    a = math.pow(math.sqrt(2 * np.pi), 4)
    b = math.sqrt(np.linalg.det(covariance))
    ktt = np.matrix(point - mean)
    c = ((ktt).dot(np.linalg.inv(covariance)).dot(ktt.transpose())) / 2
    # c=np.matmul(covariance,(np.linalg.inv(covariance)))
    # print((a*b),c,ktt,point,mean)
    d = (1 / (a * b)) * math.exp(-(c))
    return(d*prior)

def likelyhood(priors,virginica,setosa,versicolor,means,names):
    #print(means[1][0])
    prob=[]
    for i in range(len(names)):
        p=[]
        arr=np.array(names[i], dtype=float)
        post_virginica=estimateLikelyhood(virginica, means[0][0], arr, priors['Iris-virginica'])
        post_setosa=estimateLikelyhood(setosa,means[1][0],arr,priors['Iris-setosa'])
        post_versicolor=estimateLikelyhood(versicolor,means[2][0],arr,priors['Iris-versicolor'])
        p.append([post_virginica,post_setosa,post_versicolor])
        prob.append(p)
    #print(len(prob))
    classlabel=[]
    for each in prob:
        classlabel.append(each[0].index(max(each[0])))
    return  classlabel

def metrics(mat,df):

    Accuracy=[]
    precision=[]
    recall=[]
    F1score=[]
    TP = (np.diag(mat))
    #print(TruePoistive)
    for (i,j) in zip(range(3),range(3)):
        TruePoistive=TP[i]
        df1=(df.drop(i,axis=1).drop(j,axis=0))
        #print(df1,sum(sum(df1.values)))
        TrueNegative=sum(sum(df1.values))
        #print(df)
        a=df[i].values
        b=df.loc[i].values
        #print(a,b)
        falseNegative=sum(np.delete(a,i))
        falsePositive=sum(np.delete(b,i))
        #print(TruePoistive,TrueNegative,falseNegative,falsePositive)
        #print(falsePositive,falseNegative)
        p=(TruePoistive)/(TruePoistive+falsePositive)
        r=(TruePoistive)/(TruePoistive+falseNegative)
        Accuracy.append((TruePoistive+TrueNegative)/(TruePoistive+TrueNegative+falseNegative+falsePositive))
        precision.append(p)
        recall.append(r)
        F1score.append((2*((p*r)/(p+r))))
    #print(Accuracy)
    return(Accuracy,precision,recall,F1score)

def main():
    acc=[]
    prec=[]
    rec=[]
    f1score=[]
    conf=[]
    f = open(sys.argv[1], "r")
    file = f.read()
    f1=file.split("\n")
    answ=findPriors(f1)
    #print(answ[0][7])
    for each in answ:
        names = []
        labels = []
        for eac in each[7]:
            classvar = eac.split(',')
            names.append(classvar[0:4])
            labels.append(classvar[4])
        classlabel = likelyhood(each[0], each[1], each[2], each[3], each[4:7], names)
        #print(classlabel)
        lab = []
        lab1=[]
        for each in classlabel:
            if (each == 2):
                lab.append('Iris-versicolor')
            elif (each == 0):
                lab.append('Iris-virginica')
            elif (each == 1):
                lab.append('Iris-setosa')
        for each in labels:
            if (each == 'Iris-versicolor'):
                lab1.append(2)
            elif (each == 'Iris-virginica'):
                lab1.append(0)
            elif (each == 'Iris-setosa'):
                lab1.append(1)
        actual = pd.Series(labels, name='Actual')
        predicted = pd.Series(lab, name='Predicted')
        actual1 = pd.Series(lab1, name='Actual')
        predicted1 = pd.Series(classlabel, name='Predicted')
        sys.stdout=open("testfile_pavan.txt","w")
        df_confusion = pd.crosstab(predicted, actual)
        df_confusion1 = pd.crosstab(predicted1, actual1)
        conf.append(df_confusion)
        mat=np.matrix(df_confusion1.values)
        #print(conf)
        accuracy,precision,recall,F1score=metrics(mat,df_confusion1)
        #print(accuracy)
        acc.append(accuracy)
        prec.append(precision)
        rec.append(recall)
        f1score.append(F1score)
    #print(np.matrix(acc),"\n",np.matrix(prec),"\n",np.matrix(rec),"\n",np.matrix(f1score))
    print("The confusion matrices are\n", conf,"\n Metrics averaged over 3 folds\n","Accuracy :",np.matrix(acc).mean(0),
          "\n","Precision :", np.matrix(prec).mean(0),"\n","Recall:",np.matrix(rec).mean(0),"\n",
          "F1Score:",np.matrix(f1score).mean(0))



if __name__ == "__main__":
    main()