#Isaac Palacio

import sys
import pandas as pd
from csv import reader
from scipy.stats import entropy
from IPython.core import ultratb
from sklearn.model_selection import KFold 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def checkInput(sw):
    if sw!="dia" and sw!="sys" and sw!="eda" and sw!="res" and sw!="all":
        raise ValueError('Invalid option. Please provide one of this options: dia, sys, eda, res, all.')

def addToList(destinyList, data, classList, painClass):
    painCLassn = 0; #No pain
    if painClass=="Pain":
        painCLassn = 1
    classList.append(painCLassn)
    
    if data.empty:
        #print('Instance of data empty.')
        destinyList.append([0, 0, 0, 0, 0])
    else:
        mean = data.mean()
        variance = data.var()
        ent = entropy(data)
        minimum = data.min()
        maximum = data.max()
        destinyList.append([mean, variance, ent, minimum, maximum])
    

#Make error messages colorful
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)

#Script Input
if len(sys.argv) != 2:
    raise ValueError('Please provide one of this options: dia, sys, eda, res, all.')
#Get Input
switch=sys.argv[1]
checkInput(switch)

# #Manual Input
# switch="all"

allSwitch = 0
if switch=="all":
    allSwitch = 1
    
#Define lists we will use
if switch=="dia" or allSwitch:
    diaClassification=[]
    diastolicBP=[]
if switch=="sys" or allSwitch:
    sysClassification=[]
    systolicBP=[]
if switch=="eda" or allSwitch:
    edaClassification=[]
    EDA=[]
if switch=="res" or allSwitch:
    resClassification=[]
    respiration=[]
        

#Open data file
with open('Project1Data.csv', 'r') as temp_f:
    csv_reader = reader(temp_f)
    for lineList in csv_reader:
        #subjectID = lineList[0]
        dataType = lineList[1]
        painClass = lineList[2]
        data = pd.Series(lineList[3:], dtype=float)
        cleanData = data[data>0]
  
        #Calculate and save Hand Crafted data
        if   (switch=="dia" or allSwitch) and dataType=="BP Dia_mmHg":
            addToList(diastolicBP, cleanData, diaClassification, painClass)
        elif (switch=="sys" or allSwitch) and dataType=="LA Systolic BP_mmHg":
            addToList(systolicBP, cleanData, sysClassification, painClass)
        elif (switch=="eda" or allSwitch) and dataType=="EDA_microsiemens":
            addToList(EDA, cleanData, edaClassification, painClass)
        elif (switch=="res" or allSwitch) and dataType=="Respiration Rate_BPM":
            addToList(respiration, cleanData, resClassification, painClass)


if switch=="dia":
    X = diastolicBP
    Y = diaClassification
elif switch=="sys":
    X = systolicBP
    Y = sysClassification
elif switch=="eda":
    X = EDA
    Y = edaClassification
elif switch=="res":
    X = respiration
    Y = resClassification
elif allSwitch: #Fusion
    X, Y = [],[]
    if len(diastolicBP) == len(systolicBP) == len(EDA) == len(respiration):
        for i in range(len(diastolicBP)):
            X.append(diastolicBP[i]+systolicBP[i]+EDA[i]+respiration[i])
            if diaClassification[i]==1 and sysClassification[i]==1 and edaClassification[i] == 1 and resClassification[i]==1:
                Y.append(1)
            elif diaClassification[i]==0 and sysClassification[i]==0 and edaClassification[i] == 0 and resClassification[i]==0:
                Y.append(0)
            else:
                raise ValueError('There is an instance where resul data is not the same.')
    else:
        raise ValueError('There are not the same number of instances in every data type.')


#K-fold Cross Validation
k = 10
recall = []
acc_score = []
precision = []
confusion_matrices =[]
kf = KFold(n_splits=k, random_state=None)
rf = RandomForestClassifier() #max_depth=1, random_state=0


for train_index , test_index in kf.split(X):
    X_train, X_test, y_train, y_test = [],[],[],[]
    for i in train_index:
        X_train.append(X[i])
        y_train.append(Y[i])
    for i in test_index:
        X_test.append(X[i])
        y_test.append(Y[i])   
    rf.fit(X_train, y_train)
    rfPred = rf.predict(X_test)
    
    #Gather results
    recall.append(recall_score(y_test,rfPred))
    acc_score.append(accuracy_score(y_test, rfPred))
    precision.append(precision_score(y_test,rfPred))
    confusion_matrices.append(confusion_matrix(y_test,rfPred))

#Calculate average results
avg_recall = sum(recall)/k
avg_acc_score = sum(acc_score)/k
avg_precision = sum(precision)/k
avg_conf_matrix = sum(confusion_matrices)/k

print("\n Results \n")
#print('Recall of each fold: {}'.format(recall))
print('Avg recall : {}'.format(avg_recall))
#print('Accuracy of each fold: {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))
#print('Precision of each fold: {}'.format(precision))
print('Avg precision : {}'.format(avg_precision))
#print('Confusion Matrix of each fold:')
#print(confusion_matrices)
print('Avg confusion matrix:')
print(avg_conf_matrix)
print("\n")
