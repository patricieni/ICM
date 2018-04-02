import os 
from pathlib import Path
import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score, confusion_matrix, r2_score
from sklearn.metrics import log_loss,auc,classification_report,roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, label_binarize
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import gbm

pd.set_option('display.max_columns', None)

def write_to_pickle(dataframe, name):
    dataframe.to_pickle(data_path + name + ".pickle")
def read_from_pickle(name): 
    return pd.read_pickle(data_path + name + ".pickle")

data_path = str(Path(os.getcwd())) + "/data/"
results_path = str(Path(os.getcwd())) + "/results/random_forests/"
#.parents[0]

# Args Parser
parser = argparse.ArgumentParser(description='Random forests script')

# Not sure for labels if I have to say str
parser.add_argument('-d','--dataset', nargs="?", type=str, help='Path for dataset to use (csv)',required=True)
parser.add_argument('-l','--labels', nargs="+", type=str, help='The exact labels required for splitting the expectancy variable, i.e. 1.5years 4years, more for 3 categories',required=True)
parser.add_argument('-v','--values', nargs="+", type=int, help='Values where to split the dataset (in days, i.e. 500 ~ 1.5 years )',required=True)
parser.add_argument('-lr','--learning_rate', nargs="?", type=float, help='Learning rate for rf',required=True)
parser.add_argument('-o','--output', nargs="+", type=str, help='Output files',required=True)

args = parser.parse_args()
dataset = args.dataset
labels = args.labels
cut_points = args.values
learning_rate = args.learning_rate
output = args.output

# Clean up from the imputation
df = pd.read_csv(data_path+dataset)
df.drop("Unnamed: 0", axis = 1, inplace=True)


# Try with three classes first
df.loc[:,"life_expectancy_bin"] = gbm.helper.binning(df.life_expectancy, cut_points, labels)
#print(pd.value_counts(df_amelia.life_expectancy_bin, sort=False))
print(df.life_expectancy_bin.values)
print("\n")
# Print the data in the output 
for column in df:
    unique_vals = np.unique(df[column])
    nr_vals = len(unique_vals)
    if nr_vals < 20:
        print('Number of values for attribute {}: {} -- {}'.format(column, nr_vals, unique_vals))
    else:
        print('Number of values for attribute {}: {}'.format(column, nr_vals))
print("\n")        
        
non_dummy_cols = ['Tumor_grade','IDH_TERT','life_expectancy','life_expectancy_bin','Gender','IK','Age_surgery']
dummy_cols = list(set(df.columns) - set(non_dummy_cols))

df = pd.get_dummies(df,columns=dummy_cols)

df.Gender.replace(to_replace={'M':1, 'F':0},inplace=True)   

X = df.drop(["life_expectancy","life_expectancy_bin"], axis=1)
Y = df.life_expectancy_bin

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,train_size=0.8, test_size=0.2, random_state=1332)

    
# Random forests n_estimators based on validation set
estimators_range = [5, 10, 20, 50, 100, 200, 300, 400, 500, 1000, 2000]
accuracies = []
errors = []
roc_auc = []
cm = []
l = np.array(labels)
i=0

for n_estimators in estimators_range:
    rfc = RandomForestClassifier(criterion='entropy', 
                                   n_estimators=n_estimators,
                                   max_features = 30,
                                   max_depth =100,  
                                   n_jobs = -1,
                                   random_state = 1123)
    rfc.fit(X_train, Y_train)
    
    # Accuracy
    accuracies.append(rfc.score(X_test,Y_test)) 
    
    # XEntropy Error
    probas = rfc.predict_proba(X_test)
    y_pred = np.argmax(probas, axis=1)
    error = log_loss(Y_test,probas)
    errors.append(error)
    
    # Confusion matrix      
    cfm = confusion_matrix(Y_test, l[y_pred])
    cm.append(cfm)
    
    # ROC-AUC - I think this is wrong
    Y_test_binary = label_binarize(Y_test, classes=labels)
    y_pred_binary = label_binarize(l[y_pred],classes =labels)
    auc_curve = gbm.evaluation.multi_class_auc(len(l),Y_test_binary,y_pred_binary)
    roc_auc.append(auc_curve)
    i+=1
    print("Logloss {} --Random Forest Classifier with features = {}, max_depth = {}, estimators = {} -- {}/{}".format(error,rfc.max_features,rfc.max_depth,n_estimators,i,len(estimators_range)))

n_estimators_optimal_accuracies = estimators_range[np.argmax(accuracies)]
n_estimators_optimal_errors = estimators_range[np.argmin(errors)]    
print("\n")
if not os.path.isdir(results_path):
    os.mkdir(results_path)


fig, ax = plt.subplots(1, 3, figsize=(15,10))

ax[0].scatter(estimators_range, errors)
ax[0].set_ylabel('Log-loss error on validation set')
ax[0].set_xlabel('Number of estimators for Random Forest Classifier');

ax[1].scatter(estimators_range, accuracies)
ax[1].set_ylabel('Accuracies on validation set')
ax[1].set_xlabel('Number of estimators for Random Forest Classifier');

ax[2].scatter(estimators_range, roc_auc)
ax[2].set_ylabel('AUC on validation set')
ax[2].set_xlabel('Number of estimators for Random Forest Classifier');

plt.savefig(results_path + output[0])
print("Saved Results for RF")

plt.figure()
gbm.evaluation.plot_confusion_matrix(cm[8], norm=True, classes=labels)
plt.xlabel('Interpreted cluster label')
plt.savefig(results_path + output[1])
print("Saved Confusion Matrix for RF")
'''
# Gradient Boosting classifiers based on validation/test
estimators_range = [100, 200, 300, 400, 500, 1000, 2000]
accuracies = []
errors = []
roc_auc = []
cm = []
l = np.array(labels)
i = 0
for n_estimators in estimators_range:
    gbc = GradientBoostingClassifier(learning_rate = learning_rate, 
                                     n_estimators=n_estimators,
                                     max_depth =30,                          
                                     random_state = 1123)
    gbc.fit(X_train, Y_train)
    
    # Accuracy
    accuracies.append(gbc.score(X_test,Y_test)) 
    
    # XEntropy Error
    probas = gbc.predict_proba(X_test)
    y_pred = np.argmax(probas, axis=1)
    errors.append(log_loss(Y_test,probas))
    
    # Confusion matrix      
    cfm = confusion_matrix(Y_test, l[y_pred])
    cm.append(cfm)
    
    i+=1
    print("Gradient Boosting Classifier with lr = {0}, depth = {1}, estimators = {3} -- {4}/7".format(learning_rate,max_depth,n_estimators,i))
    # ROC - AUC
'''  