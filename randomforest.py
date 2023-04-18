import numpy as np                                                                                                                                                                                                          
from scipy.stats import pearsonr,spearmanr
from sklearn.model_selection import PredefinedSplit,KFold
import glob
import os
from matplotlib import pyplot as plt 
import pandas as pd
import math
import scipy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from joblib import dump, load
from scipy.stats.mstats import gmean

from sklearn import preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from joblib import load,Parallel,delayed
from sklearn.ensemble import RandomForestRegressor
from scipy.io import savemat
from scipy.stats import spearmanr,pearsonr
from scipy.optimize import curve_fit
import glob
import argparse



parser = argparse.ArgumentParser(description='Run a content-separated Random Forest model')
parser.add_argument('--score_file',help='File with video names and scores')
parser.add_argument('--feature_folder',help='Folder containing features')
parser.add_argument('--only_train',action='store_true',help='only train')
parser.add_argument('--only_test',action='store_true',help='only test')
parser.add_argument('--train_and_test',action='store_true',help='train and test')


args = parser.parse_args()


def results(all_preds,all_dmos):
    all_preds = np.asarray(all_preds)
    all_preds[np.isnan(all_preds)]=0
    all_dmos = np.asarray(all_dmos)

    try:
        f = lambda x, a, b, c, s : (a-b) / (1 + np.exp(-((x - c) / s))) + b
        init_val = np.array([np.max(all_mos), np.min(all_mos), np.mean(all_preds) , np.std(all_preds)/4])
        [[a, b, c, s], _] = curve_fit(f, all_preds, all_mos, p0=init_val, maxfev=20000)
        preds_fitted = (a-b) / (1 + np.exp(-((all_preds - c) / s))) + b
    except:
        preds_fitted = all_preds
    preds_srocc = spearmanr(preds_fitted,all_dmos)
    preds_lcc = pearsonr(preds_fitted,all_dmos)
    preds_rmse = np.sqrt(np.mean((preds_fitted-all_dmos)**2))
#    print('SROCC:')
#    print(preds_srocc[0])
#    print('LCC:')
#    print(preds_lcc[0])
#    print('RMSE:')
#    print(preds_rmse)
#    print(len(all_preds),' videos were read')
    return preds_srocc[0],preds_lcc[0],preds_rmse


scores_df = pd.read_csv(args.score_file)
scores_df.reset_index(drop=True, inplace=True)
video_names = scores_df['video']
scores = list(scores_df['mos'])
srocc_list = []

def trainval_split(trainval_content,r):
    train,val= train_test_split(trainval_content,test_size=0.2,random_state=r)
    train_features = []
    train_indices = []
    val_features = []
    train_scores = []
    val_scores = []

    feature_folder1 = args.feature_folder

    train_names = []
    val_names = [] 
    for i,vid in enumerate(video_names):
        featfile_name = vid+'.z'
        feat_file = load(os.path.join(feature_folder1,featfile_name))
        full_feature1 = np.asarray(feat_file['features'],dtype=np.float32)
        feature = full_feature1
            
        score = scores[i]
        if(scores_df.loc[i]['content'] in train):
            train_features.append(feature)
            train_scores.append(score)
            train_indices.append(i)
            train_names.append(scores_df.loc[i]['video'])
            
        elif(scores_df.loc[i]['content'] in val):
            val_features.append(feature)
            val_scores.append(score)
            val_names.append(scores_df.loc[i]['video'])
    return np.asarray(train_features),train_scores,np.asarray(val_features),val_scores,train,val_names

def single_split(trainval_content,cv_index,C):

    train_features,train_scores,val_features,val_scores,_,_ = trainval_split(trainval_content,cv_index)
    clf = RandomForestRegressor()
    X_train =train_features
    X_test = val_features
    clf.fit(X_train,train_scores)
    return clf.score(X_test,val_scores)
def grid_search(C_list,trainval_content):
    best_score = -100
    best_C = C_list[0]
    for C in C_list:
        cv_score = Parallel(n_jobs=5)(delayed(single_split)(trainval_content,cv_index,C) for cv_index in range(5))
        avg_cv_score = np.average(cv_score)
        if(avg_cv_score>best_score):
            best_score = avg_cv_score
            best_C = C
    return best_C

def train_test(r):
    train_features,train_scores,test_features,test_scores,trainval_content,test_names = trainval_split(scores_df['content'].unique(),r)
    best_C= grid_search(C_list=np.logspace(-7,2,10,base=2),trainval_content=trainval_content)
    X_train = train_features
    X_test = test_features
    best_randomforest =RandomForestRegressor() 
    best_randomforest.fit(X_train,train_scores)
    preds = best_randomforest.predict(X_test)
    srocc,lcc,rmse = results(preds,test_scores)
    return srocc,lcc,rmse
def only_train(r):
    train_features,train_scores,test_features,test_scores,trainval_content = trainval_split(scores_df['content'].unique(),r)
    all_features = np.concatenate((np.asarray(train_features),np.asarray(test_features)),axis=0) 
    all_scores = np.concatenate((train_scores,test_scores),axis=0) 
    X_train = all_features
    grid_randomforest = RandomForestRegressor()
    grid_randomforest.fit(X_train, all_scores)
    preds = grid_randomforest.predict(X_train)
    srocc_test = spearmanr(preds,all_scores)
    print(srocc_test)
    return

def only_test(r):
    train_features,train_scores,test_features,test_scores,trainval_content = trainval_split(scores_df['content'].unique(),r)
    all_features = np.concatenate((np.asarray(train_features),np.asarray(test_features)),axis=0) 
    all_scores = np.concatenate((train_scores,test_scores),axis=0) 
    X_train = all_features
    grid_randomforest = load('/home/ubuntu/ChipQA_files/zfiles/rapique_on_apv_randomforest.z')
    preds = grid_randomforest.predict(X_train)
    srocc_test = spearmanr(preds,all_scores)
    predfname = 'preds_'+str(r)+'.mat'
    out = {'pred':preds,'y':test_scores}
    srocc_val = np.nan_to_num(srocc_test[0])
    print(srocc_val)
    return

if(args.only_train):
    only_train(0)
elif(args.only_test):
    only_test(0)
elif(args.train_and_test):
    srocc_list = Parallel(n_jobs=-1,verbose=0)(delayed(train_test)(i) for i in range(100))
    print("median srocc is")
    print(np.median([s[0] for s in srocc_list]))
    print("median lcc is")
    print(np.median([s[1] for s in srocc_list]))
    print("median rmse is")
    print(np.median([s[2] for s in srocc_list]))
    print("std of srocc is")
    print(np.std([s[0] for s in srocc_list]))
    print("std of lcc is")
    print(np.std([s[1] for s in srocc_list]))
    print("std of rmse is")
    print(np.std([s[2] for s in srocc_list]))
