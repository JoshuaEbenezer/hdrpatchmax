import numpy as np                                                                                                                                                                                                          
from scipy.stats import pearsonr,spearmanr
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from joblib import dump, load

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import load,Parallel,delayed
from sklearn.svm import SVR
from scipy.io import savemat
from scipy.stats import spearmanr,pearsonr
from scipy.optimize import curve_fit
import argparse

parser = argparse.ArgumentParser(description='Run a content-separated SVR model')
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
        [[b0, b1, b2, b3, b4], _] = curve_fit(lambda t, b0, b1, b2, b3, b4: b0 * (0.5 - 1.0/(1 + np.exp(b1*(t - b2))) + b3 * t + b4),
                                              all_preds, all_dmos, p0=0.5*np.ones((5,)), maxfev=20000)
        preds_fitted = b0 * (0.5 - 1.0/(1 + np.exp(b1*(all_preds - b2))) + b3 * all_preds+ b4)
    except:
        preds_fitted = all_preds
    preds_srocc = spearmanr(preds_fitted,all_dmos)
    preds_lcc = pearsonr(preds_fitted,all_dmos)
    preds_rmse = np.sqrt(np.mean((preds_fitted-all_dmos)**2))
    return preds_srocc[0],preds_lcc[0],preds_rmse



scores_df = pd.read_csv(args.score_file)
video_names = scores_df['video']
scores = scores_df['mos']
srocc_list = []

def trainval_split(trainval_content,r):
    train,val= train_test_split(trainval_content,test_size=0.2,random_state=r)
    train_features = []
    train_indices = []
    val_features = []
    train_scores = []
    val_scores = []

    feature_folder= args.feature_folder

    train_names = []
    val_names = [] 
    for i,vid in enumerate(video_names):
        featfile_name = vid+'_upscaled.z'
        feat_file = load(os.path.join(feature_folder,featfile_name))
            
        feature = np.asarray(feat_file['features'],dtype=np.float32)

        feature = np.nan_to_num(feature)
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
    clf = svm.SVR(kernel='linear',C=C)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_features)
    X_test = scaler.transform(val_features)
    clf.fit(X_train,train_scores)
    return clf.score(X_test,val_scores)
def grid_search(C_list,trainval_content):
    best_score = -100
    best_C = C_list[0]
    for C in C_list:
        cv_score = Parallel(n_jobs=-1)(delayed(single_split)(trainval_content,cv_index,C) for cv_index in range(5))
        avg_cv_score = np.average(cv_score)
        if(avg_cv_score>best_score):
            best_score = avg_cv_score
            best_C = C
    return best_C

def train_test(r):
    train_features,train_scores,test_features,test_scores,trainval_content,test_names = trainval_split(scores_df['content'].unique(),r)
    best_C= grid_search(C_list=np.logspace(-7,2,10,base=2),trainval_content=trainval_content)
    scaler = StandardScaler()
    scaler.fit(train_features)
    X_train = scaler.transform(train_features)
    X_test = scaler.transform(test_features)
    best_svr =SVR(kernel='linear',C=best_C) 
    best_svr.fit(X_train,train_scores)
    preds = best_svr.predict(X_test)
    srocc,lcc,rmse = results(preds,test_scores)
    return srocc,lcc,rmse

def only_train(r):
    train_features,train_scores,test_features,test_scores,trainval_content,test_names = trainval_split(scores_df['content'].unique(),r)
    best_C= grid_search(C_list=np.logspace(-7,2,10,base=2),trainval_content=trainval_content)

    all_features = np.concatenate((np.asarray(train_features),np.asarray(test_features)),axis=0)
    all_scores = np.concatenate((train_scores,test_scores),axis=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(all_features)
    if(args.model=='decisiontree'):
        clf = DecisionTreeRegressor()
    elif(args.model=='svr'):
        clf = svm.SVR(kernel='linear')
    clf.fit(X_train, all_scores)
    preds = clf.predict(X_train)
    srocc_test = spearmanr(preds,all_scores)
    print(srocc_test)
    dump(scaler,args.scaler_path)
    dump(clf,args.model_path)
    return



def only_train(r):

    best_C= grid_search(C_list=np.logspace(-7,2,10,base=2),trainval_content=scores_df['content'].unique())
    train_features,train_scores,test_features,test_scores,trainval_content,test_names = trainval_split(scores_df['content'].unique(),r)
    all_features = np.concatenate((np.asarray(train_features),np.asarray(test_features)),axis=0) 
    all_scores = np.concatenate((train_scores,test_scores),axis=0) 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(all_features)
    best_svr = svm.SVR(kernel='linear',C=best_C)
    best_svr.fit(X_train, all_scores)
    preds = best_svr.predict(X_train)
    srocc_test = spearmanr(preds,all_scores)
    print(srocc_test)
    dump(scaler,"./fitted_scaler.z")
    dump(best_svr,"./trained_svr.z")
    return

def only_train_grid(r):
    train_features,train_scores,test_features,test_scores,_,_ = trainval_split(scores_df['content'].unique(),r)
    all_features = np.concatenate((np.asarray(train_features),np.asarray(test_features)),axis=0) 
    all_scores = np.concatenate((train_scores,test_scores),axis=0) 
#    scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(all_features)
    grid_svr = GridSearchCV(svm.SVR(kernel='linear'),param_grid = {"C":np.logspace(-7,2,10,base=2)},cv=5)
    grid_svr.fit(X_train, all_scores)
    preds = grid_svr.predict(X_train)
    srocc_test = spearmanr(preds,all_scores)
    print(srocc_test)
    dump(scaler,"./fitted_scaler_grid.z")
    dump(grid_svr,"./trained_svr_grid.z")
    return

def only_test(r):
    train_features,train_scores,test_features,test_scores,trainval_content,test_names = trainval_split(scores_df['content'].unique(),r)
    all_features = np.concatenate((np.asarray(train_features),np.asarray(test_features)),axis=0) 
    all_scores = np.concatenate((train_scores,test_scores),axis=0) 
    scaler = StandardScaler()
    scaler = load('wacv_livehdr_fitted_scaler_grid.z')
    X_train = scaler.transform(all_features)
    grid_svr = load('wacv_livehdr_trained_svr_grid.z')
    preds = grid_svr.predict(X_train)
    srocc_test = spearmanr(preds,all_scores)
    print(srocc_test)
    return


if(args.only_train):
    only_train_grid(0)
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
