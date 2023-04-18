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



parser = argparse.ArgumentParser(description='Run a trained HDRPATCHMAX Random Forest model')
parser.add_argument('--feature_file',help='Folder containing features')


args = parser.parse_args()

rf = load('./trained_rf.z')
features = load(args.feature_file)['features']
features = np.reshape(features,(1,-1))
score = rf.predict(features)
print(score[0], ' is the predicted MOS')
