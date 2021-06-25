import time, datetime
import requests 
import pickle as pkl 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn import metrics
import time
from sklearn.model_selection import ShuffleSplit

import itertools

import warnings
warnings.filterwarnings('ignore')


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.model_selection import train_test_split

import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from scipy import stats
import pprint
import seaborn as sns

#models 
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.ensemble import VotingClassifier  

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedShuffleSplit

from vecstack import stacking
from xgboost import XGBClassifier


class FMSAD:
    '''
    FMS-AD an automated feature and model selectionanomaly detection framework for decentralized systems This repo contains the code for the FMS-AD framework

    Parameters
    ----------
    X : Pandas DataFrame
        Data set feature input
    Y : Numpy Array
        Labels for the data
    supervised_models : List
        List with supervised models
    unsupervised_models : List
        List with unsupervised models
    '''

    def __init__(self, X, Y, supervised_models, unsupervised_models):
        self.X = X
        self.Y = Y
        self.supervised_models =supervised_models
        self.unsupervised_models =unsupervised_models
        self.all_models = unsupervised_models+supervised_models
    
    def correlation_analysis(self, plot_heatmap=False):
        '''Calculate label correlation with respect to output label
        
        Parameters
        ----------
        plot_heatmap : Bool
            Set True if you want to plot a heatmap of the correlation
        '''

        correlation_txn_falure = {}

        for label in self.X.columns:
            r,p = stats.pearsonr(self.X[label],self.Y)
            correlation_txn_falure[label] = (r,p)
            
            txn_falure_correlation = pd.DataFrame.from_dict(correlation_txn_falure, orient='index').rename(columns={0:'r', 1:'p-value'}).sort_values('r', ascending=False)
        
        if plot_heatmap:
            plt.figure(figsize=(10,6))
            sns.heatmap(txn_falure_correlation, cmap='GnBu', square=True, annot=True, linewidths=.5)
            
        df_index = list(txn_falure_correlation.index)

        txn_falure_correlation = txn_falure_correlation.dropna(axis='rows')
        return txn_falure_correlation
    
    def correlation_features(self, r_lim=0.5, p_lim=0.05):
        '''Select features based specified boundaries
        Parameters
        ----------
        r_lim : float
            set limit for R-value
        p_lim : float
            set limit for P-value
        '''
        txn_falure_correlation = self.correlation_analysis()
        selected_features = txn_falure_correlation[np.abs(txn_falure_correlation.r) >= r_lim]
        selected_features = selected_features[selected_features['p-value'] <= p_lim]

        selected_features_names = selected_features.index
        return selected_features_names
    
    def evaluate(self, y_real, y_pred):
        '''
        Evaluate model performance

        Parameters
        ----------
        y_real : NumPy Array
            array with real labels
        y_reY_predal : NumPy Array
            array with predicted labels
        '''
        from sklearn import metrics
        accuracy = accuracy_score(y_real,y_pred)
        precision = precision_score(y_real, y_pred, average='macro')
        recall = metrics.recall_score(y_real, y_pred, average='macro')
        f1_score = metrics.f1_score(y_real, y_pred, average='macro') 

        fpr, tpr, threshold = metrics.roc_curve(y_real, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        return accuracy, precision, recall, f1_score, fpr, tpr, roc_auc

    # since boolean predictions may be wrong way around
    def result(self, y_real, y_pred):
        '''
        Evaluate model performance

        Parameters
        ----------
        y_real : NumPy Array
            array with real labels
        y_reY_predal : NumPy Array
            array with predicted labels
        '''
        from sklearn import metrics
        if  metrics.f1_score(y_real, y_pred, average='macro') >  metrics.f1_score(y_real, [not y for y in y_pred], average='macro'):
            return self.evaluate(y_real, y_pred)
        else:
            return self.evaluate(y_real, [not y for y in y_pred])

    def show_res(self, res):
        '''
        Prin results of model

        Parameters
        ----------
        res : list
            list with model results
        '''
        accuracy, precision, recall, f1_score, time = res
        print("Accuracy", accuracy)
        print("Precision", precision)
        print("Recall", recall)
        print("F1 Score", f1_score)
        print("Time", time)

    def flip_if_inverted(self, Y_pred):
        '''
        Anomalies are the label with the that occur the least,
        sometimes the boolean classification is the wrong way around
        Parameters
        ----------
        Y_pred : NumPy Array
            array with predicted labels
        '''
        Y_pred = np.array([0 if i != 1 else 1 for i in Y_pred ])
        if len(Y_pred[Y_pred == 1]) < len(Y_pred[Y_pred == 0]):

            return [not elem for elem in Y_pred]
        else:
            return Y_pred

    def validate_model(self, model,undersample=False, oversample=False, flip=True):
        '''
        Validate the performance of the given models

        Parameters
        ----------
        model : List
            list of models
        undersample : Boolean
            True if data needs to be undersampled
        oversample : Boolean
            True if data needs to be oversampled
        flip : Boolean
            True if data needs to be flipt if classifications are inverted
        '''
        accuracy_, precision_, recall_, f1_score_,fpr_, tpr_, roc_auc_, time_ = [],[],[],[],[], [],[],[]

        shuffle_split = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
        for train_index, test_index in shuffle_split.split(self.X,self.Y):
            start_time = time.time()

            if type(self.X) == pd.DataFrame:
                X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
                Y_train, Y_test = self.Y.iloc[train_index], self.Y.iloc[test_index]
            else:
                X_train, X_test = self.X[train_index], self.X[test_index]
                Y_train, Y_test = self.Y[train_index], self.Y[test_index]
            if undersample:
                undersampling = RandomUnderSampler(sampling_strategy="majority")
                X_train, Y_train = undersampling.fit_resample(X_train, Y_train)

            if oversample:
                oversampling = RandomOverSampler(sampling_strategy="minority")
                X_train, Y_train = oversampling.fit_resample(X_train, Y_train)

            fitted_model = model.fit(X_train, Y_train)
            try:
                y_pred = fitted_model.predict(X_test)
            except:
                y_pred = fitted_model.fit_predict(X_test)

            if flip:
                y_pred = self.flip_if_inverted(y_pred)

            accuracy, precision, recall, f1_score, fpr, tpr, roc_auc = self.result(Y_test, y_pred)

            end_time = time.time()
            fpr_.append(fpr)
            tpr_.append(tpr)

            accuracy_.append(accuracy)
            precision_.append(precision)
            recall_.append(recall)
            f1_score_.append(f1_score)
            roc_auc_.append(roc_auc)
            time_.append(end_time - start_time)              
        return np.mean(accuracy_), np.mean(precision_), np.mean(recall_), np.mean(f1_score_), np.mean(time_)

    def optimize_voting_parameters(self, weight_permuations):
        '''
        Optimize voting parameters for (un)supervised models

        Parameters
        ----------
        weight_permuations : List
            list of weights to test
        '''
        
        tuning_res = pd.DataFrame(columns = ["w1", "w2", 'w3', "accuracy", "precision", "recall",\
                                             "f1_score", "time"])

        # for w1_, w2_, w3_ in itertools.product(*[w1, w2,w3]):
        for w1_, w2_, w3_ in weight_permuations:
            dbscan_clf = DBSCAN(eps=3, min_samples=2)
            if_clf=IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.12), \
                                    max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
            kmeans_clf = KMeans(n_clusters=2)
            KNN_clf = KNeighborsClassifier(n_neighbors=2)
            svm_clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            rf_clf = RandomForestClassifier(max_depth=10, random_state=5)      

            model = VotingClassifier(estimators=[
                    ('knn', KNN_clf), \
                    ('svm', svm_clf),('random forest', rf_clf)], voting='hard', weights=[w1_, w2_, w3_])
            self.validate_model(model)

            accuracy, precision, recall, f1_score,  time = \
                    self.validate_model(model)

            row = {"w1":w1_, "w2":w2_, 'w3':w3_,"accuracy":accuracy, "precision":precision,\
                   "recall":recall, "f1_score":f1_score, "time":time}
            tuning_res = tuning_res.append(row, ignore_index=True)
        return tuning_res.sort_values("f1_score", ascending=False)
    
    
    def weighted_vote_ensamble(self, models, weights=[1,1,1,1,1,1], undersample=False, oversample=False):
        '''
        Optimize voting parameters for all models

        Parameters
        ----------
        model : List
            list of models
        weights : List
            list of weights to test
        undersample : Boolean
            True if data needs to be undersampled
        oversample : Boolean
            True if data needs to be oversampled
        '''
        accuracy_, precision_, recall_, f1_score_,fpr_, tpr_, roc_auc_, time_ = [],[],[],[],[], [],[],[]

        shuffle_split = StratifiedShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
        for train_index, test_index in shuffle_split.split(self.X,self.Y):
            start_time = time.time()

            if type(self.X) == pd.DataFrame:
                X_train, X_test = self.X.iloc[train_index],self.X.iloc[test_index]
                Y_train, Y_test = self.Y.iloc[train_index], self.Y.iloc[test_index]
            else:
                X_train, X_test = self.X[train_index], self.X[test_index]
                Y_train, Y_test = self.Y[train_index], self.Y[test_index]
            if undersample:
                undersampling = RandomUnderSampler(sampling_strategy="majority")
                X_train, Y_train = undersampling.fit_resample(X_train, Y_train)

            if oversample:
                oversampling = RandomOverSampler(sampling_strategy="minority")
                X_train, Y_train = oversampling.fit_resample(X_train, Y_train)


            start_time = time.time()

            results = []
            for model, weight in zip(models, weights):
                try:
                    model.fit(X_train, Y_train)
                    for i in range(weight):
                        res = model.predict(X_test)
                        res = self.flip_if_inverted(res)
                        results.append(res)
                except:
                    for i in range(weight):
                        res =  model.fit_predict(X_test)
                        res[res == -1] = 1
                        res = self.flip_if_inverted(res)
                        results.append(res)

            results = np.array(results)      
            # find majority
            results=results.T

            final_res = []
            for i in list(results):
                final_res.append(int(Counter(i).most_common(1)[0][0]))
            end_time = time.time()

            accuracy, precision, recall, f1_score, fpr, tpr, roc_auc = self.result(Y_test, final_res)

            fpr_.append(fpr)
            tpr_.append(tpr)

            accuracy_.append(accuracy)
            precision_.append(precision)
            recall_.append(recall)
            f1_score_.append(f1_score)
            roc_auc_.append(roc_auc)
            time_.append(end_time - start_time)  

        return np.mean(accuracy_), np.mean(precision_), np.mean(recall_), np.mean(f1_score_), np.mean(time_)
    
    def optimize_all_voting_parameters(self, models, weight_permutations, undersample=False, oversample=False):
        '''
        Automate the optimzation for voting parameters for all models

        Parameters
        ----------
        model : List
            list of models
        weight_permutations : List
            list of weights to test
        undersample : Boolean
            True if data needs to be undersampled
        oversample : Boolean
            True if data needs to be oversampled
        '''

        tuning_res = pd.DataFrame(columns = ["w1", "w2", 'w3', 'w4', 'w5','w6', "accuracy", "precision", "recall",\
                                             "f1_score", "time"])

        for permutation in weight_permutations:
            start_time = time.time()
            w1_, w2_, w3_, w4_,w5_,w6_ = permutation

            accuracy, precision, recall, f1_score, time_ = \
                self.weighted_vote_ensamble(models, weights=list(permutation), undersample=undersample, oversample=oversample)


            end_time=time.time()
            row = {"w1":w1_, "w2":w2_, 'w3':w3_,'w4':w4_,'w5':w5_, 'w6':w6_,"accuracy":accuracy, "precision":precision,\
                   "recall":recall, "f1_score":f1_score, 'time':end_time-start_time}
            tuning_res = tuning_res.append(row, ignore_index=True)
        return tuning_res.sort_values("f1_score", ascending=False)


    def stack(self, models, undersample=False, oversample=False):
        '''
        Stacking algorithm

        Parameters
        ----------
        model : List
            list of models
        undersample : Boolean
            True if data needs to be undersampled
        oversample : Boolean
            True if data needs to be oversampled
        '''
        X_train, X_test, Y_train, Y_test = train_test_split(self.X,self.Y, test_size=0.4, random_state=22)

        if undersample:
            undersampling = RandomUnderSampler(sampling_strategy="majority")
            X_train, Y_train = undersampling.fit_resample(X_train, Y_train)

        if oversample:
            oversamping = RandomOverSampler(sampling_strategy="minority")
            X_train, Y_train = oversamping.fit_resample(X_train, Y_train)

        start_time = time.time()
        S_train, S_test = stacking(models,                   
                                    X_train, Y_train, X_test,   
                                   regression=False, 

                                    mode='oof_pred_bag', 

                                   needs_proba=False,

                                   save_dir=None, 

                                   metric=accuracy_score, 

                                   n_folds=4, 

                                   stratified=True,

                                   shuffle=True,  

                                   random_state=0,    

                                   verbose=2)

        model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
                              n_estimators=100, max_depth=3)
        model = model.fit(S_train, Y_train)

        y_pred = model.predict(S_test)

        accuracy, precision, recall, f1_score, fpr, tpr, roc_auc = self.result(Y_test, y_pred)

        end_time = time.time()

        return accuracy, precision, recall, f1_score, end_time - start_time

    def generate_models(self, weights, undersample=False, oversample=False):
        df = pd.DataFrame(columns=['model','accuracy', 'precision', 'recall', 'f1 score', 'time'])
        for model in self.all_models:
            accuracy, precision, recall, f1_score, time = self.validate_model(model, undersample=undersample, \
                                                                            oversample=oversample)
            df = df.append({'model':model,'accuracy': accuracy, 'precision':precision, 'recall':recall, \
                            'f1 score':f1_score, 'time':time}, ignore_index=True)
            
        #ensemble models
        for i,j in zip([self.unsupervised_models[1:], self.supervised_models, self.all_models[1:]],\
                            ['unsupervised stacking', 'supervised stacking', 'all model stacking']):
            accuracy, precision, recall, f1_score, time = self.stack(i, \
                                                                    undersample=undersample, \
                                                                    oversample=oversample)
            df = df.append({'model':j,'accuracy': accuracy, 'precision':precision, 'recall':recall, \
                                'f1 score':f1_score, 'time':time}, ignore_index=True)

        for i,j in zip([self.unsupervised_models[1:], self.supervised_models, self.all_models[1:]],\
                            ['unsupervised voting', 'supervised voting', 'all model voting']):
            accuracy, precision, recall, f1_score, time = self.weighted_vote_ensamble(models=i, weights=[1,1,1],\
                                                                                     oversample=oversample,\
                                                                                     undersample=undersample)
            df = df.append({'model':j,'accuracy': accuracy, 'precision':precision, 'recall':recall, \
                                'f1 score':f1_score, 'time':time}, ignore_index=True)
            
        for i,j,z in zip([self.unsupervised_models[1:], self.supervised_models, self.all_models[1:], weights],\
                            ['unsupervised weighted voting', 'supervised weighted voting', 'all model weighted voting'],\
                            weights):
            accuracy, precision, recall, f1_score, time = self.weighted_vote_ensamble(models=i, weights=z,\
                                                                                     oversample=oversample,\
                                                                                     undersample=undersample)
            df = df.append({'model':j,'accuracy': accuracy, 'precision':precision, 'recall':recall, \
                                'f1 score':f1_score, 'time':time}, ignore_index=True)

        return df.sort_values(by="f1 score", ascending=False)
