def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

import os
import numpy as np
import pandas as pd

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.feature_selection import SelectKBest, SelectFromModel, VarianceThreshold, f_classif, chi2
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.tree import DecisionTreeClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler

# Create a class to select numerical or categorical columns 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

from sklearn.base import BaseEstimator, TransformerMixin

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

#Select top k Principal Components for feature
class PCAFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k):
        self.k = k
    def fit(self, X, y=None):
        self.pca = TruncatedSVD(n_components = self.k)
        self.pca.fit(X)
        return self
    def transform(self, X):
        return self.pca.transform(X)

#Use SKLearn SelectKBest for feature selection
class KBestFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k, scorefunc):
        self.k = k
        self.scorefunc = scorefunc
    def fit(self, X, y):
        self.kbest = SelectKBest(score_func = self.scorefunc,k=self.k)
        self.kbest.fit(X, y)
        return self
    def transform(self, X):
        return self.kbest.transform(X)
    
#Select from an SKLearn built in model
class FromModelFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
    def fit(self, X, y):
        self.fromModel = SelectFromModel(estimator=self.model, threshold = self.threshold)
        self.fromModel.fit(X, y)
        return self
    def transform(self, X):
        return self.fromModel.transform(X)


#Class for hyperparameter searches across models
class EstimatorSelectionHelper:
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
    
    def fit(self, X, y, cv=2, n_jobs=-1, verbose=5, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, 
                              verbose=verbose, scoring=scoring, refit=refit)
            gs.fit(X,y)
            self.grid_searches[key] = gs    
    
    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            dict1 = d.copy()
            dict1.update(params)
            return pd.Series(dict1)
                      
        rows = [row(k, gsc.cv_validation_scores, gsc.parameters) 
                     for k in self.keys
                     for gsc in self.grid_searches[k].grid_scores_]
        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)
        
        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        
        return df[columns]