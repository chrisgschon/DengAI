from sklearn.feature_selection import RFECV
estimator = SVR(kernel="linear")
sj_selector = RFECV(estimator, step=1, cv=2, n_jobs = -1)
sj_selector = selector.fit(X_tr_sj, Y_tr_sj)
sj_selector.support_ 

selector.ranking_

from sklearn.feature_selection import RFECV
estimator = SVR(kernel="linear")
iq_selector = RFECV(estimator, step=1, cv=2, n_jobs = -1, verbose = 1)
iq_selector = selector.fit(X_tr_iq, Y_tr_iq)
iq_selector.support_ 


from sklearn.feature_selection import f_regression
sj_selector = f_regression(X_tr_sj, Y_tr_sj)
sj_p_val_pass = 1e-1

X_tr_sj_sel = X_tr_sj.loc[:,sj_selector[1]<sj_p_val_pass]
X_tr_sj_sel.shape

svrmodels = { 
    'SVR': SVR()
}

svrparams = {  
    'SVR': [
        {'kernel': ['linear'], 'C': [0.05, 0.01], 'degree':[1], 'epsilon':[0.2, 0.5, 1]}]
    
}

svr_sj_helper = EstimatorSelectionHelper(svrmodels, svrparams)
svr_sj_helper.fit(X_tr_sj_sel, Y_tr_sj, scoring='neg_mean_absolute_error', cv = 2)

svr_sj_helper.score_summary(sort_by='mean_score')

# iq feature selector

from sklearn.feature_selection import f_regression
iq_selector = f_regression(X_tr_iq, Y_tr_iq)
iq_p_val_pass = 1e-1

X_tr_iq_sel = X_tr_iq.loc[:,iq_selector[1]<p_val_pass]
X_tr_iq_sel.shape

svr_iq_helper = EstimatorSelectionHelper(svrmodels, svrparams)
svr_iq_helper.fit(X_tr_iq_sel, Y_tr_iq, scoring='neg_mean_absolute_error', cv = 2)

svr_iq_helper.score_summary(sort_by='min_score')


importance = sj_model.feature_importances_
importance = pd.DataFrame(importance, index=sj_feature_train.drop('total_cases', axis = 1).columns, 
                          columns=["Importance"])
importance.sort_values(by = 'Importance', ascending = False)