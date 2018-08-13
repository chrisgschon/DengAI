pd.date_range(np.min(test['week_start_date']), np.max(max['week_start_date']))


import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
model = sm.tsa.AR(np.asarray(train_sj['total_cases'].map(lambda x: float(x))), freq = 'W').fit()

weeks_forecast = len(test_sj)

forecast = model.predict(weeks_forecast)

from datetime import timedelta
forecast_dates = test_sj['week_start_date']

fig = plt.figure(figsize=(20,10));
plt.plot(train_sj['week_start_date'], train_sj['total_cases'], c = 'green', label = 'True');
plt.plot(train_sj['week_start_date'], model.fittedvalues, c = 'red', label = 'Fitted');
plt.plot(forecast_dates, forecast, c= 'blue', label = 'Forecast');
plt.fill_between(forecast_dates, conf_int[:,0], conf_int[:,1],
    alpha=0.5, edgecolor='#1B2ACC', facecolor='#089FFF')
plt.legend();

pre_process = Pipeline([("imputer", Imputer(strategy="mean")), ("scaler", RobustScaler())])
sj_prefit = pre_process.fit(train_sj[best_raw_features])
iq_prefit = pre_process.fit(train_iq[best_raw_features])

best_raw_features = ['reanalysis_dew_point_temp_k', 'reanalysis_min_air_temp_k', 'reanalysis_max_air_temp_k','reanalysis_specific_humidity_g_per_kg',
                    'station_min_temp_c', 'reanalysis_tdtr_k', 'station_avg_temp_c']
raw_features = numeric_cols

from SKL_search import *
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import PolynomialFeatures

num_pca = 3
num_feature_pipeline = Pipeline([("imputer", Imputer(strategy="mean")),
                    ('features', 
                    FeatureUnion([('pca', PCAFeatureSelector(k = num_pca)), ('poly', PolynomialFeatures(interaction_only = False, include_bias = True))])), 
                                 ("scaler", RobustScaler())])

num_sj_pipe_fit = num_feature_pipeline.fit(train_sj[numeric_cols], train_sj['total_cases'])
num_iq_pipe_fit = num_feature_pipeline.fit(train_iq[numeric_cols], train_iq['total_cases'])

num_sj_train = num_feature_pipeline.fit_transform(train_sj[numeric_cols], train_sj['total_cases'])
num_iq_train = num_feature_pipeline.fit_transform(train_iq[numeric_cols], train_iq['total_cases'])

num_sj_test = num_sj_pipe_fit.transform(test_sj[numeric_cols])
num_iq_test = num_iq_pipe_fit.transform(test_iq[numeric_cols])

# Add month and quarter one hots

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
cats_sj_train = enc.fit_transform(train_sj[categorical_features]).toarray()
cats_sj_test = enc.fit_transform(test_sj[categorical_features]).toarray()
cats_iq_train = enc.fit_transform(train_iq[categorical_features]).toarray()
cats_iq_test = enc.fit_transform(test_iq[categorical_features]).toarray()

from scipy import stats

sj_feature_train = pd.DataFrame(X_sj)
sj_feature_train['weekofyear'] = train_sj['weekofyear']
sj_feature_train['poly_fit'] = train[train['city'] == 'sj']['poly_fit']
sj_feature_train['total_cases'] = train_sj['total_cases']
#for f in best_raw_features:
   # sj_feature_train[f] = pre_process.fit_transform(pd.DataFrame(train_sj[f]))
    
    corr = sj_feature_train.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize = (15,15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(data = corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});



from sklearn.ensemble import RandomForestRegressor
mod = RandomForestRegressor()

X = sj_feature_train.drop(['total_cases'], axis = 1).dropna()
y = sj_feature_train.dropna()['total_cases']
mod.fit(X,y)

importance = mod.feature_importances_
importance = pd.DataFrame(importance, index=X.columns, 
                          columns=["Importance"])

importance["Std"] = np.std([tree.feature_importances_
                            for tree in mod.estimators_], axis=0)

importance = importance.sort_values(by = 'Importance')

x = range(importance.shape[0])
y = importance.ix[:, 0]
yerr = importance.ix[:, 1]

fig, ax = plt.subplots(figsize = (20,10));
plt.bar(x, y, align="center");
ax.set_xticks(np.arange(len(importance)));
ax.set_xticklabels(importance.index.tolist());
plt.xticks(rotation = 85);
plt.title('Feature Importance for City sj');

importance.sort_values(by = 'Importance', ascending = False).cumsum()[:20]