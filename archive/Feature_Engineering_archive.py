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