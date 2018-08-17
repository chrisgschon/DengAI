#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0, 1))
#X_tr_sj = scaler.fit_transform(X_tr_sj)

# split into train and test sets
values = X_tr_sj_wide.values
n_train = values.shape[0] - 50
train = values[:n_train, :]
val = values[n_train:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
val_X, val_y = val[:, :-1], val[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
val_X = test_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
print(train_X.shape, train_y.shape, val_X.shape, val_y.shape)

# design network
model = Sequential()
model.add(LSTM(25, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=1000, batch_size=72, validation_data=(val_X, val_y), verbose=2, shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

from sklearn.model_selection import train_test_split
X_tr_sj, X_val_sj  = sj_feature_train.drop('total_cases', axis = 1).iloc[:-val_len,:], sj_feature_train.drop('total_cases', axis = 1).iloc[-val_len:,:]

Y_tr_sj, Y_val_sj = sj_feature_train['total_cases'].iloc[:-val_len], sj_feature_train['total_cases'].iloc[-val_len:]

X_test_sj = sj_feature_test 

X_tr_iq, X_val_iq, = iq_feature_train.drop('total_cases', axis = 1).iloc[:-val_len,:], iq_feature_train.drop('total_cases', axis = 1).iloc[-val_len:,:]

Y_tr_iq, Y_val_iq = iq_feature_train['total_cases'].iloc[:-val_len], iq_feature_train['total_cases'].iloc[-val_len:]

X_test_iq = iq_feature_test

# Reshape with lookback features - and output data to features folder

Set lookback

n_lookback = 2

## Test SJ

X_test_sj_wide = sj_feature_train.drop('total_cases', axis = 1).iloc[-n_lookback:,:].append(sj_feature_test.drop(['city','weekofyear', 'year'], axis = 1)).reset_index(drop = True)
X_test_sj_wide = series_to_supervised(X_test_sj_wide, n_lookback, 1)
X_test_sj_wide.drop(['var1(t)', 'var2(t)', 'var3(t)', 'var4(t)', 'var5(t)'], axis = 1, inplace = True)
features_out = pd.DataFrame(X_test_sj_wide).reset_index(drop = True)
features_out['city'] = 'sj'
features_out['year'] = sj_feature_test['year']
features_out['weekofyear'] = sj_feature_test['weekofyear']
features_out.to_csv(feature_path + '\\sj_test_Lookback' + str(n_lookback) + '.csv')
X_test_sj_wide = X_test_sj_wide.values.reshape((X_test_sj_wide.shape[0], 1, X_test_sj_wide.shape[1]))
X_test_sj_wide.shape

## Train SJ

X_tr_sj_wide = series_to_supervised(sj_feature_train, n_lookback, 1)
X_tr_sj_wide.drop(['var1(t)', 'var2(t)', 'var3(t)', 'var4(t)', 'var5(t)'], axis = 1, inplace = True)
X_tr_sj_wide['total_cases'] = sj_feature_train['total_cases']
X_tr_sj_wide.to_csv(feature_path + '\\sj_train_Lookback_w_cases' + str(n_lookback) + '.csv')
X_tr_sj_wide.shape

## Test IQ

X_test_iq_wide = iq_feature_train.drop('total_cases', axis = 1).iloc[-n_lookback:,:].append(iq_feature_test.drop(['city','weekofyear', 'year'], axis = 1)).reset_index(drop = True)
X_test_iq_wide = series_to_supervised(X_test_iq_wide, n_lookback, 1)
X_test_iq_wide.drop(['var1(t)', 'var2(t)', 'var3(t)', 'var4(t)', 'var5(t)'], axis = 1, inplace = True)
features_out = pd.DataFrame(X_test_iq_wide).reset_index(drop = True)
features_out['city'] = 'iq'
features_out['year'] = iq_feature_test['year']
features_out['weekofyear'] = iq_feature_test['weekofyear']
features_out.to_csv(feature_path + '\\iq_test_Lookback' + str(n_lookback) + '.csv')
X_test_iq_wide = X_test_iq_wide.values.reshape((X_test_iq_wide.shape[0], 1, X_test_iq_wide.shape[1]))
X_test_iq_wide.shape

## Train IQ

X_tr_iq_wide = series_to_supervised(iq_feature_train, n_lookback, 1)
X_tr_iq_wide.drop(['var1(t)', 'var2(t)', 'var3(t)', 'var4(t)', 'var5(t)'], axis = 1, inplace = True)
X_tr_iq_wide['total_cases'] = iq_feature_train['total_cases']
X_tr_iq_wide.to_csv(feature_path + '\\iq_train_Lookback_w_cases' + str(n_lookback) + '.csv')
X_tr_iq_wide.shape