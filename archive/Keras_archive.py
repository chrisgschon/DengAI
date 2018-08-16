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