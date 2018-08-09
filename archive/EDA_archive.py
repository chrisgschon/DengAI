fig, axes = plt.subplots(figsize = (10,8));
#axes.xaxis.set_tick_params(labelsize=20)
#axes.yaxis.set_tick_params(labelsize=20)
plt.xlabel('Null proportion')
pd.DataFrame((len(test) - test.count())/len(test)).plot.barh(ax = axes);
axes.legend_.remove()

grid = sns.FacetGrid(combined, row = 'dataset', col = 'city', size = 4)
grid = grid.map(plt.hist, "ndvi_nw")

fig, ax = plt.subplots(figsize = (10,5))
train[train['city'] == 'sj'][['week_start_date', 'ndvi_ne']].set_index(pd.DatetimeIndex(train[train['city'] == 'sj']['week_start_date'])).groupby(pd.Grouper(freq='Q')).mean().plot(ax = ax, label = 'Train SJ')
test[test['city'] == 'sj'][['week_start_date', 'ndvi_ne']].set_index(pd.DatetimeIndex(test[test['city'] == 'sj']['week_start_date'])).groupby(pd.Grouper(freq='Q')).mean().plot(ax = ax, label = 'Test SJ')
#ax.legend_.remove()
ax.legend(labels=['Train', 'Test'])


#mistaken for loop code
fix, ax = plt.subplots(nrows = 20, ncols = 2, figsize = (30,50))
cities = ['sj', 'iq']
for i, col in enumerate(numeric_cols):
    city = cities[np.mod(i,2)]
    axes = ax[int(np.floor(i/2)),np.mod(i,2)]
    sns.distplot(combined[col][(combined['dataset'] == 'train') & (combined['city'] == city)].dropna(), kde = False, label = 'Train', norm_hist = True, ax = axes)
    sns.distplot(combined[col][(combined['dataset'] == 'test') & (combined['city'] == city)].dropna(), kde = False, label = 'Test', norm_hist = True, ax = axes)
    axes.legend(loc = 'upper right');
    axes.set_title('City = '+  city + ' | Feature = ' + col)
    plt.tight_layout()