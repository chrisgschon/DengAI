fig, axes = plt.subplots(figsize = (10,8));
#axes.xaxis.set_tick_params(labelsize=20)
#axes.yaxis.set_tick_params(labelsize=20)
plt.xlabel('Null proportion')
pd.DataFrame((len(test) - test.count())/len(test)).plot.barh(ax = axes);
axes.legend_.remove()