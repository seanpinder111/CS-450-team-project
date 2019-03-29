import numpy as np
import pandas as pd
from statsmodels import robust

data = pd.read_csv('all_time_normal.csv', 
                   index_col=False,
                   skipinitialspace=True,
                   na_values="nan"
                   )
data.fillna(0, inplace=True)

data_str = data[['Season', 'Lg', 'Tm']]
data = data.drop(data.columns[[0,1,2]], axis=1)

# discretize the data
def discretize(X, n_scale=1):

    for c in X.columns:
        loc = X[c].median()

        # median absolute deviation of the column
        scale = robust.mad(X[c])

        bins = [-np.inf, loc - (scale * n_scale),
                loc + (scale * n_scale), np.inf]
        X[c] = pd.cut(X[c], bins, labels=[-1, 0, 1])

    return X

# bin the data
binned_data = discretize(data)

print (binned_data)