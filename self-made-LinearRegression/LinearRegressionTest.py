import numpy as np
import pandas as pd
from LinearRegression import LinearRegression

adver_data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv')
X = adver_data.values[:, :3]
y = adver_data.values[:, -1:]

# fit with analytical solution
lm = LinearRegression(normalize=True)
lm.fit(X, y, fit_type='normal')
print('analytical solution predict MSE = ', lm.score(X, y))

# fit with sgd
lm.fit(X, y, max_iter=1e5, verbose=False)
print('SGD solution Last Error = ', lm.errors[-1])
print('SGD solution predict MSE = ', lm.score(X, y))
