import pandas as pd
import sklearn.preprocessing as preproc

# Load the online news popularity dataset
df = pd.read_csv('../../data/OnlineNewsPopularity.csv', delimiter=', ')

# Look at the original data - the number of words in an article
df['n_tokens_content'].as_matrix()

# Min-max scaling
df['minmax'] = preproc.minmax_scale(df[['n_tokens_content']])
df['minmax'].as_matrix()

# Standardization - note that by definition, some outputs will be negative
df['standardized'] = preproc.StandardScaler().fit_transform(df[['n_tokens_content']])
df['standardized'].as_matrix()

# L2-normalization
df['l2_normalized'] = preproc.normalize(df[['n_tokens_content']], axis=0)
df['l2_normalized'].as_matrix()

df['l1_normalized'] = preproc.normalize(df[['n_tokens_content']], norm='l1', axis=0)

print(df[['minmax', 'standardized', 'l2_normalized', 'l1_normalized']].head())

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)
fig.tight_layout(pad=0, w_pad=1.0, h_pad=2.0)
# fig.tight_layout()

df['n_tokens_content'].hist(ax=ax1, bins=100)
ax1.tick_params(labelsize=14)
ax1.set_xlabel('Article word count', fontsize=14)
# ax1.set_ylabel('Number of articles', fontsize=14)

df['minmax'].hist(ax=ax2, bins=100)
ax2.tick_params(labelsize=14)
ax2.set_xlabel('Min-max scaled word count', fontsize=14)
ax2.set_ylabel('Number of articles', fontsize=14)

df['standardized'].hist(ax=ax3, bins=100)
ax3.tick_params(labelsize=14)
ax3.set_xlabel('Standardized word count', fontsize=14)
# ax3.set_ylabel('Number of articles', fontsize=14)

df['l2_normalized'].hist(ax=ax4, bins=100)
ax4.tick_params(labelsize=14)
ax4.set_xlabel('L2-normalized word count', fontsize=14)
ax4.set_ylabel('Number of articles', fontsize=14)
plt.show()

# 交互特征
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preproc

# Select the content-based features as singleton features in the model,

features = ['n_tokens_title', 'n_tokens_content']

X = df[features]
y = df[['shares']]

X2 = preproc.PolynomialFeatures(include_bias=False).fit_transform(X)
print(X.shape, X2.shape)

X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X, X2, y, test_size=0.3, random_state=123)


def evaluate_feature(X_train, X_test, y_train, y_test):
    #   Fit a linear regression model on the training set and score on the test set
    model = linear_model.LinearRegression().fit(X_train, y_train)
    r_score = model.score(X_test, y_test)
    return model, r_score


(m1, r1) = evaluate_feature(X1_train, X1_test, y_train, y_test)
(m2, r2) = evaluate_feature(X2_train, X2_test, y_train, y_test)
print("R-squared score with singleton features: %0.5f" % r1)
print("R-squared score with pairwise features: %0.10f" % r2)

