from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# df = pd.read_csv('../../data/OnlineNewsPopularity.csv', delimiter=', ')
# print(df.head(5))
#
# df['log_n_tokens_content'] = np.log10(df['n_tokens_content'] + 1)
#
# fig, (ax1, ax2) = plt.subplots(2,1)
# fig.tight_layout(pad=0, w_pad=4.0, h_pad=4.0)
# df['n_tokens_content'].hist(ax=ax1, bins=100)
# ax1.tick_params(labelsize=14)
# ax1.set_xlabel('Number of Words in Article', fontsize=14)
# ax1.set_ylabel('Number of Articles', fontsize=14)
#
# df['log_n_tokens_content'].hist(ax=ax2, bins=100)
# ax2.tick_params(labelsize=14)
# ax2.set_xlabel('Log of Number of Words', fontsize=14)
# ax2.set_ylabel('Number of Articles', fontsize=14)
# plt.show()


biz_file = open('../../data/yelp_academic_dataset_business.json')
biz_df = pd.DataFrame([json.loads(x) for x in biz_file.readlines()])
biz_file.close()

print(biz_df.head())