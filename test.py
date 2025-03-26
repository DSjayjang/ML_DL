import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\user\Downloads\stroke data.csv (1)')

df = pd.read_csv('stroke data.csv')
print(df)


df.corr()

df.info()
df2 = df.drop(['gender', 'age', 'married', 'occupation', 'residence', 'smoking_status'], axis = 1)
df2_corr = df2.corr()


plt.figure(figsize = (12,8))
ax = sns.heatmap(df2_corr, annot = False, center = 0, cmap = 'RdBu', vmin=-1, vmax=1, cbar_kws={'shrink': 0.2, 'aspect': 20}, xticklabels=True)
colorbar = ax.collections[0].colorbar
colorbar.ax.set_title("corr", fontsize=12, pad = 10)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.show()