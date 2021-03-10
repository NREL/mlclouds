"""
Make plots of the statistics
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rex import Resource

plot_dir = './plots'
feature_data_fp = '/projects/mlclouds/data_surfrad_9/2016_east_adj/mlclouds_surfrad_2016.h5'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

with Resource(feature_data_fp) as res:
    meta = res.meta
    surfrad_ids = meta['surfrad_id']

# extract data from all available stats files for only the xval site
df = None
for i, sid in enumerate(surfrad_ids):
    fp = './outputs/stats_k_fold_{}.csv'.format(i)
    if os.path.exists(fp):
        temp = pd.read_csv(fp, index_col=0)
        mask = (temp['Site'] == sid.upper())
        print('Taking data for "{}" from: {}'.format(sid, fp))
        assert any(mask)
        temp = temp[mask]
        if df is None:
            df = temp
        else:
            df = df.append(temp, ignore_index=True)

variables = df.Variable.unique()
conditions = df.Condition.unique()
metrics = ('MAE (%)', 'MBE (%)', 'RMSE (%)')
models = ('Baseline', 'PHYGNN')

assert all([m in df for m in metrics]), 'Could not find: {}'.format(metrics)
assert all([m in df.Model.unique() for m in models]), 'Could not find: {}'.format(models)

for var in variables:
    for condition in conditions:
        for metric in metrics:
            mask = ((df.Variable == var)
                    & (df.Condition == condition)
                    & df.Model.isin(models))
            df_plot = df[mask]
            sns.barplot(x='Site', y=metric, hue='Model', data=df_plot, ci=None)
            fname = 'kfold_stats_{}_{}_{}.png'.format(metric, var, condition)
            fname = fname.lower().replace(' (%)', '')
            fname = fname.replace('-', '_').replace(' ', '_')
            plt.title(fname.replace('.png', ''))
            fp = os.path.join(plot_dir, fname)
            plt.savefig(fp)
            print('Saved: {}'.format(fname))
            plt.close()
