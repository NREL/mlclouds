"""
Make plots of the statistics
"""
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rex import Resource

plot_dir = './plots'
feature_data_fp = '/projects/pxs/mlclouds/training_data/2016_east_v321/mlclouds_surfrad_east_2016.h5'
west_sites = ['TBL', 'DRA', 'FPK', 'SRRL']
east_sites = ['BON', 'GWN', 'PSU', 'SXF', 'SGP']

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

with Resource(feature_data_fp) as res:
    meta = res.meta
    surfrad_ids = meta['surfrad_id']

# extract data from all available stats files for only the xval site
df = None
for i, sid in enumerate(surfrad_ids):
    if sid.upper() in east_sites:
        fp = './outputs/validation_stats_east_{}.csv'.format(i)
    else:
        fp = './outputs/validation_stats_west_{}.csv'.format(i)
    if os.path.exists(fp):
        temp = pd.read_csv(fp, index_col=0)
        mask = (temp['Site'] == sid.upper())
        print('Taking data for "{}" from: {}'.format(sid, fp))
        assert any(mask)
        temp = temp[mask]
        if df is None:
            df = temp
        else:
            df = pd.concat([df, temp], ignore_index=True)

variables = df.Variable.unique()
conditions = df.Condition.unique()
metrics = ('MAE (%)', 'MBE (%)', 'RMSE (%)')
models = ('Baseline', 'MLClouds')

assert all([m in df for m in metrics]), 'Could not find: {}'.format(metrics)
assert all([m in df.Model.unique() for m in models]), 'Could not find: {}'.format(models)

for var in variables:
    for condition in conditions:
        for metric in metrics:
            mask = ((df.Variable == var)
                    & (df.Condition == condition)
                    & df.Model.isin(models))
            df_plot = df[mask]
            sns.barplot(x='Site', y=metric, hue='Model', data=df_plot,
                        errorbar=None)
            fname = 'kfold_stats_{}_{}_{}.png'.format(metric, var, condition)
            fname = fname.lower().replace(' (%)', '')
            fname = fname.replace('-', '_').replace(' ', '_')
            plt.title(fname.replace('.png', ''))
            fp = os.path.join(plot_dir, fname)
            plt.savefig(fp)
            print('Saved: {}'.format(fname))
            plt.close()
