"""
Make plots of the statistics
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plot_dir = './plots'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

west_sites = ['TBL', 'DRA', 'FPK', 'SRRL']
east_sites = ['BON', 'GWN', 'PSU', 'SXF', 'SGP']
df_east = pd.read_csv('./outputs/validation_stats_east.csv')
df_west = pd.read_csv('./outputs/validation_stats_west.csv')
df_east = df_east[df_east.Site.isin(east_sites)]
df_west = df_west[df_west.Site.isin(west_sites)]
df = pd.concat([df_east, df_west], ignore_index=True)
df = df.sort_values(['Site', 'Model'])

variables = df.Variable.unique()
conditions = df.Condition.unique()
metrics = ('MAE (%)', 'MBE (%)', 'RMSE (%)')
models = ('Baseline', 'MLClouds')

assert all(m in df for m in metrics), 'Could not find: {}'.format(metrics)
assert all(
    m in df.Model.unique() for m in models
), 'Could not find: {}'.format(models)

for var in variables:
    for condition in conditions:
        for metric in metrics:
            mask = (
                (df.Variable == var)
                & (df.Condition == condition)
                & df.Model.isin(models)
            )
            df_plot = df[mask]
            sns.barplot(
                x='Site', y=metric, hue='Model', data=df_plot, errorbar=None
            )
            fname = 'stats_{}_{}_{}.png'.format(metric, var, condition)
            fname = fname.lower().replace(' (%)', '')
            fname = fname.replace('-', '_').replace(' ', '_')
            plt.title(fname.replace('.png', ''))
            fp = os.path.join(plot_dir, fname)
            plt.savefig(fp)
            print('Saved: {}'.format(fname))
            plt.close()
