import os
import os.path as osp
from copy import deepcopy
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from pgym import pf_cases
from pypower.idx_bus import BUS_I, PD, QD, VM, VA, GS, BUS_TYPE, PQ, REF, VMAX, VMIN
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS, PMAX, PMIN

sns.set(style='white', font_scale=1.2)
sns.set_style({
    'font.family': 'Times New Roman',
    'axes.unicode_minus': False
})
cname = 'case33bw'
case = getattr(pf_cases, cname)()
data_dir = 'data/profiles'
os.makedirs(osp.join(data_dir, 'pdf'), 0o755, True)
os.makedirs(osp.join(data_dir, 'png'), 0o755, True)
os.makedirs(osp.join(data_dir, 'svg'), 0o755, True)

load = case['load_p'].T * case['bus'][:, PD]
load = load * case['baseMVA']
dfl = pd.DataFrame(load)
dfl['t'] = dfl.index
pdfl = dfl.melt(id_vars=['t'], value_name='MW', var_name='node')
gen = case['gen_p'].T * case['gen'][:, PG]
gen = gen * case['baseMVA']
dfg = pd.DataFrame(gen)
dfg['t'] = dfg.index
pdfg = dfg.melt(id_vars=['t'], value_name='MW', var_name='node')
pdfl['type'] = 'Load'
pdfg['type'] = 'Generation'

pdf = pd.concat([pdfl, pdfg], ignore_index=True)
graph = sns.lineplot(
    data=pdf, x='t', y='MW', hue='type',
    errorbar=('ci', 100)
)
graph.set_xlim(pdf['t'].min(), pdf['t'].max())

# graph = sns.lineplot(data=pdfl, x='t', y='MW/bus', hue='type', ci=100)
# graph.set_xlim(pdfl['t'].min(), pdfl['t'].max())
# ax2 = plt.twinx()
# graph2 = sns.lineplot(data=pdfg, x='t', y='MW/bus', hue='type', ci=100, ax=ax2)

plt.savefig(osp.join(data_dir, 'png', f'{cname}.png'), dpi=300)
plt.savefig(osp.join(data_dir, 'svg', f'{cname}.svg'), dpi=300)
plt.savefig(osp.join(data_dir, 'pdf', f'{cname}.pdf'), dpi=300)
