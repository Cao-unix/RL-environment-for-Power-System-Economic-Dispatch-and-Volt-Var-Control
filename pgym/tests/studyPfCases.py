import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import networkx as nx
from numpy import array
import numpy as np
from pgym.pf_cases import case33bw, case5bw
from pgym.utils.power import loss, getErrorCase, current, drawNet, showNet
from pgym.pf_cases.idx_unc import UNC_L_I, UNC_L_MIN, UNC_L_MAX, UNC_P_I, UNC_P_MIN, UNC_P_MAX
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PQ, REF, BUS_I, VMAX, VMIN
from pypower.idx_brch import PF, PT, QF, QT, F_BUS, T_BUS, BR_R, BR_X, SHIFT, TAP, BR_STATUS
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS
from pypower.api import runpf, ppoption, printpf
ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0)

ppc = case5bw()
results, success = runpf(ppc, ppopt)
printpf(results)
# print(current(results))

showNet(ppc, savefig={
    'fname': 'data/case5bw.png',
    'dpi': 300,
    'transparent': True,
})
showNet(results, savefig={
    'fname': 'data/case5bw_P.png',
    'dpi': 300,
    'transparent': True,
}, value='P')
showNet(results, savefig={
    'fname': 'data/case5bw_Q.png',
    'dpi': 300,
    'transparent': True,
}, value='Q')
drawNet(results, value='P')
plt.savefig('data/case5bw_PQ.png', transparent=True, dpi=300)

showNet(ppc, savefig={
    'fname': 'data/case5bw.svg',
    'dpi': 300,
    'transparent': True,
})
showNet(results, savefig={
    'fname': 'data/case5bw_P.svg',
    'dpi': 300,
    'transparent': True,
}, value='P')
showNet(results, savefig={
    'fname': 'data/case5bw_Q.svg',
    'dpi': 300,
    'transparent': True,
}, value='Q')
drawNet(results, value='P')
plt.savefig('data/case5bw_PQ.svg', transparent=True, dpi=300)

ppc = case33bw()
## bus data
# bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin
ppc["bus"] = array([
    [1, 3, 0, 0, 0, 0, 1, 1, 0, 12.66, 1, 1, 1],
    [2, 1, 0.1, 0.06, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [3, 1, 0.09, 0.04, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [4, 1, 0.12, 0.08, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [5, 1, 0.06, 0.03, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [6, 1, 0.06, 0.02, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [7, 1, 0.2, 0.1, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [8, 1, 0.2, 0.2, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [9, 1, 0.06, 0.02, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [10, 1, 0.06, 0.02, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [11, 1, 0.045, 0.03, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [12, 1, 0.06, 0.035, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [13, 1, 0.06, 0.035, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [14, 1, 0.12, 0.08, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [15, 1, 0.06, 0.01, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [16, 1, 0.06, 0.02, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [17, 1, 0.06, 0.02, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [18, 1, 0.09, 0.04, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [19, 1, 0.09, 0.04, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [20, 1, 0.09, 2.34, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [21, 1, 0.09, 0.04, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [22, 1, 0.09, 0.04, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [23, 1, 0.09, 0.15, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [24, 1, 0.42, 0.2, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [25, 1, 0.42, 0.2, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [26, 1, 0.06, 0.125, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [27, 1, 0.06, 0.025, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [28, 1, 0.06, 0.02, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [29, 1, 0.12, 0.07, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [30, 1, 0.2, 0.6, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [31, 1, 0.15, 0.07, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [32, 1, 0.21, 0.1, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
    [33, 1, 0.06, 0.04, 0, 0, 1, 1, 0, 12.66, 1, 1.1, 0.9],
])
ppc["gen"] = array([
    [1, 0, 0, 10, -10, 1, 100, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [22, 0.6, -1.6, 4, -4, 1, 100, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [25, 0.6, 0.6, 4, -4, 1, 100, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [18, 0.6, 0.6, 4, -4, 1, 100, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [33, 0.6, 0.6, 4, -4, 1, 100, 1, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])
ppc["bus"][:, VMAX] = 1.05
ppc["bus"][:, VMIN] = 0.95
ppc["bus"][:, PD] *= 1.2
ppc["bus"][:, QD] *= -1.2
results, success = runpf(ppc, ppopt)
printpf(results)
# print(current(results))

showNet(ppc, savefig={
    'fname': 'data/case33bw.png',
    'dpi': 300,
    'transparent': True,
})
showNet(results, savefig={
    'fname': 'data/case33bw_P.png',
    'dpi': 300,
    'transparent': True,
}, value='P')
showNet(results, savefig={
    'fname': 'data/case33bw_Q.png',
    'dpi': 300,
    'transparent': True,
}, value='Q')
drawNet(results, value='P')
plt.savefig('data/case33bw_PQ.png', transparent=True, dpi=300)

showNet(ppc, savefig={
    'fname': 'data/case33bw.svg',
    'dpi': 300,
    'transparent': True,
})
showNet(results, savefig={
    'fname': 'data/case33bw_P.svg',
    'dpi': 300,
    'transparent': True,
}, value='P')
showNet(results, savefig={
    'fname': 'data/case33bw_Q.svg',
    'dpi': 300,
    'transparent': True,
}, value='Q')
drawNet(results, value='P')
plt.savefig('data/case33bw_PQ.svg', transparent=True, dpi=300)

pLoss, qLoss = loss(results)
print(f'Active power loss: {pLoss:.4} MW')

exit(0)

ppc["uncertain_branch"] = array([
        [0,     0.95,   1.05],
        [1,     0.95,   1.05],
        [2,     0.95,   1.05],
        [3,     0.95,   1.05],
        [4,     0.95,   1.05],
        [5,     0.95,   1.05],
        [6,     0.95,   1.05],
        [7,     0.95,   1.05],
        [8,     0.95,   1.05],
        [9,     0.95,   1.05],
        [10,     0.95,   1.05],
        [11,     0.95,   1.05],
        [12,     0.95,   1.05],
        [13,     0.95,   1.05],
        [14,     0.95,   1.05],
        [15,     0.95,   1.05],
        [16,     0.95,   1.05],
        [17,     0.95,   1.05],
        [18,     0.95,   1.05],
        [19,     0.95,   1.05],
        [20,     0.95,   1.05],
        [21,     0.95,   1.05],
        [22,     0.95,   1.05],
        [23,     0.95,   1.05],
        [24,     0.95,   1.05],
        [25,     0.95,   1.05],
        [26,     0.95,   1.05],
        [27,     0.95,   1.05],
        [28,     0.95,   1.05],
        [29,     0.95,   1.05],
        [30,     0.95,   1.05],
        [31,     0.95,   1.05],
        [32,     0.95,   1.05],
        [33,     0.95,   1.05],
        [34,     0.95,   1.05],
        [35,     0.95,   1.05],
        [36,     0.95,   1.05],
    ])

unbri = ppc["uncertain_branch"][:, UNC_L_I].astype(int)
unbrmin = ppc["uncertain_branch"][:, UNC_L_MIN]
unbrmax = ppc["uncertain_branch"][:, UNC_L_MAX]
ppc["branch"][unbri, BR_R] *= unbrmax
ppc["branch"][unbri, BR_X] *= unbrmax
results, success = runpf(ppc, ppopt)
# printpf(results)

pLoss, qLoss = loss(results)
print(f'Active power loss: {pLoss:.4} MW')
