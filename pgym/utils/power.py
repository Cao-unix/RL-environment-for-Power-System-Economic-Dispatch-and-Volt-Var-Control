from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pgym.pf_cases.idx_unc import UNC_L_I, UNC_L_MIN, UNC_L_MAX, UNC_P_I, UNC_P_MIN, UNC_P_MAX
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
from matplotlib.cm import get_cmap

from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PQ, REF, BUS_I, VMAX, VMIN
from pypower.idx_brch import PF, PT, QF, QT, F_BUS, T_BUS, BR_R, BR_X, SHIFT, TAP, BR_STATUS
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS

from numpy import zeros, arange, exp, pi, ones, real, imag
from numpy import flatnonzero as find

def loss(ppc=None, baseMVA=1, bus=None, branch=None):
    """
    Power loss of the case
    
    The case should follow the pypower case format, M-file like.
    You can provide extra bus or branch information to substitude
    the information in the case.
    
    Keyword Arguments:
        ppc {dict of ndarray} -- pypower case (default: {None})
        baseMVA {int} -- base / MVA (default: {1})
        bus {ndarray} -- bus information (default: {None})
        branch {ndarray} -- branch information (default: {None})
    
    Return:
        Real power loss, Imag power loss
    """     
    if ppc is None and (bus is None or branch is None):
        return None, None
    branch = ppc['branch'] if branch is None else branch
    bus = ppc['bus'] if bus is None else bus

    nl = branch.shape[0]
    i2e = bus[:, BUS_I].astype(int)
    e2i = zeros(max(i2e) + 1, int)
    e2i[i2e] = arange(bus.shape[0])
    out = find(branch[:, BR_STATUS] == 0)
    nout = len(out)

    V = bus[:, VM] * exp(-1j * pi / 180 * bus[:, VA]) 
    tap = ones(nl)
    xfmr = find(branch[:, TAP])
    tap[xfmr] = branch[xfmr, TAP]
    tap = tap * exp(1j * pi / 180 * branch[:, SHIFT])
    loss = baseMVA * abs(V[e2i[ branch[:, F_BUS].astype(int) ]] / tap - 
                V[e2i[ branch[:, T_BUS].astype(int) ]])**2 / \
                (branch[:, BR_R] - 1j * branch[:, BR_X])
    loss[out] = zeros(nout)
    return sum(loss.real), sum(loss.imag)

def current(ppc=None, baseMVA=1, bus=None, branch=None):
    """
    Current of the case
    
    The case should follow the pypower case format, M-file like.
    You can provide extra bus or branch information to substitude
    the information in the case.
    
    Keyword Arguments:
        ppc {dict of ndarray} -- pypower case (default: {None})
        baseMVA {int} -- base / MVA (default: {1})
        bus {ndarray} -- bus information (default: {None})
        branch {ndarray} -- branch information (default: {None})
    
    Return:
        ndarray -- all I (real, imag)
    """    
    if ppc is None and (bus is None or branch is None):
        return None
    branch = ppc['branch'] if branch is None else branch
    bus = ppc['bus'] if bus is None else bus

    nl = branch.shape[0]
    i2e = bus[:, BUS_I].astype(int)
    e2i = zeros(max(i2e) + 1, int)
    e2i[i2e] = arange(bus.shape[0])
    out = find(branch[:, BR_STATUS] == 0)
    nout = len(out)

    V = bus[:, VM] * exp(-1j * pi / 180 * bus[:, VA]) 
    tap = ones(nl)
    xfmr = find(branch[:, TAP])
    tap[xfmr] = branch[xfmr, TAP]
    tap = tap * exp(1j * pi / 180 * branch[:, SHIFT])
    current = (V[e2i[ branch[:, F_BUS].astype(int) ]] / tap - 
                V[e2i[ branch[:, T_BUS].astype(int) ]]) / \
                (branch[:, BR_R] + 1j * branch[:, BR_X])
    current[out] = zeros(nout)
    return current

cm = get_cmap()
# interp_color = lambda i, N: np.array([[(i + 1)/N, (1 - (i + 1)/N), (((i + 1)/N)**2)]])
interp_color = lambda i, N: cm(0) if N == 1 else cm(i / (N - 1))

def drawNet(ppc, ctrl=True, ug=True, ub=True, showOff=False, showEid=False, value=None):    
    """
    Draw Power Network

    Draw the network on the current figure. Note that plt.figure(),
    plt.show(), plt.savefig()... should be managed manually, or you
    should use `showNet`.
    
    Arguments:
        ppc {dict of ndarray} -- Case from pypower, M-file like case
    
    Keyword Arguments:
        ctrl {bool} -- show if a bus has controlled gen (default: {True})
        ug {bool} -- show if a bus has uncertain gen (default: {True})
        ub {bool} -- show if a branch is uncertain (default: {True})
        showOff {bool} -- show the off branches (default: {False})
        value {string} -- show the powerflow: 'P', 'Q' or None (default: {None})
    """
    branch = ppc['branch']
    gen = ppc['gen']
    bus = ppc['bus']
    area = ppc['area'] if 'area' in ppc.keys() else None
    tobr = lambda br: (int(br[F_BUS]), int(br[T_BUS]))
    tobri = lambda br: (int(br[T_BUS]), int(br[F_BUS]))
    # control buses
    ctrl_set = set()
    if ctrl and 'controlled_gen' in ppc.keys() and len(ppc['controlled_gen']) > 0:
        ctrl_gen = ppc['controlled_gen'][:, 0].astype(int)
        ctrl_set = {int(gen[i, GEN_BUS]) for i in ctrl_gen}
    # uncertain buses
    ug_set = set()
    if ug and 'uncertain_gen' in ppc.keys() and len(ppc['uncertain_gen']) > 0:
        uncer_gen = ppc['uncertain_gen'][:, UNC_P_I].astype(int)
        ug_set = {int(gen[i, GEN_BUS]) for i in uncer_gen}
    # uncertain branches
    unbrch_set = set()
    if ub and 'uncertain_branch' in ppc.keys() and len(ppc['uncertain_branch']) > 0:
        unbrch_set = set(ppc['uncertain_branch'][:, UNC_L_I].astype(int))
    # all buses
    bus_set = set(bus[:, BUS_I].astype(int))
    brch_set = set(range(len(branch)))
    offbrch_set = set([k for k, brch in enumerate(branch) if brch[BR_STATUS] == 0])

    if len(branch[0]) > QT:
        brP = (branch[:, PF] - branch[:, PT]) / 2
        brQ = (branch[:, QF] - branch[:, QT]) / 2
    else:
        brP = np.ones(len(branch))
        brQ = np.ones(len(branch))
    brd = {'P': brP, 'Q': brQ}[value] if value is not None else np.ones(len(branch))

    maxV = bus[:, VM].max()
    minV = bus[:, VM].min()
    if maxV == minV:
        maxV = bus[:, VMAX].mean()
        minV = bus[:, VMIN].mean()

    G0 = nx.Graph()
    for i in brch_set:
        G0.add_edge(*tobr(branch[i]))
    pos = nx.kamada_kawai_layout(G0)

    G = nx.MultiDiGraph()
    for b in bus_set:
        style = {
            'alpha': 0.9,
            'node_size': 300,
            'node_color': 'k',
            'edgecolors': 'k',
            'linewidths': 1.0,
        }
        if b in ctrl_set:
            style['edgecolors'] = 'r'
            style['node_color'] = np.array([[0.2,0.2,0.2]])
            style['linewidths'] = 2
        if b in ug_set:
            style['edgecolors'] = 'b'
            style['node_color'] = np.array([[0.2,0.2,0.2]])
            style['linewidths'] = 2
        if area is not None:
            ncolor = np.zeros((1,4))
            narea = 0
            for i, v in enumerate(area):
                if b in v['bus']:
                    ncolor += interp_color(i, len(area))
                    narea += 1
            if narea > 0:
                style['node_color'] = ncolor / narea
        style['node_size'] = 200 + 300 * (bus[bus[:, BUS_I] == b, VM] - minV) / (maxV - minV)
        nx.draw_networkx_nodes(G, pos, nodelist=[b], **style)
    
    brdMean = abs(brd).mean()
    for e in brch_set:
        style = {
            'width': 1.2,
            'alpha': 0.9,
        }
        if value is not None:
            style['width'] = abs(brd[e]) / brdMean * 1.0 + 0.1
            style['edge_color'] = {
                'P': 'r',
                'Q': 'b',
            }[value]
        if e in unbrch_set:
            style['style'] = 'dashed'
            style['alpha'] = 0.5
        if e in offbrch_set:
            style['alpha'] = 0.1
            if showOff:
                nx.draw_networkx_edges(G, pos, [tobr(branch[e])], **style)
                nx.draw_networkx_edges(G, pos, [tobri(branch[e])], **style)
                if showEid:
                    nx.draw_networkx_edge_labels(G, pos, {tobri(branch[e]) : str(e)})
        elif value is not None and brd[e] < 0:
            nx.draw_networkx_edges(G, pos, [tobri(branch[e])], **style)
            if showEid:
                nx.draw_networkx_edge_labels(G, pos, {tobri(branch[e]) : str(e)})
        else:
            nx.draw_networkx_edges(G, pos, [tobr(branch[e])], **style)
            if showEid:
                nx.draw_networkx_edge_labels(G, pos, {tobr(branch[e]) : str(e)})

    nx.draw_networkx_labels(G, pos, labels={
        k:k for k in bus_set
    }, font_color='w')

    plt.axis('off')

def showNet(ppc, savefig=None, **drawKws):
    """
    Show Power Network
    
    Show the network on a new figure, or save it on the disk.
    
    Arguments:
        ppc {dict of ndarray} -- Case from pypower, M-file like case
    
    Keyword Arguments:
        savefig {string|dict|None} -- save configuration, (default: {None})
                `None`: show on screen,
                `dict`: includes `fname` key, the others will be passed to `plt.savefig`
                `string`: the filename

    """
    nbus = len(ppc['bus'])
    sz = max(6.4 * nbus / 25, 4.8)
    plt.figure(figsize=(sz, sz))
    drawNet(ppc, **drawKws)
    if isinstance(savefig, str):
        plt.savefig(savefig, dpi=300)
    elif isinstance(savefig, dict):
        fname = savefig.pop('fname')
        plt.savefig(fname, **savefig)
    elif isinstance(savefig, list):
        for sf in savefig:
            fname = sf.pop('fname')
            plt.savefig(fname, **sf)
    else:
        plt.show()

# interpolate to query
def interp_q(data, time):
    """
    Interpolate data to query the value at time
    
    Arguments:
        data {ndarray} -- time series data, i-th row with time long cols
        time {float} -- the time to query
    
    Returns:
        ndarray -- value at time with n rows
    """    
    #! warning: 96th point, which should be given in data, is from 95th point.
    n, rng = data.shape
    orgt = np.arange(rng)
    return np.array([np.interp(time, orgt, data[i, :]) for i in range(n)])

def getErrorCase(ad_case, seed=0):
    """
    Get the case with parameter errors
    
    Arguments:
        ad_case {dict of ndarray} -- pypower case
    
    Keyword Arguments:
        seed {int} -- np.random seed (default: {0})
    
    Returns:
        dict of ndarray -- pypower case with errors
    """    
    np.random.seed(seed)
    ppc = deepcopy(ad_case)
    unbri = ppc["uncertain_branch"][:, UNC_L_I].astype(int)
    unbrmin = ppc["uncertain_branch"][:, UNC_L_MIN]
    unbrmax = ppc["uncertain_branch"][:, UNC_L_MAX]
    # unbrrnd = np.random.uniform(unbrmin, unbrmax)
    unbrrnd = np.random.uniform(unbrmin * 0.5, unbrmax * 2)
    unbrrnd = np.clip(unbrrnd, unbrmin, unbrmax)
    ppc["branch"][unbri, BR_R] *= unbrrnd
    unbrrnd = np.random.uniform(unbrmin * 0.5, unbrmax * 2)
    unbrrnd = np.clip(unbrrnd, unbrmin, unbrmax)
    ppc["branch"][unbri, BR_X] *= unbrrnd
    return ppc
