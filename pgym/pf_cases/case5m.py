# CASE5M
from numpy import array

def case():

    ## PYPOWER Case Format : Version 2
    ppc = {"version": "2"}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 1

    ## bus data
    # bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin
    ppc["bus"] = array([
        [1, 3, 0, 0, 0, 0, 1, 1, 0, 230, 1, 1.1, 0.9],
        [2, 1, 3, 0.9861, 0, 0, 1, 1.19894254, 16.0131883, 230, 1, 1.1, 0.9],
        [3, 1, 3, 0.9861, 0, 0, 1, 1.18694969, 14.7514859, 230, 1, 1.1, 0.9],
        [4, 1, 1.5, 1.3147, 0, 0, 1, 0.952287418, -2.50392167, 230, 1, 1.1, 0.9],
        [5, 1, 3, 0, 0, 0, 1, 0.997890825, -1.10247216, 230, 1, 1.1, 0.9],
    ])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2, Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = array([
        [1, -2.21873238, 5.21732694, 10, -10, 1, 100, 1, 0.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 15, 3, 5, -4.5, 1, 100, 1, 1.7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax, Pf, Qf, Pt, Qt
    ppc["branch"] = array([
        [1, 2, 0.0381, 0.0281, 0, 400, 400, 400, 0, 0, 1, -360, 360, -6.7379, 3.7114, 8.9924, -2.0486],
        [1, 4, 0.00304, 0.0304, 0, 0, 0, 0, 0, 0, 1, -360, 360, 1.5133, 1.4481, -1.5000, -1.3147],
        [1, 5, 0.00064, 0.0064, 0, 0, 0, 0, 0, 0, 1, -360, 360, 3.0058, 0.0578, -3.0000, -0.0000],
        [2, 3, 0.00108, 0.0108, 0, 0, 0, 0, 0, 0, 1, -360, 360, 3.0076, 1.0625, -3.0000, -0.9861],
    ])


def pf_case():
    ppc = case()
    # gen_id
    ppc["controlled_gen"] = array([
        [1],
    ])
    return ppc
