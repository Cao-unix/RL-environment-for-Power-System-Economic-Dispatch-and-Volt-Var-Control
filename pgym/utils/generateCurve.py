import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import time
from pgym import pf_cases

tt = np.array([0,  3, 5,  9,  11, 13, 15, 18, 20, 23, 24]) * 4
lp = np.array([20, 10, 20, 60, 70, 40, 35, 55, 30, 20, 15])
gp = np.array([1, 3, 5,  30, 50, 80, 70, 30, 2,  1, 1])

def gen(t):
    fl = interp.interpolate.interp1d(tt, lp + np.random.rand(len(tt)) * 20, kind='cubic')
    fg = interp.interpolate.interp1d(tt, gp + np.random.rand(len(tt)) * 20, kind='cubic')
    
    tlp = fl(t) + np.random.rand(len(t)) * 20
    tgp = fg(t) + np.random.rand(len(t)) * 20
    tlp = tlp / np.max(tlp)
    tgp = tgp / np.max(tgp)
    return tlp, tgp

# plot an example
def plotExample(t, data_dir):
    tlp, tgp = gen(t)

    plt.figure()
    plt.plot(t, tlp)
    plt.plot(t, tgp)
    plt.grid('both')
    plt.xlim([0, 95])
    plt.legend(['Load P', 'Generation P'])
    plt.xlabel('time / h')
    plt.ylabel('P / Pmax')
    plt.savefig(osp.join(data_dir, 'genLPLQ.png'), dpi=300)
    plt.savefig(osp.join(data_dir, 'genLPLQ.svg'))

# gen
def dump(load_len, gen_len, data_dir, seed=None, datestamp=False):
    dt = time.strftime("%Y-%m-%d_%H-%M-%S_") if datestamp else ''
    fname = f'genLPGP_s{seed}.txt' if seed is not None else 'genLPGP.txt'
    with open(osp.join(data_dir, dt + fname), 'w') as f:
        dlp = ",\n    ".join([f'[{",".join(gen(t)[0].astype(str))}]' for i in range(load_len)])
        dgp = ",\n    ".join([f'[{",".join(gen(t)[1].astype(str))}]' for i in range(gen_len)])
        print(f'[\n    {dlp}\n]', file=f)
        print('-' * 100, file=f)
        print(f'[\n    {dgp}\n]', file=f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--to', type=str, default='data')
    parser.add_argument('--case', type=str, default=None)
    parser.add_argument('--time', '-t', type=int, default=96)
    parser.add_argument('--loadlen', '-l', type=int, default=33)
    parser.add_argument('--genlen', '-g', type=int, default=5)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--datestamp', '-d', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    t = np.arange(0, args.time, 1)
    if args.plot:
        plotExample(t, args.to)
    if args.case is not None:
        if not hasattr(pf_cases, args.case):
            print('Case not found!')
            exit(1)
        ppc = getattr(pf_cases, args.case)()
        genlen = len(ppc['gen'])
        loadlen = len(ppc['bus'])
        dump(loadlen, genlen, args.to, args.seed, args.datestamp)
    else:
        dump(args.loadlen, args.genlen, args.to, args.seed, args.datestamp)
