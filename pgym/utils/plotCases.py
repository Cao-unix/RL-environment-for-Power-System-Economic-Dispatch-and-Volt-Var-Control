import os.path as osp
import os
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
import sys
sys.path.insert(0, r"H:\caoshujun\homework\pgym")  # 这一层里面要有 pgym 目录
from pgym import pf_cases
from pgym.utils.power import showNet

data_dir = 'data/cases'
os.makedirs(osp.join(data_dir, 'pdf'), 0o755, True)
os.makedirs(osp.join(data_dir, 'png'), 0o755, True)
os.makedirs(osp.join(data_dir, 'svg'), 0o755, True)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('cases', type=str, nargs='*', help='cases to plot')
    parser.add_argument('--all', '-a', action='store_true', help='plot all default cases')
    args = parser.parse_args()
    cases = args.cases if not args.all else [
        'case5bw',
        'case9',
        'case18',
        'case22',
        'case30',
        'case33bw',
        'ma_case33bw',
        'case69',
        'case85',
        'case141',
    ]
    for c in cases:
        print(f'Generating {c} image files...')
        if not hasattr(pf_cases, c):
            print(f'{c} not found!')
            continue
        case = getattr(pf_cases, c)
        ppc = case()
        showNet(ppc, savefig=[{
                'fname': osp.join(data_dir, 'pdf', f'{c}.pdf'),
                'dpi': 300,
                'transparent': True,
            }, {
                'fname': osp.join(data_dir, 'png', f'{c}.png'),
                'dpi': 300,
                'transparent': True,
            }, {
                'fname': osp.join(data_dir, 'svg', f'{c}.svg'),
                'dpi': 300,
                'transparent': True,
            },
        ])

        showNet(ppc, savefig=[{
                'fname': osp.join(data_dir, 'pdf', f'{c}_eid.pdf'),
                'dpi': 300,
                'transparent': True,
            }, {
                'fname': osp.join(data_dir, 'png', f'{c}_eid.png'),
                'dpi': 300,
                'transparent': True,
            }, {
                'fname': osp.join(data_dir, 'svg', f'{c}_eid.svg'),
                'dpi': 300,
                'transparent': True,
            },
        ], showEid=True)

