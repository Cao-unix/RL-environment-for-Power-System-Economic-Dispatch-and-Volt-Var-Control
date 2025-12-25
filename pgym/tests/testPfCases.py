from pgym.pf_cases import ma_case33bw as case33bw
from pgym.utils.power import showNet, loss, getErrorCase
ppc = case33bw()

showNet(ppc, savefig=[{
        'fname': 'data/case33bw.png',
        'dpi': 300,
        'transparent': True,
    }, {
        'fname': 'data/case33bw.svg',
        'dpi': 300,
        'transparent': True,
    }, {
        'fname': 'data/case33bw.pdf',
        'dpi': 300,
        'transparent': True,
    },
])

exit(0)

from pypower.api import runpf, ppoption, printpf
ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0)
results, success = runpf(ppc, ppopt)
printpf(results)

pLoss, qLoss = loss(results)
print(f'Active power loss: {pLoss:.4} MW')

errcase = getErrorCase(ppc)
results, success = runpf(errcase, ppopt)
printpf(results)

pLoss, qLoss = loss(results)
print(f'Active power loss: {pLoss:.4} MW')
