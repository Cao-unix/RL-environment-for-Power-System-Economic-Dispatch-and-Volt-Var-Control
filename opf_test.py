from pypower.api import runopf, runpf
from pgym.pf_cases.case9 import case9 as case9
from pgym.pf_cases.case39 import case39 as case39
from pgym.pf_cases.case5m import pf_case as case5m

def test_opf():
    """Test the optimal power flow (OPF) functionality of PyPower."""

    # Load the test case
    ppc = case9()

    # Run the OPF
    results = runopf(ppc)

if __name__ == "__main__":
    test_opf()

