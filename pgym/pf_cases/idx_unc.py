"""Defines constants for named column indices to uncertainty matrix.

The index, name and meaning of each column of the uncertainty matrix is given
below:

columns 0-2 must be included in input matrix (in case file)
    0.  C{UNC_L_I}     line branch number
    1.  C{UNC_L_MIN}   line min ratio
    2.  C{UNC_L_MAX}   line max ratio

columns 0-2 must be included in input matrix (in case file)
    0.  C{UNC_P_I}     P node number
    1.  C{UNC_P_MIN}   P min ratio
    2.  C{UNC_P_MAX}   P max ratio

@author: Haotian Liu
"""

# define the indices
UNC_L_I       = 0    # i, index
UNC_L_MIN     = 1    # min
UNC_L_MAX     = 2    # max

UNC_P_I       = 0    # i, index
UNC_P_MIN     = 1    # min
UNC_P_MAX     = 2    # max
