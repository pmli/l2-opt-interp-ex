"""Tools."""

import numpy as np

from pymor.operators.constructions import LincombOperator
from pymor.parameters.functionals import ParameterFunctional


def savetxt(fname, columns, names=None):
    """Save columns to a text file.

    Parameters
    ----------
    fname : str
        File name.
    columns : sequence of lists of floats
        Columns to save.
    names : sequence of str (optional)
        Column names to write in the header.
    """
    X = np.vstack(columns).T
    header = '' if names is None else ' '.join(names)
    np.savetxt(fname, X, fmt='%.5e', header=header, comments='')


def simplify(op):
    """Simplify LincombOperator.

    Parameters
    ----------
    op
        Operator.

    Returns
    -------
    Simplified operator.
    """
    assert isinstance(op, LincombOperator)

    scalars_idx = [i for i, (opi, ci) in enumerate(zip(op.operators,
                                                       op.coefficients))
                   if not isinstance(ci, ParameterFunctional)]

    if len(scalars_idx) == 1:
        return op

    scalar_op = op.operators[scalars_idx[0]] * op.coefficients[scalars_idx[0]]
    for i in scalars_idx[1:]:
        scalar_op += op.operators[i] * op.coefficients[i]
    scalar_op = scalar_op.assemble()

    other_ops = [opi for i, opi in enumerate(op.operators)
                 if i not in scalars_idx]
    other_coeffs = [ci for i, ci in enumerate(op.coefficients)
                    if i not in scalars_idx]

    return LincombOperator([scalar_op] + other_ops, [1] + other_coeffs)
