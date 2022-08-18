"""Models used in examples."""

from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import (
    ExpressionFunction, ConstantFunction, LincombFunction)
from pymor.discretizers.builtin import discretize_stationary_cg, RectGrid
from pymor.parameters.base import ParameterSpace
from pymor.parameters.functionals import ProjectionParameterFunctional

from .abc import ABCStationaryModel
from ..tools import simplify


def _poisson2d_stationarymodel(diameter):
    problem = StationaryProblem(
        domain=RectDomain(),
        diffusion=LincombFunction(
            [ExpressionFunction('x[0]', 2),
             ExpressionFunction('1 - x[0]', 2)],
            [1, ProjectionParameterFunctional('mu')],
        ),
        rhs=ConstantFunction(1, 2),
        dirichlet_data=ConstantFunction(0, 2),
    )

    m, _ = discretize_stationary_cg(
        analytical_problem=problem,
        grid_type=RectGrid,
        diameter=diameter,
    )

    return m


def poisson2d_output(diameter=2**(0.5) / 32):
    """Poisson 2D example with one output.

    Parameters
    ----------
    diameter
        Discretization diameter.

    Returns
    -------
    fom
        ABCStationaryModel.
    parameter_space
        ParameterSpace.
    """
    m = _poisson2d_stationarymodel(diameter=diameter)
    fom = ABCStationaryModel(simplify(m.operator), m.rhs, m.rhs.H,
                             visualizer=m.visualizer)
    parameter_space = ParameterSpace(fom.parameters, 0.1, 10)
    return fom, parameter_space
