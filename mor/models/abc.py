"""Models."""

import math

import numpy as np
import scipy.integrate as spint
import scipy.linalg as spla
import scipy.optimize as spopt

from pymor.core.exceptions import InversionError
from pymor.models.interface import Model
from pymor.operators.block import (BlockColumnOperator, BlockDiagonalOperator,
                                   BlockRowOperator)
from pymor.operators.interface import Operator


class ABCStationaryModel(Model):
    """Linear parametric stationary models.

    This class describes discrete problems given by the equation::

        A(x(μ), μ) = B(μ)
        y(μ) = C(x(μ), μ)

    with operators A, B, C.

    Parameters
    ----------
    A
        The |Operator| A.
    B
        The |Operator| B.
    C
        The |Operator| C.
    products
        A dict of inner product |Operators| defined on the discrete space the
        problem is posed on. For each product with key `'x'` a corresponding
        attribute `x_product`, as well as a norm method `x_norm` is added to
        the model.
    error_estimator
        An error estimator for the problem. This can be any object with
        an `estimate_error(U, mu, m)` method. If `error_estimator` is
        not `None`, an `estimate_error(U, mu)` method is added to the
        model which will call `error_estimator.estimate_error(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, m, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the model which forwards its arguments to the
        visualizer's `visualize` method.
    name
        Name of the model.
    """

    cache_region = 'memory'

    def __init__(self, A, B, C, products=None,
                 error_estimator=None, visualizer=None, name=None):

        assert isinstance(A, Operator)
        assert isinstance(B, Operator)
        assert isinstance(C, Operator)
        assert A.source == A.range
        assert B.range == A.source and B.linear
        assert C.source == A.source

        super().__init__(products=products, error_estimator=error_estimator,
                         visualizer=visualizer, name=name)

        self.__auto_init(locals())
        self.linear = A.linear and C.linear
        self.solution_space = A.source
        self.dim_input = B.source.dim
        self.dim_output = C.range.dim
        self.logger.setLevel('INFO')

    def __str__(self):
        return (
            f'{self.name}\n'
            f'    class: {self.__class__.__name__}\n'
            f'    {"linear" if self.linear else "nonlinear"}'
            f'    solution_space: {self.solution_space}\n'
            f'    order:          {self.order}\n'
            f'    dim_input:      {self.dim_input}\n'
            f'    dim_output:     {self.dim_output}\n'
            f'    parameters:     {self.parameters}'
        )

    def solve(self, mu=None):
        """Compute the solution.

        Parameters
        ----------
        mu
            |Parameter values| for which to solve.

        Returns
        -------
        The solution |VectorArray|.
        """
        try:
            x = self.A.apply_inverse(self.B.as_range_array(mu=mu), mu=mu)
        except InversionError:
            x = np.inf * self.solution_space.ones(self.dim_input)
        return x

    def output(self, mu=None):
        """Compute the output.

        Parameters
        ----------
        mu
            |Parameter values| for which to solve.

        Returns
        -------
        The output as a 2D |NumPy| array of shape
        `(self.dim_output, self.dim_input)`.
        """
        mu = self.parameters.parse(mu)
        return self.cached_method_call(self._output, mu=mu)

    def _output(self, mu=None):
        return self.C.apply(self.solve(mu=mu), mu=mu).to_numpy().T

    def solve_dual(self, mu=None):
        """Compute the solution of the dual system.

        Parameters
        ----------
        mu
            |Parameter values| for which to solve.

        Returns
        -------
        The solution |VectorArray|.
        """
        try:
            xd = self.A.apply_inverse_adjoint(self.C.as_source_array(mu=mu),
                                              mu=mu)
        except InversionError:
            xd = np.inf * self.solution_space.ones(self.dim_output)
        return xd

    def __sub__(self, other):
        """Subtract another model."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return ABCStationaryErrorModel(self, other)

    def l2_norm(self, parameter_space, quad_options=None):
        """Compute L2 norm of the output over a parameter space.

        Parameters
        ----------
        parameter_space
            |ParameterSpace| over which to compute the norm.
        quad_options
            Quadrature options.
            If not `None`, it must be a dict with key `'type'` with the
            following possible values:

            - `'quad'` (default for 1D parameters),
            - `'nquad'` (default for >2D parameters),
            - `'fixed_quad'`.

            Optionally, a key `'options'` can be given.
            Its value is passed to the appropriate function in
            scipy.integrate.

        Return
        ------
        L2 norm as a scalar.
        """
        if (len(parameter_space.ranges) == 1
                and quad_options is not None
                and quad_options['type'] not in ('quad', 'fixed_quad')):
            raise ValueError("For 1D parameters, quad_options['type'] must be"
                             "'quad' or 'fixed_quad'.")
        if (len(parameter_space.ranges) >= 2
                and quad_options is not None
                and quad_options['type'] != 'nquad'):
            raise ValueError("For >2D parameters, quad_options['type'] must be"
                             "'nquad'.")

        ranges = list(parameter_space.ranges.values())
        return self.cached_method_call(self._l2_norm, ranges=ranges,
                                       quad_options=quad_options)

    def _l2_norm(self, ranges, quad_options):
        def output2(mu):
            output = self.output(mu=mu)
            return np.sum(np.abs(output)**2)

        norm2 = _quad_scalar(output2, ranges, quad_options)
        norm = math.sqrt(norm2)
        return norm

    def linf_norm(self, parameter_space):
        """Compute Linf norm of the output over a parameter space.

        Parameters
        ----------
        parameter_space
            |ParameterSpace| over which to compute the norm.

        Return
        ------
        Linf norm as a scalar.
        """
        ranges = list(parameter_space.ranges.values())
        return self.cached_method_call(self._linf_norm, ranges=ranges)

    def _linf_norm(self, ranges):
        def func(x):
            return -spla.norm(self.output(mu=x))
        res = spopt.shgo(func, ranges)
        return -res.fun

    def solutions(self, ps):
        """Return model solutions.

        Parameters
        ----------
        ps
            Sequence of parameter values.

        Returns
        -------
        X
            Solutions as a single VectorArray.
        """
        X = self.A.source.empty(reserve=len(ps))
        for p in ps:
            X.append(self.solve(mu=p))
        return X

    def outputs(self, ps):
        """Return model outputs.

        Parameters
        ----------
        ps
            Sequence of parameter values.

        Returns
        -------
        y
            Outputs as a NumPy array of shape
            `(len(ps), self.dim_output, self.dim_input)`.
        """
        return np.stack([self.output(mu=p) for p in ps])

    def plot_outputs(self, ps, ax=None, **plot_options):
        """Plot model outputs.

        Parameters
        ----------
        ps
            Sequence of parameter values.
        ax
            Matplotlib axis.
            If not given, `matplotlib.pyplot.gca` is used.
        plot_options
            Keyword arguments passed to the plot function.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        outputs = [self.output(mu=mu).reshape(-1) for mu in ps]
        if self.parameters.dim == 1:
            p_vals = [next(iter(mu.values())) for mu in ps]
            ax.plot(p_vals, outputs, **plot_options)
            ax.set_xlabel('Parameter value')
        else:
            ax.plot(outputs, **plot_options)
            ax.set_xlabel('Parameter index')
        ax.set_ylabel('Output')

    def plot_outputs_2d(self, ps1, ps2, ax=None):
        """Plot model outputs (for 2D parameters).

        Parameters
        ----------
        ps1
            Sequence of parameter values for the first parameter.
        ps2
            Sequence of parameter values for the second parameter.
        ax
            Matplotlib axis.
            If not given, `matplotlib.pyplot.gca` is used.
        """
        assert self.parameters.dim == 2
        assert self.dim_input == self.dim_output == 1
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        p1_name = next(iter(ps1[0].keys()))
        p2_name = next(iter(ps2[0].keys()))
        p1_vals = [mu1[p1_name][0] for mu1 in ps1]
        p2_vals = [mu2[p2_name][0] for mu2 in ps2]
        outputs = [
            [
                self.output(mu={
                    p1_name: mu1[p1_name][0],
                    p2_name: mu2[p2_name][0],
                })[0, 0]
                for mu1 in ps1
            ]
            for mu2 in ps2
        ]
        out = ax.pcolormesh(p1_vals, p2_vals, outputs, shading='gouraud')
        ax.set_xlabel(p1_name)
        ax.set_ylabel(p2_name)
        return out

    def plot_outputs_mag(self, ps, ax=None, **plot_options):
        """Plot model output magnitudes.

        Parameters
        ----------
        ps
            Sequence of parameter values.
        ax
            Matplotlib axis.
            If not given, `matplotlib.pyplot.gca` is used.
        plot_options
            Keyword arguments passed to the plot function.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.gca()
        outputs = [spla.norm(self.output(mu=mu)) for mu in ps]
        if self.parameters.dim == 1:
            p_vals = [next(iter(mu.values())) for mu in ps]
            ax.plot(p_vals, outputs, **plot_options)
            ax.set_xlabel('Parameter value')
        else:
            ax.plot(outputs, **plot_options)
            ax.set_xlabel('Parameter index')
        ax.set_ylabel('Output magnitude')


class ABCStationaryErrorModel(ABCStationaryModel):
    """Error model.

    Parameters
    ----------
    first, second
        Two ABCStationaryModels.
    """

    def __init__(self, first, second):
        assert first.B.source == second.B.source
        assert first.C.range == second.C.range

        A = BlockDiagonalOperator([first.A, second.A])
        B = BlockColumnOperator([first.B, second.B])
        C = BlockRowOperator([first.C, -second.C])

        super().__init__(A, B, C)
        self.__auto_init(locals())

    def output(self, mu=None):
        """Compute output using the (cached) outputs."""
        return self.first.output(mu=mu) - self.second.output(mu=mu)


def _quad_scalar(func, ranges, quad_options):
    if quad_options is None or quad_options['type'] in ('quad', 'nquad'):
        options = ({} if quad_options is None
                   else quad_options.get('options', {}))
        if len(ranges) == 1:
            a, b = ranges[0]
            res = spint.quad(func, a, b, **options)[0]
        else:
            res = spint.nquad(func, ranges, **options)[0]
    else:
        a, b = ranges[0]
        n = quad_options.get('options', {}).get('n')
        res = spint.fixed_quad(np.vectorize(func), a, b, n=n)[0]
    return res
