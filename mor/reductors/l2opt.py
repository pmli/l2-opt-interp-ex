"""Reductors."""

import numpy as np
import scipy.integrate as spint
import scipy.linalg as spla
import scipy.optimize as spopt

from pymor.core.base import BasicObject
from pymor.operators.constructions import LincombOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.parameters.base import ParameterSpace
from pymor.parameters.functionals import ParameterFunctional

from ..models.abc import ABCStationaryModel, _quad_scalar


class L2OptimalReductor(BasicObject):
    """L2-optimal reductor for stationary problems.

    Parameters
    ----------
    fom
        The full-order |Model| to reduce.
    parameter_space
        |ParameterSpace| over which to optimize.
    """

    def __init__(self, fom, parameter_space):
        self.fom = fom
        self.parameter_space = parameter_space
        self.roms = None
        self.dist = None
        self.errors = None
        self.logger.setLevel('INFO')

    def reduce(self, rom0, maxit=100, tol=1e-4, method='BFGS',
               quad_options=None):
        """Reduce using optimization.

        Parameters
        ----------
        rom0
            Initial reduced-order ABCStationaryModel.
            Its operators have to be |NumpyMatrixOperators| or
            |LincombOperators| of |NumpyMatrixOperators|.
        maxit
            Maximum number of iterations.
        tol
            Tolerance for optimization.
        method
            Optimization method to use in `scipy.optimize.minimize`.
        quad_options
            See ABCStationaryModel.l2_norm.

        Returns
        -------
        rom
            Reduced-order ABCStationaryModel.
        """
        def f(x):
            rom = _vec_to_rom(x, rom0)
            return self._objective(rom, quad_options=quad_options)

        def fprime(x):
            rom = _vec_to_rom(x, rom0)
            return self._gradient(rom, quad_options=quad_options)

        x0 = _rom_to_vec(rom0)
        callback = _Callback(self, rom0, f, fprime, tol)
        try:
            spopt.minimize(f, x0, method=method, jac=fprime, tol=None,
                           callback=callback,
                           options={
                               'maxiter': maxit,
                               'disp': True,
                               'gtol': 1e-20,
                           })
        except StopIteration:
            pass
        return self.roms[-1]

    def _objective(self, rom, quad_options=None):
        """Compute objective function value.

        Parameters
        ----------
        rom
            Reduced-order model.
        quad_options
            See ABCStationaryModel.l2_norm.

        Returns
        -------
        Scalar objective value.
        """
        def func(mu):
            mu = self.fom.parameters.parse(mu)
            y = self.fom.output(mu=mu)
            yr = rom.output(mu=mu)
            return np.sum(np.abs(y - yr)**2)

        if (quad_options is not None
                and quad_options['type'] not in ('quad', 'fixed_quad')):
            raise ValueError("quad_options['type'] must be 'quad' or"
                             "'fixed_quad'.")

        return _quad_scalar(func, list(self.parameter_space.ranges.values()),
                            quad_options)

    def _gradient(self, rom, quad_options=None):
        """Compute gradient of the squared L2 error.

        Parameters
        ----------
        rom
            Reduced-order model.
        quad_options
            See ABCStationaryModel.l2_norm.

        Returns
        -------
        Gradient as a concatenated vector.
        """
        def func(mu):
            mu = self.fom.parameters.parse(mu)
            y = self.fom.output(mu=mu)
            yr = rom.output(mu=mu)
            xr = rom.solve(mu=mu).to_numpy().T
            xrd = rom.solve_dual(mu=mu).to_numpy().T
            dA = xrd @ (y - yr) @ xr.T.conj()
            dB = xrd @ (yr - y)
            dC = (yr - y) @ xr.T.conj()

            def get_matrices(mat, rom_op):
                if isinstance(rom_op, NumpyMatrixOperator):
                    return [2 * mat]
                matrices = []
                for coeff in rom_op.coefficients:
                    c = (coeff.evaluate(mu=mu)
                         if isinstance(coeff, ParameterFunctional)
                         else coeff)
                    matrices.append(2 * c.conjugate() * mat)
                return matrices

            dAs = get_matrices(dA, rom.A)
            dBs = get_matrices(dB, rom.B)
            dCs = get_matrices(dC, rom.C)

            def matrices_to_vector(matrices):
                vectors = [mat.reshape(-1) for mat in matrices]
                return np.hstack(vectors)

            return np.hstack([matrices_to_vector(matrices)
                              for matrices in [dAs, dBs, dCs]])

        return _quad_vec(func, list(self.parameter_space.ranges.values()),
                         quad_options)

    def _plot_descent_direction(self, rom, alphas, quad_options=None):
        """Plot objective function in the descent direction.

        Parameters
        ----------
        rom
            The reduced-order model from which to plot.
        alphas
            Sequence of gradient factors used for plotting.
        quad_options
            See ABCStationaryModel.l2_norm.
        """
        import matplotlib.pyplot as plt
        desc_dir = -self._gradient(rom, quad_options=quad_options)
        x = _rom_to_vec(rom)
        obj_vals = []
        for alpha in alphas:
            x_alpha = x + alpha * desc_dir
            rom_alpha = _vec_to_rom(x_alpha, rom)
            obj_vals.append(self._objective(rom_alpha,
                                            quad_options=quad_options))
        plt.plot(alphas, obj_vals, '.-')


class L2DataDrivenReductor(BasicObject):
    """L2-optimal data-driven reductor.

    Parameters
    ----------
    ps
        Sequence of parameter values.
    ys
        Outputs corresponding to parameter values.
    """

    def __init__(self, ps, ys):
        self.parameter_space = ps
        self.ys = ys
        self.roms = None
        self.dist = None
        self.errors = None
        self.logger.setLevel('INFO')

    def reduce(self, rom0, maxit=100, tol=1e-4, method='BFGS',
               constraint=None):
        """Reduce using optimization.

        Parameters
        ----------
        rom0
            Initial reduced-order ABCStationaryModel.
            Its operators have to be |NumpyMatrixOperators| or
            |LincombOperators| of |NumpyMatrixOperators|.
        maxit
            Maximum number of iterations.
        tol
            Tolerance for optimization.
        method
            Optimization method to use in SciPy's `minimize`.
        constraint
            A method that takes a reduced-order model and returns a `bool`.
            If `False`, the objective function evaluates as `np.inf`.

        Returns
        -------
        rom
            Reduced-order ABCStationaryModel.
        """
        def f(x):
            rom = _vec_to_rom(x, rom0)
            if constraint is not None and not constraint(rom):
                return 1
            return self._objective(rom)

        def fprime(x):
            rom = _vec_to_rom(x, rom0)
            return self._gradient(rom)

        x0 = _rom_to_vec(rom0)
        callback = _Callback(self, rom0, f, fprime, tol)
        try:
            options = {
                'maxiter': maxit,
                'disp': True,
                'gtol': 1e-20,
            }
            if method == 'L-BFGS-B':
                options['ftol'] = 1e-20
            spopt.minimize(f, x0, method=method, jac=fprime, tol=None,
                           callback=callback,
                           options=options)
        except StopIteration:
            pass
        return self.roms[-1]

    def _objective(self, rom):
        """Compute objective function value.

        Parameters
        ----------
        rom
            Reduced-order model.

        Returns
        -------
        Scalar objective value.
        """
        def func(mu, y):
            yr = rom.output(mu=mu)
            return np.sum(np.abs(y - yr)**2)

        return np.mean([func(mu, y)
                        for mu, y in zip(self.parameter_space, self.ys)])

    def _gradient(self, rom):
        """Compute gradient of the squared L2 error.

        Parameters
        ----------
        rom
            Reduced-order model.

        Returns
        -------
        Gradient as a concatenated vector.
        """
        def func(mu, y):
            yr = rom.output(mu=mu)
            xr = rom.solve(mu=mu).to_numpy().T
            xrd = rom.solve_dual(mu=mu).to_numpy().T
            dA = xrd @ (y - yr) @ xr.T.conj()
            dB = xrd @ (yr - y)
            dC = (yr - y) @ xr.T.conj()

            def get_matrices(mat, rom_op):
                if isinstance(rom_op, NumpyMatrixOperator):
                    return [2 * mat]
                matrices = []
                for coeff in rom_op.coefficients:
                    c = (coeff.evaluate(mu=mu)
                         if isinstance(coeff, ParameterFunctional)
                         else coeff)
                    matrices.append(2 * c.conjugate() * mat)
                return matrices

            dAs = get_matrices(dA, rom.A)
            dBs = get_matrices(dB, rom.B)
            dCs = get_matrices(dC, rom.C)

            def matrices_to_vector(matrices):
                vectors = [mat.reshape(-1).real for mat in matrices]
                return np.hstack(vectors)

            return np.hstack([matrices_to_vector(matrices)
                              for matrices in [dAs, dBs, dCs]])

        return np.mean([func(mu, y)
                        for mu, y in zip(self.parameter_space, self.ys)],
                       axis=0)

    def _plot_descent_direction(self, rom, alphas):
        """Plot objective function in the descent direction.

        Parameters
        ----------
        rom
            The reduced-order model from which to plot.
        alphas
            Sequence of gradient factors used for plotting.
        """
        import matplotlib.pyplot as plt
        desc_dir = -self._gradient(rom)
        x = _rom_to_vec(rom)
        obj_vals = []
        for alpha in alphas:
            x_alpha = x + alpha * desc_dir
            rom_alpha = _vec_to_rom(x_alpha, rom)
            obj_vals.append(self._objective(rom_alpha))
        plt.plot(alphas, obj_vals, '.-')


def _rom_to_vec(rom):
    """Convert reduced-order model to vector.

    Parameters
    ----------
    rom
        Reduced-order model.

    Returns
    -------
    vec
        Vectorized matrices of `rom`.
    """
    def op_to_vec(op):
        if isinstance(op, NumpyMatrixOperator):
            return op.matrix.reshape(-1)
        return np.hstack([opi.matrix.reshape(-1) for opi in op.operators])

    return np.hstack(list(map(op_to_vec, [rom.A, rom.B, rom.C])))


def _vec_to_rom(vec, rom0):
    """Convert vector to reduced-order model.

    Parameters
    ----------
    vec
        Vector to convert to reduced-order model.
    rom0
        Reduced-order model whose structure to follow.

    Returns
    -------
    rom
        Reduced-order model.
    """
    r = rom0.order
    m = rom0.dim_input
    r2 = r**2

    def get_q(op):
        return (1
                if isinstance(op, NumpyMatrixOperator)
                else len(op.operators))

    q_A, q_B = map(get_q, (rom0.A, rom0.B))
    vec_A, vec_B, vec_C = np.split(
        vec,
        np.cumsum([
            r2 * q_A,
            r * m * q_B,
        ]),
    )

    def vec_to_op(vec, op0):
        shape = (op0.range.dim, op0.source.dim)
        if isinstance(op0, NumpyMatrixOperator):
            return op0.with_(matrix=vec.reshape(shape))
        vecs = np.split(vec, len(op0.operators))
        operators = [op0.operators[0].with_(matrix=v.reshape(shape))
                     for v in vecs]
        return LincombOperator(operators, op0.coefficients)

    return ABCStationaryModel(
        vec_to_op(vec_A, rom0.A),
        vec_to_op(vec_B, rom0.B),
        vec_to_op(vec_C, rom0.C),
    )


def _rel_l2_dist(m1, m2, parameters):
    if isinstance(parameters, ParameterSpace):
        return (m1 - m2).l2_norm(parameters) / m2.l2_norm(parameters)
    y1 = np.stack([m1.output(mu=mu) for mu in parameters])
    y2 = np.stack([m2.output(mu=mu) for mu in parameters])
    return spla.norm(y1 - y2) / spla.norm(y2)


def _quad_vec(func, ranges, quad_options):
    a, b = ranges[0]
    if quad_options is None or quad_options['type'] == 'quad':
        options = ({} if quad_options is None
                   else quad_options.get('options', {}))
        res = spint.quad_vec(func, a, b, **options)[0]
    elif quad_options['type'] == 'fixed_quad':
        def vfunc(mu_list):
            return np.vstack([func(mu) for mu in mu_list]).T
        n = quad_options.get('options', {}).get('n')
        res = spint.fixed_quad(vfunc, a, b, n=n)[0]
    else:
        raise ValueError(f"Unknown quad. type {quad_options['type']}")
    return res


class _Callback():
    def __init__(self, reductor, rom0, obj, grad, tol):
        self.reductor = reductor
        self.obj = obj
        self.grad = grad
        self.tol = tol
        self.it = 0
        self.reductor.roms = [rom0]
        self.reductor.dist = []
        self.reductor.errors = []

    def __call__(self, x):
        self.it += 1
        with self.reductor.logger.block(f'Iteration {self.it}'):
            rom = _vec_to_rom(x, self.reductor.roms[0])
            self.reductor.roms.append(rom)
            dist_rel = _rel_l2_dist(self.reductor.roms[-2],
                                    self.reductor.roms[-1],
                                    self.reductor.parameter_space)
            self.reductor.dist.append(dist_rel)
            self.reductor.logger.info(f'ROM change:    {dist_rel:.3e}')
            error = np.sqrt(self.obj(x))
            self.reductor.errors.append(error)
            self.reductor.logger.info(f'L2 error:      {error:.3e}')
            grad_norm = spla.norm(self.grad(x))
            self.reductor.logger.info(f'Gradient norm: {grad_norm:.3e}')
        if self.reductor.dist[-1] <= self.tol:
            raise StopIteration
