# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as spla

from pymor.algorithms.to_matrix import to_matrix

from mor.models.abc import ABCStationaryModel
from mor.models.examples import poisson2d_output
from mor.reductors.l2opt import L2OptimalReductor
from mor.reductors.rb import StrongGreedyRBReductor
from mor.tools import savetxt

# %% [markdown]
# # Full-order model

# %%
fom, parameter_space = poisson2d_output(np.sqrt(2) / 32)

# %%
fom

# %%
print(fom)

# %%
ps = parameter_space.sample_uniformly(500)

# %%
fom.plot_outputs(ps)

# %%
fom.l2_norm(parameter_space)

# %%
fom.visualize(fom.solutions(parameter_space.sample_uniformly(50)))

# %% [markdown]
# # Reduced basis method

# %%
rb = StrongGreedyRBReductor(fom)

# %%
rom_rb = rb.reduce(parameter_space.sample_uniformly(100), 2, 1e-7)

# %%
rom_rb

# %%
print(rom_rb)

# %%
fom.plot_outputs(ps)
rom_rb.plot_outputs(ps, linestyle='--')

# %%
err_rb = fom - rom_rb

# %%
err_rb.plot_outputs_mag(ps)

# %%
err_rb.l2_norm(parameter_space) / fom.l2_norm(parameter_space)

# %% [markdown]
# # L2 optimization

# %% [markdown]
# ## Initialization with RB

# %%
l2opt = L2OptimalReductor(fom, parameter_space)

# %%
rom_l2 = l2opt.reduce(rom_rb, tol=1e-6)

# %%
_ = plt.semilogy(l2opt.dist, '.-')

# %%
fom.plot_outputs(ps)
rom_l2.plot_outputs(ps, linestyle='--')

# %%
err_l2 = fom - rom_l2

# %%
err_rb.plot_outputs(ps, label='RB')
err_l2.plot_outputs(ps, label='L2')
_ = plt.legend()

# %%
err_l2.l2_norm(parameter_space) / fom.l2_norm(parameter_space)

# %% [markdown]
# ## Check interpolation

# %%
np.linalg.matrix_rank(fom.A.operators[0].matrix.toarray())

# %%
np.linalg.matrix_rank(fom.A.operators[1].matrix.toarray())

# %%
np.linalg.matrix_rank(rom_l2.A.operators[0].matrix)

# %%
np.linalg.matrix_rank(rom_l2.A.operators[1].matrix)


# %%
def f(p, s, a, b):
    """Evaluate helper function for modified outputs.

    Parameters
    ----------
    p : float
        Parameter value.
    s : float
        Shift value.
    a : float
        Lower parameter space boundary.
    b : float
        Upper parameter space boundary.

    Returns
    -------
    fp : float
        Function value.
    """
    if p == s:
        return (b - a) / ((s - a) * (s - b))
    g = lambda x: np.log(np.abs((x - b) / (x - a)))
    return (g(p) - g(s)) / (p - s)


class InterpFun():
    """Interpolating function.

    Parameters
    ----------
    m : ABCStationaryModel
        The model to base the interpolating function on.
    a : float
        Lower parameter space boundary.
    b : float
        Upper parameter space boundary.
    dae : bool (optional)
        Whether to assume that A2 is singular.
    """

    def __init__(self, m, a, b, dae=False):
        assert isinstance(m, ABCStationaryModel)
        assert len(m.A.operators) == 2
        self.a = a
        self.b = b
        A1 = to_matrix(m.A.operators[0], format='dense')
        A2 = to_matrix(m.A.operators[1], format='dense')
        B = to_matrix(m.B, format='dense')
        C = to_matrix(m.C, format='dense')
        if dae:
            U, s, V = spla.svd(A2, lapack_driver='gesvd')
            V = V.T
            rank = sum(s > 1e-14 * s[0])
            U = U[:, :rank]
            s = s[:rank]
            V = V[:, :rank]
            s_sqrt = np.sqrt(s)
            U = U * s_sqrt
            V = V * s_sqrt
            A1invBU = spla.solve(A1, np.hstack((B, U)))
            A1invB = A1invBU[:, :B.shape[1]]
            A1invU = A1invBU[:, B.shape[1]:]
            H1 = C @ A1invB
            H2 = C @ A1invU
            H3 = V.T @ A1invU
            H4 = V.T @ A1invB
            self.residue0 = H1 - H2 @ spla.solve(H3, H4)
            poles, T = spla.eig(H3)
            poles = -1 / poles
            C2 = H2 @ T * poles**2
            B2 = spla.solve(T, H4)
        else:
            poles, T = spla.eig(-A1, A2)
            C2 = C @ T
            B2 = spla.solve(T, spla.solve(A2, B))
            self.residue0 = np.zeros((C.shape[0], B.shape[1]))
        residues = np.array([C2[:, i:i + 1] @ B2[i:i + 1, :]
                             for i in range(len(poles))])
        self.poles = poles.real
        self.residues = residues.real

    def __call__(self, p):
        """Evaluate the interpolating function.

        Parameters
        ----------
        p : float
            Parameter value.

        Returns
        -------
        result : float
            Interpolating function value.
        """
        result = self.residue0 * np.log(np.abs((p - self.b) / (p - self.a)))
        for pole, residue in zip(self.poles, self.residues):
            result += f(p, pole, self.a, self.b) * residue
        return result


# %%
fom_poles = spla.eigvals(-fom.A.operators[0].matrix.toarray(),
                         fom.A.operators[1].matrix.toarray())

# %%
fom_poles_finite = fom_poles[np.isfinite(fom_poles)]

# %%
fom_poles_finite.real.max()

# %%
rom_poles = spla.eigvals(-rom_l2.A.operators[0].matrix,
                         rom_l2.A.operators[1].matrix)

# %%
rom_poles

# %%
_ = plt.plot(fom_poles_finite.real, fom_poles_finite.imag, '.')
_ = plt.plot(rom_poles.real, rom_poles.imag, 'x')

# %%
Y = InterpFun(fom, 0.1, 10, dae=True)
Yr = InterpFun(rom_l2, 0.1, 10)

# %%
p_list = np.linspace(-15, 15, 2000)
Y_list = np.array([Y(p)[0, 0] for p in p_list])
Yr_list = np.array([Yr(p)[0, 0] for p in p_list])

# %%
fig, ax = plt.subplots(dpi=150)
ax.plot(p_list, Y_list)
ax.plot(p_list, Yr_list, '--')
ax.grid()
_ = ax.set_ylim([-0.08, 0.2])

# %%
savetxt('poisson_Y_Yr.txt',
        [p_list, Y_list, Yr_list],
        ['p', 'Y', 'Yr'])

# %%
fig, ax = plt.subplots(dpi=150)
ax.plot(p_list, Y_list - Yr_list)
plt.plot(rom_poles.real, rom_poles.imag, 'x')
ax.grid()
ax.set_xlim([-6, -0.01])
_ = ax.set_ylim([-1e-7, 1e-6])

# %%
p_list = np.linspace(-6, -0.01, 1000)
Y_list = np.array([Y(p)[0, 0] for p in p_list])
Yr_list = np.array([Yr(p)[0, 0] for p in p_list])

# %%
savetxt('poisson_dY.txt',
        [p_list, Y_list - Yr_list],
        ['p', 'dY'])
