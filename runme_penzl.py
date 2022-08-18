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
import scipy.sparse as sps

from pymor.models.iosys import LTIModel
from pymor.operators.constructions import LincombOperator
from pymor.parameters.functionals import ProjectionParameterFunctional
from pymor.reductors.h2 import IRKAReductor

from mor.models.abc import ABCStationaryModel
from mor.reductors.l2opt import L2DataDrivenReductor
from mor.tools import savetxt

# %% [markdown]
# # Load model

# %%
A1 = np.array([[-1, 100], [-100, -1]])
A2 = np.array([[-1, 200], [-200, -1]])
A3 = np.array([[-1, 400], [-400, -1]])
A4 = sps.diags(np.arange(-1, -1001, -1))
A = sps.block_diag((A1, A2, A3, A4), format='csc')
B = np.ones((1006, 1))
B[:6] = 10
C = B.T
fom_lti = LTIModel.from_matrices(A, B, C)

# %%
fom_lti

# %%
print(fom_lti)

# %% [markdown]
# # Visualize LTI model

# %%
w = np.logspace(0, 4, 1000)
_ = fom_lti.transfer_function.mag_plot(w)

# %% [markdown]
# # Build parametric ABC model

# %%
fom = ABCStationaryModel(
    LincombOperator(
        [fom_lti.E, fom_lti.A],
        [ProjectionParameterFunctional('s'), -1]
    ),
    fom_lti.B,
    fom_lti.C,
)

# %%
fom

# %%
print(fom)

# %% [markdown]
# # IRKA

# %%
irka = IRKAReductor(fom_lti)

# %%
rom_irka = irka.reduce(2)

# %%
rom_irka

# %%
rom_irka.poles()

# %% [markdown]
# # Data

# %%
N = 50
w_data = np.logspace(0, 4, N)
H_data = fom_lti.transfer_function.freq_resp(w_data)

# %%
fig, ax = plt.subplots()
_ = fom_lti.transfer_function.mag_plot(w, ax=ax)
_ = ax.loglog(w_data, spla.norm(H_data, axis=(1, 2)), '.')

# %% [markdown]
# # $\mathcal{L}_2$-optimal ROM

# %%
ps = [fom.parameters.parse(p)
      for p in np.concatenate((1j * w_data, -1j * w_data))]
H = np.concatenate((H_data, H_data.conj()))

# %%
l2dd = L2DataDrivenReductor(ps, H)

# %%
rom0 = ABCStationaryModel(
    LincombOperator(
        [rom_irka.E, rom_irka.A],
        [ProjectionParameterFunctional('s'), -1]
    ),
    rom_irka.B,
    rom_irka.C,
)

# %% tags=[]
rom = l2dd.reduce(rom0, maxit=1000, tol=1e-6)

# %%
_ = plt.semilogy(l2dd.dist, '.-')

# %%
_ = plt.semilogy(l2dd.errors, '.-')

# %%
rom_lti = LTIModel(rom.A.operators[1], rom.B, rom.C, E=rom.A.operators[0])

# %%
rom_lti.poles()

# %%
fig, ax = plt.subplots()
_ = fom_lti.transfer_function.mag_plot(w, ax=ax)
_ = ax.loglog(w_data, spla.norm(H_data, axis=(1, 2)), '.')
_ = rom_lti.transfer_function.mag_plot(w, ax=ax)


# %% [markdown]
# # Least-squares interpolation

# %%
class ModifiedOutput():
    """Interpolated least-squares transfer functions."""

    def __init__(self, w, H):
        assert np.all(w > 0)
        self.w = np.concatenate((1j * w, -1j * w))
        self.H = np.concatenate((H, H.conj()))

    def __call__(self, s):
        """Evaluate the transfer function.

        Parameters
        ----------
        s : float, complex
            Number in the complex plane.

        Returns
        -------
        G : float, complex
            Transfer function value.
        """
        G = np.sum([Hi / (s - wi) for wi, Hi in zip(self.w, self.H)])
        if s.imag == 0:
            G = G.real
        return G


# %%
G = ModifiedOutput(w_data, H_data)
Gr = ModifiedOutput(w_data, rom_lti.transfer_function.freq_resp(w_data))

# %%
s_list = np.logspace(-2, 4, 1000)
G_list = np.array([G(s) for s in s_list])
Gr_list = np.array([Gr(s) for s in s_list])

# %%
fig, ax = plt.subplots()
_ = ax.semilogx(s_list, G_list, label='$G$')
_ = ax.semilogx(s_list, Gr_list, '--', label=r'$\widehat{G}$')
_ = ax.legend()
ax.grid()

# %%
fig, ax = plt.subplots()
_ = ax.semilogx(s_list, G_list - Gr_list,
                label=r'$G - \widehat{G}$')
_ = ax.semilogx(-rom_lti.poles().real, [0, 0], 'x',
                label="ROM's reflected poles")
ax.grid()
_ = ax.legend()
_ = ax.set_xlim([1, 2000])
_ = ax.set_ylim([-0.1, 0.1])

# %%
savetxt('penzl_G.txt',
        (s_list, G_list, Gr_list, G_list - Gr_list),
        ('s', 'G', 'Gr', 'dG'))
