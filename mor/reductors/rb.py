import numpy as np
import scipy.linalg as spla

from pymor.core.base import BasicObject

from .basic import ABCStationaryRBReductor


class StrongGreedyRBReductor(BasicObject):
    """Strong greedy RB reductor.

    Parameters
    ----------
    fom
        The full-order |Model| to reduce.
    """

    def __init__(self, fom):
        self.fom = fom
        self.RB = None
        self.mus = None
        self.errors = None
        self.logger.setLevel('INFO')
        self._rb_reductor = None

    def reduce(self, training_set, maxit, tol):
        """Reduce using strong greedy.

        Parameters
        ----------
        training_set
            The list of parameter values to use in training.
        maxit
            Maximum number of iterations (and of the order of the ROM).
        tol
            Tolerance for the error.
        """
        self._rb_reductor = ABCStationaryRBReductor(self.fom)
        training_set = list(training_set)
        self.mus = []
        self.errors = []

        def argmax_mu_error(rom):
            errors = np.array([
                spla.norm(self.fom.output(mu=mu) - rom.output(mu=mu))
                for mu in training_set
            ])
            idx = np.argmax(errors)
            return idx, training_set[idx], errors[idx]

        for _ in range(maxit):
            rom = self._rb_reductor.reduce()
            if not training_set:
                self.logger.info('Training set is exhausted.')
                break
            idx, mu_max, err_max = argmax_mu_error(rom)
            self.logger.info(f'Parameter value with max. error: {mu_max}')
            self.mus.append(mu_max)
            self.logger.info(f'Current maximum error: {err_max:.3e}')
            self.errors.append(err_max)
            if err_max <= tol:
                break
            self._rb_reductor.extend_basis(self.fom.solve(mu=mu_max))
            del training_set[idx]

        self.RB = self._rb_reductor.bases['RB']
        return self._rb_reductor.reduce()
