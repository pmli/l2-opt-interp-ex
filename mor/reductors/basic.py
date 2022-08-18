from pymor.algorithms.projection import project, project_to_subbasis
from pymor.reductors.basic import ProjectionBasedReductor

from ..models.abc import ABCStationaryModel


class ABCStationaryRBReductor(ProjectionBasedReductor):
    """Galerkin projection of a |ABCStationaryModel|.

    Parameters
    ----------
    fom
        The full-order |Model| to reduce.
    RB
        The basis of the reduced space onto which to project.
        If `None` an empty basis is used.
    product
        Inner product |Operator| w.r.t. which `RB` is orthonormalized.
        If `None`, the Euclidean inner product is used.
    check_orthonormality
        See :class:`ProjectionBasedReductor`.
    check_tol
        See :class:`ProjectionBasedReductor`.
    """

    def __init__(self, fom, RB=None, product=None, check_orthonormality=None,
                 check_tol=None):
        assert isinstance(fom, ABCStationaryModel)
        RB = fom.solution_space.empty() if RB is None else RB
        assert RB in fom.solution_space
        super().__init__(fom, {'RB': RB}, {'RB': product},
                         check_orthonormality=check_orthonormality,
                         check_tol=check_tol)

    def project_operators(self):
        """Project operators."""
        fom = self.fom
        RB = self.bases['RB']
        projected_operators = {
            'A': project(fom.A, RB, RB),
            'B': project(fom.B, RB, None),
            'C': project(fom.C, None, RB),
            'products': {k: project(v, RB, RB)
                         for k, v in fom.products.items()},
        }
        return projected_operators

    def project_operators_to_subbasis(self, dims):
        """Project operators to subbasis."""
        rom = self._last_rom
        dim = dims['RB']
        projected_operators = {
            'A': project_to_subbasis(rom.A, dim, dim),
            'B': project_to_subbasis(rom.B, dim, None),
            'C': project_to_subbasis(rom.C, None, dim),
            'products': {k: project_to_subbasis(v, dim, dim)
                         for k, v in rom.products.items()},
        }
        return projected_operators

    def build_rom(self, projected_operators, error_estimator):
        """Build ROM."""
        return ABCStationaryModel(error_estimator=error_estimator,
                                  **projected_operators)
