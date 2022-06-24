import numpy as np
import scipy.sparse as sp
from pymatsolver import Pardiso as Solver
from discretize.utils import is_scalar, sdiag

class Inversion:
    def __init__(self, sim, data, noise_floor, beta=None):
        self._sim = sim
        self._data = data
        self._noise_floor = noise_floor
        self._beta = beta

    @property
    def data(self):
        return self._data

    @property
    def noise_floor(self):
        return self._noise_floor

    @noise_floor.setter
    def noise_floor(self, value):
        if getattr(self, "_Wd", None) is not None:
            self._Wd = None
        if getattr(self, "_WdG", None) is not None:
            self._WdG = None
        if getattr(self, "_rhs", None) is not None:
            self._rhs = None
        self._noise_floor = value

    @property
    def target_misfit(self):
        return self._sim.survey.nD

    @property
    def Wd(self):
        if getattr(self, "_Wd", None) is None:
            if is_scalar(self.noise_floor):
                self._Wd = (1./self.noise_floor) * np.eye(self._sim.survey.nD)
            else:
                self._Wd = sdiag(1./self.noise_floor)
        return self._Wd

    @property
    def G(self):
        if getattr(self, "_G", None) is None:
            self._G = np.vstack(self._sim.G)
        return self._G

    def _set_WdG(self):
        G = self.G
        self._WdG = self.Wd @ G

    @property
    def phi_d_matrix(self):
        if getattr(self, "_WdG", None) is None:
            self._set_WdG()
        if getattr(self, "_WdG2", None) is None:
            self._WdG2 = self._WdG.T @ self._WdG
        return self._WdG2

    @property
    def regularization_matrix(self):
        return np.eye(self._sim.mapping.nP)

    def get_linear_system(self):
        return self.phi_d_matrix + self._beta * self.regularization_matrix

    @property
    def rhs(self):
        if getattr(self, "_rhs", None) is None:
            if getattr(self, "_WdG", None) is None:
                self._set_WdG()
            self._rhs = self._WdG.T @ self.Wd @ self.data
        return self._rhs

    def _estimate_beta(self, beta_fact):
        mtest = np.random.randn(self._sim.mapping.nP)
        phi_d_tmp = self.phi_d_matrix @ mtest
        phi_m_tmp = self.regularization_matrix @ mtest
        return beta_fact * np.linalg.norm(phi_d_tmp) / np.linalg.norm(phi_m_tmp)

    def _solve_for_m(self):
        A = self.get_linear_system()
        Ainv = Solver(sp.csr_matrix(A))
        rhs = self.rhs
        return Ainv * rhs

    def solve(self, beta=None, beta_fact=None, beta_cooling=None, verbose=True, max_iter=10):
        if beta is not None:
            self._beta = beta
        elif beta_fact is not None:
            self._beta = self._estimate_beta(beta_fact)

        if beta_cooling is not None:
            pass
        else:
            return self._solve_for_m()





