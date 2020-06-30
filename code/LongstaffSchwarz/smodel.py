from pprint import pformat

import numpy as np
from swutil.stochastic_processes import black_scholes, r_bergomi, heston


class Heston:

    def __init__(self, d, T, nu0, theta, r, kappa, xi, rho, S0, N, payoff):
        self.d = d
        self.d_eff = d + 1
        self.T = T
        self.nu0 = nu0
        self.theta = theta
        self.kappa = kappa
        self.rho = rho
        self.xi = xi
        self.r = r
        self.S0 = S0
        self.N = N
        self.payoff = payoff
        self.payoff.d = self.d

    def __call__(self, M, random=np.random):
        times = np.linspace(0, self.T, self.N)
        securities = np.log(heston(
            times,
            mu=self.r,
            rho=self.rho,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            S0=self.S0,
            nu0=self.nu0,
            d=self.d,
            M=M,
            random=random,
        )).transpose([1,0,2])
        return securities, times

    def __repr__(self):
        return pformat(vars(self))
