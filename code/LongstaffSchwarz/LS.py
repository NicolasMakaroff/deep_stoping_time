import numpy as np
from numpy.polynomial import Polynomial


class LS:
    def __init__(self, X, t, r, strike):
        self.X = X
        self.t = t
        self.r = r
        self.strike = strike

    def _ls_american_option_quadratic_iter(self, X, t, r, strike):
        # given no prior exercise we just receive the payoff of a European option
        cashflow = np.maximum(strike - X[-1, :], 0.0)
        # iterating backwards in time
        for i in reversed(range(1, X.shape[1] - 1)):
            # discount factor between t[i] and t[i+1]
            df = np.exp(-r * (t[i+1]-t[i]))
            # discount cashflows from next period
            cashflow = cashflow * df
            x = X[:, i]
            # exercise value for time t[i]
            exercise = np.maximum(strike - x, 0.0)
            # boolean index of all in-the-money paths
            itm = exercise > 0
            # fit polynomial of degree 2
            fitted = Polynomial.fit(x[itm], cashflow[itm], 2)
            # approximate continuation value
            continuation = fitted(x)
            # boolean index where exercise is beneficial
            ex_idx = itm & (exercise > continuation)
            # update cashflows with early exercises
            cashflow[ex_idx] = exercise[ex_idx]

        yield cashflow, x, fitted, continuation, exercise, ex_idx

    def simulate(self):
        for cashflow, *_ in self._ls_american_option_quadratic_iter(self.X, self.t, self.r, self.strike):
            pass
        return cashflow.mean(axis=0) * np.exp(-self.r * (self.t[1] - self.t[0]))