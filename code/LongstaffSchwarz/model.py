import numpy as np
import scipy.stats as stats


class BS:

    def __init__(self, S, K, T, r, sigma, option='call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option = option

    def simulate(self):

        # S: spot price
        # K: strike price
        # T: time to maturity
        # r: interest rate
        # sigma: volatility of underlying asset

        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = (np.log(self.S / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

        if self.option == 'call':
            return self.S * stats.norm.cdf(d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2, 0.0, 1.0)
        if self.option == 'put':
            return self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2, 0.0, 1.0) - self.S * stats.norm.cdf(-d1, 0.0, 1.0)
        else:
            raise NameError("Option undefined")

    def LSM(self, N=10000, paths=10000, order=2, S_defined = 'bs'):
        """
        Longstaff-Schwartz Method for pricing American options

        Arguments
        ---------

        N: int
         number of time steps
        paths: int
         number of generated paths
        order: int
         order of the polynomial for the regression
        """

        if self.option != "put":
            raise ValueError("invalid type. Set 'call' or 'put'")

        dt = self.T / (N - 1)  # time interval
        df = np.exp(-self.r * dt)  # discount factor per time time interval

        if S_defined == 'noS':
            X0 = np.zeros((paths, 1))
            increments = stats.norm.rvs(loc=(self.r - self.sigma ** 2 / 2) * dt, scale=np.sqrt(dt) * self.sigma,
                                 size=(paths, N - 1))
            X = np.concatenate((X0, increments), axis=1).cumsum(1)
            S = self.S * np.exp(X)
        if S_defined == 'bs':
            S = self.S
            H = np.maximum(self.K - S, 0)  # intrinsic values for put option
            V = np.zeros_like(H)  # value matrix
            V[:, -1] = H[:, -1]
            print(V)
        if S_defined == 'fractional':
            H = self.S
            S = self.S
            V = np.zeros_like(H)  # value matrix
            V[:, -1] = H[:, -1]
        # Valuation by LS Method
        for t in range(N - 2, 0, -1):
            good_paths = H[:, t] > 0
            rg = np.polyfit(S[good_paths, t], V[good_paths, t + 1] * df, order)  # polynomial regression
            C = np.polyval(rg, S[good_paths, t])  # evaluation of regression

            exercise = np.zeros(len(good_paths), dtype=bool)
            exercise[good_paths] = H[good_paths, t] > C

            V[exercise, t] = H[exercise, t]
            V[exercise, t + 1:] = 0
            discount_path = (V[:, t] == 0)
            V[discount_path, t] = V[discount_path, t + 1] * df

        V0 = np.mean(V[:, 1]) * df  #
        bound = np.max(V[:, 1])
        return V0, bound
