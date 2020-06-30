import numpy as np
from scipy.stats.distributions import norm, lognorm, rv_frozen
from math import sqrt
from scipy.sparse import spdiags


class BrownianMotion:

    def __init__(self):
        pass

    def simulate(self, x0: np.array, n: int, dt: float, mu: float, sigma: float, out=None):
        """
            Generate an instance of Brownian motion (i.e. the Wiener process):

                X(t) = X(0) + N(mu, sigma**2 * t; 0, t)

            where N(a,b; t0, t1) is a normally distributed random variable with mean a and
            variance b.  The parameters t0 and t1 make explicit the statistical
            independence of N on different time intervals; that is, if [t0, t1) and
            [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
            are independent.

            Written as an iteration scheme,

                X(t + dt) = X(t) + N(mu, sigma**2 * dt; t, t+dt)


            If `x0` is an array (or array-like), each value in `x0` is treated as
            an initial condition, and the value returned is a numpy array with one
            more dimension than `x0`.

            Arguments
            ---------
            x0 : float or numpy array (or something that can be converted to a numpy array
                 using numpy.asarray(x0)).
                The initial condition(s) (i.e. position(s)) of the Brownian motion.
            n : int
                The number of steps to take.
            dt : float
                The time step.
            sigma: float
                delta determines the "speed" of the Brownian motion.  The random variable
                of the position at time t, X(t), has a normal distribution whose mean is
                the position at time t=0 and whose variance is delta**2*t.
            out : numpy array or None
                 If `out` is not None, it specifies the array in which to put the
                result.  If `out` is None, a new numpy array is created and returned.

            Returns
            -------
            A numpy array of floats with shape `x0.shape + (n,)`.

            """
        x0 = np.asarray(x0)

        # For each element of x0, generate a sample of n numbers from a
        # normal distribution.
        r = norm.rvs(size=x0.shape + (n,), scale=sigma * sqrt(dt))

        # If `out` was not given, create an output array.
        if out is None:
            out = np.empty(r.shape)

        # This computes the Brownian motion by forming the cumulative sum of
        # the random samples.
        np.cumsum(r, axis=-1, out=out)

        # Add the initial condition.
        out += np.expand_dims(x0, axis=-1)

        return out


class FractionalBrownianMotion:

    def __init__(self, n, hurst, length):
        self.n = n
        self.hurst = hurst
        self.length = length

    def autocovariance(self, k):
        """Autocovariance for fgn."""
        return 0.5 * (np.abs(k - 1) ** (2 * self.hurst) - 2 * np.abs(k) ** (2 * self.hurst) + np.abs(k + 1) ** (2 * self.hurst))

    def simulate(self):
        """Generate a fgn realization using the Cholesky method.
        Uses Cholesky decomposition method (exact method) from:
        Asmussen, S. (1998). Stochastic simulation with a view towards
        stochastic processes. University of Aarhus. Centre for Mathematical
        Physics and Stochastics (MaPhySto)[MPS].
        """

        # create Gaussian Noise

        scale = (1.0 * self.length / self.n) ** self.hurst
        gn = np.random.normal(0.0, 1.0, self.n)

        # If hurst == 1/2 then just return Gaussian noise
        if self.hurst == 0.5:
            return gn * scale
        else:
            # Monte carlo consideration
            # Generate covariance matrix
            g = []
            offset = []
            for i in range(self.n):
                g.append(self.autocovariance(i)*np.ones(self.n))
                offset.append(-i)
            g = np.array(g)
            offset = np.array(offset)
            G = spdiags(g, offset, self.n, self.n).toarray()

            """G = np.zeros([self.n, self.n])
            for i in range(self.n):
                for j in range(i + 1):
                    G[i, j] = self.autocovariance(i - j)"""

            # Cholesky decomposition
            chol = np.linalg.cholesky(G)

            # Generate fgn
            fgn = np.dot(chol, np.array(gn).transpose())
            fgn = np.squeeze(fgn)

            # Scale to interval [0, L]
            scaled_fgn = fgn * scale
            return np.insert(scaled_fgn.cumsum(), [0], 0)

class GeometricBrownianMotion:
    '''Geometric Brownian Motion.(with optional drift).'''
    def __init__(self, mu: float=0.0, sigma: float=1.0):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, t: np.array, n: int, rnd: np.random.RandomState) \
            -> np.array:
        assert t.ndim == 1, 'One dimensional time vector required'
        assert t.size > 0, 'At least one time point is required'
        dt = np.concatenate((t[0:1], np.diff(t)))
        assert (dt >= 0).all(), 'Increasing time vector required'
        # transposed simulation for automatic broadcasting
        dW = (rnd.normal(size=(t.size, n)).T * np.sqrt(dt)).T
        W = np.cumsum(dW, axis=0)
        return np.exp(self.sigma * W.T + (self.mu - self.sigma**2 / 2) * t).T

    def distribution(self, t: float) -> rv_frozen:
        mu_t = (self.mu - self.sigma**2/2) * t
        sigma_t = self.sigma * np.sqrt(t)
        return lognorm(scale=np.exp(mu_t), s=sigma_t)