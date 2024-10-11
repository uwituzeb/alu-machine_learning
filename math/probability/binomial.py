#!/usr/bin/env python3

'''The binomial task '''


class Binomial:
    """
    Represents a binomial distribution.

    Attributes:
        n (int): The number of Bernoulli trials.
        p (float): The probability of a "success".
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initializes the Binomial distribution with data or given n and p.

        Args:
            data: List of the data to be used to estimate the distribution.
            n: The number of Bernoulli trials.
            p: The probability of a "success".
        """
        if n < 1:
            raise ValueError("n must be a positive value")
        if p <= 0 or p >= 1:
            raise ValueError("p must be greater than 0 and less than 1")

        if data is None:
            self.n = n
            self.p = p
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            q = variance / mean
            p = 1 - q
            n = round(mean / p)
            p = mean / n
            self.n = n
            self.p = p

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given k-value.

        Args:
            k: The k-value.

        Returns:
            float: The PMF value for k.
        """
        return ((self.factorial(self.n) /
                 (self.factorial(k) * self.factorial(self.n - k))) *
                (self.p ** k) * ((1 - self.p) ** (self.n - k)))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given k-value.

        Args:
            k: The k-value.

        Returns:
            float: The CDF value for k.
        """
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf

    def factorial(self, n):
        """
        Calculates the factorial of a given integer.

        Args:
            n: The integer.

        Returns:
            int: The factorial of n.
        """
        if n == 0 or n == 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes.

        Args:
            k: The number of successes.

        Returns:
            float: The PMF value for k.
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        n_factorial = 1
        for i in range(1, self.n + 1):
            n_factorial *= i
        k_factorial = 1
        for i in range(1, k + 1):
            k_factorial *= i
        nk_factorial = 1
        for i in range(1, self.n - k + 1):
            nk_factorial *= i
        binomial_co = n_factorial // (k_factorial * nk_factorial)
        pmf = binomial_co * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pmf

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes.

        Args:
            k: The number of successes.

        Returns:
            float: The CDF value for k.
        """
        k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
