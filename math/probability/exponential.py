#!/usr/bin/env python3

'''
This module defines an exponential distribution
'''
e = 2.7182818285


class Exponential:
    '''
    Represents an Exponential distribution.

    Attributes:
        lambtha (float): The rate (λ) of the distribution, representing
        the expected number of occurrences in a given time frame.
    '''

    def __init__(self, data=None, lambtha=1.):
        '''
        Initializes the Exponential distribution with data or a given λ.

        Args:
            data: List of the data to be used to estimate the distribution.
            lambtha: The expected number of occurrences in a given time frame.

        Raises:
            ValueError: If lambtha is not a positive value.
            TypeError: If data is not a list.
            ValueError: If data does not contain multiple values.
        '''
        # Validate lambtha
        if lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        # Handle case when data is None (not provided)
        if data is None:
            self.lambtha = float(lambtha)
        else:
            # Validate data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # Calculate lambtha from data
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        '''
        Calculates the value of the PDF for a given time period.

        Args:
            x (float): The time period.

        Returns:
            float: The PDF value for x.
        '''
        if x < 0:
            return 0
        # Calculate PDF using the formula
        pdf_value = self.lambtha * e ** (-self.lambtha * x)
        return pdf_value

    def cdf(self, x):
        '''
        Calculates the value of the CDF for a given time period.

        Args:
            x (float): The time period.

        Returns:
            float: The CDF value for x.
        '''
        if x < 0:
            return 0
        # Calculate e^(-λx) using the provided value of e
        e_neg_lambda_x = e ** (-self.lambtha * x)
        return 1 - e_neg_lambda_x
