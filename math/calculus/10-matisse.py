#!/usr/bin/env python3
'''
    this function
    calculates the derivative of a polynomial
'''


def poly_derivative(poly):
    '''
        calculates the derivative of a polynomial
    '''
    if not isinstance(poly, list) or not poly:
        return None
    for coefficient in poly:
        if not isinstance(coefficient, (int, float)):
            return None

    # Calculate derivative
    if len(poly) == 1:
        return [0]
    derivative = [
        coefficient * power
        for power, coefficient in enumerate(poly)
    ][1:]
    return derivative


# Example usage:
if __name__ == "__main__":
    poly = [5, 3, 0, 1]  # Represents the polynomial 5 + 3x + x^3
    print(poly_derivative(poly))
  
