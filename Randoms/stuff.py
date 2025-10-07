import math
import time

'''
Ignore
'''
# n=5
# print(3*n)
# print(math.comb(3 * n, 3, repetition=True) + math.comb(3 * n, 4, repetition=True))

# print((-6.22298564e-6 + 8.37724820e-6)/1)

from math import comb, factorial
import numpy as np
from scipy.integrate import quad

# parameters you can adjust:
n = 20
m = 80
# denominator is (m-n)! which for m=80, n=20 is 60!
den = factorial(m - n)

total = 0
for k in range(n + 1):
    term = (-1) ** k * comb(n, k) * factorial(m - k) / den
    total += term

# If you want an integer result:
total = total

print(total)
print(factorial(m) / (den))



# numericaly integrate sin(x) from 0 to 1

x = np.linspace(0, 1, 100)  # Adjusted to integrate from 0 to 1
def func(x):
    return np.sin(x) + np.cos(x) + np.tan(.000001*x)  # Example function, you can change it as needed

time1 = time.time()
for i in range(10000):
    integral, _ = quad(func, 0, 100)  # Integrate from 0 to 1
time2 = time.time()
print(f"Time taken for integration: {time2 - time1} seconds")


print(f"Numerical integral of func from 0 to 1: {integral}")
